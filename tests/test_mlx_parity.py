"""MLX backend parity against PyTorch reference outputs (tests/fixtures/parity.npz, see make_mlx_fixture.py).

  PYTHONPATH=. python tests/test_mlx_parity.py

Policy/ref files default to the rlhf-v2 run the fixture was made from; override with
JXLART_PARITY_POLICY / JXLART_PARITY_REF.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jxlart import mlx_model as M  # noqa: E402


def _default(name: str) -> str:
    for base in ("/mnt/d/jxl-art/runs/rlhf-v2/policy", "D:/jxl-art/runs/rlhf-v2/policy"):
        if Path(base, name).exists():
            return str(Path(base, name))
    return str(Path("/mnt/d/jxl-art/runs/rlhf-v2/policy", name))


POLICY = os.environ.get("JXLART_PARITY_POLICY", _default("step_00100.safetensors"))
REF = os.environ.get("JXLART_PARITY_REF", _default("step_00000.safetensors"))
FIX = np.load(ROOT / "tests" / "fixtures" / "parity.npz")
IDS = FIX["ids"]
N_PROMPT, PAD = int(FIX["n_prompt"]), int(FIX["pad_id"])
SEQS = [IDS[i, :n].tolist() for i, n in enumerate(FIX["seq_lens"])]
EOS = SEQS[0][-1]


def check(name: str, value: float, limit: float) -> None:
    ok = value < limit
    print(f"{'PASS' if ok else 'FAIL'} {name}: {value:.3g} (limit {limit:g})")
    assert ok, name


def maxdiff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def params(policy) -> dict:
    return {k: np.array(v) for k, v in tree_flatten(policy.parameters())}


def decode_logits(policy, ids: np.ndarray, prefill: int = 0) -> np.ndarray:
    """Logits for every position of ids (B,L) via the KV-cached decode step (optionally after a prefill)."""
    B, L = ids.shape
    step = M._decode_fn(policy)
    Ks, Vs = M._empty_cache(policy, B, L)
    out = []
    if prefill:
        kv = []
        out.append(np.array(policy(mx.array(ids[:, :prefill]), kv_out=kv)))
        for i, (k, v) in enumerate(kv):
            Ks[i][:, :, :prefill] = k
            Vs[i][:, :, :prefill] = v
    for t in range(prefill, L):
        lg, Ks, Vs = step(mx.array(ids[:, t].astype(np.int32)), mx.array(t, dtype=mx.int32), Ks, Vs)
        out.append(np.array(lg)[:, None])
    return np.concatenate(out, 1)


def test_forward_and_logprobs() -> None:
    policy, _ = M.load_policy(POLICY)
    logits = np.array(policy(mx.array(IDS.astype(np.int32))))
    check("forward logits max|diff|", maxdiff(logits, FIX["logits"]), 2e-3)
    lp_sum, tok, mask = M.sequence_logprobs(policy, SEQS, N_PROMPT, PAD)
    assert np.array_equal(np.array(mask), FIX["mask"]), "mask mismatch"
    assert tok.shape == mask.shape == (len(SEQS), IDS.shape[1] - 1)
    check("sequence_logprobs sum max|diff|", maxdiff(lp_sum, FIX["logp_sum"]), 2e-3)
    check("kv-cache decode vs full forward max|diff|", maxdiff(decode_logits(policy, IDS), logits), 1e-3)
    check("prefill+kv-cache decode vs full forward max|diff|",
          maxdiff(decode_logits(policy, IDS, prefill=N_PROMPT), logits), 1e-3)


def test_rl_update() -> None:
    policy, _ = M.load_policy(POLICY)
    ref = M.freeze(M.load_policy(REF)[0])
    probe = FIX["probe"].astype(np.int32)
    decode_logits(policy, probe)  # compile the decode step before the update: it must see new weights
    opt = M.new_optimizer(policy, float(FIX["lr"]))
    stats = M.rl_update(policy, ref, opt, SEQS, N_PROMPT, PAD, FIX["adv"], float(FIX["kl_beta"]))
    want = json.loads(str(FIX["stats"]))
    print("stats mlx  ", stats)
    print("stats torch", want)
    for k in ("pg_loss", "kl", "grad_norm", "mean_len"):
        assert isinstance(stats[k], float)
        check(f"rl_update {k} rel diff", abs(stats[k] - want[k]) / max(abs(want[k]), 1e-12), 1e-3)
    post = np.array(policy(mx.array(probe)))
    check("post-update probe logits max|diff|", maxdiff(post, FIX["post_update_logits"]), 2e-3)
    check("post-update kv-cache decode vs torch max|diff|",
          maxdiff(decode_logits(policy, probe), FIX["post_update_logits"]), 2e-3)
    assert tree_flatten(ref.trainable_parameters()) == [], "freeze() left trainable parameters"


def test_policy_io(tmp: Path) -> None:
    policy, meta = M.load_policy(POLICY)
    f = tmp / "policy.safetensors"
    M.save_policy(policy, f, {"step": 7, "note": "mlx"})
    again, meta2 = M.load_policy(f)
    a, b = params(policy), params(again)
    assert meta2 == {"step": 7, "note": "mlx"}
    assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) and a[k].dtype == b[k].dtype for k in a)
    print("PASS save_policy -> load_policy bit-identical", f"({len(a)} tensors)")
    with safe_open(POLICY, "numpy") as t, safe_open(str(f), "numpy") as m:
        tk, mk = sorted(t.keys()), sorted(m.keys())
        assert tk == mk, set(tk) ^ set(mk)
        for k in tk:
            ts, ms = t.get_slice(k), m.get_slice(k)
            assert ts.get_shape() == ms.get_shape() and ts.get_dtype() == ms.get_dtype(), k
        tmd, mmd = t.metadata(), m.metadata()
        assert tmd.keys() == mmd.keys(), (tmd.keys(), mmd.keys())
        assert tmd["format"] == mmd["format"] and json.loads(tmd["config"]) == json.loads(mmd["config"])
    assert "head.weight" not in mk
    print(f"PASS MLX-saved file matches torch file: {len(tk)} names/shapes/dtypes, metadata keys {sorted(mmd)}")
    # load_weights replaces weights in place on an existing policy
    other, _ = M.load_policy(REF)
    M.load_weights(other, f)
    assert all(np.array_equal(v, params(other)[k]) for k, v in a.items())
    print("PASS load_weights in place")


def test_optimizer_resume(tmp: Path) -> None:
    ref = M.freeze(M.load_policy(REF)[0])
    args = (SEQS, N_PROMPT, PAD, FIX["adv"], float(FIX["kl_beta"]))
    lr = float(FIX["lr"])
    pa, _ = M.load_policy(POLICY)
    oa = M.new_optimizer(pa, lr)
    M.rl_update(pa, ref, oa, *args)
    pf, of = tmp / "online.safetensors", tmp / ("online_opt" + M.OPT_SUFFIX)
    M.save_policy(pa, pf, {"step": 1})
    M.save_optimizer(oa, of)
    sa = M.rl_update(pa, ref, oa, *args)
    pb, _ = M.load_policy(pf)
    ob = M.new_optimizer(pb, lr)
    M.load_optimizer(ob, of)
    assert int(ob.state["step"]) == 1
    sb = M.rl_update(pb, ref, ob, *args)
    a, b = params(pa), params(pb)
    assert sa == sb, (sa, sb)
    assert all(np.array_equal(a[k], b[k]) for k in a), "weights differ after optimizer resume"
    assert int(oa.state["step"]) == int(ob.state["step"]) == 2
    print("PASS optimizer save -> load -> rl_update identical to uninterrupted run")


def test_sample() -> None:
    policy, _ = M.load_policy(POLICY)
    prompt = SEQS[0][:N_PROMPT]
    limit = min(40, policy.ctx - len(prompt))

    def check_rows(rows, eos, n):
        assert isinstance(rows, list) and len(rows) == n
        for r in rows:
            assert isinstance(r, list) and all(isinstance(t, int) for t in r)
            assert 1 <= len(r) <= limit
            assert eos not in r[:-1], "tokens after eos"
            assert r[-1] == eos or len(r) == limit, "row stopped early without eos"
        return [len(r) for r in rows], sum(r[-1] == eos for r in rows)

    rows = M.sample(policy, prompt, n=8, max_new=40, eos_id=EOS, temperature=1.0)
    lens, n_eos = check_rows(rows, EOS, 8)
    print(f"PASS sample(n=8, max_new=40): lengths {lens}, eos rows {n_eos}")
    # the real eos rarely fits in 40 tokens: use the commonest sampled token as eos to exercise truncation
    common = int(np.bincount(np.concatenate(rows)).argmax())
    lens, n_eos = check_rows(M.sample(policy, prompt, n=8, max_new=40, eos_id=common), common, 8)
    assert n_eos > 0 and min(lens) < limit, (lens, n_eos)
    print(f"PASS sample(n=8, max_new=40, eos={common}): lengths {lens}, eos rows {n_eos}")
    # near-greedy sampling must follow the full-forward argmax path
    greedy = M.sample(policy, prompt, n=2, max_new=24, eos_id=EOS, temperature=1e-6)
    seq = list(prompt)
    for _ in range(24):
        nxt = int(np.argmax(np.array(policy(mx.array([seq]))[0, -1])))
        seq.append(nxt)
        if nxt == EOS:
            break
    assert greedy[0] == greedy[1] == seq[len(prompt):], (greedy, seq[len(prompt):])
    print("PASS greedy sample == full-forward argmax rollout")
    # context limit
    long = M.sample(policy, prompt, n=2, max_new=10**6, eos_id=-1)
    assert [len(r) for r in long] == [policy.ctx - len(prompt)] * 2
    print(f"PASS context-limited rows have length ctx - prompt = {policy.ctx - len(prompt)}")


def bench_sample() -> None:
    policy, _ = M.load_policy(POLICY)
    prompt = SEQS[0][:N_PROMPT]
    M.sample(policy, prompt, n=64, max_new=16, eos_id=-1)  # compile / warm up
    t0 = time.perf_counter()
    rows = M.sample(policy, prompt, n=64, max_new=128, eos_id=-1)
    dt = time.perf_counter() - t0
    steps = len(rows[0])
    print(f"BENCH sample n=64 prompt={len(prompt)} max_new=128: {steps} steps in {dt:.2f}s "
          f"= {1000 * dt / steps:.2f} ms/token-step ({64 * steps / dt:.0f} tok/s)")


def main() -> None:
    print("mlx", mx.__version__, mx.default_device(), "| policy", POLICY, "| ref", REF)
    with tempfile.TemporaryDirectory() as d:
        test_forward_and_logprobs()
        test_rl_update()
        test_policy_io(Path(d))
        test_optimizer_resume(Path(d))
        test_sample()
    if "--no-bench" not in sys.argv:
        bench_sample()
    print("ALL PASSED")


if __name__ == "__main__":
    main()
