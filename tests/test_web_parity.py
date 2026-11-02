"""Parity of the in-browser model/tokenizer (web/blind/lib/{slm,tok}.js) with the PyTorch policy and HF tokenizer.

  PYTHONPATH=. python -m pytest tests/test_web_parity.py -q -s      (or: python tests/test_web_parity.py)

Uses the exported assets in web/blind/assets (`python -m jxlart.export_web`) and runs the JS side through
tests/web_parity.mjs under bun (node if bun is missing). The torch reference runs on CPU with the policy's weights
rounded to float16 and back, so the logit comparison isolates the JS implementation error; the effect of the
float16 export itself is reported separately. Training programs come from JXLART_WEB_DATA (default
D:/jxl-art/data/synth-v2).
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from functools import cache
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jxlart import model as M  # noqa: E402
from jxlart.corpus import PROMPT, format_example  # noqa: E402
from jxlart.export_web import _sha256  # noqa: E402

ASSETS = ROOT / "web" / "blind" / "assets"
RUNNER = ROOT / "tests" / "web_parity.mjs"
DATA = Path(os.environ.get("JXLART_WEB_DATA", "D:/jxl-art/data/synth-v2"))
JS = shutil.which("bun") or shutil.which("node")


def check(name: str, value: float, limit: float) -> None:
    ok = value < limit
    print(f"{'PASS' if ok else 'FAIL'} {name}: {value:.3g} (limit {limit:g})")
    assert ok, name


def run_js(*args: str) -> str:
    assert JS, "neither bun nor node on PATH"
    r = subprocess.run([JS, str(RUNNER), *args], capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    assert r.returncode == 0, f"{JS} {RUNNER.name} {args[0]} failed:\n{r.stderr}"
    return r.stdout


@cache
def tok():
    return M.load_tokenizer()


@cache
def manifest(name: str) -> dict:
    return json.loads((ASSETS / f"{name}.json").read_text())


@cache
def torch_policy(name: str, fp16: bool) -> torch.nn.Module:
    model, _ = M.load_policy(ROOT / manifest(name)["source"]["file"], device="cpu")
    if fp16:
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(p.half().float())
    return M.freeze(model)


@cache
def training_texts(n: int = 2000) -> list[str]:
    out = []
    for shard in sorted(DATA.glob("shard_*.jsonl")):
        with open(shard, encoding="utf-8") as f:
            for line in f:
                out.append(json.loads(line)["text"])
                if len(out) == n:
                    return out
    assert out, f"no training data under {DATA}"
    return out


def encode(text: str) -> list[int]:
    t = tok()
    return t(format_example(text))["input_ids"] + [t.eos_token_id]


@torch.no_grad()
def torch_logits(model, seqs: list[list[int]]) -> list[np.ndarray]:
    return [model(torch.tensor([s]))[0][0].numpy() for s in seqs]


def js_logits(name: str, seqs: list[list[int]]) -> list[np.ndarray]:
    V = manifest(name)["config"]["vocab_size"]
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "seqs.json").write_text(json.dumps(seqs))
        info = json.loads(run_js("logits", str(ASSETS / f"{name}.json"), str(Path(d) / "seqs.json"),
                                 str(Path(d) / "out.f32")))
        flat = np.fromfile(Path(d) / "out.f32", dtype="<f4")
    print(f"js logits ({Path(JS).stem}): {info['tokens']} positions in {info['seconds']:.2f}s "
          f"({info['tok_per_s']:.0f} tok/s, full-vocab head at every position)")
    out, o = [], 0
    for s in seqs:
        out.append(flat[o:o + len(s) * V].reshape(len(s), V))
        o += len(s) * V
    assert o == flat.size
    return out


def js_decode(seqs: list[list[int]]) -> list[str]:
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "seqs.json").write_text(json.dumps(seqs))
        run_js("decode", str(ASSETS / "tokenizer.json"), str(Path(d) / "seqs.json"), str(Path(d) / "out.json"))
        return json.loads((Path(d) / "out.json").read_text(encoding="utf-8"))


class _CPUDecoder(M.StaticDecoder):
    """StaticDecoder step without the CUDA graph, so `M.sample` runs unchanged on CPU."""

    @torch.no_grad()
    def step(self, tokens, pos):
        self.tok.copy_(tokens.view(-1, 1))
        self.pos.fill_(pos)
        return self._step()


def torch_sample(model, n: int, max_new: int, seed: int = 0) -> list[list[int]]:
    prompt = tok()(PROMPT)["input_ids"]
    max_len = min(model.ctx, len(prompt) + max_new)
    model.__dict__.setdefault("_decoders", {})[(n, max_len)] = _CPUDecoder(model, n, max_len)
    torch.manual_seed(seed)
    return M.sample(model, prompt, n, max_new, tok().eos_token_id, temperature=1.0)


def logit_seqs() -> list[list[int]]:
    """Prompt alone, the longest of the first 2000 training programs, three random ones, and training programs
    packed back to back up to the full context (RoPE and the KV cache at every position)."""
    texts = training_texts()
    ctx = manifest("tuned")["config"]["max_position_embeddings"]
    rng = random.Random(0)
    packed = []
    for t in texts:
        if len(packed) >= ctx:
            break
        packed += encode(t)
    seqs = [encode(max(texts, key=len))[:ctx]] + [encode(t) for t in rng.sample(texts, 3)] + [packed[:ctx]]
    return [tok()(PROMPT)["input_ids"]] + seqs


def test_assets() -> None:
    t = json.loads((ASSETS / "tokenizer.json").read_text())
    assert t["promptIds"] == tok()(PROMPT)["input_ids"] and t["eosId"] == tok().eos_token_id
    for name in ("base", "tuned"):
        m = manifest(name)
        src = ROOT / m["source"]["file"]
        assert _sha256(src) == m["source"]["sha256"], f"{name}: {src} changed since export"
        n = sum(int(np.prod(x["shape"])) for x in m["tensors"])
        assert (ASSETS / m["bin"]).stat().st_size == m["bin_bytes"] == 2 * n
        print(f"PASS {name}: {src.name} sha256 matches, {n:,} params")


def test_logits() -> None:
    seqs = logit_seqs()
    print("sequence lengths:", [len(s) for s in seqs])
    for name in ("tuned", "base"):
        ref = torch_logits(torch_policy(name, fp16=True), seqs)
        got = js_logits(name, seqs)
        diff = max(float(np.abs(g.astype(np.float64) - r).max()) for g, r in zip(got, ref))
        check(f"{name}: JS vs torch (fp16-rounded weights) logits max|diff|", diff, 5e-4)
        n_pos = sum(len(s) for s in seqs)
        am = sum(int((g.argmax(-1) == r.argmax(-1)).sum()) for g, r in zip(got, ref))
        top5, near_ties = 0, 0
        for g, r in zip(got, ref):
            for gi, ri in zip(g, r):
                tg, tr = np.argsort(-gi)[:5], np.argsort(-ri)[:5]
                if np.array_equal(tg, tr):
                    top5 += 1
                elif np.abs(ri[tg] - ri[tr]).max() <= 2 * diff:
                    near_ties += 1   # order flip between reference logits closer than the measured error
        print(f"{name}: argmax equal at {am}/{n_pos} positions, top-5 (ordered) equal at {top5}/{n_pos}"
              f" (+{near_ties} where the order differs only between reference logits within 2x the max diff)")
        assert am == n_pos, f"{name}: argmax mismatch"
        assert top5 + near_ties == n_pos, f"{name}: top-5 mismatch"
        full = torch_logits(torch_policy(name, fp16=False), seqs)
        d16 = max(float(np.abs(f - r).max()) for f, r in zip(full, ref))
        am16 = sum(int((f.argmax(-1) == r.argmax(-1)).sum()) for f, r in zip(full, ref))
        print(f"{name}: [info] torch fp32 weights vs fp16-rounded weights: logits max|diff| {d16:.3g}, "
              f"argmax equal at {am16}/{n_pos}")


def test_decode() -> None:
    t = tok()
    eos = t.eos_token_id

    def compare(label: str, seqs: list[list[int]]) -> None:
        want = [t.decode([i for i in s if i != eos]) for s in seqs]
        got = js_decode(seqs)
        bad = [i for i, (g, w) in enumerate(zip(got, want)) if g != w]
        print(f"{'PASS' if not bad else 'FAIL'} decode {label}: {len(seqs) - len(bad)}/{len(seqs)} equal")
        assert not bad, f"{label}: first mismatch #{bad[0]}: {got[bad[0]]!r} != {want[bad[0]]!r}"

    # max_new 256 instead of the game's 700: the CPU StaticDecoder attends over (and copies) the whole cache every
    # step, and samples are far shorter than 256 tokens anyway (the eos rate is printed).
    sampled = [s for i, name in enumerate(("tuned", "base"))
               for s in torch_sample(torch_policy(name, fp16=False), 100, max_new=256, seed=i)]
    print(f"torch CPU samples: mean length {np.mean([len(s) for s in sampled]):.1f}, "
          f"eos rate {np.mean([s[-1] == eos for s in sampled]):.2f}")
    compare("200 torch samples (100 tuned + 100 base)", sampled)
    compare("500 training programs", [encode(x) for x in training_texts()[:500]])
    rng = random.Random(1)
    rand = [[rng.randrange(t.vocab_size) for _ in range(rng.randrange(1, 60))] for _ in range(300)]
    compare("300 random id sequences (len tokens, specials, broken UTF-8)", rand)


def main() -> None:
    print("runner:", JS, "| assets:", ASSETS)
    test_assets()
    test_logits()
    test_decode()
    print("ALL PASSED")


if __name__ == "__main__":
    main()
