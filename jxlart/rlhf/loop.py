"""Online RLHF with a human: label continuously while the model trains in the background.

One long-lived process serves the labeling UI and runs three loops:

  taste model   refit after every pick (CPU, Bradley-Terry on CLIP features); rescoring the
                pool so the next pair and the "newest" grid use your latest taste
  cycle         alternate: you label --picks-per-batch pairs drawn from a fresh batch sampled from the
                current model; labeling pauses while RL runs --steps-per-batch KL-regularised
                policy-gradient steps; then a new batch is sampled from the updated model. Training
                only starts once the taste model passes the gate (>= --rl-min-labels picks and
                >= --rl-min-acc held-out accuracy); otherwise the label batch is extended.

  python -m jxlart.rlhf.loop --run runs/rlhf-v2 --sft runs/sft-gfree/best_policy.safetensors   # new run
  python -m jxlart.rlhf.loop --run runs/rlhf-v2                                               # resume

Durable state (under --run):
  items.jsonl, items/    every candidate shown (PNG preview + .jxl), tagged with the policy step
  prefs.jsonl            every pick, append-only (undo records pop the previous pick)
  rm/fit_N.npz           every taste-model version; rm_fits.jsonl one line per refit (accuracy, picks)
  policy/online.pt       current policy + optimizer + step (rewritten after every RL step)
  policy/step_N.pt       snapshots every --snapshot-every steps (rollback targets; step_00000 = SFT)
  rollouts.jsonl         every sample ever drawn (broken ones too): recipe, outcome, reward, taste
                         score, and the .jxl itself (base64) for valid ones
  rl_metrics.jsonl       per-step RL statistics; rl_samples/ contact sheets
"step" on items and rollouts is the policy step that generated the sample.
Stop: STOP file / Ctrl+C. The current RL step finishes and is saved; picks are never lost.
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import base64
import json
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch  # taste-model fits (CPU) and CLIP features; the policy lives in self.B
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image

from .. import compiler, model as M
from ..novelty import Reference
from ..corpus import PROMPT
from ..runctl import Run, atomic_write_json
from .reward import BTReward, Featurizer, cross_val_accuracy, load_prefs
from .ui import PAGE

PENALTY = {"truncated": -3.0, "parse": -3.0, "compile": -3.0, "limits": -2.0, "decode": -2.0,
           "degenerate": -1.5}
CHOICE_Y = {"a": 1.0, "b": 0.0, "tie": 0.5, "bad": 0.5}
CLIP_DIM = 512  # leading columns of Featurizer output: L2-normalised CLIP image embedding


def mean_pairwise_sim(E: np.ndarray) -> float:
    """Mean cosine similarity over distinct pairs of normalised rows (1.0 = all identical)."""
    n = len(E)
    S = E @ E.T
    return float((S.sum() - np.trace(S)) / (n * (n - 1))) if n > 1 else 1.0


def crowding(E: np.ndarray, k: int) -> np.ndarray:
    """Per row: mean similarity to its k most similar other rows."""
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)
    k = min(k, len(E) - 1)
    return np.sort(S, 1)[:, -k:].mean(1)


def farthest_first(E: np.ndarray, n: int) -> list[int]:
    """Greedy max-min selection of n mutually dissimilar rows, starting from a random one."""
    if len(E) <= n:
        return list(range(len(E)))
    pick = [random.randrange(len(E))]
    best = E @ E[pick[0]]
    while len(pick) < n:
        best[pick] = np.inf
        i = int(np.argmin(best))  # least similar to everything chosen so far
        pick.append(i)
        best = np.maximum(best, E @ E[i])
    return pick


class Loop:
    def __init__(self, args, run: Run):
        self.a, self.run = args, run
        self.dir = Path(args.run)
        for sub in ("items", "rm", "policy", "rl_samples"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.items: dict[str, dict] = {}
        p = self.dir / "items.jsonl"
        if p.exists():
            for line in p.open(encoding="utf-8"):
                try:
                    it = json.loads(line)
                except json.JSONDecodeError:
                    break
                self.items[it["id"]] = it
        self.phashes = {it["phash"] for it in self.items.values()}
        # next numeric id; items may be an imported subset, so count != max id
        self.next_item = 1 + max((int(i[1:]) for i in self.items if i.startswith("c") and i[1:].isdigit()), default=-1)
        self.prefs: list[dict] = []
        for rec in load_prefs(self.dir / "prefs.jsonl"):
            if rec.get("undo"):
                if self.prefs:
                    self.prefs.pop()
            else:
                self.prefs.append(rec)
        self.pref_version = 0
        self.prefs_changed = threading.Event()
        if self.prefs:
            self.prefs_changed.set()
        self.feats: dict[str, np.ndarray] = {}
        self.rm_snapshot: tuple[BTReward, float, float, dict] | None = None
        self.rl_hist: list[dict] = []
        p = self.dir / "rl_metrics.jsonl"
        if p.exists():
            self.rl_hist = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        p = self.dir / "rm_fits.jsonl"
        self.n_fits = sum(1 for l in p.open(encoding="utf-8") if l.strip()) if p.exists() else 0
        self.rollback_req: int | None = None
        # (snapshot restored, step counter at that moment); snapshots in (to, at] are an abandoned branch
        self.rollbacks: list[tuple[int, int]] = []
        p = self.dir / "events.jsonl"
        if p.exists():
            for line in p.open(encoding="utf-8"):
                if '"rollback"' in line:
                    e = json.loads(line)
                    if e.get("event") == "rollback":
                        self.rollbacks.append((e["to_snapshot"], e["from_step"]))
        self.fitted_version = -1
        self.activity = "starting"

        self.tok = M.load_tokenizer()
        self.prompt = self.tok(PROMPT)["input_ids"]
        self.feat = Featurizer()
        self.gallery = Reference.load_or_build(self.feat)  # known JXL art: penalised if reproduced
        # policy backend: jxlart.model (PyTorch/CUDA) or jxlart.mlx_model (Apple MLX); same file format
        self.B = importlib.import_module("jxlart.mlx_model" if args.backend == "mlx" else "jxlart.model")
        pol = self.dir / "policy"
        step0 = pol / "step_00000.safetensors"
        if not step0.exists():  # new run: snapshot 0 is the SFT policy and the KL reference
            if not args.sft:
                raise SystemExit("new run: pass --sft <policy.safetensors>")
            shutil.copyfile(args.sft, step0)
        self.ref = self.B.freeze(self.B.load_policy(step0)[0])
        online = pol / "online.safetensors"
        self.opt_file = pol / ("online_opt" + self.B.OPT_SUFFIX)
        self.policy, meta = self.B.load_policy(online if online.exists() else step0)
        self.opt = self.B.new_optimizer(self.policy, args.rl_lr)
        self.step = meta.get("step", 0) if online.exists() else 0
        if online.exists():
            if self.opt_file.exists():
                self.B.load_optimizer(self.opt, self.opt_file)
            else:  # e.g. continuing on another device/backend: weights carry over, Adam state restarts
                run.event("optimizer_reset", reason=f"no {self.opt_file.name}", backend=args.backend)
            run.event("resume", step=self.step, items=len(self.items), picks=len(self.prefs), backend=args.backend)
        p = self.dir / "cycle.json"
        self.cycle = json.loads(p.read_text()) if p.exists() else {
            "phase": "label", "start_picks": len(self.prefs), "start_step": self.step, "batch_step": None}

    # ------------------------------------------------------------ pool helpers
    def _counts(self) -> Counter:
        cnt = Counter()
        for p in self.prefs:
            cnt[p["a"]] += 1
            cnt[p["b"]] += 1
        return cnt

    def rollout(self, n: int, level: str):
        gens = self.B.sample(self.policy, self.prompt, n, self.a.max_new, self.tok.eos_token_id,
                             temperature=self.a.temperature)
        texts = [self.tok.decode([t for t in g if t != self.tok.eos_token_id]) for g in gens]
        truncated = [self.tok.eos_token_id not in g for g in gens]
        self.run.level(level, n, "renders")

        def work(t):
            r = compiler.compile_tree(t, timeout=8, max_bytes=compiler.MAX_JXL_BYTES)
            self.run.advance(level)
            return r

        with ThreadPoolExecutor(self.a.workers) as ex:
            res = list(ex.map(work, texts))
        self.run.drop_level(level)
        return gens, texts, truncated, res

    def known_art(self, results: list[compiler.Result], E: np.ndarray) -> list[dict]:
        """Closest gallery piece for each valid result, and a copy penalty in [0, 1].

        Recipe similarity is the primary signal (distinct gallery pieces: median 0.62 to their
        nearest neighbour; reproduced pieces: 0.9-1.0). Image similarity only flags exact
        visual copies: CLIP rates distinct gallery pieces ~0.93 alike, so lower values are style.
        """
        a = self.a
        img_sim, img_idx = self.gallery.nearest_image(E)
        out = []
        for r, s, ii in zip(results, img_sim, img_idx):
            rs, ri = self.gallery.nearest_recipe(r.canonical.split())
            pen = max(np.clip((rs - a.copy_recipe) / (1 - a.copy_recipe), 0, 1),
                      np.clip((s - a.copy_image) / (1 - a.copy_image), 0, 1))
            title = self.gallery.titles[ri] if ri >= 0 and rs >= a.copy_recipe else self.gallery.titles[ii]
            out.append({"title": title, "recipe_sim": round(float(rs), 3), "image_sim": round(float(s), 3),
                        "penalty": round(float(pen), 3),
                        "drop": bool(rs >= a.copy_drop_recipe or s >= a.copy_drop_image)})
        return out

    def archive_rollout(self, kind: str, step: int, texts, trunc, res, rewards=None, rm_raw=None,
                        known: dict | None = None) -> None:
        """Append every sample of a batch (broken ones too) to rollouts.jsonl.

        Valid samples carry their .jxl (<= 256 bytes, base64): the artwork itself, so previews
        can be re-rendered later without storing PNGs.
        """
        ts = time.time()
        with open(self.dir / "rollouts.jsonl", "a", encoding="utf-8") as f:
            for k, (text, t, r) in enumerate(zip(texts, trunc, res)):
                rec = {"ts": ts, "kind": kind, "step": step, "k": k, "outcome": self.outcome(t, r),
                       "error": r.error or None, "text": text, "bytes": r.nbytes or None}
                if r.ok:
                    rec.update(jxl_b64=base64.b64encode(r.jxl).decode(), size=list(r.size), phash=r.stats["phash"])
                if rewards is not None:
                    rec["reward"] = float(rewards[k])
                if rm_raw is not None and k in rm_raw:
                    rec["rm"] = rm_raw[k]
                if known is not None and k in known:
                    rec["known"] = known[k]
                f.write(json.dumps(rec) + "\n")

    @staticmethod
    def outcome(trunc: bool, r: compiler.Result) -> str:
        if trunc:
            return "truncated"
        if not r.ok:
            return r.stage
        return "degenerate" if compiler.is_degenerate(r.stats) else "ok"

    def featurize(self, results: list[compiler.Result]) -> np.ndarray:
        return self.feat([r.thumb.resize((224, 224)) for r in results], [r.stats for r in results])

    def featurize_missing(self) -> None:
        with self.lock:
            missing = [i for i in self.items if i not in self.feats]
        for k in range(0, len(missing), 64):
            chunk = missing[k:k + 64]
            imgs = [Image.open(self.dir / "items" / f"{i}.png").convert("RGB").resize((224, 224)) for i in chunk]
            X = self.feat(imgs, [self.items[i]["stats"] for i in chunk])
            with self.lock:
                self.feats.update(zip(chunk, X))

    def add_items(self, results: list[compiler.Result], X: np.ndarray, source: str, step: int,
                  batch: int | None = None, known: list[dict] | None = None) -> int:
        snap = self.rm_snapshot
        scores = snap[0].score(X) if snap is not None and len(results) else None
        added = 0
        with self.lock, open(self.dir / "items.jsonl", "a", encoding="utf-8") as f:
            for k, r in enumerate(results):
                ph = r.stats["phash"]
                if ph in self.phashes:
                    continue
                self.phashes.add(ph)
                iid = f"c{self.next_item:05d}"
                self.next_item += 1
                r.thumb.save(self.dir / "items" / f"{iid}.png")
                (self.dir / "items" / f"{iid}.jxl").write_bytes(r.jxl)
                it = {"id": iid, "step": step, "batch": batch, "source": source, "text": r.canonical,
                      "jxl_bytes": r.nbytes, "size": list(r.size),
                      "stats": {k2: v for k2, v in r.stats.items() if k2 != "phash"}, "phash": ph}
                if known is not None:
                    it["known"] = {k2: v for k2, v in known[k].items() if k2 != "drop"}
                f.write(json.dumps(it) + "\n")
                if scores is not None:
                    it["rm"] = float(scores[0][k])
                self.items[iid] = it
                self.feats[iid] = X[k]
                added += 1
        return added

    # ------------------------------------------------------------ taste model (own thread, CPU only)
    def rm_loop(self) -> None:
        while not self.run.should_stop():
            if not self.prefs_changed.wait(1.0):
                continue
            self.prefs_changed.clear()
            with self.lock:
                version = self.pref_version
            if version == self.fitted_version:
                continue
            try:
                self.fit_rm(version)
                self.fitted_version = version
            except Exception as e:  # keep labeling alive; the error is visible in events/log
                self.run.event("rm_error", error=f"{type(e).__name__}: {e}")

    def fit_rm(self, version: int) -> None:
        t0 = time.perf_counter()
        with self.lock:
            prefs = list(self.prefs)
            feats = dict(self.feats)
        prefs = [p for p in prefs if p["a"] in feats and p["b"] in feats]
        if sum(CHOICE_Y[p["choice"]] != 0.5 for p in prefs) < 2:
            return
        ids = sorted({p["a"] for p in prefs} | {p["b"] for p in prefs})
        index = {i: k for k, i in enumerate(ids)}
        X = np.stack([feats[i] for i in ids])
        pairs = [(index[p["a"]], index[p["b"]], CHOICE_Y[p["choice"]]) for p in prefs]
        cv = cross_val_accuracy(X, pairs, self.a.rm_l2)
        rm = BTReward(n_models=1, l2=self.a.rm_l2)  # one full-data fit; ensemble spread is unused
        fit = rm.fit(X, pairs)
        all_ids = list(feats)
        s, _ = rm.score(np.stack([feats[i] for i in all_ids]))
        mu, sd = float(s.mean()), float(s.std() + 1e-6)
        meta = {"picks": len(prefs), "items": len(ids), **fit, "cv_acc": cv, "norm_mu": mu, "norm_sd": sd,
                "fit_ms": round((time.perf_counter() - t0) * 1000), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "rl_step": self.step}
        with self.lock:
            for i, v in zip(all_ids, s):
                self.items[i]["rm"] = float(v)
            self.rm_snapshot = (rm, mu, sd, meta)
        meta["fit"] = self.n_fits
        rm.save(self.dir / "rm" / f"fit_{self.n_fits:05d}.npz")
        self.n_fits += 1
        with open(self.dir / "rm_fits.jsonl", "a") as f:
            f.write(json.dumps(meta) + "\n")
        self.run.metric(rm_picks=len(prefs), rm_cv_acc=cv if cv is not None else "n/a", rm_fit_ms=meta["fit_ms"])

    def gate(self) -> tuple[bool, str]:
        n = len(self.prefs)
        if n < self.a.rl_min_labels:
            return False, f"collecting picks: {n}/{self.a.rl_min_labels} before training starts"
        snap = self.rm_snapshot
        if snap is None:
            return False, "fitting taste model"
        cv = snap[3]["cv_acc"]
        if cv is None or cv < self.a.rl_min_acc:
            shown = "n/a" if cv is None else f"{cv:.0%}"
            return False, f"paused: taste model held-out accuracy {shown} < {self.a.rl_min_acc:.0%}, keep picking"
        return True, "training"

    # ------------------------------------------------------------ worker (owns the GPU)
    def generate_batch(self) -> None:
        """Sample a label batch: draw ~--batch-oversample x --batch-items valid images from the current
        model, then keep the --batch-items most mutually different ones (farthest-first in CLIP space)."""
        a = self.a
        self.activity = f"sampling a new batch from model step {self.step}"
        self.run.set_state("running", self.activity)
        want = a.batch_items * a.batch_oversample
        cands, seen, oc = [], set(self.phashes), Counter()
        for _ in range(a.max_gen_batches):
            if len(cands) >= want or self.run.should_stop():
                break
            gens, texts, trunc, res = self.rollout(a.gen_n, "batch renders")
            self.archive_rollout("fresh", self.step, texts, trunc, res)
            oc.update(self.outcome(t, r) for t, r in zip(trunc, res))
            for t, r in zip(trunc, res):
                if self.outcome(t, r) == "ok" and r.stats["phash"] not in seen:
                    seen.add(r.stats["phash"])
                    cands.append(r)
        if self.run.should_stop():
            return
        added, sim, dropped, bid = 0, None, 0, self.cycle.get("batches_made", 0) + 1
        if cands:
            X = self.featurize(cands)
            known = self.known_art(cands, X[:, :CLIP_DIM])
            keep = [i for i, kn in enumerate(known) if not kn["drop"]]  # near-copies of known art never shown
            dropped = len(cands) - len(keep)
            if keep:
                sub = farthest_first(X[keep, :CLIP_DIM], a.batch_items)
                pick = [keep[i] for i in sub]
                sim = mean_pairwise_sim(X[pick, :CLIP_DIM]) if len(pick) > 1 else None
                added = self.add_items([cands[i] for i in pick], X[pick], "fresh", self.step, batch=bid,
                                       known=[known[i] for i in pick])
        self.run.event("generated", step=self.step, candidates=len(cands), dropped_known_art=dropped,
                       added=added, batch_sim=sim, outcomes=dict(oc))
        if added < 2:  # nothing labelable (e.g. all near-copies): sample again
            self.set_cycle(batches_made=bid)
            return
        self.set_cycle(batch_id=bid, batches_made=bid, batch_step=self.step, start_picks=len(self.prefs), note=None)

    def set_cycle(self, **kw) -> None:
        with self.lock:
            self.cycle.update(kw)
            atomic_write_json(self.dir / "cycle.json", self.cycle)

    def rl_step_once(self) -> None:
        a = self.a
        rm, mu, sd, meta = self.rm_snapshot
        self.activity = "training"
        self.run.set_state("running", f"RL step {self.step + 1}")
        gens, texts, trunc, res = self.rollout(a.rl_batch, "rollout renders")
        outcomes = [self.outcome(t, r) for t, r in zip(trunc, res)]
        rewards = np.array([PENALTY.get(o, 0.0) for o in outcomes], dtype=np.float64)
        ok = [k for k, o in enumerate(outcomes) if o == "ok"]
        X_ok = self.featurize([res[k] for k in ok]) if ok else None
        rm_raw = {}
        if ok:
            raw = rm.score(X_ok)[0]
            rm_raw = {k: float(v) for k, v in zip(ok, raw)}
            z = np.clip((raw - mu) / sd, -3, 3)
            E = X_ok[:, :CLIP_DIM]
            known = self.known_art([res[k] for k in ok], E)
            if len(ok) >= 3:  # variety: penalise samples crowded by look-alikes in this batch
                crowd = crowding(E, a.div_k)
                div = (crowd - crowd.mean()) / (crowd.std() + 1e-6)
                batch_sim = mean_pairwise_sim(E)
            else:
                div, batch_sim = np.zeros(len(ok)), None
            seen = Counter()
            for j, (k, v) in enumerate(zip(ok, z)):
                ph = res[k].stats["phash"]
                rewards[k] = (v - a.dup_penalty * seen[ph] - a.div_coef * div[j]
                              - a.copy_coef * known[j]["penalty"])
                seen[ph] += 1
        else:
            batch_sim = None
        B = len(rewards)
        adv = rewards - (rewards.sum() - rewards) / (B - 1)
        adv = adv / (adv.std() + 1e-6)
        seqs = [self.prompt + g for g in gens]
        upd = self.B.rl_update(self.policy, self.ref, self.opt, seqs, len(self.prompt), self.tok.pad_token_id,
                               adv, a.kl_beta)
        self.archive_rollout("rl", self.step, texts, trunc, res, rewards, rm_raw,
                             known={k: kn for k, kn in zip(ok, known)} if ok else None)
        self.step += 1
        self.save_online()
        if self.step % a.snapshot_every == 0:
            self.B.save_policy(self.policy, self.dir / "policy" / f"step_{self.step:05d}.safetensors",
                               {"step": self.step})
        rec = {"step": self.step, "ts": time.time(), "reward_mean": float(rewards.mean()),
               "rm_z_mean_ok": float(np.mean([rewards[k] for k in ok])) if ok else None,
               "ok_frac": len(ok) / B, "unique_ok": len({res[k].stats["phash"] for k in ok}),
               **upd,
               "rm_picks": meta["picks"], "batch_sim": batch_sim,
               "copy_frac": float(np.mean([kn["penalty"] > 0 for kn in known])) if ok else None,
               "outcomes": dict(Counter(outcomes))}
        with open(self.dir / "rl_metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        with self.lock:
            self.rl_hist.append(rec)
        self.run.advance("rl steps")
        self.run.metric(**{k: v for k, v in rec.items() if isinstance(v, float) and k != "ts"})
        if self.step % a.sheet_every == 0:
            self._sheet([res[k] for k in ok], self.dir / "rl_samples" / f"step{self.step:05d}.png")

    def save_online(self) -> None:
        self.B.save_policy(self.policy, self.dir / "policy" / "online.safetensors", {"step": self.step})
        self.B.save_optimizer(self.opt, self.opt_file)

    def do_rollback(self, target: int) -> None:
        self.B.load_weights(self.policy, self.dir / "policy" / f"step_{target:05d}.safetensors")
        self.opt = self.B.new_optimizer(self.policy, self.a.rl_lr)
        self.run.event("rollback", from_step=self.step, to_snapshot=target)
        self.rollbacks.append((target, self.step))
        self.save_online()

    @staticmethod
    def _sheet(results, path: Path) -> None:
        if not results:
            return
        cols = 8
        sheet = Image.new("RGB", (128 * cols, 128 * ((len(results) + cols - 1) // cols)), (40, 40, 40))
        for i, r in enumerate(results):
            sheet.paste(r.thumb.resize((128, 128)), ((i % cols) * 128, (i // cols) * 128))
        sheet.save(path)

    def _wait(self, msg: str) -> None:
        if self.activity != msg:
            self.activity = msg
            self.run.set_state("waiting", msg)
        time.sleep(0.5)

    def run_forever(self) -> None:
        self.featurize_missing()
        threading.Thread(target=self.rm_loop, daemon=True).start()
        self.run.level("rl steps", None, "steps", done=self.step)
        while not self.run.should_stop():
            if self.rollback_req is not None:
                self.do_rollback(self.rollback_req)
                self.rollback_req = None
                self.set_cycle(phase="label", start_picks=len(self.prefs), batch_step=None, batch_id=None)
            if self.a.max_steps is not None and self.step >= self.a.max_steps:
                self.run.event("max_steps_reached", step=self.step)
                return
            c = self.cycle
            done = len(self.prefs) - c["start_picks"]
            if c["phase"] == "train":
                if self.step - c["start_step"] >= self.a.steps_per_batch:
                    self.run.event("training_done", step=self.step)
                    self.set_cycle(phase="label", start_picks=len(self.prefs))
                elif self.rm_snapshot is None or self.fitted_version != self.pref_version:
                    self._wait("fitting the taste model before training")  # e.g. resumed mid-training
                else:
                    self.rl_step_once()
            elif c.get("batch_step") != self.step or c.get("batch_id") is None:
                self.generate_batch()
            elif done >= self.a.picks_per_batch and self.fitted_version == self.pref_version:
                ok, reason = self.gate()
                if ok:
                    self.run.event("training_started", step=self.step, picks=len(self.prefs))
                    self.set_cycle(phase="train", start_step=self.step, note=None)
                else:  # taste model not trustworthy yet: keep labeling this batch
                    self.run.event("batch_extended", reason=reason)
                    self.set_cycle(start_picks=len(self.prefs), note=reason)
            else:
                self._wait(f"labeling: {max(done, 0)}/{self.a.picks_per_batch} picks for model step {self.step}")
            self.run.metric(picks=len(self.prefs), pool=len(self.items), phase=self.cycle["phase"])
        if self.rollback_req is not None:  # requested just before a stop: still honour it
            self.do_rollback(self.rollback_req)
            self.rollback_req = None

    # ------------------------------------------------------------ labeling API
    def next_pair(self) -> dict:
        with self.lock:
            if self.cycle["phase"] == "train":
                return {"none": True, "reason": "training on your picks; new images appear when it finishes"}
            if len(self.prefs) - self.cycle["start_picks"] >= self.a.picks_per_batch:
                return {"none": True, "reason": "batch complete; training starts in a moment"}
            items = list(self.items.values())
            bid = self.cycle.get("batch_id")
            batch = [it for it in items if bid is not None and it.get("batch") == bid]
            if len(batch) < 2:
                return {"none": True, "reason": "sampling a new batch from the model"}
            cnt = self._counts()
            done = {frozenset((p["a"], p["b"])) for p in self.prefs}
            recent = batch
            for a_it in sorted(recent, key=lambda it: (cnt[it["id"]], random.random())):
                cands = [b for b in recent if b["id"] != a_it["id"] and frozenset((a_it["id"], b["id"])) not in done]
                if not cands:
                    cands = [b for b in items if b["id"] != a_it["id"] and frozenset((a_it["id"], b["id"])) not in done]
                if not cands:
                    continue
                if "rm" in a_it:  # informative: least-compared, then closest predicted taste
                    cands.sort(key=lambda b: (cnt[b["id"]], abs(b.get("rm", 0.0) - a_it["rm"])))
                else:
                    cands.sort(key=lambda b: cnt[b["id"]])
                pair = [a_it, random.choice(cands[:3])]
                random.shuffle(pair)
                return {"a": self._pub(pair[0]), "b": self._pub(pair[1])}
            return {"none": True, "reason": "every pair has been labeled"}

    @staticmethod
    def _pub(it: dict) -> dict:
        return {"id": it["id"], "text": it["text"], "bytes": it["jxl_bytes"], "size": it["size"],
                "step": it.get("step", 0), "rm": it.get("rm"), "known": it.get("known")}

    def add_pref(self, body: dict) -> dict:
        with self.lock:
            if not body.get("undo") and (self.cycle["phase"] == "train"
                                         or len(self.prefs) - self.cycle["start_picks"] >= self.a.picks_per_batch):
                raise HTTPException(409, "batch complete; labeling resumes after training")
            if body.get("undo"):
                if not self.prefs:
                    return {"ok": False}
                rec = {"undo": True, "ts": time.time()}
                self.prefs.pop()
            else:
                if body.get("choice") not in CHOICE_Y or body.get("a") not in self.items or body.get("b") not in self.items:
                    raise HTTPException(400, "bad preference")
                rec = {"a": body["a"], "b": body["b"], "choice": body["choice"], "rl_step": self.step,
                       "ts": time.time()}
                self.prefs.append(rec)
            with open(self.dir / "prefs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            self.pref_version += 1
        self.prefs_changed.set()
        return {"ok": True}

    def ui_state(self) -> dict:
        ok, reason = self.gate()
        snap = self.rm_snapshot
        with self.lock:
            c = dict(self.cycle)
            last = self.rl_hist[-1] if self.rl_hist else None
            ts = [r["ts"] for r in self.rl_hist[-11:] if "ts" in r]
            step_s = (ts[-1] - ts[0]) / (len(ts) - 1) if len(ts) > 1 else None
            train_done = self.step - c["start_step"] if c["phase"] == "train" else 0
            return {"picks": len(self.prefs), "pool": len(self.items), "step": self.step,
                    "phase": c["phase"], "activity": self.activity, "gate_ok": ok, "gate": reason,
                    "note": c.get("note"),
                    "batch_done": max(0, len(self.prefs) - c["start_picks"]), "batch_size": self.a.picks_per_batch,
                    "batch_ready": c.get("batch_step") == self.step and c.get("batch_id") is not None,
                    "train_done": train_done, "train_total": self.a.steps_per_batch,
                    "eta_s": step_s * (self.a.steps_per_batch - train_done) if step_s and c["phase"] == "train" else None,
                    "rm": {k: snap[3][k] for k in ("picks", "cv_acc", "train_acc", "fit_ms")} if snap else None,
                    "last": {k: last.get(k) for k in ("reward_mean", "ok_frac", "kl", "batch_sim", "copy_frac")} if last else None,
                    "snapshots": [{"step": s, "old_branch": any(to < s <= at for to, at in self.rollbacks)}
                                  for s in sorted(int(p.stem.split("_")[1])
                                                  for p in (self.dir / "policy").glob("step_*.safetensors"))],
                    "rollback": {"to": self.rollbacks[-1][0], "at": self.rollbacks[-1][1]} if self.rollbacks else None}

    def batch_items(self) -> list[dict]:
        """The current label batch, best first by the taste model."""
        with self.lock:
            bid = self.cycle.get("batch_id")
            items = [it for it in self.items.values() if bid is not None and it.get("batch") == bid]
        items.sort(key=lambda it: -it.get("rm", 0.0))
        return [self._pub(it) for it in items]

    def metrics(self) -> list[dict]:
        with self.lock:
            return [{k: r.get(k) for k in ("step", "reward_mean", "kl", "ok_frac", "batch_sim")} for r in self.rl_hist[-300:]]

    def request_rollback(self, body: dict) -> dict:
        target = int(body.get("step", -1))
        if not (self.dir / "policy" / f"step_{target:05d}.safetensors").exists():
            raise HTTPException(404, f"no snapshot for step {target}")
        self.rollback_req = target
        return {"ok": True}


def make_app(loop: Loop) -> FastAPI:
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get("/api/state")
    def state():
        return loop.ui_state()

    @app.get("/api/pair")
    def pair():
        return loop.next_pair()

    @app.post("/api/pref")
    def pref(body: dict):
        return loop.add_pref(body)

    @app.get("/api/recent")
    def recent():
        return loop.batch_items()

    @app.get("/api/metrics")
    def metrics():
        return loop.metrics()

    @app.post("/api/rollback")
    def rollback(body: dict):
        return loop.request_rollback(body)

    @app.get("/items/{name}")
    def item_file(name: str):
        p = (loop.dir / "items" / name).resolve()
        if p.parent != (loop.dir / "items").resolve() or not p.exists():
            raise HTTPException(404)
        return FileResponse(p)

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sft", help="SFT policy .safetensors; needed only to start a new run (becomes snapshot 0)")
    ap.add_argument("--backend", choices=["torch", "mlx"], default="mlx" if sys.platform == "darwin" else "torch",
                    help="policy backend: torch (CUDA) or mlx (Apple silicon)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--picks-per-batch", type=int, default=25, help="picks per label batch")
    ap.add_argument("--steps-per-batch", type=int, default=20, help="RL steps between label batches")
    ap.add_argument("--batch-items", type=int, default=32, help="images sampled for each label batch")
    ap.add_argument("--batch-oversample", type=int, default=3,
                    help="draw N x --batch-items valid images, keep the most varied --batch-items")
    ap.add_argument("--max-gen-batches", type=int, default=6, help="sampling attempts to fill a batch")
    ap.add_argument("--rl-min-labels", type=int, default=30, help="picks before RL may start")
    ap.add_argument("--rl-min-acc", type=float, default=0.65, help="held-out taste-model accuracy needed for RL")
    ap.add_argument("--rl-batch", type=int, default=64)
    ap.add_argument("--rl-lr", type=float, default=3e-5)
    ap.add_argument("--kl-beta", type=float, default=0.05)
    ap.add_argument("--dup-penalty", type=float, default=0.5)
    ap.add_argument("--div-coef", type=float, default=1.0,
                    help="variety penalty: reward minus coef x (within-batch CLIP crowding, standardised)")
    ap.add_argument("--div-k", type=int, default=5, help="neighbours used for crowding")
    ap.add_argument("--copy-coef", type=float, default=2.0, help="reward penalty for reproducing known gallery art")
    ap.add_argument("--copy-recipe", type=float, default=0.8, help="recipe similarity where the copy penalty starts")
    ap.add_argument("--copy-image", type=float, default=0.97, help="image similarity where the copy penalty starts")
    ap.add_argument("--copy-drop-recipe", type=float, default=0.9, help="never show candidates at/above this")
    ap.add_argument("--copy-drop-image", type=float, default=0.98, help="never show candidates at/above this")
    ap.add_argument("--rm-l2", type=float, default=10.0)
    ap.add_argument("--gen-n", type=int, default=64, help="samples per sampling attempt")
    ap.add_argument("--snapshot-every", type=int, default=5)
    ap.add_argument("--sheet-every", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=700)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-live", action="store_true")
    args = ap.parse_args()
    # Taste-model fits are tiny CPU problems: intra-op threads fighting the renderers made a
    # 46-pick refit take 20 s instead of ~0.2 s. GPU work is unaffected.
    torch.set_num_threads(1)
    run = Run(args.run, "rlhf-online", vars(args), live=not args.no_live)
    with run:
        loop = Loop(args, run)
        server = uvicorn.Server(uvicorn.Config(make_app(loop), host="127.0.0.1", port=args.port,
                                               log_level="warning"))
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        for _ in range(100):  # the labeling page is the point of this process: no page, no run
            if server.started or not t.is_alive():
                break
            time.sleep(0.1)
        if not server.started:
            raise RuntimeError(f"labeling UI could not start on port {args.port} (in use?); pass --port N")
        run.event("ui", url=f"http://127.0.0.1:{args.port}")
        loop.run_forever()
        server.should_exit = True


if __name__ == "__main__":
    code = 0
    try:
        main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if e.code is not None and not isinstance(e.code, int):
            print(e.code, file=sys.stderr)
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    # Everything is saved by now (Run.__exit__ wrote the final status/event). Skip interpreter
    # teardown: with MLX and PyTorch in one process, native thread pools abort ("terminate called
    # without an active exception") while daemon threads (UI, taste model) are still alive.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
