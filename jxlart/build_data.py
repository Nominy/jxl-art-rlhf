"""Build the gallery-free SFT corpus: random grammar programs and mutations of accepted ones.

No gallery program is copied, mutated or used as a crossover donor; the gallery only sets the
sampling priors (how often each header/property/predictor appears). Candidates whose recipe is
>= --known-art-max similar to any gallery recipe are rejected.

Resumable at single-sample granularity: accepted samples are appended (and fsynced)
to the current shard as they arrive; on restart, existing lines are reloaded,
dedup state is rebuilt and generation continues with a fresh RNG stream.

  python -m jxlart.build_data --run runs/data-v2 --out data/synth-v2 --target 24000
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from . import compiler, dsl
from .corpus import load_seeds
from .novelty import recipe_sim
from .runctl import Run
from .synth import Generator

MAX_PER_PHASH = 3


def load_existing(out: Path) -> tuple[list[Path], int, set[str], Counter, deque]:
    shards = sorted(out.glob("shard_*.jsonl"))
    n, texts, phashes, pool = 0, set(), Counter(), deque(maxlen=5000)
    for sh in shards:
        good = []
        with open(sh, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    break  # torn final line from a hard kill
                good.append(line)
                texts.add(hashlib.sha1(rec["text"].encode()).hexdigest())
                phashes[rec["phash"]] += 1
                pool.append(rec["text"])
        raw = sh.read_text(encoding="utf-8")
        if "".join(good) != raw:  # truncate torn tail so appends stay valid JSONL
            sh.write_text("".join(good), encoding="utf-8")
        n += len(good)
    return shards, n, texts, phashes, pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=40000)
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-bytes", type=int, default=compiler.MAX_JXL_BYTES)
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--mix", default="rand=0.4,evo=0.6", help="rand: fresh grammar program; evo: mutate an accepted one")
    ap.add_argument("--known-art-max", type=float, default=0.8, help="reject recipes this similar to a gallery recipe")
    ap.add_argument("--no-live", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mix = {k: float(v) for k, v in (kv.split("=") for kv in args.mix.split(","))}
    if set(mix) - {"rand", "evo"}:
        raise SystemExit(f"unknown --mix kinds {set(mix) - {'rand', 'evo'}}")
    run = Run(args.run, "build_data", vars(args), live=not args.no_live)
    with run:
        run.set_state("running", "loading")
        seeds = load_seeds()
        gallery_recipes = [s.text.split() for s in seeds]
        shards, n_done, texts, phashes, pool = load_existing(out)
        run.event("resume" if n_done else "fresh", accepted=n_done, shards=len(shards), seeds=len(seeds))
        # Fresh RNG stream per (re)start so a resume never replays earlier candidates.
        rng = random.Random(f"{args.seed}:{n_done}:{time.time_ns()}")
        gen = Generator([s.program for s in seeds], rng,
                        donors=lambda: dsl.parse(rng.choice(pool)) if pool else gen.random_program())

        def known_art(text: str) -> bool:
            toks = text.split()
            return any(recipe_sim(toks, g) >= args.known_art_max for g in gallery_recipes)

        total = run.level("accepted", args.target, "samples", done=n_done)
        shard_idx = n_done // args.shard_size
        shard_lv = run.level("shard", args.shard_size, "samples", done=n_done % args.shard_size)
        run.level("candidates", None, "cands")
        shard_f = open(out / f"shard_{shard_idx:05d}.jsonl", "a", encoding="utf-8")
        run.set_state("running", f"shard {shard_idx}")

        def make_candidate() -> tuple[str, str]:
            kind = rng.choices(list(mix), weights=list(mix.values()))[0]
            if kind == "evo" and pool:
                return "evo", dsl.to_text(gen.mutate(dsl.parse(rng.choice(pool))))
            return "rand", dsl.to_text(gen.random_program())

        def work(cand: tuple[str, str]):
            origin, text = cand
            return origin, compiler.compile_tree(text, timeout=args.timeout, max_bytes=args.max_bytes)

        inflight = set()
        with ThreadPoolExecutor(args.workers) as ex:
            while True:
                stopping = run.should_stop()
                while not stopping and total.done < args.target and len(inflight) < args.workers * 2:
                    inflight.add(ex.submit(work, make_candidate()))
                if not inflight:
                    break
                done, inflight = wait(inflight, timeout=1.0, return_when=FIRST_COMPLETED)
                for fut in done:
                    origin, r = fut.result()
                    run.advance("candidates")
                    if not r.ok:
                        run.count(f"rej_{r.stage}")
                        continue
                    if compiler.is_degenerate(r.stats):
                        run.count("rej_degenerate")
                        continue
                    h = hashlib.sha1(r.canonical.encode()).hexdigest()
                    if h in texts:
                        run.count("rej_dup_text")
                        continue
                    if phashes[r.stats["phash"]] >= MAX_PER_PHASH:
                        run.count("rej_dup_image")
                        continue
                    if known_art(r.canonical):
                        run.count("rej_known_art")
                        continue
                    if total.done >= args.target:
                        continue
                    texts.add(h)
                    phashes[r.stats["phash"]] += 1
                    pool.append(r.canonical)
                    rec = {"text": r.canonical, "origin": origin, "jxl_bytes": r.nbytes,
                           "size": r.size, "compile_ms": round(r.compile_ms, 1),
                           "phash": r.stats["phash"],
                           "stats": {k: v for k, v in r.stats.items() if k != "phash"},
                           "jxl_b64": base64.b64encode(r.jxl).decode()}
                    shard_f.write(json.dumps(rec) + "\n")
                    shard_f.flush()
                    os.fsync(shard_f.fileno())
                    run.count("accepted_" + origin.split(":")[0])
                    run.advance("accepted")
                    run.advance("shard")
                    if shard_lv.done >= args.shard_size and total.done < args.target:
                        shard_f.close()
                        run.event("checkpoint", shard=shard_idx, accepted=total.done)
                        shard_idx += 1
                        shard_f = open(out / f"shard_{shard_idx:05d}.jsonl", "a", encoding="utf-8")
                        shard_lv = run.level("shard", args.shard_size, "samples")
                        run.set_state("running", f"shard {shard_idx}")
                cands = run.levels["candidates"].done
                run.metric(accept_rate=(total.done - total.base_done) / max(1, cands))
                if total.done >= args.target and not inflight:
                    break
        shard_f.close()
        run.event("checkpoint", shard=shard_idx, accepted=total.done)


if __name__ == "__main__":
    main()
