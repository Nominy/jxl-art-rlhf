"""Supervised fine-tuning of Supra2.5-Pico (or the same architecture from scratch) on tree DSL.

Resumable: ckpt/last.pt holds model+optimizer+schedule+data position; rerunning the
same command continues from it. Graceful stop (STOP file / Ctrl+C) saves a checkpoint.

  python -m jxlart.sft --run runs/sft-v1 --data data/synth-v1 --init pretrained
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import compiler, model as M
from .corpus import PROMPT, format_example, load_seeds
from .runctl import Run, atomic_write_json

GALLERY_CACHE = compiler.ROOT / "data" / "gallery_seeds.jsonl"


def gallery_records() -> list[dict]:
    """Gallery trees that compile to <= MAX_JXL_BYTES with our libjxl (cached)."""
    if GALLERY_CACHE.exists():
        return [json.loads(l) for l in GALLERY_CACHE.open(encoding="utf-8")]
    seeds = load_seeds()
    with ThreadPoolExecutor(10) as ex:
        res = list(ex.map(lambda s: compiler.compile_tree(s.text, timeout=30, render=False), seeds))
    recs = [{"text": s.text, "title": s.title, "file": s.file, "jxl_bytes": r.nbytes}
            for s, r in zip(seeds, res) if r.ok and r.nbytes <= compiler.MAX_JXL_BYTES]
    GALLERY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    GALLERY_CACHE.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    return recs


def encode(tok, texts: list[str], max_len: int) -> list[list[int]]:
    out = []
    for t in texts:
        ids = tok(format_example(t))["input_ids"] + [tok.eos_token_id]
        if len(ids) <= max_len:
            out.append(ids)
    return out


def make_batches(seqs: list[list[int]], batch_tokens: int, rng: random.Random) -> list[list[int]]:
    order = sorted(range(len(seqs)), key=lambda i: (len(seqs[i]), rng.random()))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = len(seqs[i])
        if cur and max(cur_max, L) * (len(cur) + 1) > batch_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


def collate(seqs, idxs, n_prompt, pad_id, device):
    L = max(len(seqs[i]) for i in idxs)
    x = torch.full((len(idxs), L), pad_id, dtype=torch.long)
    y = torch.full((len(idxs), L), -100, dtype=torch.long)
    for r, i in enumerate(idxs):
        s = torch.tensor(seqs[i])
        x[r, : len(s)] = s
        y[r, n_prompt: len(s)] = s[n_prompt:]
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def lm_loss(model, x, y):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(x)
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1), ignore_index=-100)


@torch.no_grad()
def evaluate(model, seqs, n_prompt, pad_id, device, batch_tokens) -> float:
    model.eval()
    tot, n = 0.0, 0
    for idxs in make_batches(seqs, batch_tokens, random.Random(0)):
        x, y = collate(seqs, idxs, n_prompt, pad_id, device)
        ntok = int((y[:, 1:] != -100).sum())
        tot += float(lm_loss(model, x, y)) * ntok
        n += ntok
    model.train()
    return tot / max(1, n)


def sample_eval(model, tok, n: int, temperature: float, out_png: Path | None, max_new: int = 700,
                save_dir: Path | None = None, backend=M) -> dict:
    """Sample n programs with `backend` (jxlart.model or jxlart.mlx_model), compile, and score validity."""
    prompt = tok(PROMPT)["input_ids"]
    gens = backend.sample(model, prompt, n, max_new, tok.eos_token_id, temperature=temperature)
    texts = [tok.decode([t for t in g if t != tok.eos_token_id]) for g in gens]
    with ThreadPoolExecutor(10) as ex:
        res = list(ex.map(lambda t: compiler.compile_tree(t, timeout=8, max_bytes=compiler.MAX_JXL_BYTES), texts))
    stages = [r.stage for r in res]
    good = [r for r in res if r.ok and not compiler.is_degenerate(r.stats)]
    m = {
        "eos_rate": float(np.mean([tok.eos_token_id in g for g in gens])),
        "parse_ok": 1 - stages.count("parse") / n,
        "valid_le256": sum(r.ok for r in res) / n,
        "good": len(good) / n,
        "unique_images": len({r.stats["phash"] for r in good}) / n,
        "mean_bytes": float(np.mean([r.nbytes for r in res if r.ok])) if any(r.ok for r in res) else 0.0,
        "compile_ok": float(np.mean([s not in ("parse", "compile") for s in stages])),
    }
    if save_dir:
        for i, r in enumerate(good):
            (save_dir / f"{i:03d}.jxl").write_bytes(r.jxl)
            (save_dir / f"{i:03d}.txt").write_text(r.canonical, encoding="utf-8")
            r.thumb.save(save_dir / f"{i:03d}.png")
        m["texts"] = [r.canonical for r in good]
    if out_png and good:
        cols = 8
        sheet = Image.new("RGB", (128 * cols, 128 * math.ceil(len(good) / cols)), (40, 40, 40))
        for i, r in enumerate(good):
            sheet.paste(r.thumb.resize((128, 128)), ((i % cols) * 128, (i // cols) * 128))
        out_png.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out_png)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True, help="directory of build_data shards")
    ap.add_argument("--init", default="pretrained", help="pretrained | scratch")
    ap.add_argument("--epochs", type=float, default=8)
    ap.add_argument("--batch-tokens", type=int, default=24576)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--gallery-repeat", type=int, default=10)
    ap.add_argument("--gallery-val", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--sample-n", type=int, default=48)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-live", action="store_true")
    args = ap.parse_args()
    device = "cuda"
    run_dir = Path(args.run)
    ck_dir = run_dir / "ckpt"
    ck_dir.mkdir(parents=True, exist_ok=True)
    run = Run(run_dir, "sft", vars(args), live=not args.no_live)
    with run:
        run.set_state("running", "data")
        tok = M.load_tokenizer()
        n_prompt = len(tok(PROMPT)["input_ids"])
        probe = tok(format_example("Bitdepth 8\n- N 1\n"))["input_ids"]
        assert probe[:n_prompt] == tok(PROMPT)["input_ids"], "prompt tokenization is not a prefix"
        synth = [json.loads(l)["text"] for f in sorted(Path(args.data).glob("shard_*.jsonl"))
                 for l in f.open(encoding="utf-8") if l.strip()]
        gal = [r["text"] for r in gallery_records()]
        rng = random.Random(args.seed)
        rng.shuffle(gal)
        gal_val, gal_train = gal[: args.gallery_val], gal[args.gallery_val:]
        rng.shuffle(synth)
        n_val = max(64, len(synth) // 50)
        syn_val, syn_train = synth[:n_val], synth[n_val:]
        train = encode(tok, syn_train + gal_train * args.gallery_repeat, args.max_len)
        val_syn, val_gal = encode(tok, syn_val, args.max_len), encode(tok, gal_val, args.max_len)
        ntok = sum(len(s) for s in train)
        run.event("data", synth=len(synth), gallery_train=len(gal_train), gallery_val=len(gal_val),
                  train_seqs=len(train), train_tokens=ntok)
        steps_per_epoch = len(make_batches(train, args.batch_tokens, random.Random(0)))
        total_steps = int(args.epochs * steps_per_epoch)

        torch.manual_seed(args.seed)
        if args.init == "pretrained":
            model = M.load_pretrained()
        elif args.init == "scratch":
            model = M.build(M.base_config())
        else:
            model, _ = M.load_policy(Path(args.init))
        model = model.to(device).train()
        decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
        no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                                 {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))

        def lr_at(step):
            if step < args.warmup:
                return args.lr * (step + 1) / args.warmup
            p = min(1.0, (step - args.warmup) / max(1, total_steps - args.warmup))
            return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))

        step, epoch, bidx, best = 0, 0, 0, float("inf")
        last = ck_dir / "last.pt"
        if last.exists():
            ck = torch.load(last, map_location="cpu", weights_only=False)
            if ck["data_fingerprint"] != [len(train), ntok]:
                raise RuntimeError("checkpoint was trained on a different dataset; use a new --run dir")
            model.load_state_dict(ck["model"])
            model.head.weight = model.embed.weight
            opt.load_state_dict(ck["opt"])
            step, epoch, bidx, best = ck["step"], ck["epoch"], ck["bidx"], ck["best"]
            run.event("resume", step=step, epoch=epoch, bidx=bidx, best=best)

        def save(tag="last"):
            path = ck_dir / f"{tag}.pt"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "epoch": epoch,
                        "bidx": bidx, "best": best, "data_fingerprint": [len(train), ntok],
                        "args": vars(args)}, path.with_suffix(".tmp"))
            path.with_suffix(".tmp").replace(path)
            run.event("checkpoint", step=step, epoch=epoch, bidx=bidx)

        run.level("steps", total_steps, "steps", done=step)
        run.level("epoch", math.ceil(args.epochs), "epochs", done=epoch)
        batches = make_batches(train, args.batch_tokens, random.Random(f"{args.seed}:{epoch}"))
        run.level("batches in epoch", len(batches), "batches", done=bidx)
        run.set_state("running", "train")
        t_tok, t0 = 0, time.time()
        while step < total_steps and not run.should_stop():
            if bidx >= len(batches):
                epoch, bidx = epoch + 1, 0
                run.advance("epoch")
                batches = make_batches(train, args.batch_tokens, random.Random(f"{args.seed}:{epoch}"))
                run.level("batches in epoch", len(batches), "batches")
            x, y = collate(train, batches[bidx], n_prompt, tok.pad_token_id, device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            loss = lm_loss(model, x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step, bidx = step + 1, bidx + 1
            t_tok += int((y != -100).sum())
            run.advance("steps")
            run.advance("batches in epoch")
            if step % 10 == 0:
                run.metric(loss=float(loss), grad_norm=float(gn), lr=lr_at(step),
                           tok_per_s=t_tok / (time.time() - t0))
                with open(run_dir / "train_metrics.jsonl", "a") as f:
                    f.write(json.dumps({"step": step, "loss": float(loss), "lr": lr_at(step)}) + "\n")
            if step % args.eval_every == 0 or step == total_steps:
                run.set_state("running", "eval")
                vs = evaluate(model, val_syn, n_prompt, tok.pad_token_id, device, args.batch_tokens)
                vg = evaluate(model, val_gal, n_prompt, tok.pad_token_id, device, args.batch_tokens)
                rec = {"step": step, "val_synth": vs, "val_gallery": vg}
                if step % args.sample_every == 0 or step == total_steps:
                    run.set_state("running", "sampling")
                    rec.update(sample_eval(model, tok, args.sample_n, 1.0, run_dir / "samples" / f"step_{step:06d}.png"))
                    model.train()
                run.metric(**{k: v for k, v in rec.items() if k != "step"})
                run.event("eval", **rec)
                with open(run_dir / "eval_metrics.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                if vs < best:
                    best = vs
                    M.save_policy(model, run_dir / "best_policy.safetensors", {"step": step, **rec, "init": args.init})
                run.set_state("running", "train")
            if step % args.ckpt_every == 0:
                save()
        save()
        M.save_policy(model, run_dir / "final_policy.safetensors", {"step": step, "init": args.init})
        atomic_write_json(run_dir / "summary.json", {"step": step, "total_steps": total_steps, "best_val_synth": best})


if __name__ == "__main__":
    main()
