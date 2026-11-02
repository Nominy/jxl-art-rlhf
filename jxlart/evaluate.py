"""Evaluate policy checkpoints: uniqueness, interestingness, quality, and closeness to known JXL art.

  python -m jxlart.evaluate --run runs/rlhf-v2 --steps 0,120,400,online --out reports/eval

Per checkpoint it samples --n programs and compares the valid images against the SFT training data
(gallery-free, --train) and the jpegxl.info gallery (human artists):

  uniqueness       distinct images, mean pairwise CLIP similarity, novelty vs training data and gallery
  interestingness  PNG bytes of the render (visual complexity), colours, edges, PNG bytes per JXL byte
  quality          the run's taste model (your picks) and an independent CLIP zero-shot judge
  known art        nearest gallery piece by image and by recipe; "convergent" = looks alike, built differently
  techniques       which JXL coding tools programs use, vs the human gallery and the training data
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from . import compiler, dsl, model as M
from .corpus import PROMPT, load_seeds
from .novelty import REF_DIR, Reference, recipe_sim
from .rlhf.reward import BTReward, Featurizer
from .runctl import Run, atomic_write_json

GOOD_PROMPTS = ["a beautiful abstract digital artwork", "stunning generative art", "an intricate colorful pattern",
                "a striking minimalist composition"]
BAD_PROMPTS = ["random noise", "a boring flat image", "a glitchy corrupted image", "a blank screen"]
NEIGHBOUR_PROPS = {"W-NW", "N-NE", "W+N-NW", "NW-N", "N-NN", "W-WW", "W-WW-NW+NWW"}


# ---------------------------------------------------------------- techniques
def techniques(prog: dsl.Program) -> dict[str, bool]:
    f0 = prog.frames[0]
    hv = lambda n, d=None: dsl.header_value(f0, n, d)  # noqa: E731
    nodes = [n for f in prog.frames for n in dsl.tree_nodes(f.tree)]
    splits = [n for n in nodes if isinstance(n, dsl.Split)]
    leaves = [n for n in nodes if isinstance(n, dsl.Leaf)]
    props = {s.prop for s in splits}
    preds = {lf.predictor for lf in leaves}

    def depth(n):
        return 0 if isinstance(n, dsl.Leaf) else 1 + max(depth(n.left), depth(n.right))

    bd = hv("Bitdepth", (8,))[0]
    w, h = hv("Width", (1024,))[0], hv("Height", (1024,))[0]
    return {
        "colour transform (RCT)": hv("RCT", (0,))[0] != 0,
        "Squeeze (multi-resolution)": hv("Squeeze") is not None,
        "Upsample (smooth upscaling)": hv("Upsample", (1,))[0] > 1,
        "DeltaPalette": hv("DeltaPalette") is not None,
        "Gaborish/EPF filters": hv("Gaborish") is not None or hv("EPF", (0,))[0] > 0,
        "XYB colour space": hv("XYB") is not None,
        "Orientation flip/rotate": hv("Orientation", (1,))[0] != 1,
        "low bit depth (<=4)": bd <= 4,
        "high bit depth (>=10)": bd >= 10,
        "non-square canvas": w != h,
        "multiple frames / layers": len(prog.frames) > 1,
        "splines": any(f.splines for f in prog.frames),
        "per-channel branches (c)": "c" in props,
        "position branches (x/y)": bool(props & {"x", "y"}),
        "neighbour-gradient branches": bool(props & NEIGHBOUR_PROPS),
        "cross-channel refs (Prev*)": any(p.startswith("Prev") or p.startswith("PPrev") for p in props),
        "error-adaptive (WGH/*Err)": any(p == "WGH" or p.endswith("Err") for p in props),
        "Weighted predictor": "Weighted" in preds,
        "Select/Gradient predictor": bool(preds & {"Select", "Gradient"}),
        "only Set leaves (no prediction)": preds <= {"Set"},
        "deep tree (depth >= 4)": depth(f0.tree) >= 4,
        "large tree (>= 12 nodes)": len(nodes) >= 12,
    }


def technique_rates(progs: list[dsl.Program]) -> dict[str, float]:
    rows = [techniques(p) for p in progs]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}


# ---------------------------------------------------------------- image metrics
def interest(img: Image.Image, jxl_bytes: int) -> dict:
    t = img.resize((256, 256), Image.Resampling.BOX)
    buf = io.BytesIO()
    t.save(buf, "PNG", optimize=True)
    a = np.asarray(t, dtype=np.float32) / 255
    g = a.mean(2)
    return {"png_kb": len(buf.getvalue()) / 1024,
            "colours": int(len(np.unique((a * 255).astype(np.uint8).reshape(-1, 3), axis=0))),
            "edge": float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()),
            "amplification": len(buf.getvalue()) / max(1, jxl_bytes)}


@torch.no_grad()
def text_embed(feat: Featurizer, prompts: list[str]) -> np.ndarray:
    inp = feat.proc(text=prompts, return_tensors="pt", padding=True).to(feat.device)
    e = feat.model.get_text_features(**inp)
    if not isinstance(e, torch.Tensor):
        e = e.pooler_output
    return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()


def clip_judge(E: np.ndarray, good: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """P(good prompt) under CLIP's zero-shot softmax (logit scale 100)."""
    logits = 100 * np.concatenate([E @ good.T, E @ bad.T], 1)
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return p[:, : len(good)].sum(1)


def mean_pairwise(E: np.ndarray) -> float:
    S = E @ E.T
    n = len(E)
    return float((S.sum() - np.trace(S)) / (n * (n - 1))) if n > 1 else float("nan")


def sheet(images: list[Image.Image], labels: list[str], path: Path, side: int = 160, cols: int = 6) -> None:
    rows = (len(images) + cols - 1) // cols
    s = Image.new("RGB", (cols * side, rows * (side + 14)), (30, 30, 30))
    d = ImageDraw.Draw(s)
    for i, (im, lab) in enumerate(zip(images, labels)):
        x, y = (i % cols) * side, (i // cols) * (side + 14)
        s.paste(im.resize((side, side)), (x, y))
        d.text((x + 2, y + side + 1), lab[:28], fill=(230, 230, 230))
    s.save(path)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--steps", default="0,120,400,online", help="snapshot steps and/or 'online'")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--train", type=Path, default=compiler.ROOT / "data" / "synth-v2")
    ap.add_argument("--train-sample", type=int, default=600)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-live", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run = Run(args.out, "evaluate", {k: str(v) for k, v in vars(args).items()}, live=not args.no_live)
    with run:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        run.set_state("running", "loading")
        tok = M.load_tokenizer()
        prompt = tok(PROMPT)["input_ids"]
        feat = Featurizer()
        ref = Reference.load_or_build(feat)
        gal_thumbs = [Image.open(REF_DIR / "thumbs" / f"{i:03d}.png").convert("RGB") for i in range(len(ref.titles))]
        good_t, bad_t = text_embed(feat, GOOD_PROMPTS), text_embed(feat, BAD_PROMPTS)
        fits = [json.loads(l) for l in (args.run / "rm_fits.jsonl").open() if l.strip()]
        fit = max(f.get("fit", -1) for f in fits)
        rm = BTReward.load(args.run / "rm" / f"fit_{fit:05d}.npz")
        rm_meta = next(f for f in fits if f.get("fit") == fit)
        taste = lambda X: (rm.score(X)[0] - rm_meta["norm_mu"]) / rm_meta["norm_sd"]  # noqa: E731

        sets: dict[str, dict] = {}

        def add_set(name, imgs, stats, texts, jxl_sizes, valid_frac=None):
            X = feat([im.resize((224, 224)) for im in imgs], stats)
            progs = [dsl.parse(t) for t in texts]
            sets[name] = {"imgs": imgs, "X": X, "E": X[:, :512], "texts": texts, "progs": progs,
                          "jxl": jxl_sizes, "valid": valid_frac,
                          "phash": [compiler.phash(im) for im in imgs]}

        # training data sample (gallery-free SFT corpus)
        recs = [json.loads(l) for f in sorted(args.train.glob("shard_*.jsonl")) for l in f.open(encoding="utf-8")]
        if not recs:
            raise SystemExit(f"no training shards in {args.train}; pass --train <build_data output dir>")
        train_texts = {r["text"] for r in recs}
        sample = random.Random(args.seed).sample(recs, min(args.train_sample, len(recs)))
        run.set_state("running", "rendering training sample")
        with ThreadPoolExecutor(10) as ex:
            imgs = list(ex.map(lambda r: compiler.thumbnail(compiler.decode(base64.b64decode(r["jxl_b64"]))), sample))
        add_set("training data", imgs, [compiler.image_stats(i) for i in imgs], [r["text"] for r in sample],
                [r["jxl_bytes"] for r in sample])

        # gallery (only pieces with a parseable recipe, for technique stats; all for image similarity)
        seeds = load_seeds()
        seed_by_file = {s.file: s for s in seeds}
        g_idx = [i for i, f in enumerate(ref.files) if f in seed_by_file]
        add_set("gallery (human)", [gal_thumbs[i] for i in g_idx], [compiler.image_stats(gal_thumbs[i]) for i in g_idx],
                [seed_by_file[ref.files[i]].text for i in g_idx], [seed_by_file[ref.files[i]].jxl_bytes for i in g_idx])

        # checkpoints
        steps = [s.strip() for s in args.steps.split(",")]
        lv = run.level("checkpoints", len(steps), "ckpts")
        for s in steps:
            path = args.run / "policy" / ("online.safetensors" if s == "online" else f"step_{int(s):05d}.safetensors")
            policy, meta = M.load_policy(path)
            # snapshot files carry their RL step in meta; step 0 is a copy of the SFT file (SFT meta)
            label = f"step {meta['step']}" if s == "online" else f"step {int(s)}"
            last = label
            run.set_state("running", f"sampling {label}")
            gens = M.sample(policy, prompt, args.n, 700, tok.eos_token_id)
            texts = [tok.decode([t for t in g if t != tok.eos_token_id]) for g in gens]
            run.level("renders", len(texts), "renders")

            def work(t):
                r = compiler.compile_tree(t, timeout=8, max_bytes=compiler.MAX_JXL_BYTES)
                run.advance("renders")
                return r

            with ThreadPoolExecutor(10) as ex:
                res = list(ex.map(work, texts))
            good = [r for r in res if r.ok and not compiler.is_degenerate(r.stats)]
            add_set(label, [r.thumb for r in good], [r.stats for r in good], [r.canonical for r in good],
                    [r.nbytes for r in good], valid_frac=len(good) / len(res))
            del policy
            torch.cuda.empty_cache()
            lv.done += 1
            run.event("sampled", checkpoint=label, good=len(good), n=len(res))

        # ---------------------------------------------------------- metrics
        run.set_state("running", "scoring")
        train_E = sets["training data"]["E"]
        gal_rates = technique_rates(sets["gallery (human)"]["progs"])
        metrics, rates = {}, {}
        for name, S in sets.items():
            E = S["E"]
            img_sim, img_idx = ref.nearest_image(E)
            if name == "gallery (human)":  # nearest *other* gallery piece; E rows are a subset of ref rows
                Sg = E @ ref.emb.T
                for r, gi in enumerate(g_idx):
                    Sg[r, gi] = -np.inf
                img_sim, img_idx = Sg.max(1), Sg.argmax(1)
            rec = []
            for k, t in enumerate(S["texts"]):
                toks = t.split()
                best = max((recipe_sim(toks, g) for j, g in enumerate(ref.recipes)
                            if g and not (name == "gallery (human)" and j == g_idx[k])), default=0.0)
                rec.append(best)
            rec = np.array(rec)
            tr = E @ train_E.T
            if name == "training data":
                np.fill_diagonal(tr, -np.inf)
            inter = [interest(im, b) for im, b in zip(S["imgs"], S["jxl"])]
            tz = taste(S["X"])
            cj = clip_judge(E, good_t, bad_t)
            S.update(img_sim=img_sim, img_idx=img_idx, rec=rec, taste=tz, clip=cj, inter=inter)
            r = technique_rates(S["progs"])
            rates[name] = r
            med = lambda k: float(np.median([i[k] for i in inter]))  # noqa: E731
            metrics[name] = {
                "n": len(E), "valid_frac": S["valid"],
                "distinct_frac": len(set(S["phash"])) / max(1, len(E)),
                "mean_pairwise_sim": mean_pairwise(E),
                "nearest_training_sim_median": float(np.median(tr.max(1))),
                "exact_training_program": float(np.mean([t in train_texts for t in S["texts"]])),
                "gallery_image_sim_median": float(np.median(img_sim)),
                "gallery_image_sim_ge_0.95": float(np.mean(img_sim >= 0.95)),
                "gallery_recipe_sim_median": float(np.median(rec)),
                "gallery_recipe_sim_ge_0.8": float(np.mean(rec >= 0.8)),
                "png_kb_median": med("png_kb"), "colours_median": med("colours"), "edge_median": med("edge"),
                "amplification_median": med("amplification"), "jxl_bytes_median": float(np.median(S["jxl"])),
                "taste_z_median": float(np.median(tz)), "taste_z_top10pct": float(np.quantile(tz, 0.9)),
                "clip_judge_median": float(np.median(cj)),
                "technique_distance_to_gallery": float(np.mean([abs(r[k] - gal_rates[k]) for k in r])),
            }

        # ---------------------------------------------------------- pictures
        L = sets[last]
        order = np.argsort(-L["taste"])[:18]
        sheet([L["imgs"][i] for i in order], [f"taste {L['taste'][i]:.1f}" for i in order], args.out / "best_by_taste.png")
        order = np.argsort(-L["clip"])[:18]
        sheet([L["imgs"][i] for i in order], [f"clip {L['clip'][i]:.2f}" for i in order], args.out / "best_by_clip.png")
        # most unique: far from training data and gallery, but liked (taste above median)
        uniq = (1 - (L["E"] @ train_E.T).max(1)) + (1 - L["img_sim"])
        cand = [i for i in np.argsort(-uniq) if L["taste"][i] > np.median(L["taste"])][:18]
        sheet([L["imgs"][i] for i in cand], [f"u {uniq[i]:.2f} t {L['taste'][i]:.1f}" for i in cand],
              args.out / "most_unique_liked.png")
        # convergent: looks like a gallery piece, recipe is different -> independent rediscovery candidates
        conv = [i for i in np.argsort(-L["img_sim"]) if L["rec"][i] < 0.55][:12]
        pair_imgs, pair_labels, convergent = [], [], []
        for i in conv:
            gi = L["img_idx"][i]
            pair_imgs += [L["imgs"][i], gal_thumbs[gi]]
            pair_labels += [f"model img {L['img_sim'][i]:.2f}", ref.titles[gi]]
            convergent.append({"model_recipe": L["texts"][i], "gallery_title": ref.titles[gi],
                               "gallery_file": ref.files[gi], "image_sim": float(L["img_sim"][i]),
                               "recipe_sim": float(L["rec"][i]),
                               "shared_techniques": sorted(k for k, v in techniques(L["progs"][i]).items()
                                                           if v and gi in g_idx and
                                                           techniques(sets["gallery (human)"]["progs"][g_idx.index(gi)])[k])})
        sheet(pair_imgs, pair_labels, args.out / "convergent_pairs.png", cols=6)

        # ---------------------------------------------------------- what the user prefers
        prefs, items = [], {}
        for l in (args.run / "prefs.jsonl").open(encoding="utf-8"):
            p = json.loads(l)
            if p.get("undo"):
                prefs and prefs.pop()
            else:
                prefs.append(p)
        for l in (args.run / "items.jsonl").open(encoding="utf-8"):
            it = json.loads(l)
            items[it["id"]] = it
        wins, games = {}, {}
        for p in prefs:
            for side, y in (("a", 1.0), ("b", 0.0)):
                i = p[side]
                score = {"a": y, "b": 1 - y, "tie": 0.5, "bad": 0.0}[p["choice"]]
                wins[i] = wins.get(i, 0) + score
                games[i] = games.get(i, 0) + 1
        pref_rows = []
        for i, n in games.items():
            if n >= 2 and i in items:
                try:
                    pref_rows.append((wins[i] / n, techniques(dsl.parse(items[i]["text"]))))
                except dsl.ParseError:
                    pass
        lift = {}
        for k in gal_rates:
            with_k = [w for w, t in pref_rows if t[k]]
            without = [w for w, t in pref_rows if not t[k]]
            if len(with_k) >= 15 and len(without) >= 15:
                lift[k] = {"win_with": float(np.mean(with_k)), "win_without": float(np.mean(without)),
                           "n_with": len(with_k)}

        atomic_write_json(args.out / "metrics.json", {"metrics": metrics, "technique_rates": rates,
                                                      "convergent": convergent, "preference_lift": lift,
                                                      "taste_model_fit": rm_meta})
        _report(args.out, metrics, rates, convergent, lift, rm_meta)


def _report(out: Path, metrics: dict, rates: dict, convergent: list, lift: dict, rm_meta: dict) -> None:
    names = list(metrics)
    f = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))  # noqa: E731
    rows = [("valid & non-blank", "valid_frac"), ("distinct images", "distinct_frac"),
            ("mean pairwise CLIP sim (lower = more varied)", "mean_pairwise_sim"),
            ("nearest training image sim (median)", "nearest_training_sim_median"),
            ("exact training-data program", "exact_training_program"),
            ("nearest gallery image sim (median)", "gallery_image_sim_median"),
            ("gallery image sim >= 0.95", "gallery_image_sim_ge_0.95"),
            ("nearest gallery recipe sim (median)", "gallery_recipe_sim_median"),
            ("gallery recipe sim >= 0.8 (near-copy)", "gallery_recipe_sim_ge_0.8"),
            ("render complexity, PNG KB (median)", "png_kb_median"), ("colours (median)", "colours_median"),
            ("edge density (median)", "edge_median"), ("PNG bytes per JXL byte (median)", "amplification_median"),
            ("JXL bytes (median)", "jxl_bytes_median"), ("taste z (median)", "taste_z_median"),
            ("taste z (top 10%)", "taste_z_top10pct"), ("CLIP judge P(good) (median)", "clip_judge_median"),
            ("technique-mix distance to gallery", "technique_distance_to_gallery")]
    lines = ["# Checkpoint evaluation", "", f"Taste model: fit {rm_meta.get('fit')} on {rm_meta.get('picks')} picks, "
             f"held-out accuracy {f(rm_meta.get('cv_acc'))}.", "",
             "| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    lines += [f"| {lab} | " + " | ".join(f(metrics[n][k]) for n in names) + " |" for lab, k in rows]
    lines += ["", "## Technique usage (share of valid programs)", "",
              "| technique | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k in rates[names[0]]:
        lines.append(f"| {k} | " + " | ".join(f"{rates[n][k]:.0%}" for n in names) + " |")
    lines += ["", "## What wins your comparisons (items with >= 2 comparisons)", "",
              "| technique | win rate with | without | n with |", "|---|---|---|---|"]
    for k, v in sorted(lift.items(), key=lambda kv: -(kv[1]["win_with"] - kv[1]["win_without"])):
        lines.append(f"| {k} | {v['win_with']:.2f} | {v['win_without']:.2f} | {v['n_with']} |")
    lines += ["", "## Convergent pairs (looks like a gallery piece, recipe differs)", "", "![](convergent_pairs.png)", ""]
    for c in convergent:
        lines.append(f"- image {c['image_sim']:.2f} / recipe {c['recipe_sim']:.2f} ~ *{c['gallery_title']}*; "
                     f"shared techniques: {', '.join(c['shared_techniques']) or '-'}")
    lines += ["", "![best by taste](best_by_taste.png) ![best by CLIP](best_by_clip.png) "
              "![most unique liked](most_unique_liked.png)", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
