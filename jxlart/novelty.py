"""Novelty against known JXL art: a reference index of the jpegxl.info gallery and a report.

Reference index (cached at <ref-dir>/gallery.npz): every downloaded gallery piece rendered with
djxl, embedded with CLIP, plus its canonical recipe tokens when the source is known.

Two similarity signals per generated image:
  image   cosine similarity of CLIP embeddings to the nearest gallery piece
  recipe  difflib ratio of recipe tokens to the nearest gallery recipe (1.0 = same program)

  python -m jxlart.novelty --run runs/rlhf-v2 --out reports/novelty --item c00994
"""
from __future__ import annotations

import argparse
import difflib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import compiler, dsl
from .compiler import ROOT

REF_DIR = ROOT / "cache" / "reference"  # rendered gallery thumbs + CLIP index, rebuilt on demand


def recipe_tokens(text: str) -> list[str]:
    try:
        return dsl.canonicalize(text).split()
    except dsl.ParseError:
        return text.split()


def recipe_sim(a: list[str], b: list[str]) -> float:
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    if sm.real_quick_ratio() < 0.5 or sm.quick_ratio() < 0.5:
        return 0.0  # far apart; exact value irrelevant below this
    return sm.ratio()


class Reference:
    """Gallery embeddings + recipes; nearest-neighbour queries for images and recipe text."""

    def __init__(self, emb: np.ndarray, titles: list[str], files: list[str], recipes: list[list[str]]):
        self.emb, self.titles, self.files, self.recipes = emb, titles, files, recipes

    @classmethod
    def load_or_build(cls, feat, ref_dir: Path = REF_DIR) -> "Reference":
        path = ref_dir / "gallery.npz"
        if path.exists():
            d = np.load(path, allow_pickle=False)
            meta = json.loads((ref_dir / "gallery.json").read_text(encoding="utf-8"))
            return cls(d["emb"], meta["titles"], meta["files"], meta["recipes"])
        ref_dir.mkdir(parents=True, exist_ok=True)
        (ref_dir / "thumbs").mkdir(exist_ok=True)
        art = ROOT / "art"
        manifest = json.loads((art / "manifest.json").read_text(encoding="utf-8"))

        def render(m):
            try:
                return compiler.thumbnail(compiler.decode((art / m["file"]).read_bytes(), timeout=60))
            except Exception:
                return None

        with ThreadPoolExecutor(8) as ex:
            thumbs = list(ex.map(render, manifest))
        keep = [(m, t) for m, t in zip(manifest, thumbs) if t is not None]
        for i, (m, t) in enumerate(keep):
            t.save(ref_dir / "thumbs" / f"{i:03d}.png")
        emb = feat.embed([t.resize((224, 224)) for _, t in keep])
        recipes = [recipe_tokens((art / m["tree"]).read_text(encoding="utf-8")) if m.get("tree") else []
                   for m, _ in keep]
        meta = {"titles": [m["title"] for m, _ in keep], "files": [m["file"] for m, _ in keep], "recipes": recipes}
        np.savez(path, emb=emb)
        (ref_dir / "gallery.json").write_text(json.dumps(meta), encoding="utf-8")
        return cls(emb, meta["titles"], meta["files"], meta["recipes"])

    def nearest_image(self, E: np.ndarray, exclude_self: bool = False) -> tuple[np.ndarray, np.ndarray]:
        S = E @ self.emb.T
        if exclude_self:
            np.fill_diagonal(S, -np.inf)
        idx = S.argmax(1)
        return S[np.arange(len(E)), idx], idx

    def nearest_recipe(self, tokens: list[str]) -> tuple[float, int]:
        best, bi = 0.0, -1
        for i, r in enumerate(self.recipes):
            if r:
                s = recipe_sim(tokens, r)
                if s > best:
                    best, bi = s, i
        return best, bi


def _pair_sheet(rows: list[tuple[Image.Image, Image.Image, str]], path: Path, side: int = 192) -> None:
    sheet = Image.new("RGB", (side * 2 + 8, (side + 18) * len(rows)), (30, 30, 30))
    d = ImageDraw.Draw(sheet)
    for i, (a, b, label) in enumerate(rows):
        y = i * (side + 18)
        sheet.paste(a.resize((side, side)), (0, y))
        sheet.paste(b.resize((side, side)), (side + 8, y))
        d.text((2, y + side + 2), label[:60], fill=(230, 230, 230))
    sheet.save(path)


def main() -> None:
    from .rlhf.reward import Featurizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--item", action="append", default=[], help="item ids to rank")
    ap.add_argument("--train", default=str(ROOT / "data" / "synth-v1"), help="SFT shards, for exact-copy checks")
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    feat = Featurizer()
    ref = Reference.load_or_build(feat)
    ref_thumbs = [Image.open(REF_DIR / "thumbs" / f"{i:03d}.png").convert("RGB") for i in range(len(ref.titles))]

    # baseline: how close is each gallery piece to its nearest *other* gallery piece
    g_sim, _ = ref.nearest_image(ref.emb, exclude_self=True)
    g_rec = []
    for i, r in enumerate(ref.recipes):
        if r:
            others = Reference(ref.emb, ref.titles, ref.files, [x if j != i else [] for j, x in enumerate(ref.recipes)])
            g_rec.append(others.nearest_recipe(r)[0])

    items = [json.loads(l) for l in (run / "items.jsonl").open(encoding="utf-8")]
    imgs = [Image.open(run / "items" / f"{it['id']}.png").convert("RGB") for it in items]
    E = feat.embed([im.resize((224, 224)) for im in imgs])
    i_sim, i_idx = ref.nearest_image(E)
    with ThreadPoolExecutor(8) as ex:
        rec = list(ex.map(lambda it: ref.nearest_recipe(recipe_tokens(it["text"])), items))
    r_sim = np.array([r[0] for r in rec])
    train = set()
    for f in Path(args.train).glob("shard_*.jsonl"):
        train |= {json.loads(l)["text"] for l in f.open(encoding="utf-8") if l.strip()}
    canon = []
    for it in items:
        try:
            canon.append(dsl.canonicalize(it["text"]))
        except dsl.ParseError:
            canon.append(it["text"])
    in_train = np.array([c in train for c in canon])
    exact_gallery = r_sim >= 0.999

    q = lambda a, p: float(np.quantile(a, p))  # noqa: E731
    lines = ["# Novelty vs the jpegxl.info gallery", "",
             f"Gallery pieces: {len(ref.titles)} ({sum(1 for r in ref.recipes if r)} with recipes). "
             f"Generated images checked: {len(items)}.", "",
             "## Baseline: gallery piece vs its nearest other gallery piece",
             f"image sim median {q(g_sim, .5):.3f}, p90 {q(g_sim, .9):.3f}; "
             f"recipe sim median {q(np.array(g_rec), .5):.3f}, p90 {q(np.array(g_rec), .9):.3f}", "",
             "## Generated images by model step", "",
             "| steps | n | image sim median | image p90 | ≥ gallery-p90 image | recipe sim median | recipe ≥0.9 | exact gallery recipe | exact SFT-data recipe |",
             "|---|---|---|---|---|---|---|---|---|"]
    steps = np.array([it.get("step", 0) for it in items])
    thr_img = q(g_sim, .9)
    for lo in range(0, int(steps.max()) + 1, 40):
        m = (steps >= lo) & (steps < lo + 40)
        if m.sum() < 5:
            continue
        lines.append(f"| {lo}-{lo + 39} | {m.sum()} | {np.median(i_sim[m]):.3f} | {q(i_sim[m], .9):.3f} | "
                     f"{(i_sim[m] >= thr_img).mean():.0%} | {np.median(r_sim[m]):.3f} | {(r_sim[m] >= .9).mean():.0%} | "
                     f"{exact_gallery[m].mean():.0%} | {in_train[m].mean():.0%} |")
    # novelty score: distance to the gallery on both signals
    novelty = (1 - i_sim) + 0.5 * (1 - r_sim)
    order = np.argsort(novelty)
    lines += ["", "## Closest copies (generated | nearest gallery)", "", "![](closest.png)", "",
              "## Most original (generated | nearest gallery)", "", "![](most_original.png)", ""]
    _pair_sheet([(imgs[i], ref_thumbs[i_idx[i]],
                  f"{items[i]['id']} img {i_sim[i]:.2f} rec {r_sim[i]:.2f} ~ {ref.titles[i_idx[i]]}")
                 for i in order[:16]], out / "closest.png")
    _pair_sheet([(imgs[i], ref_thumbs[i_idx[i]],
                  f"{items[i]['id']} img {i_sim[i]:.2f} rec {r_sim[i]:.2f} ~ {ref.titles[i_idx[i]]}")
                 for i in order[::-1][:16]], out / "most_original.png")
    rank = {items[i]["id"]: k for k, i in enumerate(order[::-1])}
    for iid in args.item:
        i = next(k for k, it in enumerate(items) if it["id"] == iid)
        lines.append(f"- **{iid}**: originality rank {rank[iid] + 1}/{len(items)}; image sim {i_sim[i]:.3f} "
                     f"to '{ref.titles[i_idx[i]]}'; recipe sim {r_sim[i]:.3f}"
                     + (f" to '{ref.titles[rec[i][1]]}'" if rec[i][1] >= 0 else ""))
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    per_item = [{"id": it["id"], "step": it.get("step", 0), "image_sim": float(a), "nearest": ref.titles[b],
                 "recipe_sim": float(c), "in_sft_data": bool(d), "novelty": float(n)}
                for it, a, b, c, d, n in zip(items, i_sim, i_idx, r_sim, in_train, novelty)]
    (out / "items.json").write_text(json.dumps(per_item), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
