"""Seed corpus: gallery trees downloaded from jpegxl.info/art (`python -m jxlart.fetch_gallery`)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import dsl
from .compiler import ROOT

PROMPT = "JXL tree:\n"


@dataclass
class Seed:
    title: str
    file: str
    text: str  # canonical
    program: dsl.Program
    jxl_bytes: int


def load_seeds(art_dir: Path = ROOT / "art") -> list[Seed]:
    manifest = art_dir / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} missing: run `python -m jxlart.fetch_gallery` first")
    out = []
    for m in json.loads(manifest.read_text(encoding="utf-8")):
        if not m.get("tree"):
            continue
        raw = (art_dir / m["tree"]).read_text(encoding="utf-8")
        prog = dsl.parse(raw)
        w, h = dsl.output_size(prog)
        if w * h > 4 * 1024 * 1024 or max(w, h) > 4096:
            continue
        out.append(Seed(m["title"], m["file"], dsl.to_text(prog), prog, m["bytes"]))
    return out


def format_example(tree_text: str) -> str:
    return PROMPT + tree_text
