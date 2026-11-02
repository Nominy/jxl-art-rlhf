"""Compile tree DSL to JXL with libjxl's jxl_from_tree and render previews with djxl."""
from __future__ import annotations

import io
import os
import subprocess
import tempfile
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import dsl

ROOT = Path(__file__).resolve().parent.parent
_EXE = ".exe" if os.name == "nt" else ""
JXL_FROM_TREE = ROOT / "tools" / "bin" / f"jxl_from_tree{_EXE}"  # see README: Windows release zip / scripts/build_libjxl.sh
DJXL = ROOT / "tools" / "bin" / f"djxl{_EXE}"

MAX_JXL_BYTES = 256
MAX_SIDE = 4096
MAX_PIXELS = 4 * 1024 * 1024
THUMB = 512
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class Result:
    ok: bool
    stage: str  # parse | limits | compile | decode | ok
    error: str = ""
    canonical: str = ""
    jxl: bytes = b""
    size: tuple[int, int] = (0, 0)
    compile_ms: float = 0.0
    decode_ms: float = 0.0
    thumb: Image.Image | None = None
    stats: dict = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        return len(self.jxl)


def _run(args: list[str], timeout: float, stdin: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, input=stdin, capture_output=True, timeout=timeout,
                          creationflags=_NO_WINDOW)


def compile_tree(text: str, *, timeout: float = 10.0, render: bool = True,
                 max_bytes: int | None = None, workdir: str | None = None) -> Result:
    try:
        prog = dsl.parse(text)
        canonical = dsl.to_text(prog)
    except dsl.ParseError as e:
        return Result(False, "parse", str(e))
    w, h = dsl.output_size(prog)
    if w <= 0 or h <= 0 or w > MAX_SIDE or h > MAX_SIDE or w * h > MAX_PIXELS:
        return Result(False, "limits", f"output size {w}x{h}", canonical, size=(w, h))
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        src, out = os.path.join(td, "t.txt"), os.path.join(td, "o.jxl")
        Path(src).write_text(canonical, encoding="utf-8")
        t0 = time.perf_counter()
        try:
            r = _run([str(JXL_FROM_TREE), src, out], timeout)
        except subprocess.TimeoutExpired:
            return Result(False, "compile", "timeout", canonical, size=(w, h))
        cms = (time.perf_counter() - t0) * 1000
        if r.returncode or not os.path.exists(out):
            err = r.stderr.decode("utf-8", "replace").strip().splitlines()
            return Result(False, "compile", (err[0] if err else f"rc={r.returncode}")[:200],
                          canonical, size=(w, h), compile_ms=cms)
        data = Path(out).read_bytes()
    res = Result(True, "ok", "", canonical, data, (w, h), cms)
    if max_bytes is not None and len(data) > max_bytes:
        return Result(False, "limits", f"{len(data)} bytes > {max_bytes}", canonical, data, (w, h), cms)
    if render:
        return render_into(res, timeout)
    return res


def decode(jxl: bytes, timeout: float = 10.0) -> Image.Image:
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "i.jxl")
        Path(src).write_bytes(jxl)
        r = _run([str(DJXL), src, "-", "--output_format", "png", "--bits_per_sample=8",
                  "--num_threads=1", "--color_space=RGB_D65_SRG_Rel_SRG", "--quiet"], timeout)
    if r.returncode or not r.stdout:
        raise RuntimeError(r.stderr.decode("utf-8", "replace").strip()[:200] or f"djxl rc={r.returncode}")
    img = Image.open(io.BytesIO(r.stdout))
    img.load()
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (128, 128, 128, 255))
        img = Image.alpha_composite(bg, img.convert("RGBA"))
    return img.convert("RGB")


def thumbnail(img: Image.Image, side: int = THUMB) -> Image.Image:
    t = img.copy()
    t.thumbnail((side, side), Image.Resampling.BOX)
    return t


def image_stats(img: Image.Image) -> dict:
    """Cheap descriptors used to reject degenerate outputs (flat, single-colour)."""
    a = np.asarray(img.resize((64, 64), Image.Resampling.BOX), dtype=np.float32) / 255.0
    gray = a.mean(axis=2)
    q = (a * 255).astype(np.uint8)
    comp = len(zlib.compress(q.tobytes(), 6)) / q.size
    return {
        "std": float(gray.std()),
        "color_std": float(a.std(axis=(0, 1)).mean()),
        "edge": float(np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()),
        "compressibility": float(comp),
        "unique": int(len(np.unique(q.reshape(-1, 3), axis=0))),
    }


def phash(img: Image.Image) -> int:
    """64-bit average hash over an 8x8 grayscale grid."""
    g = np.asarray(img.convert("L").resize((8, 8), Image.Resampling.BOX), dtype=np.float32)
    bits = (g > g.mean()).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def render_into(res: Result, timeout: float = 10.0) -> Result:
    t0 = time.perf_counter()
    try:
        img = decode(res.jxl, timeout)
    except (RuntimeError, subprocess.TimeoutExpired, OSError) as e:
        res.ok, res.stage, res.error = False, "decode", str(e)[:200]
        return res
    res.decode_ms = (time.perf_counter() - t0) * 1000
    res.thumb = thumbnail(img)
    res.stats = image_stats(res.thumb)
    res.stats["phash"] = phash(res.thumb)
    return res


def is_degenerate(stats: dict) -> bool:
    return stats["std"] < 0.02 and stats["color_std"] < 0.02 or stats["unique"] < 3
