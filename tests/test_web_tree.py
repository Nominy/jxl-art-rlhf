"""Parity of the in-browser tree pipeline (web/blind/lib/{tree,codec}.js) with jxlart.dsl + tools/bin libjxl.

  PYTHONPATH=. python -m pytest tests/test_web_tree.py -q -s     (or: python tests/test_web_tree.py)

test_canonicalize (~1 s): >=2000 texts (training programs from D:/jxl-art/data/synth-v2, gallery programs from
art/manifest.json, mutated/broken variants, float-heavy spline programs): same accept/reject, byte-identical
canonical text and identical output size. test_compile (~1.5 min on 12 cores; launch it via jxlart.launch):
>=300 valid programs compiled by the wasm build and by tools/bin/jxl_from_tree.exe and decoded by the wasm and by
djxl (compiler.decode flags). Prints identical-byte rate, <=256-byte agreement, decoder pixel parity (same JXL),
end-to-end pixel parity, PNG formats, timing; checks that libjxl failures (including an abort) throw without
hanging and that the codec recovers, and that Neon Chevron matches the native pipeline exactly.
"""
from __future__ import annotations

import io
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jxlart import compiler, dsl  # noqa: E402
from jxlart.fetch_gallery import decode_zcode  # noqa: E402

RUNNER = ROOT / "tests" / "web_tree.mjs"
DATA = Path(os.environ.get("JXLART_WEB_DATA", "D:/jxl-art/data/synth-v2"))
JS = shutil.which("bun") or shutil.which("node")
NEON_CHEVRON = ("eNodxzEKgDAMBdA9p_gXKNgWHR1U0FFUcC422kARkQweX3F7rxGNfGmCdbRK_OBdQQPLkfTn1C6wnjrOGsaQWZVJdjyo4VxFgMHMmTe"
                "Fsf_6O0Th83tJLwomGhg")
# Programs that make libjxl fail natively: a regular error, and a crash (abort) in jxl_from_tree.
LIBJXL_FAILS = ["RCT 100\n- Set 0\n", "- Set 99999999999999\n"]


def check(name: str, ok: bool, detail: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    assert ok, f"{name}: {detail}"


def run_js(*args: str, timeout: float = 300) -> str:
    assert JS, "neither bun nor node on PATH"
    r = subprocess.run([JS, str(RUNNER), *args], capture_output=True, text=True, encoding="utf-8",
                       timeout=timeout, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


@cache
def training_texts() -> list[str]:
    out = []
    for shard in sorted(DATA.glob("shard_*.jsonl")):
        with open(shard, encoding="utf-8") as f:
            out += [json.loads(line)["text"] for line in f]
    assert out, f"no training programs in {DATA}"
    return out


@cache
def gallery_texts() -> list[str]:
    man = json.loads((ROOT / "art" / "manifest.json").read_text(encoding="utf-8"))
    paths = [ROOT / "art" / e["tree"] for e in man if e.get("tree")]
    return [p.read_text(encoding="utf-8") for p in paths if p.exists()]


# --- canonicalize ---------------------------------------------------------------------------------

ODD_TOKENS = ["-", "+", "if", ">", "/*", "*/", "NotLast", "Spline", "EndSpline", "1e999", "\u0663", "-0", "+5",
              "007", "99999999999999999999999", "Bitdepth", "Upsample", "EPF", "Noise", "Width", "FramePos",
              "BlendMode", "kAdd", "constructor", "__proto__", "toString", "Prev19Err", "Prev20", "Set", "AvgAll",
              "x", "c", "0.5", "\u00a0", "x\u2003y", "\ufeff", "1.5\x1c2"]
ODD_LINES = ["NotLast", "Upsample 3", "Upsample 8", "EPF 4", "Bitdepth 0", "Bitdepth 33", "HiddenChannel 17",
             "Noise 0 0 0 0 0 0 0 1.5", "Noise 0.1 0.2 0 0 0 0 0 1", "XYBFactors 1e-3 2 -3", "FramePos -5 +7",
             "BlendMode kMul", "BlendMode kFoo", "/* comment */", "/* open", "Width 99999999999999999999",
             "AnimationFPS 30 1", "Duration +0", "GroupShift \u0663", "Width \u0661\u0662\u0668"]
FLOAT_EDGES = ["0", "-0", "+0", ".5", "5.", "1e999", "-1e999", "1e-400", "1234567.5", "1234568.5", "9999999.5",
               "0.00001", "0.0001", "1e7", "123456.75", "-.0", "+.5e+3", "\u0661.\u0665", "4.9e-324",
               "1.7976931348623157e308", "0.30000000000000004", "2.5e-5", "999999.95"]


def mutate(t: str, rng: random.Random) -> str:
    toks, lines = t.split(), t.split("\n")
    k = rng.randrange(9)
    if k == 0:
        return t[:rng.randrange(len(t) + 1)]
    if k == 1 and len(toks) > 1:
        i, j = rng.randrange(len(toks)), rng.randrange(len(toks))
        toks[i], toks[j] = toks[j], toks[i]
        return " ".join(toks)
    if k == 2 and len(lines) > 1:
        del lines[rng.randrange(len(lines))]
        return "\n".join(lines)
    if k == 3:
        toks[rng.randrange(len(toks))] = rng.choice(ODD_TOKENS)
        return " ".join(toks)
    if k == 4:
        lines.insert(rng.randrange(len(lines) + 1), rng.choice(ODD_LINES))
        return "\n".join(lines)
    if k == 5:
        return t.replace("\n", rng.choice(["\x1c", "\ufeff", "\x85", "\u3000", "\r\n", "\t"]))
    if k == 6:
        return t + t
    if k == 7:
        return "NotLast\n" + t + "\n" + rng.choice(training_texts())
    return re.sub(r"-?\d+", lambda m: str(int(m.group()) * rng.choice([1, -1, 10 ** 12])), t, count=3)


def random_float(rng: random.Random) -> str:
    k = rng.randrange(7)
    if k == 0:
        return repr(rng.uniform(-1, 1))
    if k == 1:
        return "%.*e" % (rng.randrange(12), rng.uniform(-1e3, 1e3) * 10.0 ** rng.randrange(-30, 30))
    if k == 2:
        return rng.choice(FLOAT_EDGES)
    if k == 3:
        return str(rng.randrange(-10 ** 9, 10 ** 9))
    if k == 4:  # arbitrary finite bit patterns (subnormals, huge exponents)
        v = np.frombuffer(rng.getrandbits(64).to_bytes(8, "little"), "<f8")[0]
        return repr(float(v)) if np.isfinite(v) else "1e300"
    if k == 5:  # exact ties at the 8th significant digit
        return f"{rng.randrange(10 ** 6, 10 ** 7)}.5"
    return "%.8g" % rng.uniform(-5000, 5000)


def spline_program(rng: random.Random) -> str:
    coeffs = " ".join(random_float(rng) for _ in range(dsl.SPLINE_COEFFS))
    points = " ".join(f"{random_float(rng)} {random_float(rng)}" for _ in range(rng.randrange(1, 4)))
    noise = " ".join(random_float(rng) if rng.random() < 0.02 else f"{rng.random():.{rng.randrange(1, 18)}g}"
                     for _ in range(8))
    return f"Noise {noise}\nXYBFactors {random_float(rng)} {random_float(rng)} {random_float(rng)}\n" \
           f"Spline {coeffs} {points} EndSpline\n- Set 0\n"


def canonicalize_cases(seed: int = 0) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    train = rng.sample(training_texts(), 1500)
    cases = [("training", t) for t in train[:1000]] + [("gallery", t) for t in gallery_texts()]
    cases += [("mutated", mutate(t, rng)) for t in train[500:1500]]
    cases += [("mutated2", mutate(mutate(t, rng), rng)) for t in train[:400]]
    cases += [("floats", spline_program(rng)) for _ in range(300)]
    return cases


def py_canonicalize(text: str) -> dict:
    try:
        prog = dsl.parse(text)
    except dsl.ParseError:
        return {"ok": False}
    return {"ok": True, "canonical": dsl.to_text(prog), "size": dsl.output_size(prog)}


def test_canonicalize() -> None:
    cases = canonicalize_cases()
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "in.jsonl", Path(d) / "out.jsonl"
        src.write_text("".join(json.dumps({"text": t}) + "\n" for _, t in cases), encoding="utf-8")
        run_js("canon", str(src), str(dst))
        js = [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines()]
    assert len(js) == len(cases)
    stats: dict[str, list[int]] = {}
    bad = []
    for (kind, text), j in zip(cases, js):
        p = py_canonicalize(text)
        same = p["ok"] == j["ok"] and (not p["ok"] and j["kind"] == "ParseError" or p["ok"] and (
            p["canonical"] == j["canonical"] and [float(v) for v in p["size"]] == [j["width"], j["height"]]))
        s = stats.setdefault(kind, [0, 0, 0])
        s[0] += 1
        s[1] += p["ok"]
        s[2] += not same
        if not same:
            bad.append((text, p, j))
    for kind, (n, acc, nbad) in stats.items():
        print(f"  {kind:9s} n={n:5d} accepted={acc:5d} rejected={n - acc:5d} mismatches={nbad}")
    for text, p, j in bad[:3]:
        print("MISMATCH", repr(text[:300]), "\n  py:", repr(p)[:300], "\n  js:", repr(j)[:300])
    check("canonicalize parity", not bad and len(cases) >= 2000,
          f"{len(cases) - len(bad)}/{len(cases)} identical (accept/reject, canonical text, width/height)")


# --- compile + decode -----------------------------------------------------------------------------

def djxl_png(jxl: bytes) -> bytes:
    """compiler.decode's djxl invocation, returning the raw PNG (no alpha compositing)."""
    r = subprocess.run([str(compiler.DJXL), "-", "-", "--output_format", "png", "--bits_per_sample=8",
                        "--num_threads=1", "--color_space=RGB_D65_SRG_Rel_SRG", "--quiet"],
                       input=jxl, capture_output=True, timeout=60)
    assert r.returncode == 0 and r.stdout, r.stderr
    return r.stdout


def pixels(png: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(png))
    assert im.mode in ("RGB", "RGBA"), im.mode
    return np.asarray(im).astype(np.int16)


def pixel_diff(a_png: bytes, b_png: bytes) -> tuple[int, float]:
    """(max abs difference, fraction of differing pixels); (256, 1.0) when the shapes differ."""
    a, b = pixels(a_png), pixels(b_png)
    if a.shape != b.shape:
        return 256, 1.0
    d = np.abs(a - b)
    return int(d.max()), float(d.max(-1).astype(bool).mean())


def png_info(png: bytes) -> tuple[int, int]:
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and png[12:16] == b"IHDR"
    return png[24], png[25]  # bit depth, colour type


def compile_programs(seed: int = 1, n_train: int = 160) -> list[str]:
    """Valid programs within the size limits: random training programs plus the gallery programs whose
    output is at most 1 MPixel (HDR, alpha, splines, XYB, noise; keeps the native side fast)."""
    rng = random.Random(seed)
    out = []
    for cap, texts in ((compiler.MAX_PIXELS, rng.sample(training_texts(), n_train)), (1 << 20, gallery_texts())):
        for t in texts:
            p = py_canonicalize(t)
            if p["ok"]:
                w, h = p["size"]
                if 0 < w <= compiler.MAX_SIDE and 0 < h <= compiler.MAX_SIDE and w * h <= cap:
                    out.append(p["canonical"])
    return out


def native(text: str) -> dict:
    r = compiler.compile_tree(text, render=False)
    if not r.ok:
        return {"ok": False, "error": r.error}
    return {"ok": True, "jxl": r.jxl, "png": djxl_png(r.jxl)}


def native_reproduces(text: str, jxl: bytes, tries: int = 12) -> bool:
    """jxl_from_tree encodes an uninitialised Image3F (Image3F::Create in tools/jxl_from_tree.cc); for some
    programs the encoder's choices depend on that heap garbage, so the native output changes with the
    process environment. True if some environment makes the native tool emit exactly `jxl`."""
    with tempfile.TemporaryDirectory() as d:
        Path(d, "t.txt").write_text(text, encoding="utf-8")
        for k in range(tries):
            env = dict(os.environ, JXLART_PAD="x" * (k * 997))
            subprocess.run([str(compiler.JXL_FROM_TREE), "t.txt", "o.jxl"], cwd=d, env=env, capture_output=True,
                           timeout=60)
            if Path(d, "o.jxl").read_bytes() == jxl:
                return True
    return False


def run_wasm(texts: list[str], out: Path, shards: int = 6) -> tuple[list[dict], list[float]]:
    items = [{"id": i, "text": t} for i, t in enumerate(texts)]
    chunks = [items[k::shards] for k in range(shards)]

    def go(k: int) -> list[dict]:
        src = out / f"in{k}.jsonl"
        src.write_text("".join(json.dumps(x) + "\n" for x in chunks[k]), encoding="utf-8")
        return [json.loads(line) for line in run_js("compile", str(src), str(out)).splitlines()]

    with ThreadPoolExecutor(shards) as ex:
        recs = [r for part in ex.map(go, range(shards)) for r in part]
    by_id = {r["id"]: r for r in recs if "id" in r}
    for i, r in by_id.items():
        if r["ok"]:
            r["jxl"] = (out / f"{i}.jxl").read_bytes()
            r["png"] = (out / f"{i}.png").read_bytes()
    return [by_id[i] for i in range(len(texts))], [r["init_ms"] for r in recs if "init_ms" in r]


def test_compile() -> None:
    progs = compile_programs()
    neon = dsl.canonicalize(decode_zcode(NEON_CHEVRON))
    texts = [neon] + progs + LIBJXL_FAILS + [neon]  # Neon Chevron again: the codec recovered after the abort
    with tempfile.TemporaryDirectory() as d, ThreadPoolExecutor(7) as ex:
        wasm_job = ex.submit(run_wasm, texts, Path(d))
        nat = list(ex.map(native, texts))
        wasm, init = wasm_job.result()
        fails = range(1 + len(progs), 1 + len(progs) + len(LIBJXL_FAILS))
        valid = [i for i in range(len(texts)) if i not in fails]
        both = [i for i in valid if nat[i]["ok"] and wasm[i]["ok"]]
        differ = [i for i in both if nat[i]["jxl"] != wasm[i]["jxl"]]
        explained = list(ex.map(lambda i: native_reproduces(texts[i], wasm[i]["jxl"]), differ))
        # Decoder alone: wasm PNG vs djxl on the same (wasm-encoded) JXL.
        dec = list(ex.map(lambda i: pixel_diff(wasm[i]["png"], djxl_png(wasm[i]["jxl"])), both))
    e2e = {i: pixel_diff(wasm[i]["png"], nat[i]["png"]) for i in both}  # vs the native pipeline as run

    ok_agree = sum(nat[i]["ok"] == wasm[i]["ok"] for i in valid)
    limit_agree = sum((len(nat[i]["jxl"]) <= compiler.MAX_JXL_BYTES) == (len(wasm[i]["jxl"]) <= compiler.MAX_JXL_BYTES)
                      for i in both)
    formats = {png_info(wasm[i]["png"]) for i in both}
    alpha = sum(png_info(wasm[i]["png"])[1] == 6 for i in both)
    ms = [wasm[i]["compile_ms"] + wasm[i]["decode_ms"] for i in both]
    dec_same = sum(m == 0 for m, _ in dec)
    dec_max, dec_frac = max(m for m, _ in dec), max(f for _, f in dec)
    e2e_same = sum(m == 0 for m, _ in e2e.values())
    e2e_big = [i for i, (m, _) in e2e.items() if m > 1]
    print(f"  {len(valid)} valid programs ({len(progs)} + Neon Chevron twice), runner {Path(JS).name}, "
          f"wasm init {statistics.mean(init):.0f} ms")
    print(f"  wasm compile+decode ms: median {statistics.median(ms):.0f}, mean {statistics.mean(ms):.0f}, "
          f"p95 {sorted(ms)[int(0.95 * len(ms))]:.0f}, max {max(ms):.0f}")
    print(f"  PNG (bit depth, colour type) from wasm: {sorted(formats)}; RGBA (alpha) outputs: {alpha}")
    print(f"  JXL bytes identical to native: {len(both) - len(differ)}/{len(both)}; the {len(differ)} others are "
          f"reproduced by the native tool under another environment: {sum(explained)}/{len(differ)}")
    print(f"  end-to-end vs native pipeline: {e2e_same}/{len(both)} pixel-identical; max abs diff > 1 in "
          f"{len(e2e_big)} (all with differing JXL bytes: {set(e2e_big) <= set(differ)})")
    check("compile ok/fail agreement", ok_agree == len(valid), f"{ok_agree}/{len(valid)}")
    check("<=256-byte agreement", limit_agree == len(both), f"{limit_agree}/{len(both)}")
    check("JXL bytes identical or a native outcome", all(explained),
          f"{len(both) - len(differ)} identical + {sum(explained)} native-nondeterministic of {len(both)}")
    check("decoder parity (same JXL)", dec_max <= 1 and dec_frac < 0.01,
          f"{dec_same}/{len(both)} pixel-identical, max abs diff {dec_max}, max differing-pixel fraction "
          f"{dec_frac:.2%}")
    check("end-to-end large differences only where native bytes differ", set(e2e_big) <= set(differ),
          f"{len(e2e_big)} programs")
    check("8-bit sRGB PNG", all(depth == 8 for depth, _ in formats), str(sorted(formats)))
    for i in fails:
        check(f"libjxl failure throws: {texts[i]!r}", not nat[i]["ok"] and not wasm[i]["ok"],
              f"native: {nat[i]['error'][:60]!r}; wasm: {wasm[i]['error'].splitlines()[0][:80]!r}")
    for i in (0, len(texts) - 1):
        check("Neon Chevron", wasm[i]["ok"] and wasm[i]["jxl"] == nat[i]["jxl"] and e2e[i][0] == 0,
              f"{len(wasm[i]['jxl'])} bytes identical to native, PNG pixel-identical to the native pipeline")


def main() -> None:
    print("runner:", JS)
    test_canonicalize()
    test_compile()
    print("ALL PASSED")


if __name__ == "__main__":
    main()
