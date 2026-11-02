"""Export policies and the tokenizer for the in-browser blind A/B game (web/blind/).

  python -m jxlart.export_web --run runs/rlhf-v2 --out web/blind/assets

Writes, for each policy, `<name>.json` (config, tensor table, source file + sha256) and `<name>.bin`
(float16 little-endian weights, tensors back to back at the byte offsets in the table), plus
`tokenizer.json`: the per-id byte strings of the byte-level BPE (base64) and the length-token /
digit ids that `web/blind/lib/tok.js` needs to reproduce `tok.decode`.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .compiler import ROOT
from .corpus import PROMPT

WEB_FORMAT = "jxlart-web-v1"
TOKENIZER_FORMAT = "jxlart-web-tokenizer-v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def export_policy(src: Path, out: Path, name: str) -> dict:
    """Write out/<name>.json + out/<name>.bin from a jxlart-policy-v1 safetensors file; returns the manifest."""
    with safe_open(str(src), "numpy") as f:
        md = f.metadata() or {}
        if md.get("format") != "jxlart-policy-v1":
            raise ValueError(f"{src}: not a jxlart-policy-v1 file (format={md.get('format')!r})")
        tensors = {k: f.get_tensor(k) for k in sorted(f.keys())}
    table, chunks, offset = [], [], 0
    for k, t in tensors.items():
        b = np.ascontiguousarray(t, dtype="<f2")
        if not np.isfinite(b).all():
            raise ValueError(f"{src}: {k} overflows float16")
        b = b.tobytes()
        table.append({"name": k, "shape": list(t.shape), "offset": offset})
        chunks.append(b)
        offset += len(b)
    manifest = {
        "format": WEB_FORMAT,
        "dtype": "float16",
        "config": json.loads(md["config"]),
        "meta": json.loads(md.get("meta", "{}")),
        "source": {"file": _rel(src), "sha256": _sha256(src)},
        "bin": f"{name}.bin",
        "bin_bytes": offset,
        "tensors": table,
    }
    _write_atomic(out / f"{name}.bin", b"".join(chunks))
    _write_atomic(out / f"{name}.json", json.dumps(manifest, indent=1).encode())
    return manifest


def _bytes_to_unicode() -> dict[str, int]:
    """GPT-2 byte-level alphabet: printable char -> byte (inverse of the encoder's byte -> char map)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def export_tokenizer(out: Path) -> dict:
    from .model import BASE_DIR, load_tokenizer
    tok = load_tokenizer()
    mod = sys.modules[type(tok).__module__]  # tokenization_pascaline (loaded via trust_remote_code)
    tj = json.loads((BASE_DIR / "tokenizer.json").read_text(encoding="utf-8"))
    if tj["decoder"]["type"] != "ByteLevel":
        raise ValueError(f"unsupported decoder {tj['decoder']['type']}")
    id_to_token = {i: t for t, i in tj["model"]["vocab"].items()}
    id_to_token.update({a["id"]: a["content"] for a in tj["added_tokens"]})
    char_bytes = _bytes_to_unicode()
    vocab = []
    for i in range(mod.LEN_BASE):
        t = id_to_token[i]
        # tokenizers' ByteLevel decoder: map every char back to its byte, or keep the UTF-8 if any char is unmapped
        b = bytes(char_bytes[c] for c in t) if all(c in char_bytes for c in t) else t.encode("utf-8")
        vocab.append(base64.b64encode(b).decode())
    data = {
        "format": TOKENIZER_FORMAT,
        "source": {"file": "models/Supra2.5-Pico/tokenizer.json", "sha256": _sha256(BASE_DIR / "tokenizer.json")},
        "eosId": tok.eos_token_id,
        "padId": tok.pad_token_id,
        "lenBase": mod.LEN_BASE,
        "lenCount": mod.LEN_MAX,
        "digitLo": mod.DIGIT_LO,
        "digitHi": mod.DIGIT_HI,
        "prompt": PROMPT,
        "promptIds": tok(PROMPT)["input_ids"],
        "vocab": vocab,
    }
    _write_atomic(out / "tokenizer.json", json.dumps(data, separators=(",", ":")).encode())
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/rlhf-v2", help="run dir containing policy/")
    ap.add_argument("--out", default="web/blind/assets")
    ap.add_argument("--base", default="policy/step_00000.safetensors", help="relative to --run")
    ap.add_argument("--tuned", default="policy/step_00820.safetensors", help="relative to --run")
    args = ap.parse_args()
    run, out = ROOT / args.run, ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for name, rel in (("base", args.base), ("tuned", args.tuned)):
        m = export_policy(run / rel, out, name)
        n = sum(int(np.prod(t["shape"])) for t in m["tensors"])
        print(f"{name}: {run / rel} -> {out / name}.bin ({n:,} params, {m['bin_bytes']:,} bytes)")
    t = export_tokenizer(out)
    print(f"tokenizer: {len(t['vocab'])} ids, promptIds={t['promptIds']} -> {out / 'tokenizer.json'}")
    total = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    print(f"total {out}: {total:,} bytes")


if __name__ == "__main__":
    main()
