"""Sample JXL art from a policy checkpoint; save valid <=256-byte images and report quality metrics.

  python -m jxlart.generate --policy runs/rlhf-v2/policy/online.safetensors --n 128 --out out/latest
  python -m jxlart.generate --policy ... --backend mlx      # Apple silicon

`--data` adds `novel`: fraction of good samples whose tree text is not in the training corpus.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from . import model as M
from .sft import gallery_records, sample_eval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", help="build_data shard dir, for the novelty metric")
    ap.add_argument("--backend", choices=["torch", "mlx"], default="mlx" if sys.platform == "darwin" else "torch")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    B = importlib.import_module("jxlart.mlx_model" if args.backend == "mlx" else "jxlart.model")
    tok = M.load_tokenizer()
    policy, _ = B.load_policy(Path(args.policy))
    m = sample_eval(policy, tok, args.n, args.temperature, out / "sheet.png", save_dir=out, backend=B)
    texts = m.pop("texts")
    if args.data:
        corpus = {r["text"] for r in gallery_records()}
        for f in Path(args.data).glob("shard_*.jsonl"):
            corpus |= {json.loads(l)["text"] for l in f.open(encoding="utf-8") if l.strip()}
        m["novel"] = sum(t not in corpus for t in texts) / len(texts) if texts else None
    m |= {"policy": args.policy, "n": args.n, "temperature": args.temperature, "backend": args.backend}
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
