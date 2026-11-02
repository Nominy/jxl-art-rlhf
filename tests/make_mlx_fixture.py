"""Generate PyTorch reference outputs for MLX parity checks (CPU, float32).

  python tests/make_mlx_fixture.py --policy <step_N.safetensors> --ref <step_0.safetensors> --out tests/fixtures/parity.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from jxlart import model as M
from jxlart.corpus import PROMPT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    tok = M.load_tokenizer()
    prompt = tok(PROMPT)["input_ids"]
    texts = ["Bitdepth 10\nDeltaPalette\nGaborish\nWidth 512\nHeight 512\nif W+N-NW > -77\n  - AvgN+NW -1\n  - AvgW+N 90\n",
             "Bitdepth 8\nRCT 6\nif c > 0\n  if x > 300\n    - Set 40\n    - N 3\n  - Gradient -2\n",
             "Width 256\nHeight 256\nUpsample 4\n- Weighted 7\n"]
    seqs = [prompt + tok(t)["input_ids"] + [tok.eos_token_id] for t in texts]
    pad = tok.pad_token_id
    policy, _ = M.load_policy(args.policy, "cpu")
    ref = M.freeze(M.load_policy(args.ref, "cpu")[0])
    L = max(len(s) for s in seqs)
    ids = np.full((len(seqs), L), pad, np.int64)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = s
    with torch.no_grad():
        logits = policy(torch.tensor(ids))[0].numpy()
        lp_sum, _, mask = M.sequence_logprobs(policy, seqs, len(prompt), pad)
    adv = np.array([1.0, -0.5, -0.5], np.float32)
    probe = torch.tensor(ids[:1, :12])
    opt = M.new_optimizer(policy, 3e-5)
    stats = M.rl_update(policy, ref, opt, seqs, len(prompt), pad, adv, 0.05)
    with torch.no_grad():
        post = policy(probe)[0].numpy()
    np.savez(args.out, ids=ids, n_prompt=len(prompt), pad_id=pad, logits=logits, logp_sum=lp_sum.numpy(),
             mask=mask.numpy(), adv=adv, kl_beta=0.05, lr=3e-5, probe=probe.numpy(), post_update_logits=post,
             stats=json.dumps(stats), seq_lens=np.array([len(s) for s in seqs]))
    print("saved", args.out, stats)


if __name__ == "__main__":
    main()
