"""Supra2.5-Pico policy, PyTorch backend: checkpoint I/O, CUDA-graph sampling, RL update.

Backend interface (mirrored by `mlx_model.py`, selected with `--backend`):
  load_policy(path) -> (policy, meta)        save_policy(policy, path, meta)
  load_weights(policy, path)                 freeze(policy)
  sample(policy, prompt_ids, n, max_new, eos_id, temperature) -> list[list[int]]
  new_optimizer(policy, lr)                  rl_update(policy, ref, opt, seqs, n_prompt, pad_id, adv, kl_beta)
  save_optimizer(opt, path)                  load_optimizer(opt, path)        OPT_SUFFIX

Policy files are framework-neutral safetensors: PyTorch state_dict names (tied `head.weight`
omitted), with the model config and run metadata as JSON strings in the header.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from .compiler import ROOT

BASE_DIR = ROOT / "models" / "Supra2.5-Pico"
POLICY_FORMAT = "jxlart-policy-v1"
OPT_SUFFIX = ".torch.pt"
_mod = None


def slm_module():
    global _mod
    if _mod is None:
        spec = importlib.util.spec_from_file_location("modeling_slm", BASE_DIR / "modeling_slm.py")
        _mod = importlib.util.module_from_spec(spec)
        sys.modules["modeling_slm"] = _mod
        spec.loader.exec_module(_mod)
    return _mod


def base_config() -> dict:
    return json.loads((BASE_DIR / "config.json").read_text())


def build(cfg: dict) -> torch.nn.Module:
    m = slm_module()
    return m.Model(vocab=cfg["vocab_size"], d=cfg["hidden_size"], blocks=cfg["num_hidden_layers"],
                   inter=cfg["intermediate_size"], n_q=cfg["num_attention_heads"],
                   n_kv=cfg["num_key_value_heads"], rotary_dim=cfg["rotary_dim"],
                   ctx=cfg["max_position_embeddings"], attnres=True)


def load_pretrained() -> torch.nn.Module:
    from safetensors.torch import load_file
    model = build(base_config())
    sd = load_file(str(BASE_DIR / "model.safetensors"))
    sd = {k.removeprefix("model."): v for k, v in sd.items()}
    sd.setdefault("head.weight", sd["embed.weight"])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k != "head.weight"]
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing} unexpected={unexpected}")
    model.head.weight = model.embed.weight
    return model


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(BASE_DIR), trust_remote_code=True)


def save_policy(model: torch.nn.Module, path: Path, meta: dict) -> None:
    from safetensors.torch import save_file
    path = Path(path)
    sd = {k: v.detach().float().contiguous().cpu() for k, v in model.state_dict().items() if k != "head.weight"}
    tmp = path.with_name(path.name + ".tmp")
    save_file(sd, str(tmp), metadata={"format": POLICY_FORMAT, "config": json.dumps(base_config()),
                                      "meta": json.dumps(meta)})
    os.replace(tmp, path)


def read_policy_file(path: Path) -> tuple[dict, dict, dict]:
    """(state_dict tensors, config, meta) from a policy safetensors file."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(str(path), "pt") as f:
        md = f.metadata() or {}
    if md.get("format") != POLICY_FORMAT:
        raise ValueError(f"{path} is not a {POLICY_FORMAT} file")
    return load_file(str(path)), json.loads(md["config"]), json.loads(md["meta"])


def load_policy(path: Path, device="cuda") -> tuple[torch.nn.Module, dict]:
    sd, cfg, meta = read_policy_file(path)
    model = build(cfg)
    sd["head.weight"] = sd["embed.weight"]
    model.load_state_dict(sd)
    model.head.weight = model.embed.weight
    return model.to(device), meta


def load_weights(model: torch.nn.Module, path: Path) -> None:
    """Copy weights in place (tensor identity kept, so cached CUDA graphs stay valid)."""
    sd, _, _ = read_policy_file(path)
    sd["head.weight"] = sd["embed.weight"]
    model.load_state_dict(sd)


def freeze(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def new_optimizer(model: torch.nn.Module, lr: float):
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)


def save_optimizer(opt, path: Path) -> None:
    tmp = Path(str(path) + ".tmp")
    torch.save(opt.state_dict(), tmp)
    os.replace(tmp, path)


def load_optimizer(opt, path: Path) -> None:
    opt.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))


# ---------------------------------------------------------------- static-cache decoding
class StaticDecoder:
    """One-token decode step over a preallocated KV cache, captured as a CUDA graph.

    Shapes are fixed per (batch, max_len), so the whole 10-layer step replays as a
    single graph launch. Math matches `Model.forward` (fp32, causal, XSA, AttnRes).
    """

    def __init__(self, model, batch: int, max_len: int):
        self.m, self.B, self.T = model, batch, max_len
        a0 = model.blocks[0].attn
        dev = model.embed.weight.device
        self.K = torch.zeros(len(model.blocks), batch, a0.n_kv, max_len, a0.hd, device=dev)
        self.V = torch.zeros_like(self.K)
        cos, sin = model.rope(max_len, dev)
        self.cos, self.sin = cos.contiguous(), sin.contiguous()
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.tok = torch.zeros(batch, 1, dtype=torch.long, device=dev)
        self.ar = torch.arange(max_len, device=dev)
        self.graph = None
        self.logits = None

    def _attn(self, li, attn, x, cos, sin, mask):
        mod = slm_module()
        B = x.shape[0]
        q = attn.wq(x).view(B, 1, attn.n_q, attn.hd).transpose(1, 2)
        k = attn.wk(x).view(B, 1, attn.n_kv, attn.hd).transpose(1, 2)
        v = attn.wv(x).view(B, 1, attn.n_kv, attn.hd).transpose(1, 2)
        q, k = attn.q_norm(q), attn.k_norm(k)
        q = mod.apply_rope(q, cos, sin, attn.rotary_dim)
        k = mod.apply_rope(k, cos, sin, attn.rotary_dim)
        self.K[li].index_copy_(2, self.pos, k)
        self.V[li].index_copy_(2, self.pos, v)
        rep = attn.n_q // attn.n_kv
        kk, vv = self.K[li].repeat_interleave(rep, 1), self.V[li].repeat_interleave(rep, 1)
        y = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
        if attn.use_xsa:
            vn = F.normalize(v.repeat_interleave(rep, 1), dim=-1)
            y = y - (y * vn).sum(-1, keepdim=True) * vn
        y = y.transpose(1, 2).reshape(B, 1, -1)
        y = y * torch.sigmoid(attn.gate(x))
        return attn.wo(y)

    def _step(self):
        m = self.m
        cos, sin = self.cos.index_select(2, self.pos), self.sin.index_select(2, self.pos)
        mask = (self.ar <= self.pos).view(1, 1, 1, -1)
        h = m.embed(self.tok)
        hs, hs_n = [h], [F.normalize(h, dim=-1)]
        for i, blk in enumerate(m.blocks):
            q = F.normalize(m.attnres_q[i], dim=-1)
            lg = torch.einsum("lbtd,d->lbt", torch.stack(hs_n, 0), q) / math.sqrt(m.embed.embedding_dim)
            x = torch.einsum("lbt,lbtd->btd", torch.softmax(lg, 0), torch.stack(hs, 0))
            x = x + self._attn(i, blk.attn, blk.n1(x), cos, sin, mask)
            h = x + blk.mlp(blk.n2(x))
            hs.append(h)
            hs_n.append(F.normalize(h, dim=-1))
        return m.head(m.final_norm(h))[:, -1]

    @torch.no_grad()
    def step(self, tokens: torch.Tensor, pos: int) -> torch.Tensor:
        """Feed tokens (B,) at position pos; returns logits (B, vocab) (a reused buffer)."""
        self.tok.copy_(tokens.view(-1, 1))
        self.pos.fill_(pos)
        if self.graph is None:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._step()
            torch.cuda.current_stream().wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.logits = self._step()
        self.graph.replay()
        return self.logits


@torch.no_grad()
def sample(model, prompt_ids: list[int], n: int, max_new: int, eos_id: int,
           temperature: float = 1.0) -> list[list[int]]:
    """Sample n continuations of one prompt; returns generated ids (eos included when emitted).

    The policy's parameters are captured by reference, so in-place optimizer updates are
    visible to a cached decoder; a decoder is rebuilt only when (model, n, length) changes.
    """
    model.eval()
    max_len = min(model.ctx, len(prompt_ids) + max_new)
    key = (n, max_len)
    cache = model.__dict__.setdefault("_decoders", {})
    if key not in cache:
        cache[key] = StaticDecoder(model, n, max_len)
    dec = cache[key]
    dev = model.embed.weight.device
    for pos, t in enumerate(prompt_ids):
        logits = dec.step(torch.full((n,), t, device=dev), pos)
    out = []
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    for pos in range(len(prompt_ids), max_len):
        probs = torch.softmax(logits / max(temperature, 1e-5), -1)
        nxt = torch.multinomial(probs, 1).squeeze(1)
        nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)
        out.append(nxt)
        done |= nxt == eos_id
        if pos % 16 == 0 and bool(done.all()):
            break
        if pos + 1 < max_len:
            logits = dec.step(nxt, pos)
    res = []
    for row in torch.stack(out, 1).tolist():
        res.append(row[: row.index(eos_id) + 1] if eos_id in row else row)
    return res


def sequence_logprobs(model, seqs: list[list[int]], n_prompt: int, pad_id: int, device=None):
    """Sum of token log-probs of each sequence's completion (positions >= n_prompt).

    Returns (sum_logp[B], token_logp[B, L-1] masked, mask[B, L-1]). bf16 autocast on CUDA, fp32 elsewhere.
    """
    device = device or model.embed.weight.device.type
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros(len(seqs), L - 1, device=device)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, device=device)
        mask[i, n_prompt - 1: len(s) - 1] = 1
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        logits, _ = model(ids)
    logp = torch.log_softmax(logits[:, :-1].float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    tok = logp * mask
    return tok.sum(-1), tok, mask


def rl_update(policy, ref, opt, seqs: list[list[int]], n_prompt: int, pad_id: int, adv, kl_beta: float) -> dict:
    """One KL-regularised policy-gradient step.

    loss = -mean(adv * sum_logp) / mean_len + kl_beta * mean(sum_t k3(ref || policy)) / mean_len,
    k3(d) = exp(d) - d - 1 with d = logp_ref - logp_policy per token; grad-norm clipped at 1.0.
    """
    policy.train()
    logp_sum, tok_logp, mask = sequence_logprobs(policy, seqs, n_prompt, pad_id)
    with torch.no_grad():
        _, ref_tok, _ = sequence_logprobs(ref, seqs, n_prompt, pad_id)
    diff = (ref_tok - tok_logp) * mask
    kl_tok = (torch.exp(diff) - diff - 1) * mask
    norm = mask.sum(-1).mean()
    pg = -(torch.as_tensor(adv, device=logp_sum.device, dtype=torch.float32) * logp_sum).mean() / norm
    kl = kl_tok.sum(-1).mean() / norm
    loss = pg + kl_beta * kl
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()
    return {"pg_loss": float(pg), "kl": float(kl), "grad_norm": float(gn), "mean_len": float(norm)}
