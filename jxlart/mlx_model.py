"""Supra2.5-Pico policy, Apple MLX backend: drop-in replacement for `jxlart.model` (PyTorch).

Same interface and the same framework-neutral policy files (PyTorch state_dict names, tied
`head.weight` omitted, config/meta JSON in the safetensors header), so policies move freely
between backends. Optimizer state is MLX-specific (`OPT_SUFFIX`).

Math mirrors `models/Supra2.5-Pico/modeling_slm.py` in float32: pre-norm RMSNorm, GQA with
per-head QK-norm, partial interleaved RoPE, XSA, sigmoid output gate, SwiGLU, AttnRes, tied head.
Imports only mlx / numpy / safetensors (no torch).
"""
from __future__ import annotations

import json
import math
import os
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from safetensors import safe_open
from safetensors.numpy import save_file

POLICY_FORMAT = "jxlart-policy-v1"
OPT_FORMAT = "jxlart-mlx-adamw-v1"
OPT_SUFFIX = ".mlx.safetensors"
ROPE_THETA = 10000.0


def _normalize(x: mx.array) -> mx.array:
    """F.normalize(x, dim=-1): x / max(||x||, 1e-12)."""
    return x / mx.maximum(mx.sqrt((x * x).sum(-1, keepdims=True)), 1e-12)


class Attention(nn.Module):
    def __init__(self, d: int, n_q: int, n_kv: int, rotary_dim: int):
        super().__init__()
        self.n_q, self.n_kv, self.hd, self.rotary_dim = n_q, n_kv, d // n_q, rotary_dim
        self.wq = nn.Linear(d, n_q * self.hd, bias=False)
        self.wk = nn.Linear(d, n_kv * self.hd, bias=False)
        self.wv = nn.Linear(d, n_kv * self.hd, bias=False)
        self.wo = nn.Linear(n_q * self.hd, d, bias=False)
        self.gate = nn.Linear(d, n_q * self.hd, bias=True)
        self.q_norm = nn.RMSNorm(self.hd, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.hd, eps=1e-6)

    def _qkv(self, x: mx.array, offset):
        B, T, _ = x.shape
        q = self.wq(x).reshape(B, T, self.n_q, self.hd).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_kv, self.hd).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_kv, self.hd).transpose(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)
        rope = partial(mx.fast.rope, dims=self.rotary_dim, traditional=True, base=ROPE_THETA, scale=1.0,
                       offset=offset)
        return rope(q), rope(k), v

    def _attend(self, x, q, k, v, v_self, mask):
        """q (B,n_q,T,hd), k/v (B,n_kv,S,hd), v_self (B,n_kv,T,hd), mask bool broadcastable to (T,S)."""
        B, _, T, hd = q.shape
        rep = self.n_q // self.n_kv
        qg = q.reshape(B, self.n_kv, rep, T, hd)  # head h uses kv head h // rep (repeat_interleave)
        s = (qg * (1.0 / math.sqrt(hd))) @ k[:, :, None].swapaxes(-1, -2)
        s = mx.where(mask, s, -mx.inf)
        y = mx.softmax(s, axis=-1, precise=True) @ v[:, :, None]
        vn = _normalize(v_self)[:, :, None]  # XSA: drop the component along the own value vector
        y = y - (y * vn).sum(-1, keepdims=True) * vn
        y = y.reshape(B, self.n_q, T, hd).transpose(0, 2, 1, 3).reshape(B, T, self.n_q * hd)
        return self.wo(y * mx.sigmoid(self.gate(x)))

    def __call__(self, x: mx.array):
        """Full causal attention; also returns (k, v) for cache prefill."""
        T = x.shape[1]
        q, k, v = self._qkv(x, 0)
        ar = mx.arange(T)
        return self._attend(x, q, k, v, v, ar[None, :] <= ar[:, None]), k, v

    def step(self, x: mx.array, pos: mx.array, K: mx.array, V: mx.array):
        """One token at position `pos` against a fixed-size cache K/V (B,n_kv,Tmax,hd)."""
        q, k, v = self._qkv(x, pos)
        start = pos.reshape(1)
        K = mx.slice_update(K, k, start, (2,))
        V = mx.slice_update(V, v, start, (2,))
        mask = mx.arange(K.shape[2]) <= pos
        return self._attend(x, q, K, V, v, mask), K, V


class SwiGLU(nn.Module):
    def __init__(self, d: int, inter: int):
        super().__init__()
        self.w1 = nn.Linear(d, inter, bias=False)
        self.w3 = nn.Linear(d, inter, bias=False)
        self.w2 = nn.Linear(inter, d, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, d, n_q, n_kv, rotary_dim, inter):
        super().__init__()
        self.n1 = nn.RMSNorm(d, eps=1e-6)
        self.attn = Attention(d, n_q, n_kv, rotary_dim)
        self.n2 = nn.RMSNorm(d, eps=1e-6)
        self.mlp = SwiGLU(d, inter)


class Model(nn.Module):
    """Parameter names match the PyTorch state_dict (minus the tied `head.weight`)."""

    def __init__(self, cfg: dict):
        super().__init__()
        d, L = cfg["hidden_size"], cfg["num_hidden_layers"]
        self.d, self.ctx = d, cfg["max_position_embeddings"]
        self.embed = nn.Embedding(cfg["vocab_size"], d)
        self.blocks = [Block(d, cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["rotary_dim"],
                             cfg["intermediate_size"]) for _ in range(L)]
        self.attnres_q = mx.zeros((L, d))
        self.final_norm = nn.RMSNorm(d, eps=1e-6)

    def _mix(self, i: int, hs: list, hs_n: list) -> mx.array:
        """AttnRes: softmax over all previous layer outputs with the normalized query of layer i."""
        q = _normalize(self.attnres_q[i])
        w = mx.softmax((mx.stack(hs_n, 0) @ q) / math.sqrt(self.d), axis=0, precise=True)
        return (w[..., None] * mx.stack(hs, 0)).sum(0)

    def _run(self, ids: mx.array, attn_fn) -> mx.array:
        h = self.embed(ids)
        hs, hs_n = [h], [_normalize(h)]
        for i, blk in enumerate(self.blocks):
            x = self._mix(i, hs, hs_n)
            x = x + attn_fn(i, blk.attn, blk.n1(x))
            h = x + blk.mlp(blk.n2(x))
            hs.append(h)
            hs_n.append(_normalize(h))
        return self.embed.as_linear(self.final_norm(h))

    def __call__(self, ids: mx.array, kv_out: list | None = None) -> mx.array:
        """Full forward (B,T) -> logits (B,T,vocab); appends per-layer (k, v) to kv_out if given."""
        def attn_fn(i, attn, x):
            y, k, v = attn(x)
            if kv_out is not None:
                kv_out.append((k, v))
            return y
        return self._run(ids, attn_fn)

    def step(self, tok: mx.array, pos: mx.array, Ks: list, Vs: list):
        """Decode tokens (B,) at scalar position pos; returns (logits (B,vocab), Ks, Vs)."""
        Ks, Vs = list(Ks), list(Vs)

        def attn_fn(i, attn, x):
            y, Ks[i], Vs[i] = attn.step(x, pos, Ks[i], Vs[i])
            return y
        return self._run(tok[:, None], attn_fn)[:, 0], Ks, Vs


# ---------------------------------------------------------------- checkpoint I/O
def _read_policy_file(path) -> tuple[dict, str, dict]:
    """(numpy tensors, raw config JSON, meta) from a policy safetensors file."""
    with safe_open(str(path), "numpy") as f:
        md = f.metadata() or {}
        if md.get("format") != POLICY_FORMAT:
            raise ValueError(f"{path} is not a {POLICY_FORMAT} file")
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    return tensors, md["config"], json.loads(md["meta"])


def load_weights(policy: Model, path) -> None:
    """Replace the weights in place (same module, cached decode steps stay valid)."""
    tensors, _, _ = _read_policy_file(path)
    policy.load_weights([(k, mx.array(v.astype(np.float32))) for k, v in tensors.items()], strict=True)
    mx.eval(policy.parameters())


def load_policy(path) -> tuple[Model, dict]:
    _, cfg_json, meta = _read_policy_file(path)
    policy = Model(json.loads(cfg_json))
    policy.config_json = cfg_json  # written back verbatim by save_policy
    load_weights(policy, path)
    return policy, meta


def save_policy(policy: Model, path, meta: dict) -> None:
    path = Path(path)
    sd = {k: np.array(v.astype(mx.float32)) for k, v in tree_flatten(policy.parameters())}
    tmp = path.with_name(path.name + ".tmp")
    save_file(sd, str(tmp), metadata={"format": POLICY_FORMAT, "config": policy.config_json,
                                      "meta": json.dumps(meta)})
    os.replace(tmp, path)


def freeze(policy: Model) -> Model:
    policy.freeze()
    return policy


# ---------------------------------------------------------------- optimizer
def new_optimizer(policy: Model, lr: float) -> optim.AdamW:
    """AdamW with PyTorch semantics (bias-corrected moments)."""
    return optim.AdamW(learning_rate=lr, betas=[0.9, 0.95], eps=1e-8, weight_decay=0.0, bias_correction=True)


def save_optimizer(opt: optim.Optimizer, path) -> None:
    """Moments, step count and learning rate (restored on load, like torch's state_dict)."""
    path = Path(path)
    flat = {k: np.array(v) for k, v in tree_flatten(opt.state)}
    tmp = path.with_name(path.name + ".tmp")
    save_file(flat, str(tmp), metadata={"format": OPT_FORMAT})
    os.replace(tmp, path)


def load_optimizer(opt: optim.Optimizer, path) -> None:
    with safe_open(str(path), "numpy") as f:
        if (f.metadata() or {}).get("format") != OPT_FORMAT:
            raise ValueError(f"{path} is not a {OPT_FORMAT} file")
        flat = [(k, mx.array(f.get_tensor(k))) for k in f.keys()]
    opt.state = tree_unflatten(flat)


# ---------------------------------------------------------------- decoding
def _decode_fn(policy: Model):
    """Compiled one-token decode step; re-reads the policy's current weights on every call."""
    fn = policy.__dict__.get("_decode_step")
    if fn is None:
        fn = mx.compile(policy.step, inputs=policy.state)
        policy.__dict__["_decode_step"] = fn
    return fn


def _empty_cache(policy: Model, batch: int, max_len: int) -> tuple[list, list]:
    a = policy.blocks[0].attn
    shape = (batch, a.n_kv, max_len, a.hd)
    return [mx.zeros(shape) for _ in policy.blocks], [mx.zeros(shape) for _ in policy.blocks]


def sample(policy: Model, prompt_ids: list[int], n: int, max_new: int, eos_id: int,
           temperature: float = 1.0) -> list[list[int]]:
    """Sample n continuations of one prompt; returns generated ids (eos included when emitted)."""
    P = len(prompt_ids)
    max_len = min(policy.ctx, P + max_new)
    if max_len <= P:
        return [[] for _ in range(n)]
    step = _decode_fn(policy)
    kv: list = []
    logits = policy(mx.array([prompt_ids]), kv_out=kv)[:, -1]  # prefill once, share across rows
    Ks, Vs = _empty_cache(policy, n, max_len)
    for i, (k, v) in enumerate(kv):
        Ks[i][:, :, :P] = k
        Vs[i][:, :, :P] = v
    logits = mx.broadcast_to(logits, (n, logits.shape[-1]))
    inv_t = 1.0 / max(temperature, 1e-5)
    done = mx.zeros((n,), dtype=mx.bool_)
    out = []
    for pos in range(P, max_len):
        nxt = mx.random.categorical(logits * inv_t).astype(mx.int32)
        nxt = mx.where(done, eos_id, nxt)
        done = done | (nxt == eos_id)
        out.append(nxt)
        if pos + 1 < max_len:
            logits, Ks, Vs = step(nxt, mx.array(pos, dtype=mx.int32), Ks, Vs)
        mx.async_eval(logits, Ks, Vs, done)
        if (pos - P) % 8 == 7 and done.all().item():
            break
    res = []
    for row in np.array(mx.stack(out, 1)).tolist():
        res.append(row[: row.index(eos_id) + 1] if eos_id in row else row)
    return res


# ---------------------------------------------------------------- scoring / RL
def _batch(seqs: list[list[int]], n_prompt: int, pad_id: int) -> tuple[mx.array, mx.array]:
    L = max(len(s) for s in seqs)
    ids = np.full((len(seqs), L), pad_id, np.int32)
    mask = np.zeros((len(seqs), L - 1), np.float32)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = s
        mask[i, n_prompt - 1: len(s) - 1] = 1
    return mx.array(ids), mx.array(mask)


def _log_softmax(x: mx.array) -> mx.array:
    """log_softmax over the vocab with a two-level sum. MLX's flat float32 CPU sum over the 4104-entry
    vocab is off by ~1e-4 relative (~1e-3 over a sequence's log-prob sum); summing 64 zero-padded
    blocks matches float64 to ~3e-6 per token."""
    m = mx.stop_gradient(x.max(-1, keepdims=True))
    e = mx.exp(x - m)
    pad = -x.shape[-1] % 64
    if pad:
        e = mx.pad(e, [(0, 0)] * (e.ndim - 1) + [(0, pad)])
    s = e.reshape(*e.shape[:-1], 64, -1).sum(-1).sum(-1, keepdims=True)
    return x - m - mx.log(s)


def _token_logp(policy: Model, ids: mx.array, mask: mx.array) -> mx.array:
    logp = _log_softmax(policy(ids)[:, :-1])
    return mx.take_along_axis(logp, ids[:, 1:, None], axis=-1)[..., 0] * mask


def sequence_logprobs(policy: Model, seqs: list[list[int]], n_prompt: int, pad_id: int):
    """Sum of token log-probs of each sequence's completion (positions >= n_prompt).

    Returns (sum_logp[B], token_logp[B, L-1] masked, mask[B, L-1]) as float32 mx arrays.
    """
    ids, mask = _batch(seqs, n_prompt, pad_id)
    tok = _token_logp(policy, ids, mask)
    return tok.sum(-1), tok, mask


def rl_update(policy: Model, ref: Model, opt, seqs: list[list[int]], n_prompt: int, pad_id: int, adv,
              kl_beta: float) -> dict:
    """One KL-regularised policy-gradient step (same loss as `jxlart.model.rl_update`).

    loss = -mean(adv * sum_logp) / mean_len + kl_beta * mean(sum_t k3(ref || policy)) / mean_len,
    k3(d) = exp(d) - d - 1 with d = logp_ref - logp_policy per token; grad-norm clipped at 1.0.
    """
    ids, mask = _batch(seqs, n_prompt, pad_id)
    ref_tok = mx.stop_gradient(_token_logp(ref, ids, mask))
    adv = mx.array(np.asarray(adv, np.float32))
    norm = mask.sum(-1).mean()

    def loss_fn():
        tok = _token_logp(policy, ids, mask)
        diff = (ref_tok - tok) * mask
        kl_tok = (mx.exp(diff) - diff - 1) * mask
        pg = -(adv * tok.sum(-1)).mean() / norm
        kl = kl_tok.sum(-1).mean() / norm
        return pg + kl_beta * kl, pg, kl

    (_, pg, kl), grads = nn.value_and_grad(policy, loss_fn)()
    grads, gn = optim.clip_grad_norm(grads, 1.0)
    opt.update(policy, grads)
    mx.eval(policy.parameters(), opt.state, pg, kl, gn, norm)
    return {"pg_loss": pg.item(), "kl": kl.item(), "grad_norm": gn.item(), "mean_len": norm.item()}
