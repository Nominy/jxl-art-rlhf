// Supra2.5-Pico inference in plain JS (float32 storage, one sequence, KV cache).
// Mirrors jxlart/mlx_model.py / models/Supra2.5-Pico/modeling_slm.py: pre-norm RMSNorm (eps 1e-6), GQA with
// per-head q/k RMSNorm, interleaved partial RoPE, XSA, sigmoid output gate, SwiGLU, AttnRes mixing of all
// previous layer outputs, final RMSNorm, head tied to the embedding.
// Weights: `<name>.json` manifest + `<name>.bin` float16 written by `python -m jxlart.export_web`.

const EPS = 1e-6;
const ROPE_THETA = 10000;

export async function fetchBytes(url) {
  const u = new URL(url, globalThis.location?.href);
  if (u.protocol === 'file:' && typeof process !== 'undefined') {
    const { readFile } = await import('node:fs/promises');
    return new Uint8Array(await readFile(u));
  }
  const r = await fetch(u);
  if (!r.ok) throw new Error(`${u}: HTTP ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

let F16 = null;
function f16Table() {   // all 65536 half-precision values as float32
  if (F16) return F16;
  F16 = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    F16[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1024 + m) * 2 ** (e - 25);
  }
  return F16;
}

export async function loadModel(url) {
  const absUrl = new URL(url, globalThis.location?.href);
  const manifest = JSON.parse(new TextDecoder().decode(await fetchBytes(absUrl)));
  if (manifest.dtype !== 'float16') throw new Error(`unsupported dtype ${manifest.dtype}`);
  const bin = await fetchBytes(new URL(manifest.bin, absUrl));
  if (bin.length !== manifest.bin_bytes) throw new Error(`${manifest.bin}: ${bin.length} bytes, expected ${manifest.bin_bytes}`);
  const half = new Uint16Array(bin.buffer, bin.byteOffset, bin.length >> 1);   // little-endian hosts only (all browsers)
  const table = f16Table(), w = {};
  for (const t of manifest.tensors) {
    const n = t.shape.reduce((a, b) => a * b, 1), src = half.subarray(t.offset >> 1, (t.offset >> 1) + n);
    const dst = new Float32Array(n);
    for (let i = 0; i < n; i++) dst[i] = table[src[i]];
    w[t.name] = dst;
  }
  return new Model(manifest.config, w);
}

// y[0..rows) = W (rows x cols, row-major) . x; eight rows at a time so each x[i] load feeds eight products
function matvec(W, x, y, rows, cols) {
  let r = 0;
  for (; r + 7 < rows; r += 8) {
    const o0 = r * cols, o1 = o0 + cols, o2 = o1 + cols, o3 = o2 + cols;
    const o4 = o3 + cols, o5 = o4 + cols, o6 = o5 + cols, o7 = o6 + cols;
    let a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    for (let i = 0; i < cols; i++) {
      const xi = x[i];
      a0 += W[o0 + i] * xi; a1 += W[o1 + i] * xi; a2 += W[o2 + i] * xi; a3 += W[o3 + i] * xi;
      a4 += W[o4 + i] * xi; a5 += W[o5 + i] * xi; a6 += W[o6 + i] * xi; a7 += W[o7 + i] * xi;
    }
    y[r] = a0; y[r + 1] = a1; y[r + 2] = a2; y[r + 3] = a3; y[r + 4] = a4; y[r + 5] = a5; y[r + 6] = a6; y[r + 7] = a7;
  }
  for (; r < rows; r++) {
    let a = 0;
    for (let i = 0, o = r * cols; i < cols; i++) a += W[o + i] * x[i];
    y[r] = a;
  }
}

// out[oo..oo+n) = x[xo..xo+n) / rms * g
function rmsnorm(x, g, out, n, xo = 0, oo = xo) {
  let ss = 0;
  for (let i = 0; i < n; i++) ss += x[xo + i] * x[xo + i];
  const s = 1 / Math.sqrt(ss / n + EPS);
  for (let i = 0; i < n; i++) out[oo + i] = x[xo + i] * s * g[i];
}

function invNorm(x, off, n) {   // 1 / max(||x||, 1e-12), as F.normalize
  let ss = 0;
  for (let i = 0; i < n; i++) ss += x[off + i] * x[off + i];
  return 1 / Math.max(Math.sqrt(ss), 1e-12);
}

class Model {
  constructor(config, w) {
    const c = this.config = config;
    const d = this.d = c.hidden_size, L = this.L = c.num_hidden_layers;
    this.V = c.vocab_size; this.ctx = c.max_position_embeddings; this.inter = c.intermediate_size;
    this.nq = c.num_attention_heads; this.nkv = c.num_key_value_heads; this.hd = d / this.nq; this.rot = c.rotary_dim;
    this.embed = w['embed.weight'];
    this.finalNorm = w['final_norm.weight'];
    this.blocks = [];
    for (let i = 0; i < L; i++) {
      const p = (n) => { const t = w[`blocks.${i}.${n}`]; if (!t) throw new Error(`missing blocks.${i}.${n}`); return t; };
      this.blocks.push({
        n1: p('n1.weight'), n2: p('n2.weight'), wq: p('attn.wq.weight'), wk: p('attn.wk.weight'), wv: p('attn.wv.weight'),
        wo: p('attn.wo.weight'), gw: p('attn.gate.weight'), gb: p('attn.gate.bias'), qn: p('attn.q_norm.weight'),
        kn: p('attn.k_norm.weight'), w1: p('mlp.w1.weight'), w2: p('mlp.w2.weight'), w3: p('mlp.w3.weight'),
      });
    }
    // normalized AttnRes queries, pre-scaled by 1/sqrt(d)
    this.arq = new Float32Array(L * d);
    for (let i = 0; i < L; i++) {
      const s = invNorm(w.attnres_q, i * d, d) / Math.sqrt(d);
      for (let j = 0; j < d; j++) this.arq[i * d + j] = w.attnres_q[i * d + j] * s;
    }
    // RoPE tables, float32 like torch: inv = 1 / theta^(2j/rot), angle = pos * inv
    const half = this.rot / 2;
    this.cos = new Float32Array(this.ctx * half); this.sin = new Float32Array(this.ctx * half);
    for (let j = 0; j < half; j++) {
      const inv = Math.fround(1 / Math.fround(ROPE_THETA ** Math.fround((2 * j) / this.rot)));
      for (let t = 0; t < this.ctx; t++) {
        const a = Math.fround(t * inv);
        this.cos[t * half + j] = Math.cos(a); this.sin[t * half + j] = Math.sin(a);
      }
    }
    const kvd = this.nkv * this.hd;
    this.K = new Float32Array(L * this.ctx * kvd); this.Vc = new Float32Array(L * this.ctx * kvd);
    this.hs = new Float32Array((L + 1) * d);   // embedding + every block output of the current token
    this.hsn = new Float32Array(L + 1);        // their inverse norms
    this.x = new Float32Array(d); this.xn = new Float32Array(d); this.q = new Float32Array(d);
    this.k = new Float32Array(kvd); this.v = new Float32Array(kvd); this.y = new Float32Array(d);
    this.g = new Float32Array(d); this.o = new Float32Array(d); this.a = new Float32Array(this.inter);
    this.b = new Float32Array(this.inter); this.mix = new Float64Array(L + 1); this.sc = new Float64Array(this.ctx);
    this.out = new Float32Array(this.V);
    this.probs = new Float64Array(this.V);
  }

  _rope(x, off, pos) {
    const half = this.rot / 2, base = pos * half;
    for (let j = 0; j < half; j++) {
      const c = this.cos[base + j], s = this.sin[base + j], x1 = x[off + 2 * j], x2 = x[off + 2 * j + 1];
      x[off + 2 * j] = x1 * c - x2 * s;
      x[off + 2 * j + 1] = x1 * s + x2 * c;
    }
  }

  // Feed token `tok` at position `pos` (cache holds positions < pos); returns logits (reused buffer) if wanted.
  step(tok, pos, wantLogits = true) {
    const { d, L, hd, nq, nkv, inter, hs, hsn, x, xn, q, k, v, y, g, o, a, b, mix, sc } = this;
    const kvd = nkv * hd, rep = nq / nkv, scale = 1 / Math.sqrt(hd);
    hs.set(this.embed.subarray(tok * d, tok * d + d), 0);
    hsn[0] = invNorm(hs, 0, d);
    for (let i = 0; i < L; i++) {
      const B = this.blocks[i];
      // AttnRes: x = sum_l softmax_l(<h_l/|h_l|, q_i>/sqrt(d)) h_l over l = 0..i
      let mx = -Infinity;
      for (let l = 0; l <= i; l++) {
        let s = 0;
        for (let j = 0; j < d; j++) s += hs[l * d + j] * this.arq[i * d + j];
        mix[l] = s * hsn[l];
        if (mix[l] > mx) mx = mix[l];
      }
      let tot = 0;
      for (let l = 0; l <= i; l++) tot += (mix[l] = Math.exp(mix[l] - mx));
      x.fill(0);
      for (let l = 0; l <= i; l++) {
        const wl = mix[l] / tot;
        for (let j = 0; j < d; j++) x[j] += wl * hs[l * d + j];
      }
      // attention
      rmsnorm(x, B.n1, xn, d);
      matvec(B.wq, xn, q, d, d);
      matvec(B.wk, xn, k, kvd, d);
      matvec(B.wv, xn, v, kvd, d);
      matvec(B.gw, xn, g, d, d);
      for (let h = 0; h < nq; h++) { rmsnorm(q, B.qn, q, hd, h * hd); this._rope(q, h * hd, pos); }
      for (let h = 0; h < nkv; h++) { rmsnorm(k, B.kn, k, hd, h * hd); this._rope(k, h * hd, pos); }
      const cbase = i * this.ctx * kvd, K = this.K, Vc = this.Vc;
      K.set(k, cbase + pos * kvd); Vc.set(v, cbase + pos * kvd);
      for (let h = 0; h < nq; h++) {
        const qo = h * hd, ko = (h / rep | 0) * hd;
        let m = -Infinity;
        for (let t = 0; t <= pos; t++) {
          const kb = cbase + t * kvd + ko;
          let s = 0;
          for (let j = 0; j < hd; j++) s += q[qo + j] * K[kb + j];
          sc[t] = s * scale;
          if (sc[t] > m) m = sc[t];
        }
        let z = 0;
        for (let t = 0; t <= pos; t++) z += (sc[t] = Math.exp(sc[t] - m));
        for (let j = 0; j < hd; j++) y[qo + j] = 0;
        for (let t = 0; t <= pos; t++) {
          const p = sc[t] / z, vb = cbase + t * kvd + ko;
          for (let j = 0; j < hd; j++) y[qo + j] += p * Vc[vb + j];
        }
        // XSA: drop the component along the token's own (normalized) value vector
        const vi = invNorm(v, ko, hd);
        let dot = 0;
        for (let j = 0; j < hd; j++) dot += y[qo + j] * v[ko + j] * vi;
        for (let j = 0; j < hd; j++) y[qo + j] -= dot * v[ko + j] * vi;
      }
      for (let j = 0; j < d; j++) y[j] *= 1 / (1 + Math.exp(-(g[j] + B.gb[j])));
      matvec(B.wo, y, o, d, d);
      for (let j = 0; j < d; j++) x[j] += o[j];
      // SwiGLU MLP
      rmsnorm(x, B.n2, xn, d);
      matvec(B.w1, xn, a, inter, d);
      matvec(B.w3, xn, b, inter, d);
      for (let j = 0; j < inter; j++) a[j] = a[j] / (1 + Math.exp(-a[j])) * b[j];
      matvec(B.w2, a, o, d, inter);
      const ho = (i + 1) * d;
      for (let j = 0; j < d; j++) hs[ho + j] = x[j] + o[j];
      hsn[i + 1] = invNorm(hs, ho, d);
    }
    if (!wantLogits) return null;
    rmsnorm(hs, this.finalNorm, xn, d, L * d, 0);   // final_norm(h_L)
    matvec(this.embed, xn, this.out, this.V, d);
    return this.out;
  }

  // Logits at every position of ids (full forward through the KV-cached step path).
  logits(ids) {
    if (ids.length > this.ctx) throw new Error(`sequence longer than ${this.ctx}`);
    return ids.map((t, pos) => this.step(t, pos).slice());
  }

  // Sample one continuation of promptIds (same loop as jxlart.model.sample with n = 1); returns the generated
  // ids, eos included when emitted.
  sample(promptIds, { maxNew = 700, temperature = 1, eosId, rng = Math.random } = {}) {
    if (eosId === undefined) throw new Error('sample: eosId required');
    const maxLen = Math.min(this.ctx, promptIds.length + maxNew), invT = 1 / Math.max(temperature, 1e-5);
    let logits = null;
    promptIds.forEach((t, pos) => { logits = this.step(t, pos, pos === promptIds.length - 1); });
    const out = [], p = this.probs, V = this.V;
    for (let pos = promptIds.length; pos < maxLen; pos++) {
      let m = -Infinity;
      for (let i = 0; i < V; i++) if (logits[i] > m) m = logits[i];
      let z = 0;
      for (let i = 0; i < V; i++) z += (p[i] = Math.exp((logits[i] - m) * invT));
      let u = rng() * z, nxt = V - 1;
      for (let i = 0; i < V; i++) { u -= p[i]; if (u < 0) { nxt = i; break; } }
      out.push(nxt);
      if (nxt === eosId) break;
      if (pos + 1 < maxLen) logits = this.step(nxt, pos);
    }
    return out;
  }
}
