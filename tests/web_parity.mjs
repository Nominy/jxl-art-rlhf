// JS side of tests/test_web_parity.py; also a quick sampling demo/benchmark. Runs under bun or node.
//   bun tests/web_parity.mjs logits <model.json> <seqs.json> <out.f32>   all-position logits, float32 LE
//   bun tests/web_parity.mjs decode <tokenizer.json> <seqs.json> <out.json>
//   bun tests/web_parity.mjs sample <model.json> <tokenizer.json> [n=20] [seed=1] [--quiet]
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { loadModel } from '../web/blind/lib/slm.js';
import { loadTokenizer } from '../web/blind/lib/tok.js';

const [mode, ...args] = process.argv.slice(2);
const url = (p) => pathToFileURL(p).href;

function mulberry32(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

if (mode === 'logits') {
  const [modelPath, seqPath, outPath] = args;
  const model = await loadModel(url(modelPath));
  const seqs = JSON.parse(readFileSync(seqPath, 'utf8'));
  const n = seqs.reduce((a, s) => a + s.length, 0), out = new Float32Array(n * model.config.vocab_size);
  let o = 0;
  const t0 = performance.now();
  for (const s of seqs) for (const row of model.logits(s)) { out.set(row, o); o += row.length; }
  const dt = (performance.now() - t0) / 1000;
  writeFileSync(outPath, new Uint8Array(out.buffer));
  console.log(JSON.stringify({ tokens: n, seconds: dt, tok_per_s: n / dt }));
} else if (mode === 'decode') {
  const [tokPath, seqPath, outPath] = args;
  const tok = await loadTokenizer(url(tokPath));
  writeFileSync(outPath, JSON.stringify(JSON.parse(readFileSync(seqPath, 'utf8')).map(tok.decode)));
} else if (mode === 'sample') {
  const [modelPath, tokPath, n = '20', seed = '1'] = args.filter((a) => !a.startsWith('--'));
  const quiet = args.includes('--quiet');
  const [model, tok] = await Promise.all([loadModel(url(modelPath)), loadTokenizer(url(tokPath))]);
  const rng = mulberry32(Number(seed));
  let tokens = 0, seconds = 0, eos = 0;
  const lens = [];
  for (let i = 0; i < Number(n); i++) {
    const t0 = performance.now();
    const ids = model.sample(tok.promptIds, { maxNew: 700, temperature: 1, eosId: tok.eosId, rng });
    const dt = (performance.now() - t0) / 1000;
    tokens += tok.promptIds.length + ids.length; seconds += dt; lens.push(ids.length);
    eos += ids.at(-1) === tok.eosId;
    if (!quiet) console.log(`--- sample ${i} (${ids.length} tokens, ${(ids.length / dt).toFixed(0)} tok/s)\n${tok.decode(ids)}`);
  }
  const mean = lens.reduce((a, b) => a + b, 0) / lens.length;
  console.log(JSON.stringify({ samples: lens.length, mean_len: mean, eos_rate: eos / lens.length,
    tok_per_s: tokens / seconds, seconds }));
} else {
  console.error('usage: web_parity.mjs logits|decode|sample ...');
  process.exit(2);
}
