// Generation worker: samples tree programs from both checkpoints on the reader's device and turns the
// valid ones into images with libjxl (wasm). Validity mirrors jxlart/compiler.py: parses, fits the
// size limits, compiles, is at most 256 bytes, decodes, and is not a flat/near-single-colour image.
import { loadModel } from "./lib/slm.js";
import { loadTokenizer } from "./lib/tok.js";
import { canonicalize, ParseError, MAX_JXL_BYTES, MAX_SIDE, MAX_PIXELS } from "./lib/tree.js";
import { initCodec, compileTree, decodeJxl } from "./lib/codec.js";

const MAX_TRIES = 40;
let tok, models, codecBase;

const post = (msg, transfer) => self.postMessage(msg, transfer || []);

async function init(base) {
  post({ type: "progress", text: "Loading tokenizer and libjxl…" });
  codecBase = new URL("vendor/jxl/", base);
  [tok] = await Promise.all([loadTokenizer(new URL("assets/tokenizer.json", base)), initCodec(codecBase)]);
  models = {};
  for (const [i, side] of ["base", "tuned"].entries()) {
    post({ type: "progress", text: `Loading model ${i + 1} of 2…` });
    models[side] = await loadModel(new URL(`assets/${side}.json`, base));
  }
  post({ type: "ready" });
}

// Cheap descriptors on a 64x64 downscale, as in compiler.image_stats / is_degenerate.
function degenerate(rgba) {
  const n = rgba.length / 4, mean = [0, 0, 0];
  let g = 0, g2 = 0;
  const sum = [0, 0, 0], sum2 = [0, 0, 0], colours = new Set();
  for (let i = 0; i < n; i++) {
    const r = rgba[4 * i] / 255, gg = rgba[4 * i + 1] / 255, b = rgba[4 * i + 2] / 255, y = (r + gg + b) / 3;
    g += y; g2 += y * y;
    sum[0] += r; sum[1] += gg; sum[2] += b;
    sum2[0] += r * r; sum2[1] += gg * gg; sum2[2] += b * b;
    colours.add((rgba[4 * i] << 16) | (rgba[4 * i + 1] << 8) | rgba[4 * i + 2]);
  }
  const std = Math.sqrt(Math.max(0, g2 / n - (g / n) ** 2));
  let cstd = 0;
  for (let c = 0; c < 3; c++) {
    mean[c] = sum[c] / n;
    cstd += Math.sqrt(Math.max(0, sum2[c] / n - mean[c] ** 2)) / 3;
  }
  return (std < 0.02 && cstd < 0.02) || colours.size < 3;
}

// PNG -> displayable PNG (alpha composited over grey 128, like compiler.decode) + degeneracy check.
async function render(png) {
  const bmp = await createImageBitmap(new Blob([png], { type: "image/png" }));
  const small = new OffscreenCanvas(64, 64).getContext("2d", { willReadFrequently: true });
  small.fillStyle = "rgb(128,128,128)";
  small.fillRect(0, 0, 64, 64);
  small.imageSmoothingQuality = "high";
  small.drawImage(bmp, 0, 0, 64, 64);
  const flat = degenerate(small.getImageData(0, 0, 64, 64).data);
  const colourType = png[25]; // IHDR colour type: 4 = grey+alpha, 6 = RGBA
  let out = png;
  if (!flat && (colourType === 4 || colourType === 6)) {
    const c = new OffscreenCanvas(bmp.width, bmp.height), ctx = c.getContext("2d");
    ctx.fillStyle = "rgb(128,128,128)";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.drawImage(bmp, 0, 0);
    out = new Uint8Array(await (await c.convertToBlob({ type: "image/png" })).arrayBuffer());
  }
  const size = [bmp.width, bmp.height];
  bmp.close();
  return { flat, png: out, size };
}

async function one(side) {
  const t0 = performance.now();
  let tokens = 0;
  for (let tries = 1; tries <= MAX_TRIES; tries++) {
    const ids = models[side].sample(tok.promptIds, { maxNew: 700, temperature: 1, eosId: tok.eosId, rng: Math.random });
    tokens += ids.length;
    const text = tok.decode(ids.filter((t) => t !== tok.eosId));
    let prog;
    try {
      prog = canonicalize(text);
    } catch (e) {
      if (e instanceof ParseError) continue;
      throw e;
    }
    const { canonical, width: w, height: h } = prog;
    if (w <= 0 || h <= 0 || w > MAX_SIDE || h > MAX_SIDE || w * h > MAX_PIXELS) continue;
    let jxl, img;
    try {
      await initCodec(codecBase); // no-op unless a previous program aborted the wasm instance
      jxl = compileTree(canonical);
      if (jxl.length > MAX_JXL_BYTES) continue;
      img = await render(decodeJxl(jxl));
    } catch {
      continue;
    }
    if (img.flat) continue;
    return { text: canonical, bytes: jxl.length, png: img.png, size: img.size, tries, tokens,
             ms: Math.round(performance.now() - t0) };
  }
  throw new Error(`no valid image from the ${side} model in ${MAX_TRIES} tries`);
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "init") await init(data.base);
    else if (data.type === "pair") {
      const base = await one("base"), tuned = await one("tuned");
      post({ type: "pair", id: data.id, base, tuned }, [base.png.buffer, tuned.png.buffer]);
    }
  } catch (e) {
    post({ type: "error", id: data.id, message: String(e && e.message || e) });
  }
};
