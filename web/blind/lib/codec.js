// libjxl (WebAssembly) wrapper: tree program -> JXL bytes -> 8-bit sRGB PNG.
// Mirrors jxlart/compiler.py: compileTree == tools/bin/jxl_from_tree, decodeJxl ==
// `djxl - --output_format png --bits_per_sample=8 --color_space=RGB_D65_SRG_Rel_SRG`
// (RGBA PNG when the image has alpha; no compositing here).
// Works in browser (window or module Worker), Bun and Node. If libjxl aborts (e.g. an
// allocation failure), the call throws and the module is re-instantiated in the background;
// `await initCodec(baseUrl)` before each compile to pick up the fresh instance.
let mod = null;
let loading = null;
let base = null;
let wasmModule = null;
let stderr = [];

async function fetchBytes(url) {
  if (url.protocol === "file:") {
    const [{ readFile }, { fileURLToPath }] = await Promise.all([import("node:fs/promises"), import("node:url")]);
    return new Uint8Array(await readFile(fileURLToPath(url)));
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return new Uint8Array(await r.arrayBuffer());
}

async function instantiate() {
  const [{ default: createJxlModule }, compiled] = await Promise.all([
    import(new URL("jxl.js", base).href),
    wasmModule ?? fetchBytes(new URL("jxl.wasm", base)).then((b) => WebAssembly.compile(b)),
  ]);
  wasmModule = compiled;
  mod = await createJxlModule({
    instantiateWasm(imports, done) {
      WebAssembly.instantiate(compiled, imports).then(done);
      return {};
    },
    print() {},
    printErr: (t) => stderr.push(t),
  });
}

export function initCodec(baseUrl) {
  if (!loading) {
    base = new URL(baseUrl, globalThis.location?.href);
    loading = instantiate().catch((e) => {
      loading = null;
      throw e;
    });
  }
  return loading;
}

function call(fn, arg, what) {
  if (!mod) throw new Error("codec not ready: await initCodec() first");
  stderr = [];
  let out;
  try {
    out = mod[fn](arg);
  } catch (e) {
    // libjxl aborted: the instance is unusable; start a fresh one.
    mod = null;
    loading = null;
    initCodec(base).catch(() => {});
    throw new Error(stderr.join("\n") || `${what} aborted: ${e?.message ?? e}`);
  }
  if (typeof out === "string") throw new Error(stderr.length ? `${out}: ${stderr.join("\n")}` : out);
  return new Uint8Array(out); // copy out of wasm memory
}

export function compileTree(canonical) {
  return call("jxl_from_tree", canonical, "jxl_from_tree");
}

export function decodeJxl(jxl) {
  return call("decode", jxl, "decode");
}
