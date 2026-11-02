// Runner for tests/test_web_tree.py (Bun or Node):
//   web_tree.mjs canon <in.jsonl> <out.jsonl>    in: {"text"} -> out: {"ok","canonical","width","height","error","kind"}
//   web_tree.mjs compile <in.jsonl> <outdir>      in: {"id","text"} -> <outdir>/<id>.jxl/.png + stdout JSON lines
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { canonicalize, ParseError } from "../web/blind/lib/tree.js";
import { initCodec, compileTree, decodeJxl } from "../web/blind/lib/codec.js";

const [mode, inPath, out] = process.argv.slice(2);
const items = readFileSync(inPath, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));

if (mode === "canon") {
  const lines = items.map(({ text }) => {
    try {
      return JSON.stringify({ ok: true, ...canonicalize(text) });
    } catch (e) {
      return JSON.stringify({ ok: false, error: e.message, kind: e instanceof ParseError ? "ParseError" : e.name });
    }
  });
  writeFileSync(out, lines.join("\n") + "\n");
} else if (mode === "compile") {
  const base = new URL("../web/blind/vendor/jxl/", import.meta.url);
  let t = performance.now();
  await initCodec(base);
  console.log(JSON.stringify({ init_ms: performance.now() - t }));
  for (const { id, text } of items) {
    await initCodec(base); // no-op unless a previous call aborted
    const rec = { id, ok: false };
    try {
      t = performance.now();
      const jxl = compileTree(text);
      rec.compile_ms = performance.now() - t;
      writeFileSync(join(out, `${id}.jxl`), jxl);
      t = performance.now();
      const png = decodeJxl(jxl);
      rec.decode_ms = performance.now() - t;
      writeFileSync(join(out, `${id}.png`), png);
      rec.ok = true;
    } catch (e) {
      rec.error = e.message.slice(0, 500);
    }
    console.log(JSON.stringify(rec));
  }
} else {
  throw new Error(`unknown mode ${mode}`);
}
