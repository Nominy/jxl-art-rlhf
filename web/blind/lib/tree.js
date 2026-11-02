// Port of jxlart/dsl.py (parse + to_text + output_size) for the jxl_from_tree language.
// canonicalize(text) returns exactly dsl.to_text(dsl.parse(text)) and dsl.output_size, and
// throws ParseError exactly when dsl.parse raises. Integers are BigInt (Python ints are
// unbounded); floats are formatted like Python's format(v, ".7g").

export const MAX_JXL_BYTES = 256;
export const MAX_SIDE = 4096;
export const MAX_PIXELS = 4 * 1024 * 1024;

export class ParseError extends Error {
  constructor(message) {
    super(message);
    this.name = "ParseError";
  }
}

const PROPERTIES = [
  "c", "g", "y", "x", "|N|", "|W|", "N", "W", "W-WW-NW+NWW", "W+N-NW", "W-NW",
  "NW-N", "N-NE", "N-NN", "W-WW", "WGH", "PrevAbs", "Prev", "PrevAbsErr", "PrevErr",
  "PPrevAbs", "PPrev", "PPrevAbsErr", "PPrevErr",
];
const PROPERTY_SET = new Set(PROPERTIES);
for (let i = 1; i < 20; i++) for (const s of ["Abs", "", "AbsErr", "Err"]) PROPERTY_SET.add(`Prev${i}${s}`);
const PREDICTOR_SET = new Set([
  "Set", "W", "N", "AvgW+N", "Select", "Gradient", "Weighted", "NE", "NW", "WW",
  "AvgW+NW", "AvgN+NW", "AvgN+NE", "AvgAll",
]);
// Header name -> argument kinds ("u" unsigned int, "i" signed int, "f" float, "s" symbol).
const HEADER_ARGS = new Map(Object.entries({
  Width: "u", Height: "u", Squeeze: "", GroupShift: "u", XYB: "", CbYCr: "",
  HiddenChannel: "u", RCT: "u", Orientation: "u", Alpha: "", Bitdepth: "u",
  FloatExpBits: "u", FramePos: "ii", NotLast: "", Upsample: "u", Upsample_EC: "u",
  Animation: "", AnimationFPS: "uu", Duration: "u", BlendMode: "s",
  SplineQuantizationAdjustment: "u", Gaborish: "", DeltaPalette: "", EPF: "u",
  Noise: "ffffffff", XYBFactors: "fff", PQ: "", HLG: "", Rec2100: "", P3: "", "16BitBuffers": "",
}));
const BLEND_MODES = new Set(["kAdd", "kReplace", "kBlend", "kAlphaWeightedAdd", "kMul"]);
const SPLINE_COEFFS = 32 * 4;

// Python's str.split() whitespace and re's Unicode \d (int()/float() accept any Nd digit).
const WS = /[\t\n\v\f\r \x1c-\x1f\x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+/u;
const D = String.raw`\p{Nd}`;
const INT_RE = new RegExp(`^[+-]?${D}+$`, "u");
const UINT_RE = new RegExp(`^\\+?${D}+$`, "u");
const FLOAT_RE = new RegExp(`^[+-]?(${D}+\\.?${D}*|\\.${D}+)([eE][+-]?${D}+)?$`, "u");
const ND = /^\p{Nd}$/u;

function asciiDigits(t) {
  if (/^[\x00-\x7f]*$/.test(t)) return t;
  let out = "";
  for (const ch of t) {
    if (ch >= "0" && ch <= "9" || !ND.test(ch)) { out += ch; continue; }
    // Nd digits come in aligned runs of 0..9; find the start of this contiguous run.
    const cp = ch.codePointAt(0);
    let start = cp;
    while (ND.test(String.fromCodePoint(start - 1))) start--;
    out += String((cp - start) % 10);
  }
  return out;
}

function toBigInt(t) {
  const s = asciiDigits(t);
  if (s.replace(/^[+-]/, "").length > 4300) throw new ParseError("integer too long"); // Python's int() limit
  const v = BigInt(s.replace(/^[+-]/, ""));
  return s[0] === "-" ? -v : v;
}

function parseUint(t) {
  if (!UINT_RE.test(t)) throw new ParseError(`invalid unsigned int ${JSON.stringify(t)}`);
  return toBigInt(t);
}

function parseInt_(t) {
  if (!INT_RE.test(t)) throw new ParseError(`invalid int ${JSON.stringify(t)}`);
  return toBigInt(t);
}

function parseFloat_(t) {
  if (!FLOAT_RE.test(t)) throw new ParseError(`invalid float ${JSON.stringify(t)}`);
  return Number(asciiDigits(t));
}

class Tokens {
  constructor(text) {
    this.toks = text.split(WS).filter((s) => s.length);
    this.i = 0;
  }
  next() {
    if (this.i >= this.toks.length) throw new ParseError("unexpected end of input");
    return this.toks[this.i++];
  }
}

function parseHeader(name, tk) {
  const args = [];
  for (const kind of HEADER_ARGS.get(name)) {
    const t = tk.next();
    if (kind === "u") args.push(parseUint(t));
    else if (kind === "i") args.push(parseInt_(t));
    else if (kind === "f") args.push(parseFloat_(t));
    else {
      if (!BLEND_MODES.has(t)) throw new ParseError(`invalid BlendMode ${JSON.stringify(t)}`);
      args.push(t);
    }
  }
  checkHeader(name, args);
  return args;
}

function checkHeader(name, args) {
  if (name === "Bitdepth" && !(args[0] >= 1n && args[0] <= 32n)) throw new ParseError("Bitdepth out of range");
  if ((name === "Upsample" || name === "Upsample_EC") && ![1n, 2n, 4n, 8n].includes(args[0]))
    throw new ParseError(`invalid ${name}`);
  if (name === "EPF" && args[0] > 3n) throw new ParseError("invalid EPF");
  if (name === "Noise" && args.some((v) => !(v >= 0 && v <= 1))) throw new ParseError("invalid Noise");
  if (name === "HiddenChannel" && args[0] > 16n) throw new ParseError("invalid HiddenChannel");
}

function parseNode(tk, frame, depth) {
  for (;;) {
    const t = tk.next();
    if (t === "if") {
      const prop = tk.next();
      if (!PROPERTY_SET.has(prop)) throw new ParseError(`unexpected property ${JSON.stringify(prop)}`);
      if (tk.next() !== ">") throw new ParseError("expected '>'");
      const value = parseInt_(tk.next());
      if (depth > 64) throw new ParseError("tree too deep");
      const left = parseNode(tk, frame, depth + 1);
      const right = parseNode(tk, frame, depth + 1);
      return { prop, value, left, right };
    }
    if (t === "-") {
      const predictor = tk.next();
      if (!PREDICTOR_SET.has(predictor)) throw new ParseError(`unexpected predictor ${JSON.stringify(predictor)}`);
      let o = tk.next();
      let sign = 1n;
      if (o === "-" || o === "+") {
        sign = o === "-" ? -1n : 1n;
        o = tk.next();
      }
      return { predictor, offset: sign * parseInt_(o) };
    }
    if (t === "/*") {
      while (tk.i < tk.toks.length && tk.next() !== "*/");
      continue;
    }
    if (t === "Spline") {
      const coeffs = [];
      for (let k = 0; k < SPLINE_COEFFS; k++) coeffs.push(parseFloat_(tk.next()));
      const points = [];
      for (;;) {
        const x = tk.next();
        if (x === "EndSpline") break;
        const px = parseFloat_(x);
        points.push([px, parseFloat_(tk.next())]);
      }
      if (!points.length) throw new ParseError("spline without control points");
      frame.splines.push({ coeffs, points });
      continue;
    }
    if (HEADER_ARGS.has(t)) {
      frame.headers.push([t, parseHeader(t, tk)]);
      continue;
    }
    throw new ParseError(`unexpected node type ${JSON.stringify(t)}`);
  }
}

function parse(text) {
  const tk = new Tokens(text);
  const frames = [];
  for (;;) {
    const frame = { headers: [], splines: [], tree: null };
    frame.tree = parseNode(tk, frame, 0);
    frames.push(frame);
    if (!frame.headers.some(([name]) => name === "NotLast")) return frames;
  }
}

// Python format(v, ".7g") with "-0" -> "0": exact decimal expansion, round half to even.
function fmtFloat(v) {
  if (v === 0) return "0";
  if (!Number.isFinite(v)) return v > 0 ? "inf" : "-inf";
  const P = 7;
  const dv = new DataView(new ArrayBuffer(8));
  dv.setFloat64(0, Math.abs(v));
  const hi = dv.getUint32(0), lo = dv.getUint32(4);
  const bexp = hi >>> 20;
  let m = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  let k = bexp - 1075;
  if (bexp) m |= 1n << 52n;
  else k = -1074;
  // |v| = digits * 10^e10 exactly
  let digits, e10;
  if (k >= 0) [digits, e10] = [(m << BigInt(k)).toString(), 0];
  else [digits, e10] = [(m * 5n ** BigInt(-k)).toString(), k];
  if (digits.length > P) {
    const drop = digits.length - P;
    const pow = 10n ** BigInt(drop);
    const n = BigInt(digits);
    let q = n / pow;
    const r2 = (n % pow) * 2n;
    if (r2 > pow || (r2 === pow && q % 2n === 1n)) q += 1n;
    digits = q.toString();
    e10 += drop;
  }
  const x = digits.length - 1 + e10; // decimal exponent of the leading digit
  digits = digits.replace(/0+$/, "") || "0";
  const sign = v < 0 ? "-" : "";
  if (x >= -4 && x < P) {
    if (x >= 0) {
      const int = digits.slice(0, x + 1).padEnd(x + 1, "0");
      const frac = digits.slice(x + 1);
      return sign + int + (frac ? "." + frac : "");
    }
    return sign + "0." + "0".repeat(-x - 1) + digits;
  }
  const mant = digits.length > 1 ? digits[0] + "." + digits.slice(1) : digits;
  return `${sign}${mant}e${x < 0 ? "-" : "+"}${String(Math.abs(x)).padStart(2, "0")}`;
}

function fmtArgs(args) {
  return args.map((a) => (typeof a === "number" ? fmtFloat(a) : String(a))).join(" ");
}

function fmtTree(node, depth, out) {
  const pad = "  ".repeat(depth);
  if (node.predictor !== undefined) {
    out.push(`${pad}- ${node.predictor} ${node.offset}`);
    return;
  }
  out.push(`${pad}if ${node.prop} > ${node.value}`);
  fmtTree(node.left, depth + 1, out);
  fmtTree(node.right, depth + 1, out);
}

function toText(frames) {
  const out = [];
  frames.forEach((frame, fi) => {
    if (fi) out.push("");
    for (const [name, args] of frame.headers) out.push(`${name} ${fmtArgs(args)}`.trimEnd());
    for (const sp of frame.splines) {
      out.push("Spline");
      for (let row = 0; row < 4; row++) out.push(sp.coeffs.slice(row * 32, (row + 1) * 32).map(fmtFloat).join(" "));
      for (const [x, y] of sp.points) out.push(`${fmtFloat(x)} ${fmtFloat(y)}`);
      out.push("EndSpline");
    }
    fmtTree(frame.tree, 0, out);
  });
  return out.join("\n") + "\n";
}

function headerValue(frame, name, dflt) {
  let val = dflt;
  for (const [n, args] of frame.headers) if (n === name) val = args;
  return val;
}

function outputSize(frames) {
  const f0 = frames[0];
  const w = headerValue(f0, "Width", [1024n])[0];
  const h = headerValue(f0, "Height", [1024n])[0];
  const [x0, y0] = headerValue(f0, "FramePos", [0n, 0n]);
  const up = headerValue(f0, "Upsample", [1n])[0];
  return [Number((w + x0) * up), Number((h + y0) * up)];
}

export function canonicalize(text) {
  const frames = parse(text);
  const [width, height] = outputSize(frames);
  return { canonical: toText(frames), width, height };
}
