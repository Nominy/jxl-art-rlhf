// Blind A/B game: mount into an element with id "blind-game". Pairs come from worker.js; one image of
// each pair is from the RLHF start checkpoint (step 0), the other from the final one (step 820).
// Options on the element: data-rounds (default 10), data-autostart (load as soon as it is on screen,
// unless the browser asks to save data), data-reference (a line shown with the results).
const root = document.getElementById("blind-game");
const ROUNDS = Number(root.dataset.rounds) || 10;
const BASE = new URL("./", import.meta.url);

const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) el.setAttribute(k, v === true ? "" : v);
  }
  el.append(...kids.flat().filter((k) => k != null));
  return el;
};

const LABEL = { base: "before RLHF (step 0)", tuned: "after RLHF (step 820)" };

async function editorLink(text) {
  const z = new Response(new Blob([text]).stream().pipeThrough(new CompressionStream("deflate")));
  const b = new Uint8Array(await z.arrayBuffer());
  let s = "";
  for (const x of b) s += String.fromCharCode(x);
  return "https://www.januschka.com/jxl-art/?zcode=" + btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

let worker, nextId = 0;
const waiting = new Map(), queue = [];
let round = 0, history = [], total = { tuned: 0, base: 0, skip: 0 }, current = null, busy = false;

function request() {
  const id = nextId++;
  const p = new Promise((resolve, reject) => waiting.set(id, { resolve, reject }));
  worker.postMessage({ type: "pair", id });
  queue.push(p);
}

// Same footprint as a round, so the page below does not jump while models load or a pair generates.
function skeleton(text) {
  const msg = h("p", { class: "jb-round" }, text);
  const off = (label) => h("button", { type: "button", disabled: true }, label);
  view(msg, h("div", { class: "jb-pair" }, h("div", { class: "jb-slot jb-wait" }), h("div", { class: "jb-slot jb-wait" })),
    h("div", { class: "jb-actions" }, off("← Left"), off("Can't decide"), off("Right →")),
    h("p", { class: "jb-note" }, "Generated in your browser by a 3M-parameter model; nothing is sent anywhere."));
  return msg;
}

let started = false;
function start() {
  if (started) return;
  started = true;
  const status = skeleton("Loading the model…");
  const fail = (msg) => view(h("p", { class: "jb-status" }, "Could not start: " + msg));
  worker = new Worker(new URL("worker.js", BASE), { type: "module" });
  worker.onmessage = ({ data }) => {
    if (data.type === "progress") status.textContent = data.text;
    else if (data.type === "ready") {
      request();
      request();
      play();
    } else if (data.type === "pair") {
      waiting.get(data.id).resolve(data);
      waiting.delete(data.id);
    } else if (data.type === "error") {
      const w = waiting.get(data.id);
      if (w) { w.reject(new Error(data.message)); waiting.delete(data.id); }
      else fail(data.message);
    }
  };
  worker.onerror = (e) => fail(e.message);
  worker.postMessage({ type: "init", base: BASE.href });
}

function slot(item) {
  const url = URL.createObjectURL(new Blob([item.png], { type: "image/png" }));
  const img = h("img", { src: url, alt: "" });
  img.onload = () => img.classList.toggle("jb-small", img.naturalWidth < img.clientWidth);
  item.url = url;
  return img;
}

async function play() {
  const pairPromise = queue.shift();
  request();
  skeleton(`Round ${round + 1} of ${ROUNDS}: generating on your device…`);
  let pair;
  try {
    pair = await pairPromise;
  } catch (e) {
    view(h("p", { class: "jb-status" }, "Generation failed: " + e.message));
    return;
  }
  const order = Math.random() < 0.5 ? ["base", "tuned"] : ["tuned", "base"];
  current = { pair, order };
  const choose = (pos) => vote(pos);
  const slots = order.map((side, i) =>
    h("button", { class: "jb-slot", type: "button", "aria-label": i ? "Right image" : "Left image", onclick: () => choose(i) },
      slot(pair[side])));
  const secs = ((pair.base.ms + pair.tuned.ms) / 1000).toFixed(1);
  view(
    h("p", { class: "jb-round" }, `Round ${round + 1} of ${ROUNDS}. Which one do you like more?`),
    h("div", { class: "jb-pair" }, slots),
    h("div", { class: "jb-actions" },
      h("button", { type: "button", onclick: () => choose(0) }, "← Left"),
      h("button", { type: "button", onclick: () => choose(-1) }, "Can't decide"),
      h("button", { type: "button", onclick: () => choose(1) }, "Right →")),
    h("p", { class: "jb-note" }, `Both generated in your browser in ${secs} s (${pair.base.tries + pair.tuned.tries} programs sampled, invalid ones discarded).`));
}

function vote(pos) {
  if (!current || busy) return;
  busy = true;
  const { pair, order } = current;
  const pick = pos < 0 ? "skip" : order[pos];
  total[pick]++;
  history.push({ pair, order, pick });
  current = null;
  round++;
  busy = false;
  if (round % ROUNDS === 0) reveal();
  else play();
}

function reveal() {
  const last = history.slice(-ROUNDS);
  const tuned = last.filter((r) => r.pick === "tuned").length, base = last.filter((r) => r.pick === "base").length;
  const rows = last.map((r, i) =>
    h("div", { class: "jb-row" },
      h("span", { class: "jb-n" }, String(i + 1)),
      r.order.map((side) => {
        const it = r.pair[side], fig = h("figure", { class: "jb-thumb" + (r.pick === side ? " jb-picked" : "") },
          h("img", { src: it.url, alt: "" }),
          h("figcaption", {}, h("strong", {}, side === "tuned" ? "after" : "before"),
            ` · ${it.bytes} bytes`, r.pick === side ? " · your pick" : ""));
        editorLink(it.text).then((u) => fig.querySelector("figcaption").append(" · ", h("a", { href: u, target: "_blank", rel: "noopener" }, "program")));
        return fig;
      })));
  const all = total.tuned + total.base;
  const ref = root.dataset.reference;
  view(
    h("p", { class: "jb-result" },
      `You preferred the model after RLHF in ${tuned} of ${tuned + base} decided rounds`,
      last.length - tuned - base ? ` (${last.length - tuned - base} undecided).` : "."),
    history.length > ROUNDS ? h("p", {}, `All rounds so far: after ${total.tuned}, before ${total.base}, undecided ${total.skip}.`) : null,
    ref ? h("p", { class: "jb-note" }, ref) : null,
    h("p", { class: "jb-note" }, `“After” is ${LABEL.tuned}, trained on one person's 1,442 picks; “before” is ${LABEL.base}, the model trained only on random programs. Each image is its own short program; “program” opens it in the jxl-art editor.`),
    h("div", { class: "jb-grid" }, rows),
    h("div", { class: "jb-actions" }, h("button", { type: "button", onclick: play }, `Play ${ROUNDS} more`)));
  if (all) root.dataset.score = `${total.tuned}/${all}`;
}

function view(...nodes) {
  root.replaceChildren(...nodes.filter((n) => n != null));
}

// ← / → vote while the game is on screen (ArrowDown is left to page scrolling).
document.addEventListener("keydown", (e) => {
  if (!current || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
  const pos = { ArrowLeft: 0, ArrowRight: 1 }[e.key];
  const r = root.getBoundingClientRect();
  if (pos === undefined || r.bottom < 0 || r.top > innerHeight) return;
  e.preventDefault();
  if (!e.repeat) vote(pos);
});

if ("autostart" in root.dataset && !navigator.connection?.saveData) {
  skeleton("Starting…");
  const io = new IntersectionObserver((entries) => {
    if (!entries.some((en) => en.isIntersecting)) return;
    io.disconnect();
    start();
  });
  io.observe(root);
} else {
  view(h("div", { class: "jb-actions" }, h("button", { type: "button", class: "jb-start", onclick: start }, "Start the blind test")),
    h("p", { class: "jb-note" }, "Downloads two copies of the 3M-parameter model (about 12 MB) and libjxl compiled to WebAssembly (2 MB). Nothing is sent anywhere."));
}
