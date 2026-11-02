"""Blind A/B test between two policy checkpoints, judged by a person in the browser.

  python -m jxlart.blind_ab --run runs/rlhf-v2 --a step_00000 --b online --pairs 50 --out reports/blind-ab

Each side contributes --pairs distinct valid images (sampled once, then frozen in --out). Pairs are
random, the side shown left is random, and images are served under opaque ids, so neither the page
nor its URLs reveal which checkpoint produced which image. Images are shown at their natural aspect
ratio on the page background. Votes are appended (fsync) to votes.jsonl; rerunning resumes.
At the end, result.json holds B's wins over A with an exact two-sided binomial test (ties excluded).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from . import compiler, model as M
from .corpus import PROMPT
from .runctl import Run, atomic_write_json


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p-value for k successes in n trials at p = 0.5."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in pmf if p <= pmf[k] * (1 + 1e-9)))


def sample_side(backend, path: Path, need: int, seed: int, run: Run, label: str) -> list:
    tok = M.load_tokenizer()
    prompt = tok(PROMPT)["input_ids"]
    policy, _ = backend.load_policy(path)
    if backend.__name__ == "jxlart.model":
        import torch
        torch.manual_seed(seed)
    else:
        import mlx.core as mx
        mx.random.seed(seed)
    got, seen, draws, n_valid = [], set(), 0, 0
    lv = run.level(f"valid images: {label}", need, "images")
    while len(got) < need and draws < 40 * need:
        gens = backend.sample(policy, prompt, 64, 700, tok.eos_token_id)
        draws += 64
        texts = [tok.decode([t for t in g if t != tok.eos_token_id]) for g in gens]
        with ThreadPoolExecutor(10) as ex:
            res = list(ex.map(lambda t: compiler.compile_tree(t, timeout=8, max_bytes=compiler.MAX_JXL_BYTES), texts))
        for r in res:
            if r.ok and not compiler.is_degenerate(r.stats):
                n_valid += 1
                if r.stats["phash"] not in seen and len(got) < need:
                    seen.add(r.stats["phash"])
                    got.append(r)
        lv.done = len(got)
        run.status()
    run.event("sampled", side=label, valid=n_valid, draws=draws)
    return got, draws, n_valid


def prepare(args, run: Run) -> dict:
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    backend = importlib.import_module("jxlart.mlx_model" if args.backend == "mlx" else "jxlart.model")
    pol = Path(args.run) / "policy"
    files = {s: pol / (f"{getattr(args, s)}.safetensors") for s in ("a", "b")}
    items, stats = {}, {}
    for side in ("a", "b"):
        run.set_state("running", f"sampling side {side.upper()} ({files[side].name})")
        res, draws, n_valid = sample_side(backend, files[side], args.pairs, args.seed, run, side)
        stats[side] = {"file": files[side].name, "kept": len(res), "valid": n_valid, "draws": draws}
        for i, r in enumerate(res):
            oid = hashlib.sha256(f"{args.seed}:{side}:{i}:{r.canonical}".encode()).hexdigest()[:16]
            compiler.decode(r.jxl).save(out / "img" / f"{oid}.png")
            items[oid] = {"side": side, "text": r.canonical, "bytes": r.nbytes, "size": list(r.size)}
    rng = random.Random(args.seed)
    a = [k for k, v in items.items() if v["side"] == "a"]
    b = [k for k, v in items.items() if v["side"] == "b"]
    rng.shuffle(a)
    rng.shuffle(b)
    pairs = []
    for x, y in zip(a, b):
        pairs.append([x, y] if rng.random() < 0.5 else [y, x])
    plan = {"pairs": pairs, "items": items, "stats": stats, "seed": args.seed,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    atomic_write_json(out / "plan.json", plan)
    return plan


def result(plan: dict, votes: list[dict]) -> dict:
    b_wins = a_wins = ties = 0
    for v in votes:
        left, right = plan["pairs"][v["pair"]]
        if v["choice"] == "tie":
            ties += 1
            continue
        chosen = left if v["choice"] == "left" else right
        if plan["items"][chosen]["side"] == "b":
            b_wins += 1
        else:
            a_wins += 1
    n = a_wins + b_wins
    return {"a": plan["stats"]["a"]["file"], "b": plan["stats"]["b"]["file"], "votes": len(votes),
            "b_wins": b_wins, "a_wins": a_wins, "ties": ties,
            "b_win_rate_excl_ties": b_wins / n if n else None, "p_two_sided": binom_two_sided(b_wins, n),
            "valid_rate": {s: plan["stats"][s]["valid"] / plan["stats"][s]["draws"] for s in ("a", "b")}}


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Blind A/B</title>
<style>
 body{background:#faf9f5;color:#1f1e1d;font:16px system-ui,sans-serif;margin:0;padding:20px}
 header{display:flex;gap:16px;align-items:center;justify-content:center;color:#6b6a65}
 .pair{display:flex;gap:24px;justify-content:center;margin-top:16px}
 .slot{width:min(46vw,620px);height:min(46vw,620px);display:flex;align-items:center;justify-content:center;cursor:pointer;border-radius:8px}
 .slot:hover{background:#efede4}
 .slot img{width:100%;height:100%;object-fit:contain}
 .slot img.small{image-rendering:pixelated}
 .btns{display:flex;gap:10px;justify-content:center;margin:18px}
 button{font:15px system-ui;padding:8px 16px;border:1px solid #d6d3c8;background:#fff;border-radius:6px;cursor:pointer}
 #done{max-width:620px;margin:60px auto;font-size:18px;line-height:1.6;display:none}
</style></head><body>
<header><b>Blind A/B</b><span id="prog"></span><span>← left · → right · ↓ about equal · u undo</span></header>
<div class="pair" id="pair"><div class="slot" id="L"><img id="imgL"></div><div class="slot" id="R"><img id="imgR"></div></div>
<div class="btns"><button onclick="vote('left')">← left is better</button><button onclick="vote('tie')">↓ about equal</button>
<button onclick="vote('right')">right is better →</button><button onclick="undo()">undo</button></div>
<div id="done"></div>
<script>
let cur=null,busy=false;const $=i=>document.getElementById(i);
async function j(u,o){const r=await fetch(u,o);if(!r.ok)throw new Error(await r.text());return r.json()}
async function load(){const s=await j('/api/next');$('prog').textContent=`${s.done} / ${s.total}`;
 if(s.finished){$('pair').style.display='none';document.querySelector('.btns').style.display='none';
  const r=s.result;$('done').style.display='block';
  $('done').innerHTML=`<p>Done. The two sources were <b>${r.a}</b> and <b>${r.b}</b>.</p><p>You preferred <b>${r.b}</b> in <b>${r.b_wins}</b> pairs and <b>${r.a}</b> in <b>${r.a_wins}</b>; ${r.ties} were about equal.</p><p>Two-sided binomial test (ties excluded): p = ${r.p_two_sided.toPrecision(2)}.</p>`;return}
 cur=s;$('imgL').src='/img/'+s.left+'.png';$('imgR').src='/img/'+s.right+'.png'}
async function vote(c){if(!cur||busy)return;busy=true;try{await j('/api/vote',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pair:cur.pair,choice:c})});cur=null;await load()}finally{busy=false}}
async function undo(){await j('/api/undo',{method:'POST'});cur=null;$('pair').style.display='';document.querySelector('.btns').style.display='';$('done').style.display='none';load()}
$('L').onclick=()=>vote('left');$('R').onclick=()=>vote('right');
for(const i of ['imgL','imgR'])$(i).onload=e=>e.target.classList.toggle('small',e.target.naturalWidth<e.target.clientWidth);
document.onkeydown=e=>{const f={ArrowLeft:()=>vote('left'),ArrowRight:()=>vote('right'),ArrowDown:()=>vote('tie'),u:undo}[e.key];if(f){e.preventDefault();if(!e.repeat)f()}};
load();
</script></body></html>"""


def make_app(out: Path, plan: dict, run: Run) -> FastAPI:
    app = FastAPI()
    lock = threading.Lock()
    vpath = out / "votes.jsonl"

    def votes() -> list[dict]:
        vs: list[dict] = []
        if vpath.exists():
            for line in vpath.open(encoding="utf-8"):
                rec = json.loads(line)
                if rec.get("undo"):
                    vs and vs.pop()
                else:
                    vs.append(rec)
        return vs

    def finish(vs):
        res = result(plan, vs)
        atomic_write_json(out / "result.json", res)
        run.metric(**{k: v for k, v in res.items() if isinstance(v, (int, float))})
        return res

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get("/api/next")
    def nxt():
        with lock:
            vs = votes()
            done = {v["pair"] for v in vs}
            total = len(plan["pairs"])
            run.levels["votes"].done = len(vs)
            run.status()
            todo = [i for i in range(total) if i not in done]
            if not todo:
                return {"finished": True, "done": len(vs), "total": total, "result": finish(vs)}
            i = todo[0]
            left, right = plan["pairs"][i]
            return {"pair": i, "left": left, "right": right, "done": len(vs), "total": total}

    @app.post("/api/vote")
    def vote(body: dict):
        if body.get("choice") not in ("left", "right", "tie") or not 0 <= int(body.get("pair", -1)) < len(plan["pairs"]):
            raise HTTPException(400, "bad vote")
        with lock, vpath.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"pair": int(body["pair"]), "choice": body["choice"], "ts": time.time()}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return {"ok": True}

    @app.post("/api/undo")
    def undo():
        with lock, vpath.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"undo": True, "ts": time.time()}) + "\n")
        return {"ok": True}

    @app.get("/img/{name}")
    def img(name: str):
        p = (out / "img" / name).resolve()
        if p.parent != (out / "img").resolve() or not p.exists():
            raise HTTPException(404)
        return FileResponse(p)

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--a", default="step_00000", help="policy file stem under <run>/policy (shown blind)")
    ap.add_argument("--b", default="online")
    ap.add_argument("--pairs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--backend", choices=["torch", "mlx"], default="mlx" if sys.platform == "darwin" else "torch")
    ap.add_argument("--no-live", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    run = Run(out, "blind-ab", vars(args), live=not args.no_live)
    with run:
        plan = json.loads((out / "plan.json").read_text()) if (out / "plan.json").exists() else prepare(args, run)
        run.level("votes", len(plan["pairs"]), "votes")
        server = uvicorn.Server(uvicorn.Config(make_app(out, plan, run), host="127.0.0.1", port=args.port,
                                               log_level="warning"))
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        for _ in range(100):
            if server.started or not t.is_alive():
                break
            time.sleep(0.1)
        if not server.started:
            raise RuntimeError(f"port {args.port} unavailable; pass --port N")
        run.set_state("waiting", f"judge pairs at http://127.0.0.1:{args.port}")
        run.event("ui", url=f"http://127.0.0.1:{args.port}")
        while not run.should_stop() and not (out / "result.json").exists():
            time.sleep(1)
        if (out / "result.json").exists():
            run.event("finished", **json.loads((out / "result.json").read_text()))
        server.should_exit = True


if __name__ == "__main__":
    main()
