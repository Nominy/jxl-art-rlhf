"""Single-page labeling UI served by the online RLHF loop."""

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>JXL art RLHF</title>
<style>
 body{background:#16161a;color:#ddd;font:14px system-ui,sans-serif;margin:0;padding:16px}
 header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
 .pill{background:#26262c;border-radius:6px;padding:4px 10px}
 .pill:empty{display:none}
 .on{color:#7ee787} .off{color:#f2cc60}
 .pair{display:flex;gap:16px;justify-content:center}
 .card{background:#202026;border:2px solid #333;border-radius:8px;padding:8px;width:min(46vw,540px)}
 .card img{width:100%;aspect-ratio:1;object-fit:contain;background:#000;border-radius:4px;cursor:pointer}
 .card:hover{border-color:#6a8cff}
 .meta{display:flex;justify-content:space-between;margin-top:6px;color:#999}
 a{color:#8fb0ff}
 pre{max-height:180px;overflow:auto;background:#111;padding:6px;font-size:11px;display:none;white-space:pre-wrap}
 .show pre{display:block}
 .btns{display:flex;gap:10px;justify-content:center;margin:14px 0}
 button,select{background:#2d2d36;color:#eee;border:1px solid #444;border-radius:6px;padding:6px 12px;font-size:14px;cursor:pointer}
 button:hover{background:#3a3a48}
 #msg{text-align:center;color:#fc6;min-height:20px}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(128px,1fr));gap:6px;margin-top:10px}
 .grid figure{margin:0;position:relative}
 .grid img{width:100%;border-radius:3px;display:block}
 .grid figcaption{position:absolute;left:3px;bottom:3px;background:#000a;font-size:11px;padding:1px 4px;border-radius:3px}
 svg{vertical-align:middle}
</style></head><body>
<header>
 <b>JXL art RLHF</b>
 <span class="pill" id="phase"></span>
 <span class="pill">batch <progress id="prog"></progress> <span id="batch"></span></span>
 <span class="pill" id="picks"></span>
 <span class="pill" id="rm"></span>
 <span class="pill" id="rl"></span>
 <span class="pill"><svg id="chart" width="220" height="34"></svg> <span style="color:#6a8cff">reward</span> · <span style="color:#999">KL</span></span>
 <span class="pill">roll back to <select id="snap"></select> <button id="rb">↺</button></span>
 <button id="code">show code (c)</button>
</header>
<div id="msg"></div>
<div class="pair" id="pair">
 <div class="card"><img id="imgA" title="prefer left (←)"><div class="meta"><span>A</span><span id="mA"></span><a id="dA" download>.jxl</a></div><pre id="tA"></pre></div>
 <div class="card"><img id="imgB" title="prefer right (→)"><div class="meta"><span>B</span><span id="mB"></span><a id="dB" download>.jxl</a></div><pre id="tB"></pre></div>
</div>
<div class="btns">
 <button onclick="pick('a')">← A is better</button>
 <button onclick="pick('tie')">↓ about equal</button>
 <button onclick="pick('bad')">x both bad</button>
 <button onclick="pick('b')">B is better →</button>
 <button onclick="undo()">u undo</button>
 <button onclick="loadPair()">s skip</button>
</div>
<h3>This batch, best first by your taste model</h3><div class="grid" id="grid"></div>
<script>
let cur=null, st=null, busy=false;
const $=id=>document.getElementById(id);
async function j(url,opt){const r=await fetch(url,opt);const b=await r.json();if(!r.ok)throw new Error(b.detail||r.status);return b}
const post=(u,b)=>j(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
function pct(x){return x==null?'n/a':(x*100).toFixed(0)+'%'}
async function refresh(){
 st=await j('/api/state');
 const rb=st.rollback;
 $('picks').textContent=`${st.picks} picks total · model step ${st.step}`+
  (rb?` (step-${rb.to} weights + ${st.step-rb.at} new steps)`:'');
 $('rm').textContent=st.rm?`taste model: held-out ${pct(st.rm.cv_acc)} · refit ${st.rm.fit_ms} ms`:'taste model: needs a few picks';
 const l=st.last;
 const sim=l&&l.batch_sim!=null?` · variety: similarity <span class="${l.batch_sim>0.9?'off':'on'}">${l.batch_sim.toFixed(2)}${l.batch_sim>0.9?' ⚠ samey':''}</span>`:'';
 const cp=l&&l.copy_frac!=null?` · near-copies of known art ${pct(l.copy_frac)}`:'';
 $('rl').innerHTML=l?`last step: reward ${l.reward_mean.toFixed(2)} · valid ${pct(l.ok_frac)} · KL ${l.kl.toFixed(3)}${sim}${cp}`:'';
 const training=st.phase==='train';
 if(training){
  const left=st.eta_s==null?'':` · ~${Math.ceil(st.eta_s/60)} min left`;
  $('phase').innerHTML=`<span class="off">● training</span> ${st.train_done}/${st.train_total} steps${left}`;
  $('prog').max=st.train_total;$('prog').value=st.train_done;$('batch').textContent='paused';
  cur=null;$('pair').style.opacity=.3;
  $('msg').textContent=`Training on your picks (${st.train_done}/${st.train_total} steps${left}). A new batch from the updated model appears when it finishes.`;
 }else{
  $('phase').innerHTML=st.batch_ready?`<span class="on">● your turn</span>`:`<span class="off">● sampling a new batch…</span>`;
  $('prog').max=st.batch_size;$('prog').value=st.batch_done;$('batch').textContent=`${st.batch_done}/${st.batch_size}`;
  if(st.note)$('msg').textContent=`Batch extended: ${st.note}`;
 }
 const sel=$('snap'), key=st.snapshots.map(s=>s.step+(s.old_branch?'o':'')).join();
 if(sel.dataset.key!==key){const keep=sel.value;sel.dataset.key=key;
  sel.innerHTML='<option value="">choose…</option>'+st.snapshots.slice().reverse().map(s=>
   `<option value="${s.step}">step ${s.step}${s.step===0?' (SFT)':''}${s.old_branch?' (old branch)':''}</option>`).join('');
  sel.value=keep}
 drawChart(await j('/api/metrics'));
 if(!training&&!cur)loadPair();
}
function drawChart(m){
 const svg=$('chart'),W=220,H=34;
 if(m.length<2){svg.innerHTML='';return}
 const line=(key,color)=>{const v=m.map(r=>r[key]);const lo=Math.min(...v),hi=Math.max(...v),d=hi-lo||1;
  return `<polyline fill="none" stroke="${color}" stroke-width="1.5" points="${v.map((x,i)=>`${(i/(v.length-1)*W).toFixed(1)},${(H-2-(x-lo)/d*(H-4)).toFixed(1)}`).join(' ')}"/>`};
 svg.innerHTML=line('kl','#999')+line('reward_mean','#6a8cff');
}
async function loadPair(){
 const p=await j('/api/pair');
 if(p.none){cur=null;$('pair').style.opacity=.3;$('msg').textContent=p.reason;return}
 cur=p;$('pair').style.opacity=1;$('msg').textContent='';
 for(const [k,it] of [['A',p.a],['B',p.b]]){
  $('img'+k).src='/items/'+it.id+'.png';$('t'+k).textContent=it.text;
  const kn=it.known, close=kn&&kn.penalty>0;
  $('m'+k).textContent=`${it.bytes} B · ${it.size[0]}×${it.size[1]} · model step ${it.step}`+(close?` · ≈ "${kn.title}"`:'');
  $('m'+k).title=kn?`closest known art: "${kn.title}" (recipe ${kn.recipe_sim.toFixed(2)}, image ${kn.image_sim.toFixed(2)})`:'';
  $('m'+k).style.color=close?'#f2cc60':'';
 }
 loadGrid();
}
async function loadGrid(){
 const items=await j('/api/recent');
 $('grid').innerHTML=items.map(it=>`<figure><img src="/items/${it.id}.png" title="${it.bytes} B${it.rm!=null?' · taste '+it.rm.toFixed(2):''}"><figcaption>step ${it.step}</figcaption></figure>`).join('');
}
async function pick(choice){
 if(!cur||busy)return;busy=true;
 try{await post('/api/pref',{a:cur.a.id,b:cur.b.id,choice});cur=null;
  if(st&&st.batch_done+1>=st.batch_size){await refresh();return}
  await loadPair()}
 catch(e){$('msg').textContent=e.message;cur=null;refresh()}
 finally{busy=false}
}
async function undo(){await post('/api/pref',{undo:true});cur=null;await loadPair();refresh()}
$('imgA').onclick=()=>pick('a');$('imgB').onclick=()=>pick('b');
$('rb').onclick=async()=>{if($('snap').value==='')return;const s=+$('snap').value;if(!confirm(`Roll the model back to step ${s}? Your picks are kept.`))return;
 try{await post('/api/rollback',{step:s});$('msg').textContent=`Rolling back to step ${s}…`}catch(e){$('msg').textContent=e.message}};
$('code').onclick=()=>document.body.classList.toggle('show');
const KEYS={ArrowLeft:()=>pick('a'),ArrowRight:()=>pick('b'),ArrowDown:()=>pick('tie'),
 x:()=>pick('bad'),u:undo,s:loadPair,c:()=>$('code').click()};
document.onkeydown=e=>{if(e.target.tagName==='SELECT'||e.ctrlKey||e.metaKey||e.altKey)return;
 const f=KEYS[e.key];if(!f)return;e.preventDefault();if(!e.repeat)f()};
refresh();setInterval(refresh,3000);
</script></body></html>
"""
