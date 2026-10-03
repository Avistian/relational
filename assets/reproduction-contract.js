/* Expected states: released+a shifts b/c/d to1/2/3; released+z leaves0/1/2.
 * Frozen is an illustrative repair, not a repaired RDBLearn run.
 * Synthetic label query day0, available day365: event history is earlier for t>0,
 * availability additionally requires t>=365. Actual L192 annual cuts have no overlap.
 */
(function(root){
 function encoding(category,policy){
  if(!['a','c','z'].includes(category)||!['released','frozen'].includes(policy))throw new Error('Invalid encoding scenario');
  const known=['b','c','d'],classes=policy==='released'?[...new Set([...known,category])].sort():known;
  return {category,policy,codes:known.map(x=>classes.indexOf(x)),classes,unknownCode:classes.includes(category)?classes.indexOf(category):3};
 }
 function clock(day){day=Math.max(0,Math.min(400,Math.round(Number(day))));if(!Number.isFinite(day))throw new Error('Invalid day');return {day,released:day>0,available:day>0&&day>=365};}
 function mountEncoding(host){
  host.classList.add('rc-widget');host.innerHTML='<h3>Does another query change this query?</h3><p>Fixed support: b → 0, c → 1, d → 2. Only the arriving category changes.</p><div class="rc-controls"><label>Arriving category <select data-category><option>a</option><option>c</option><option>z</option></select></label><label>Encoding policy <select data-policy><option value="released">Released sorted expansion</option><option value="frozen">Illustrative frozen vocabulary</option></select></label><button type="button" data-reset>Reset</button></div><div data-codes class="rc-codes"></div><p aria-live="polite" data-readout></p>';
  const c=host.querySelector('[data-category]'),p=host.querySelector('[data-policy]');
  function update(){const r=encoding(c.value,p.value);host.dataset.result=JSON.stringify(r);host.querySelector('[data-codes]').innerHTML=['b','c','d'].map((name,i)=>'<div><strong>'+name+'</strong><span>Support '+i+'</span><span>Query '+r.codes[i]+'</span></div>').join('');const changed=r.codes.some((v,i)=>v!==i);host.querySelector('[data-readout]').textContent=(changed?'FAIL: known query codes differ from stored support codes.':'Known codes remain aligned in this scenario.')+' '+(p.value==='frozen'?'Illustrative repair only; no repaired model was run.':'This reproduces the released encoder rule.');}
  c.addEventListener('change',update);p.addEventListener('change',update);host.querySelector('[data-reset]').addEventListener('click',()=>{c.value='a';p.value='released';update();});update();return {update};
 }
 function mountClock(host){
  host.classList.add('rc-widget');host.innerHTML='<h3>Two clocks for one historical label</h3><p>Synthetic label window: day 0 through day 365. Query time moves; the window stays fixed.</p><div class="rc-controls"><label>New query day <input type="range" min="0" max="400" value="180" data-day></label><button type="button" data-reset>Reset</button></div><p aria-live="polite" data-readout></p><meter min="0" max="400" value="180" aria-label="Query day"></meter><p>0 · historical prediction ───── 365 · window ends</p><p>The official L192 cutoffs are annual. Their audited schedule has zero past training labels with unfinished windows.</p>';
  const input=host.querySelector('[data-day]');function update(){const r=clock(input.value);host.dataset.result=JSON.stringify(r);host.querySelector('meter').value=r.day;host.querySelector('[data-readout]').textContent='Day '+r.day+': earlier prediction = '+(r.released?'YES':'NO')+'; label admitted by window-end policy = '+(r.available?'YES':'NO')+'.';}input.addEventListener('input',update);host.querySelector('[data-reset]').addEventListener('click',()=>{input.value=180;update();});update();return {update};
 }
 const api={encoding,clock,mountEncoding,mountClock};if(typeof module!=='undefined')module.exports=api;root.ReproductionContract=api;
})(typeof window==='undefined'?globalThis:window);
