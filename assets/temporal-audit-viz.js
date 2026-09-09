/* Label delay 0..4, unchanged event partitions. Baseline delay=0 stays visible.
 * Training freezes at time 8; validation labels must arrive by selection time 11.
 * Table reflows through horizontal scrolling on phones; controls are native.
 */
(function(g){'use strict';
function compute(delay){
  delay=Math.max(0,Math.min(4,Math.round(Number(delay)||0)));
  return Array.from({length:12},(_,i)=>{
    const t=i+1,part=t<8?'train':t<11?'validation':'test',available=t+delay;
    const cutoff=part==='train'?8:part==='validation'?11:null;
    return {t,part,available,eligible:cutoff===null?null:available<=cutoff};
  });
}
function mount(el){
  el.className='temporal-audit';
  el.innerHTML='<label>Label delay: <span class="delay">3</span> time units <input type="range" min="0" max="4" value="3" aria-label="Label delay"></label><button type="button">Reset</button><p>Fixed: train events 1–7; validation 8–10; test 11–12. Fit at 8, select at 11. Delay-zero baseline: 7 training labels and 3 validation labels available.</p><div class="scroll"><table><thead><tr><th>Event time</th><th>Partition</th><th>Label arrives</th><th>Usable by deadline?</th></tr></thead><tbody></tbody></table></div><output aria-live="polite"></output>';
  const input=el.querySelector('input'),body=el.querySelector('tbody'),out=el.querySelector('output');
  function render(){
    const rows=compute(input.value);el.querySelector('.delay').textContent=input.value;
    body.innerHTML=rows.map(r=>'<tr'+(r.eligible===false?' class="unavailable"':'')+'><td>'+r.t+'</td><td>'+r.part+'</td><td>'+r.available+'</td><td>'+(r.eligible===null?'Score later':r.eligible?'Yes':'No — wait or exclude')+'</td></tr>').join('');
    out.textContent=rows.filter(r=>r.part==='train'&&r.eligible).length+' / 7 training labels; '+rows.filter(r=>r.part==='validation'&&r.eligible).length+' / 3 validation labels usable. Event ordering alone does not enforce label availability.';
  }
  input.addEventListener('input',render);el.querySelector('button').addEventListener('click',()=>{input.value=3;render();});render();
  return {setDelay(d){input.value=Math.max(0,Math.min(4,Math.round(Number(d)||0)));render();}};
}
g.TemporalAuditViz={mount,compute};
})(window);
