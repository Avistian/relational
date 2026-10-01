/* Expected states: table/column have four semantic copies; cell has one.
 * Global corruption removes all copies. Root-only leaves three schema copies.
 * Cached clean embeddings retain all original copies. Decoder w=2; frozen
 * parameters still transmit w*(sigmoid(w*h)-1), detached path transmits none.
 */
(function(global){
'use strict';
function maskMount(host){
 host.innerHTML=`<section class="objective-widget"><h3>Which copy still carries the answer?</h3><div class="objective-controls"><label>Target identity<select data-target><option value="table">Table name</option><option value="column">Column name</option><option value="cell">Root cell</option></select></label><label>Corruption policy<select data-policy><option value="global">Mask every semantic copy</option><option value="root">Mask only the root row</option><option value="cache">Mask text after clean caching</option></select></label></div><div class="objective-rows" data-rows></div><p class="objective-answer" data-answer aria-live="polite"></p><p class="objective-baseline">Baseline: global masking before encoding exposes 0 direct copies. Orange marks a visible target occurrence. No accuracy is inferred.</p><button type="button">Reset masking</button></section>`;
 const target=host.querySelector('[data-target]'),policy=host.querySelector('[data-policy]');
 function update(){
  const col={table:0,column:1,cell:2}[target.value];let exposed=0;
  const text=['Earth','Mars','Earth','Venus'];
  host.querySelector('[data-rows]').innerHTML=text.map((value,i)=>{
   const parts=['Moons','planet',value];const isCopy=target.value!=='cell'||i===0;
   const masked=isCopy&&(policy.value!=='root'||i===0);
   const leaks=isCopy&&(policy.value==='cache'||!masked);if(leaks)exposed++;
   if(masked)parts[col]='[MASK]';else if(isCopy)parts[col]='<mark>'+parts[col]+'</mark>';
   return `<div class="objective-row">Row ${i}${i===0?' · target row':''}<br>${parts.join(' · ')}${policy.value==='cache'&&isCopy?'<br><mark>Cached vector still encodes the original target</mark>':''}</div>`;
  }).join('');
  host.querySelector('[data-answer]').textContent=exposed?`${exposed} direct target ${exposed===1?'copy remains':'copies remain'} accessible. The declared hidden-identity task is violated.`:'0 direct target copies accessible. Other cells with the same value remain valid context; this is not a guarantee against every shortcut.';
  host.dataset.exposed=String(exposed);
 }
 target.addEventListener('change',update);policy.addEventListener('change',update);
 host.querySelector('button').addEventListener('click',()=>{target.value='table';policy.value='global';update();});update();
}
function gradientMount(host){
 host.innerHTML=`<section class="objective-widget"><h3>Freeze the decoder; keep the learning signal</h3><div class="objective-controls"><label>Graph output h <input data-h type="range" min="-1" max="1" step=".1" value=".4"></label><label><span><input data-frozen type="checkbox" checked> Freeze decoder weight</span></label><label><span><input data-detach type="checkbox"> Detach decoder input</span></label></div><div class="objective-math"><div>Graph output h<output data-h-value></output></div><div>p(target = 1)<output data-p></output></div><div>Gradient reaching graph<output data-grad></output></div><div>Decoder parameter gradient<output data-w-grad></output></div></div><p class="objective-answer" data-answer aria-live="polite"></p><p class="objective-baseline">Fixed w = 2; target = 1. Baseline h = 0.4 gives p = 0.690 and graph gradient −0.620. This is one analytic step; toggles do not fit a model.</p><button type="button">Reset gradient</button></section>`;
 const h=host.querySelector('[data-h]'),frozen=host.querySelector('[data-frozen]'),detach=host.querySelector('[data-detach]');
 function update(){
  const value=Math.max(-1,Math.min(1,Number(h.value)));const p=1/(1+Math.exp(-2*value));
  host.querySelector('[data-h-value]').textContent=value.toFixed(1);host.querySelector('[data-p]').textContent=p.toFixed(3);
  host.querySelector('[data-grad]').textContent=detach.checked?'Absent (path severed)':(2*(p-1)).toFixed(3);
  host.querySelector('[data-w-grad]').textContent=frozen.checked?'Absent (weight frozen)':(value*(p-1)).toFixed(3);
  host.querySelector('[data-answer]').textContent=detach.checked?'Prediction still works, but this loss cannot teach the graph.':frozen.checked?'The decoder weight stays fixed while the graph receives a gradient.':'Both the graph and decoder can learn; this changes the second-stage protocol.';
 }
 for(const input of [h,frozen,detach])input.addEventListener('input',update);
 host.querySelector('button').addEventListener('click',()=>{h.value='.4';frozen.checked=true;detach.checked=false;update();});update();
}
global.PretrainingObjective={maskMount,gradientMount};
})(window);
