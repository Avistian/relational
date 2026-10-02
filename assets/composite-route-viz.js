/* L182: two legal messages, optional duplicates, dated third purchase,
   a deliberately merged route, and row permutation. Baseline:0/11/false/false. */
(function(g){
 'use strict';
 function calculate(duplicates,time,mix,reverse){
  duplicates=Math.max(0,Math.min(2,Math.round(Number(duplicates)||0)));
  time=Number(time)===9?9:11;
  let messages=[[1,1],[2,1]];
  for(let i=0;i<duplicates;i++)messages.push([1,1]);
  if(time<10)messages.push([100,99]);
  if(mix)messages.push([8,0]);
  if(reverse)messages.reverse();
  const logits=messages.map(z=>z[0]/Math.sqrt(2));const maximum=Math.max(...logits);
  const exps=logits.map(v=>Math.exp(v-maximum));const total=exps.reduce((a,b)=>a+b,0);
  const weights=exps.map(v=>v/total);const output=[0,1].map(k=>messages.reduce((s,z,i)=>s+weights[i]*z[k],0));
  return {messages,weights,output,duplicates,time,mix:!!mix,reverse:!!reverse};
 }
 function mount(host){
  host.classList.add('route-widget','composite-route');
  host.innerHTML='<h3>One route, one customer</h3><p>Query [1, 0] · cutoff 10 · identity projections. The baseline output is [1.6698, 1.0000].</p><div class="cr-controls"><label>Copies of first legal purchase <input data-duplicates type="range" min="0" max="2" value="0"><output data-count>0</output></label><label>Third purchase time <select data-time><option value="11">11 — after cutoff</option><option value="9">9 — before cutoff</option></select></label><label><input type="checkbox" data-mix> Merge an unrelated route [8, 0]</label><label><input type="checkbox" data-reverse> Reverse message order</label><button type="button" data-reset>Reset baseline</button></div><div class="cr-messages" aria-label="Legal messages and weights"></div><p class="route-readout" aria-live="polite"></p><p class="cr-caption">Synthetic operator trace. Changing a timestamp changes legality; it is not a real availability audit. No trained accuracy is shown.</p>';
  const d=host.querySelector('[data-duplicates]'),t=host.querySelector('[data-time]'),m=host.querySelector('[data-mix]'),r=host.querySelector('[data-reverse]');
  function render(){
   const s=calculate(d.value,t.value,m.checked,r.checked);host.dataset.result=JSON.stringify(s);host.querySelector('[data-count]').textContent=s.duplicates;
   host.querySelector('.cr-messages').innerHTML=s.messages.map((z,i)=>'<div><strong>Message '+(i+1)+'</strong><span>['+z.join(', ')+']</span><span>weight '+s.weights[i].toFixed(4)+'</span><meter min="0" max="1" value="'+s.weights[i]+'" aria-label="Message '+(i+1)+' weight"></meter></div>').join('');
   host.querySelector('.route-readout').textContent=s.messages.length+' legal/merged messages → output ['+s.output.map(v=>v.toFixed(4)).join(', ')+']. Difference from baseline: '+(s.output[0]-1.6697615493266569).toFixed(4)+' in coordinate 1.';
  }
  [d,t,m,r].forEach(x=>x.addEventListener('input',render));
  host.querySelector('[data-reset]').addEventListener('click',()=>{d.value=0;t.value=11;m.checked=false;r.checked=false;render();});render();
  return {render,calculate};
 }
 g.CompositeRoute={mount,calculate};
})(window);
