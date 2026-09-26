/* L113: scalar SAGE trace. Path features [2,4,8,16,32,64]; node1 root=4.
   q=1: neighbor mean2 + root4 =6. q>=2: mean(2,8)+4 =9.
   Memory calculator is a no-overlap node-occurrence upper bound, not a profiler. */
(function(){'use strict';
 document.querySelectorAll('[data-scaling-trace]').forEach(el=>{
  el.innerHTML='<label>Clusters in the induced batch <select aria-label="Clusters in the induced batch"><option value="1">C0 only</option><option value="2">C0 + C1</option><option value="3">C0 + C1 + C2</option></select></label> <button type="button">Reset</button><div class="scaling-trace"></div><output aria-live="polite"></output><p>Fixed full-graph baseline: node 1 receives (2 + 8) / 2 + 4 = <strong>9</strong>. Both weights are 1; bias is 0.</p>';
  const select=el.querySelector('select'),out=el.querySelector('output'),trace=el.querySelector('.scaling-trace');
  function update(){const q=Number(select.value);trace.innerHTML=[0,1,2].map(c=>'<div class="scaling-cluster '+(c<q?'active':'')+'"><strong>C'+c+'</strong><span>node '+(c*2)+' · x='+2**(c*2+1)+'</span><span>node '+(c*2+1)+' · x='+2**(c*2+2)+'</span></div>').join('');out.textContent=q===1?'Edge 1–2 is cut. Neighbor mean = 2. Root = 4. Output = 6; gap from full = −3.':'Edge 1–2 is restored. Neighbor mean = (2 + 8) / 2 = 5. Root = 4. Output = 9; gap from full = 0.';}
  select.addEventListener('change',update);el.querySelector('button').addEventListener('click',()=>{select.value='1';update();});update();
 });
 document.querySelectorAll('[data-scaling-memory]').forEach(el=>{
  el.innerHTML='<label>Training roots <input aria-label="Training roots" type="range" min="64" max="1024" step="64" value="256"></label><label>Fanout per hop <input aria-label="Fanout per hop" type="range" min="2" max="20" value="10"></label><output aria-live="polite"></output><p>Three hops; 256 float32 channels; no overlap assumption. This counts one feature buffer only. Full-products baseline: 2,449,029 × 256 × 4 / 2³⁰ = 2.34 GiB per hidden buffer.</p>';
  const inputs=el.querySelectorAll('input'),out=el.querySelector('output');function update(){const b=+inputs[0].value,f=+inputs[1].value,n=b*(1+f+f*f+f*f*f);out.textContent=b+' roots × (1 + '+f+' + '+f*f+' + '+f*f*f+') = '+n.toLocaleString('en-US')+' node occurrences; one feature buffer = '+(n*256*4/2**30).toFixed(3)+' GiB. Repeated nodes may be deduplicated; gradients, edges and workspace add memory.';}inputs.forEach(x=>x.addEventListener('input',update));update();
 });
})();
