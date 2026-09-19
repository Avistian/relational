/* Information interfaces: defaults 50% balanced bound, missing delta, one hop.
   Controls retain the baseline and recompute the exact affected quantity. */
(function(global){
 'use strict';
 function bound(n0,n1){return Math.max(n0,n1)/(n0+n1);}
 function mount(host,kind){
  host.classList.add('ceiling-widget');
  if(kind==='bound'){
   host.innerHTML='<p>Predict how class imbalance changes the bound. One collision class; ten identical vectors.</p><label>Number with label 1 <input type="range" min="1" max="9" value="5"></label><button type="button">Reset</button><output aria-live="polite"></output>';
   const input=host.querySelector('input'),out=host.querySelector('output');
   function draw(){const n=Number(input.value);out.textContent=`Labels: ${10-n} zeros, ${n} ones. Best correct = max(${10-n}, ${n}) = ${Math.max(10-n,n)}. Ceiling ${(100*bound(10-n,n)).toFixed(0)}%. Baseline: 5 + 5 → 50%.`;}
   input.addEventListener('input',draw);host.querySelector('button').onclick=()=>{input.value=5;draw();};draw();
  }else if(kind==='repair'){
   host.innerHTML='<p>Hold histories and targets fixed. Predict which feature distinguishes them.</p><label>Information supplied <select><option value="flat">Count, sum, mean, max</option><option value="delta">Also last minus first</option></select></label><div class="path">A: 10 → 30 → 50; label 1<br>B: 50 → 30 → 10; label 0</div><button type="button">Reset</button><output aria-live="polite"></output>';
   const input=host.querySelector('select'),out=host.querySelector('output');
   function draw(){out.textContent=input.value==='flat'?'Both: [3, 90, 30, 50]. One collision class; ceiling 50%. Baseline retained.':'A delta = 50 − 10 = +40; B delta = 10 − 50 = −40. Rule delta > 0 separates them; ceiling 100%. Flat baseline: 50%.';}
   input.onchange=draw;host.querySelector('button').onclick=()=>{input.value='flat';draw();};draw();
  }else if(kind==='reach'){
   host.innerHTML='<p>Predict when the merchant risk becomes accessible. All records are eligible; order amounts are identical.</p><label>Allowed hops <select><option value="0">0 · customer only</option><option value="1" selected>1 · reach order</option><option value="2">2 · reach merchant</option></select></label><div class="path">World A: customer → order amount 30 → merchant risk 1<br>World B: customer → order amount 30 → merchant risk 0</div><button type="button">Reset</button><output aria-live="polite"></output>';
   const input=host.querySelector('select'),out=host.querySelector('output');
   function draw(){const n=Number(input.value);out.textContent=n===2?'2 hops: risk 1 versus risk 0 is accessible. A preserving reducer can distinguish the worlds. Baseline: one-hop amounts both 30.':n===1?'1 hop: both orders carry amount 30. Merchant risk is unavailable to this interface. Baseline: one hop.':'0 hops: customer attributes are identical. Both orders and merchants are outside the interface. Baseline: one-hop amounts both 30.';}
   input.onchange=draw;host.querySelector('button').onclick=()=>{input.value='1';draw();};draw();
  }
 }
 global.CeilingViz={mount,bound};
})(window);
