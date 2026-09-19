/* Synthetic budget default 5 -> Trees; 10 -> TabM; 40 -> ICL; 1 -> none.
   Cohort default all11 -> random CatBoost 1.818; matched3 -> CatBoost 1.000.
   Both figures retain their declared baseline. No test losses drive selection. */
(function(global){
'use strict';
const candidates=[{model:'Trees',loss:.32,ms:2},{model:'TabM',loss:.29,ms:8},{model:'ICL',loss:.27,ms:35}];
function budget(el){
 el.className='dg-panel';
 el.innerHTML='<h3>Which candidates can serve?</h3><p>Synthetic validation and latency fixture; not model measurements.</p><label>Latency budget: <strong class="budget-value"></strong> ms<input aria-label="Latency budget" type="range" min="1" max="40" value="5"></label><div class="candidate-list"></div><output aria-live="polite"></output><p class="baseline">Baseline at 5 ms: Trees, validation loss .32. Validation losses remain fixed; only the latency constraint changes.</p><button type="button">Reset to 5 ms</button>';
 const slider=el.querySelector('input');
 function draw(){const b=Number(slider.value),valid=candidates.filter(r=>r.ms<=b),winner=valid.slice().sort((a,b)=>a.loss-b.loss)[0];
 el.querySelector('.budget-value').textContent=b;
 el.querySelector('.candidate-list').replaceChildren(...candidates.map(r=>{const p=document.createElement('p');p.textContent=`${r.model}: loss ${r.loss.toFixed(2)} · p95 ${r.ms} ms · ${r.ms<=b?'eligible':'excluded'}`;return p;}));
 el.querySelector('output').textContent=winner?`Choose ${winner.model}: lowest validation loss among ${valid.length} feasible candidate(s).`:'No feasible candidate. Revise the implementation or constraint; do not ignore it.';
 }
 slider.addEventListener('input',draw);el.querySelector('button').addEventListener('click',()=>{slider.value=5;draw();});draw();
}
function cohort(el,data){
 el.className='dg-panel';el.innerHTML='<h3>Hold the dataset population in view</h3><label>Random panel <select aria-label="Random cohort"><option value="all">All 11 datasets</option><option value="matched">Matched 3 datasets</option></select></label><div class="dg-table"><table><thead><tr><th>Method</th><th>Random rank</th><th>Temporal rank (3)</th></tr></thead><tbody></tbody></table></div><output aria-live="polite"></output><p class="baseline">Baseline: all 11 random datasets. Temporal panel always has Ecom Offers, Homesite and Sberbank. Three seeds are averaged before ranking each dataset.</p>';
 const select=el.querySelector('select');
 function draw(){const random=select.value==='all'?data.all:data.matched;
 el.querySelector('tbody').replaceChildren(...Object.keys(random).map(a=>{const tr=document.createElement('tr');[a,random[a].toFixed(3),data.temporal[a].toFixed(3)].forEach(x=>{const td=document.createElement('td');td.textContent=x;tr.append(td);});return tr;}));
 el.querySelector('output').textContent=select.value==='all'?'Unequal populations: 11 random versus 3 temporal datasets. Do not attribute this gap solely to chronology.':'Matched names: 3 random versus 3 temporal datasets. Rows and time periods still differ; this is not a causal intervention.';
 }select.addEventListener('change',draw);draw();
}
global.DecisionGuideViz={budget,cohort};
})(window);
