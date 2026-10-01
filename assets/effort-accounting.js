/* Synthetic fixture: FE 2h marginal, RDL .5h marginal +1.5h shared; MAE4/3.8.
   States: marginal/amortized, tasks1..20, observed/missing. Baseline stays4x. */
(function(){'use strict';
 function mount(host){
 host.innerHTML=`<p class="effort-tag">SYNTHETIC · change accounting, hold accuracy fixed</p><div class="effort-controls"><label>Accounting <select data-scope><option value="marginal">Marginal human work</option><option value="amortized">Include shared setup per task</option></select></label><label>Tasks sharing setup <input data-tasks type="range" min="1" max="20" value="1"><output data-n>1</output></label><label><input data-observed type="checkbox" checked> Both human logs observed and complete</label><button type="button" data-reset>Reset example</button></div><div class="effort-cards"><div><strong>FE</strong><p>2.00 h marginal + 0 h shared</p><p>Fixed synthetic MAE: 4.0</p></div><div><strong>RDL</strong><p>0.50 h marginal + 1.50 h shared</p><p>Fixed synthetic MAE: 3.8</p></div></div><p>Marginal baseline: <strong>4.00×</strong>; 75% less active human time.</p><p class="effort-result" aria-live="polite" data-result></p><p data-formula></p><p>Training runtime is excluded from human hours. This fixture is not the measured F1 result.</p>`;
 const scope=host.querySelector('[data-scope]'),tasks=host.querySelector('[data-tasks]'),observed=host.querySelector('[data-observed]');
 function draw(){const n=Math.max(1,Math.min(20,Number(tasks.value)||1));tasks.value=n;host.querySelector('[data-n]').textContent=n;
 const h=.5+(scope.value==='amortized'?1.5/n:0),ratio=2/h;
 host.dataset.hours=h;host.dataset.ratio=observed.checked?ratio:'';
 host.querySelector('[data-result]').textContent=observed.checked?`${ratio.toFixed(2)}× · ${(100*(1-h/2)).toFixed(1)}% less human time`:'NOT_OBSERVED · no ratio can be reported';
 host.querySelector('[data-formula]').textContent=observed.checked?`FE 2.00 h ÷ RDL (${scope.value==='amortized'?`0.50 + 1.50 / ${n}`:'0.50'} h) = ${ratio.toFixed(2)}×`:'Unknown effort is not zero. Accuracy values alone cannot supply the missing hours.';
 }
 for(const el of [scope,tasks,observed])el.addEventListener('input',draw);
 host.querySelector('[data-reset]').addEventListener('click',()=>{scope.value='marginal';tasks.value=1;observed.checked=true;draw();});draw();return {draw};
 }
 window.EffortAccounting={mount};
})();
