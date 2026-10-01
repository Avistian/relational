/* Synthetic fixed population: baseline p=1, outcomes[0,1,1,9].
   Slider changes p only; population control replaces two tied outcomes with0.
   Default/reset has MAE2.25, below/equal/above=.25/.50/.25, violation0.
   A constant prediction illustrates marginal median balance, not conditional proof. */
(function(root){
'use strict';
function compute(p,mode){
 p=Math.max(0,Math.min(10,Number(p)));const y=mode==='shift'?[0,0,0,9]:[0,1,1,9];
 const below=y.filter(v=>v<p).length/4,equal=y.filter(v=>v===p).length/4,above=y.filter(v=>v>p).length/4;
 return {p,y,below,equal,above,violation:Math.max(0,below-.5,above-.5),mae:y.reduce((s,v)=>s+Math.abs(v-p),0)/4,rmse:Math.sqrt(y.reduce((s,v)=>s+(v-p)**2,0)/4),residual:y.reduce((s,v)=>s+v-p,0)/4};
}
function mount(host){
 host.innerHTML='<p><strong>Predict first:</strong> will moving the prediction to the mean improve MAE?</p><label>Prediction <input type="range" min="0" max="10" step="0.25" value="1" aria-label="Scalar prediction"></label> <label>Outcomes <select aria-label="Outcome population"><option value="ties">0, 1, 1, 9</option><option value="shift">0, 0, 0, 9</option></select></label> <button type="button">Reset</button><p class="reg-readout" aria-live="polite"></p><div class="reg-bars"></div><p>Baseline: p=1, outcomes [0,1,1,9], MAE=2.25, median violation=0. Moving p changes predictions only. Population changes are synthetic interventions.</p>';
 const range=host.querySelector('input'),select=host.querySelector('select');
 function draw(){const r=compute(range.value,select.value);range.value=r.p;host.dataset.violation=r.violation;host.dataset.mae=r.mae;
 host.querySelector('.reg-readout').textContent=`Prediction ${r.p.toFixed(2)} · MAE ${r.mae.toFixed(3)} · RMSE ${r.rmse.toFixed(3)} · mean(y−p) ${r.residual.toFixed(3)} · median violation ${r.violation.toFixed(2)}`;
 host.querySelector('.reg-bars').innerHTML=['below','equal','above'].map(k=>`<div><span>${k}: ${(100*r[k]).toFixed(0)}%</span><meter min="0" max="1" value="${r[k]}" aria-label="Fraction ${k}">${r[k]}</meter></div>`).join('');}
 range.addEventListener('input',draw);select.addEventListener('change',draw);host.querySelector('button').addEventListener('click',()=>{range.value=1;select.value='ties';draw()});draw();return {compute,draw};
}
root.RegressionCalibrationViz={compute,mount};if(typeof module!=='undefined')module.exports={compute,mount};
})(typeof window==='undefined'?globalThis:window);
