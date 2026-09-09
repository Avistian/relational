/* Synthetic intervention: y=[0,0,1,1], A=[.1,.8,.9,.2].
 * Complementary B=[.8,.1,.2,.9] yields p=.45/.55 and loss .5978 at w=.5;
 * duplicate B=A yields loss .8574 for every weight. Native controls, fixed A baseline. */
(function(g){
'use strict';
const y=[0,0,1,1],a=[.1,.8,.9,.2];
function loss(p){return -p.reduce((s,v,i)=>s+(y[i]?Math.log(v):Math.log(1-v)),0)/4;}
function compute(weight,duplicate){const w=Math.max(0,Math.min(1,Number(weight)));const b=duplicate?a.slice():[.8,.1,.2,.9];const p=a.map((v,i)=>w*v+(1-w)*b[i]);return {w,a,b,p,baseline:loss(a),mixed:loss(p)};}
function mount(el){
 el.className='ensemble-panel';el.innerHTML='<p><label>Weight on A <input type="range" min="0" max="1" step="0.05" value="0.5"></label> <output></output></p><p><label><input type="checkbox"> Replace B with a duplicate of A</label> <button type="button">Reset</button></p><div class="ensemble-scroll"><table><thead><tr><th>Label y</th><th>A</th><th>B</th><th>wA + (1−w)B</th></tr></thead><tbody></tbody></table></div><p class="ensemble-readout" aria-live="polite"></p>';
 const slider=el.querySelector('input[type=range]'),box=el.querySelector('input[type=checkbox]');
 function draw(){let r=compute(slider.value,box.checked);el.querySelector('output').textContent=r.w.toFixed(2);el.querySelector('tbody').innerHTML=y.map((v,i)=>'<tr><td>'+v+'</td><td>'+r.a[i].toFixed(2)+'</td><td>'+r.b[i].toFixed(2)+'</td><td>'+r.p[i].toFixed(3)+'</td></tr>').join('');el.querySelector('.ensemble-readout').textContent='A alone: log loss '+r.baseline.toFixed(4)+' · mixture: '+r.mixed.toFixed(4)+' · change: '+(r.mixed-r.baseline).toFixed(4)+' (lower is better).';}
 slider.addEventListener('input',draw);box.addEventListener('change',draw);el.querySelector('button').addEventListener('click',()=>{slider.value='.5';box.checked=false;draw();});draw();return {draw};
}
g.CrossEnsembleViz={compute,mount};
})(window);
