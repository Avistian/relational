/* Synthetic scalar head. Baseline query time 4, event times [1,3], values [2,8].
   Slider changes query time only, in [4,10]; logits = cos(query - event).
   Exposes every intermediate and retains the baseline for causal comparison. */
(function(root){
'use strict';
function calculate(t){const gaps=[t-1,t-3],keys=gaps.map(Math.cos),exp=keys.map(Math.exp),sum=exp[0]+exp[1],weights=exp.map(x=>x/sum);return {t,gaps,keys,weights,value:2*weights[0]+8*weights[1]};}
function mount(el){if(!el)return;el.className='tgat-widget';el.innerHTML='<label>Query time <output>4.0</output><input aria-label="Query time" type="range" min="4" max="10" step="0.1" value="4"></label><div class="tgat-trace"></div><p class="tgat-readout" aria-live="polite"></p><button type="button">Reset to time 4</button>';
const slider=el.querySelector('input'),base=calculate(4);
function draw(){const t=Math.max(4,Math.min(10,Number(slider.value))),r=calculate(t);el.querySelector('output').textContent=t.toFixed(1);el.querySelector('.tgat-trace').innerHTML=r.gaps.map((gap,i)=>'<section><h3>Record '+(i+1)+' · event time '+[1,3][i]+'</h3>Elapsed: '+gap.toFixed(1)+'<br>Key = logit: '+r.keys[i].toFixed(4)+'<br>Weight: '+r.weights[i].toFixed(4)+'<div class="tgat-bar" style="width:'+100*r.weights[i]+'%"></div>Value: '+[2,8][i]+'<br>Contribution: '+(r.weights[i]*[2,8][i]).toFixed(4)+'</section>').join('');el.querySelector('.tgat-readout').textContent='Baseline at t=4: '+base.value.toFixed(4)+'. Current weighted value: '+r.value.toFixed(4)+'. Change: '+(r.value-base.value).toFixed(4)+'. Older record weight: '+r.weights[0].toFixed(4)+'.';}
slider.addEventListener('input',draw);el.querySelector('button').addEventListener('click',()=>{slider.value=4;draw();});draw();return {setTime(t){slider.value=Math.max(4,Math.min(10,t));draw();},state(){return calculate(Number(slider.value));}};
}
root.TemporalAttentionViz={mount,calculate};
})(typeof window==='undefined'?globalThis:window);
