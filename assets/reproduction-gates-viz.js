/* L140: synthetic selection fixture and hypothetical evidence gates.
   Baseline epoch2; 5/5 seeds, 0.2pp gap, known archive mismatch. */
(function(){'use strict';
const s=document.getElementById('l140-selection');
if(s){s.innerHTML='<label for="l140-epoch">Inspect epoch <input id="l140-epoch" type="range" min="1" max="3" value="2"></label><p class="audit-result" aria-live="polite"></p><button type="button">Reset</button>';
 const input=s.querySelector('input'),out=s.querySelector('p');
 function draw(){const e=Number(input.value);out.textContent=`Epoch ${e}: validation ${[.70,.75,.75][e-1].toFixed(2)}, test ${[.90,.60,1][e-1].toFixed(2)}. Frozen choice: epoch 2. ${e===2?'Valid first maximum.':e===3?'Reject: later validation tie.':'Reject: lower validation AUROC.'}`;}
 input.addEventListener('input',draw);s.querySelector('button').onclick=()=>{input.value=2;draw();};draw();}
const g=document.getElementById('l140-gates');
if(g){g.innerHTML='<p><strong>Hypothetical evidence:</strong> controls do not change the measured author results.</p><label for="l140-seeds">Completed seeds <input id="l140-seeds" type="range" min="0" max="5" value="5"></label><label for="l140-gap">Absolute mean gap (percentage points) <input id="l140-gap" type="range" min="0" max="2" step="0.1" value="0.2"></label><label for="l140-match"><input id="l140-match" type="checkbox"> Archive matches reported population</label><p class="audit-result" aria-live="polite"></p><p>Fixed baseline: 5/5 seeds · gap 0.2pp · protocol GAPPED · historical identity NOT_ESTABLISHED.</p><button type="button">Reset</button>';
 const n=g.querySelector('#l140-seeds'),d=g.querySelector('#l140-gap'),m=g.querySelector('#l140-match'),out=g.querySelector('.audit-result');
 function draw(){const complete=Number(n.value)===5;out.textContent=`${n.value}/5 seeds; gap ${Number(d.value).toFixed(1)}pp. Execution: ${complete?'COMPLETE':'INCOMPLETE'}. Score: ${complete?(Number(d.value)<=1?'CLOSE':'OUTSIDE_TOLERANCE'):'NOT_EVALUATED'}. Protocol: ${m.checked?'ALIGNED_WITH_RELEASE':'GAPPED'}. Historical identity: NOT_ESTABLISHED.`;}
 [n,d,m].forEach(x=>x.addEventListener('input',draw));g.querySelector('button').onclick=()=>{n.value=5;d.value=.2;m.checked=false;draw();};draw();}
if(window.RetrievalBank)RetrievalBank.mount(document.getElementById('warmup'),{upTo:140,count:3});
if(window.Teachback)Teachback.mount(document.getElementById('l140-teachback'),{prompt:'Explain how all seeds can finish and the score can be close while historical reproduction remains unestablished.',points:['Every planned seed completed the released schedule.','Closeness concerns the mean metric only.','Archive and historical availability gaps remain.'],model:'Complete execution establishes that the planned runs finished. A close mean concerns a numerical tolerance. Neither restores missing historical data, clock information or RNG identity.'});
})();
