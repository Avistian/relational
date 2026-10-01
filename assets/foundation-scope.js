/* Protocol illustrations only. Routes: no updates/0 or 8 labels, head-only,
   backbone updates. Database states: known disjoint, known overlap, unknown.
   Eight target-context labels do not change the pretraining inventory. */
(function(g){
'use strict';
const routes={none:['ZERO_SHOT','IN_CONTEXT'],head:['FROZEN_ENCODER_HEAD','FROZEN_ENCODER_HEAD'],backbone:['FINE_TUNING','FINE_TUNING']};
function mode(update,labels){return routes[update][labels>0?1:0];}
function adaptation(el){
 el.classList.add('fm-explorer');
 el.innerHTML='<p class="fm-eyebrow">What can change the prediction?</p><div class="fm-controls"><label>Target adaptation <select data-update><option value="none">No weight updates</option><option value="head">20 head updates</option><option value="backbone">20 backbone updates</option></select></label><label>Target labels <select data-labels><option value="0">0 examples</option><option value="8" selected>8 examples</option></select></label><button type="button" data-reset>Reset</button></div><div class="fm-flow"><div><small>Reusable start</small><strong>A + B → checkpoint θ₀</strong><span>Same starting weights in every state</span></div><div data-middle></div><div><small>Prediction input</small><strong>Target query from C</strong><span>Test target remains withheld</span></div></div><p class="fm-baseline">Fixed baseline: θ₀ + 8 labeled context examples → IN_CONTEXT</p><output aria-live="polite"></output>';
 const update=el.querySelector('[data-update]'),labels=el.querySelector('[data-labels]');
 function draw(){let u=update.value,n=Number(labels.value),m=mode(u,n);el.querySelector('[data-middle]').innerHTML='<small>Adaptation operation</small><strong>'+(u==='backbone'?'θ₀ → θC':u==='head'?'θ₀ frozen; head h → hC':'θ₀ unchanged')+'</strong><span>'+n+' target labels; '+(u==='none'?'context input may change':u==='head'?'head parameters may change':'backbone parameters may change')+'</span>';el.querySelector('output').textContent=m+' · '+(n?'8 labels is a few-shot budget, not a unique algorithm.':'0 labels does not mean no pretraining. Updates can be unsupervised.')+' No performance is measured.';}
 [update,labels].forEach(x=>x.addEventListener('change',draw));el.querySelector('[data-reset]').addEventListener('click',()=>{update.value='none';labels.value='8';draw();});draw();
}
function boundary(el){
 el.classList.add('fm-explorer');
 el.innerHTML='<p class="fm-eyebrow">Which boundary did C cross?</p><div class="fm-controls"><label>Pretraining inventory <select data-inventory><option value="held">Known: A and B</option><option value="seen">Known: A, B and C</option><option value="unknown">Undocumented</option></select></label><label><input type="checkbox" data-context checked> Supply 8 legal C adaptation labels</label><button type="button" data-reset>Reset</button></div><div class="fm-flow"><div><small>Earlier learning</small><strong data-corpus></strong><span>Corpus membership is about databases</span></div><div><small>Target C</small><strong data-adapt></strong><span>Adaptation examples precede the query cutoff</span></div><div><small>Evaluation</small><strong>C test queries</strong><span>Test targets never enter adaptation</span></div></div><p class="fm-baseline">Fixed baseline: pretrain A/B; adapt on C; test on disjoint C queries.</p><output aria-live="polite"></output><p data-limitation>These are declared inventories. Renamed copies and undisclosed data need external review.</p>';
 const inventory=el.querySelector('[data-inventory]'),context=el.querySelector('[data-context]');
 function draw(){const state=inventory.value;el.querySelector('[data-corpus]').textContent=state==='held'?'A + B':state==='seen'?'A + B + C':'?';el.querySelector('[data-adapt]').textContent=context.checked?'8 labeled context examples':'0 labeled context examples';el.querySelector('output').textContent=(state==='held'?'HELD_OUT · C is absent from declared pretraining.':state==='seen'?'SEEN · New test rows do not remove database exposure.':'UNKNOWN · No inventory cannot establish disjointness.')+' '+(context.checked?'Legal adaptation labels do not change this membership.':'Removing adaptation labels does not change this membership.');}
 [inventory,context].forEach(x=>x.addEventListener('change',draw));el.querySelector('[data-reset]').addEventListener('click',()=>{inventory.value='held';context.checked=true;draw();});draw();
}
g.FoundationScope={adaptation,boundary,mode};
})(window);
