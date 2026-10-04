/* B18: measured seed0, degree4096, concentrated, budget32 default. All96 states
   use saved complete-run metrics. This component never generates substitute results. */
(function(global){'use strict';
function mount(el,conditions){
 const selects=[...el.querySelectorAll('select')];const out=el.querySelector('output');
 function draw(){const state=Object.fromEntries(selects.map(s=>[s.name,s.value]));const c=conditions.find(c=>c.seed===+state.seed&&c.degree===+state.degree&&c.shape===state.shape&&c.budget===+state.budget);if(!c){out.textContent='Measured evidence unavailable.';return;}
 out.dataset.key=[c.seed,c.degree,c.shape,c.budget].join('/');out.dataset.correctedRmse=c.metrics.corrected.rmse;
 const f=x=>Math.abs(x)<1e-8?'0.00':x.toFixed(2);
 out.textContent=`Visible events: ${c.k} / ${c.degree}. True historical total: $1,200.\n`+['exact','sampled','corrected','aggregate'].map(a=>`${a}: bias $${f(c.metrics[a].bias)} · RMSE $${f(c.metrics[a].rmse)}`).join('\n');
 el.querySelector('[data-interpret]').textContent=c.k===c.degree?'All eligible events fit: sampling uncertainty vanishes.':c.shape==='uniform'?'Every event has the same amount. Exact N plus one observed event identifies this special total.':'A rare large event can be missed. N/k correction removes design bias, but finite-sample error can remain large. Complete aggregation processes all events.';
 }
 selects.forEach(s=>s.addEventListener('change',draw));el.querySelector('button').onclick=()=>{const defaults={seed:'0',degree:'4096',shape:'concentrated',budget:'32'};selects.forEach(s=>{s.value=defaults[s.name];});draw();};draw();return{draw};
}
global.ContextBudget={mount};
const board=document.querySelector('[data-context-budget]');if(board&&global.B18_EVIDENCE)mount(board,global.B18_EVIDENCE.conditions);
if(global.TemporalVisibility&&document.getElementById('b18-time'))global.TemporalVisibility.mount(document.getElementById('b18-time'));
if(global.RetrievalBank)global.RetrievalBank.mount(document.getElementById('b18-warmup'),{upTo:200.18,count:3});
if(global.Predict)global.Predict.mount(document.getElementById('b18-predict'),{prompt:'Multiply a uniform sample sum by N/k. What is guaranteed?',options:[{label:'Correct across possible samples',value:'average'},{label:'Correct for every sample',value:'every'}],correct:'average',reveal:'The expectation equals the fixed historical total. One sample can underestimate or overshoot, especially when one event carries most of the money.'});
if(global.Teachback)global.Teachback.mount(document.getElementById('b18-teachback'),{prompt:'Explain in 150–200 words: sampling bias versus variance, the extra information in N and aggregation, the time boundary, and the untested forecasting claim.',points:['N/k is justified by uniform inclusion probabilities.','Unbiased estimates can have large per-draw error.','Full counts and aggregates require full-history information.','Time filtering precedes sampling and aggregation.','Historical reconstruction does not reproduce Animus forecasting.'],model:'A sampled sum loses expected mass because each event is included with probability k/N. Scaling by N/k restores the expectation, not each individual total. When a rare event carries most of the money, missing or selecting it produces large errors. Exact N and complete aggregates supply information beyond the sampled rows and have a computation cost. Filter eligible history before creating any of them, using arrival as well as event time when necessary. Exact reconstruction of a past sum does not establish prediction of a future month; that requires a separate temporal forecast evaluation. The Animus source inconsistencies remain unresolved.'});
})(window);
