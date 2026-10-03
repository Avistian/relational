/* B07 defaults: meaningful center [1,.5]; anonymous [.36,.94]; numeric-only [.72,1.28].
   Toy vectors only. Values and row identity never change; numeric-only removes country.
   All target candidates are valid; supplying only the true query answer leaks its label. */
(function(global){'use strict';
function intervention(el){
 const pick=el.querySelector('select'),out=el.querySelector('output'),view=el.querySelector('[data-changed]');
 function draw(){const arm=pick.value,natural=arm==='meaningful',numeric=arm==='numeric_only';const e1=natural?[1,0]:[.6,.8],e2=natural?[0,1]:[.8,.6],leaf=e1.map(v=>2*v),product=leaf.map((v,i)=>v*e1[i]),text=[0,e2[1]],n=numeric?1:2;const center=product.map((v,i)=>(v+(numeric?0:text[i]))/n);const fmt=v=>Number(v.toFixed(2));
 view.textContent=natural?'volume: 2; country: France':numeric?'6: 2; country column removed':'6: 2; 7: France';
 out.dataset.center=center.map(fmt).join(',');out.dataset.features=String(n);
 out.textContent='Numeric leaf = 2 × ['+e1+'] = ['+leaf.map(fmt)+']\nEdge × leaf = ['+product.map(fmt)+']\n'+(numeric?'No text leaf remains.':'Text edge × leaf = ['+text.map(fmt)+']')+'\nMean center = ['+center.map(fmt)+']\nBaseline center = [1, 0.5]. '+(natural?'Original natural names.':numeric?'Text facts and graph size changed.':'Only header vectors changed; all cell values remain.');
 }
 pick.addEventListener('change',draw);el.querySelector('button').onclick=()=>{pick.value='meaningful';draw();};draw();
}
function adaptation(el){const model=el.querySelector('[data-model]'),target=el.querySelector('[data-target]'),out=el.querySelector('output');function draw(){const leak=target.value==='answer';const policies={carte:'CARTE paper: task fitting updates its predictor. B07 freezes the graph encoder and fits only ridge.',contexttab:'ConTextTab: change labeled context; keep predictor weights fixed during ordinary ICL.',tabstar:'TabSTAR: downstream LoRA adapters learn through gradients, including eligible encoder blocks.'};out.dataset.valid=String(!leak);out.textContent=policies[model.value]+'\n'+(leak?'INVALID: revealing only the actual class identifies the unknown query answer.':'VALID target-candidate rule: include every candidate for every row, independently of its true label.')+'\nThis is an information contract, not a measured three-model ranking.';}
 model.onchange=draw;target.onchange=draw;el.querySelector('button').onclick=()=>{model.value='contexttab';target.value='all';draw();};draw();}
function mount(){document.querySelectorAll('[data-b07=intervention]').forEach(intervention);document.querySelectorAll('[data-b07=adaptation]').forEach(adaptation);
 const warm=document.getElementById('b07-warmup');if(warm&&global.RetrievalBank)global.RetrievalBank.mount(warm,{upTo:200.07,count:3});
 const pred=document.getElementById('b07-predict');if(pred&&global.Predict)global.Predict.mount(pred,{prompt:'Will meaningful headers improve mean R² on every wine table?',options:[{label:'Names help every table',value:'all'},{label:'Effects vary across tables',value:'mixed'}],correct:'mixed',reveal:'Meaningful headers help wine_pl, but their mean R² is lower on wine_dot_com_prices and wine_vivino_price. Split effects also vary.'});
 const tb=document.getElementById('b07-teachback');if(tb&&global.Teachback)global.Teachback.mount(tb,{prompt:'Why is a name ablation different from text removal, and why are all-class target tokens not query-label leakage? Connect your answer to context adaptation versus LoRA updates.',points:['Name ablation preserves values and paired row identities.','Text removal changes available facts and graph size.','Every class candidate appears without revealing the actual answer.','Context changes and gradient updates are different adaptation regimes.'],model:'Renaming removes natural header meanings while preserving values. Removing text drops predictors as well. Listing every allowed class identifies the task without identifying a query’s answer. ConTextTab adapts through labeled context, whereas TabSTAR learns task-specific LoRA updates. Our mixed CARTE probe results test this bounded intervention, not general architecture superiority.'});
}
global.SemanticTransfer={intervention,adaptation};if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',mount);else mount();
})(window);
