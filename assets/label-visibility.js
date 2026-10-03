/* B15 states: visibility 2 scopes x 3 cutoffs; rule 2 worlds x 2 access;
   column 2 relevant columns x 2 support regimes. Baselines remain visible. */
(function(){
 const v=document.querySelector('[data-label-visibility]');
 if(v){const scope=v.querySelector('[name=scope]'),cut=v.querySelector('[name=cutoff]'),out=v.querySelector('output');
 function update(){const t=Number(cut.value),remote=scope.value==='expanded'&&t>=4,future=scope.value==='expanded'&&t>=11;
 out.dataset.remote=String(remote);out.dataset.future=String(future);out.textContent='At day '+t+': remote label '+(remote?'VISIBLE':'HIDDEN')+'; late label '+(future?'VISIBLE':'HIDDEN')+'. Query label HIDDEN in every legal state. '+(remote?'The remote example can identify the rule.':'The distinguishing support is unavailable.');}
 scope.addEventListener('change',update);cut.addEventListener('change',update);v.querySelector('button').addEventListener('click',()=>{scope.value='expanded';cut.value='10';update()});update();}
 const r=document.querySelector('[data-label-rules]');
 if(r){const world=r.querySelector('[name=world]'),access=r.querySelector('[name=access]'),out=r.querySelector('output');
 function update(){const theta=Number(world.value),p=access.value==='expanded'?1-theta:.5;out.dataset.probability=p;out.textContent='Neighbor label b = 1. True query label = '+(1-theta)+'. '+(access.value==='expanded'?'External pair (0, '+theta+') identifies '+(theta?'Flip':'Copy')+'. ':'Both rules remain compatible. ')+'P(query label = 1) = '+p+'.';r.querySelector('[data-support]').textContent=access.value==='expanded'?'Visible pair: input 0, label '+theta:'External support hidden';}
 world.addEventListener('change',update);access.addEventListener('change',update);r.querySelector('button').addEventListener('click',()=>{world.value='0';access.value='local';update()});update();}
 const c=document.querySelector('[data-label-columns]');
 if(c){const s=c.querySelector('[name=column]'),access=c.querySelector('[name=examples]'),out=c.querySelector('output');
 function update(){const col=Number(s.value),expanded=access.value==='expanded',p=expanded?col:.5;out.dataset.probability=p;out.dataset.bits=expanded?'1':'0';out.textContent=(expanded?'Add (0,1) → '+col+': only column '+(col?'B':'A')+' remains.':'Rows (0,0) → 0 and (1,1) → 1 fit both columns.')+' Query (0,1): P(label = 1) = '+p+'. Information about S = '+(expanded?1:0)+' bits.';}
 s.addEventListener('change',update);access.addEventListener('change',update);c.querySelector('button').addEventListener('click',()=>{s.value='0';access.value='local';update()});update();}
 if(window.RetrievalBank)RetrievalBank.mount(document.getElementById('b15-warmup'),{upTo:200.15,count:3});
 if(window.Predict)Predict.mount(document.getElementById('b15-predict'),{prompt:'With b=1 and equally likely Copy/Flip worlds, which local prediction is justified?',options:[{label:'Assign probability one half',value:'half'},{label:'Assign probability exactly one',value:'one'}],correct:'half',reveal:'The same observed input occurs with opposite query labels. Averaging the two worlds gives probability one half.'});
 if(window.Teachback)Teachback.mount(document.getElementById('b15-teachback'),{prompt:'Why does the paper not prove that training an encoder is always useless? Explain using the two worlds and a legal support example.',points:['Existence result, not every distribution','Identical local input hides the task rule','External support changes the information boundary','Query truth and unavailable labels stay hidden'],model:'The paper constructs distributions where local labels cannot resolve the task. This prevents a universal guarantee, not gains on particular distributions. A legal external pair can identify Copy versus Flip because it supplies missing task information. Reading query truth would instead violate the prediction problem.'});
})();
