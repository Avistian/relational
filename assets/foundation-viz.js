/* Numerical exhibits for L058–070. No network requests.
 * Each control changes one declared quantity and keeps a fixed baseline visible.
 * Expected states: posterior s=3 -> 4/6; mask query keys always blocked;
 * SCM w=2 -> [1,2.2,-2.3]; eligibility delay=0 admits all events <=4;
 * missing classes: unsupported-label penalty -ln(1e-12).
 */
(function(global){
  'use strict';
  function mount(root,config){
    const mode=config.mode;root.classList.add('foundation-exhibit');root.innerHTML='';
    const title=document.createElement('strong');title.textContent='Predict, change one assumption, then trace the result';root.appendChild(title);
    const label=document.createElement('label'),control=document.createElement('input');control.type='range';
    const output=document.createElement('output');output.setAttribute('aria-live','polite');
    const visual=document.createElement('div'),baseline=document.createElement('p');baseline.className='baseline';
    const settings={posterior:[0,4,1,3,'Successes among four observations'],mask:[1,4,1,3,'Labeled context rows among five rows'],scm:[-3,3,.5,2,'Edge weight 0 → 1'],
      folds:[0,2,1,0,'Held-out query fold'],inducing:[1,4,1,2,'Number of inducing summaries'],neighbors:[1,3,1,2,'Eligible neighbors k'],
      time:[0,6,1,0,'Added label delay'],classes:[0,4,1,1,'Unsupported labels among five required rows'],
      cost:[60,600,60,180,'Labeled context rows'],ranks:[1,3,1,3,'Number of task blocks included'],selection:[1,5,1,1,'Search-budget setting']};
    const s=settings[mode];if(!s)throw Error('Unknown exhibit: '+mode);
    [control.min,control.max,control.step,control.value]=s.slice(0,4);control.id='control-'+config.lesson;
    label.htmlFor=control.id;label.textContent=s[4];root.append(label,control,output,visual,baseline);
    const reset=document.createElement('button');reset.type='button';reset.textContent='Reset';root.appendChild(reset);
    function table(headers,rows){
      visual.innerHTML='';const t=document.createElement('table'),head=document.createElement('tr');
      headers.forEach(x=>{const th=document.createElement('th');th.textContent=x;head.appendChild(th)});t.appendChild(head);
      rows.forEach(row=>{const tr=document.createElement('tr');row.forEach(x=>{const td=document.createElement('td');td.textContent=x;tr.appendChild(td)});t.appendChild(tr)});visual.appendChild(t);
    }
    function draw(){
      const x=Number(control.value);visual.innerHTML='';
      if(mode==='posterior'){
        const p=(x+1)/6;output.textContent=`s=${x}, failures=${4-x}\nPosterior: (${x}+1)/(4+2) = ${p.toFixed(3)}\nEmpirical rate: ${x}/4 = ${(x/4).toFixed(3)}`;
        baseline.textContent='Fixed: Beta(1,1) prior and four observations. Prior-only prediction remains 0.500.';
        table(['Quantity','Success mass','Failure mass'],[['Prior',1,1],['Observed',x,4-x],['Posterior',x+1,5-x]]);
      }else if(mode==='mask'){
        const values=[0,1,0,1],prediction=values.slice(0,x).reduce((a,b)=>a+b,0)/x;
        output.textContent=`C=${x}, queries=${5-x}. With equal context-key logits, the final query reads values [${values.slice(0,x).join(', ')}] and returns ${prediction.toFixed(3)}.`;
        const grid=document.createElement('div');grid.className='foundation-grid';
        for(let i=0;i<5;i++)for(let j=0;j<5;j++){const c=document.createElement('span');c.className=j<x?'allowed':'blocked';c.textContent=(i===4?'→ ':'')+(j<x?'READ':'BLOCK');grid.appendChild(c)}visual.appendChild(grid);
        baseline.textContent='Fixed five feature rows and context values [0,1,0,1]. Reveal only a longer labeled prefix. Baseline C=3 gives 1/3. Query keys stay blocked; residual features remain available. This scalar attention trace is not a trained PFN.';
      }else if(mode==='scm'){
        const v=x+.2,out=-v-.1;output.textContent=`V0 = 1\nV1 = ${x} × 1 + .2 = ${v.toFixed(2)}\nV2 = −1 × ${v.toFixed(2)} − .1 = ${out.toFixed(2)}`;
        baseline.textContent='Baseline w=2 gives [1, 2.2, −2.3]. Fixed noises [1,.2,−.1], downstream weight −1 and linear activation.';
        table(['Node','Baseline','Changed'],[['0',1,1],['1',2.2,v.toFixed(2)],['2',-2.3,out.toFixed(2)]]);
      }else if(mode==='folds'){
        const rows=Array.from({length:6},(_,i)=>[i,i%3,i%3===x?'query / target hidden':'context / target visible']);table(['Original row','Fold','Information role'],rows);
        output.textContent=`Fold ${x} is extracted as unlabeled queries. Scatter its outputs to row IDs ${rows.filter(r=>r[1]===x).map(r=>r[0]).join(', ')}.`;
        baseline.textContent='Fixed row IDs and pretrained encoder. Change only which fold is withheld; every row is a query exactly once over all three folds.';
      }else if(mode==='inducing'){
        const context=[-1,0,1],ind=Array.from({length:x},(_,i)=>x===1?0:-1+2*i/(x-1));
        const soft=a=>{const m=Math.max(...a),e=a.map(v=>Math.exp(v-m)),z=e.reduce((a,b)=>a+b,0);return e.map(v=>v/z)};
        const memories=ind.map(q=>{const w=soft(context.map(k=>q*k));return w.reduce((a,v,i)=>a+v*context[i],0)});
        const q=.5,w=soft(memories.map(k=>q*k)),out=w.reduce((a,v,i)=>a+v*memories[i],0);
        table(['Inducing query','Context summary'],ind.map((v,i)=>[v.toFixed(2),memories[i].toFixed(3)]));
        output.textContent=`Stage 1: I reads fixed context [−1,0,1].\nStage 2: query .5 reads M with weights [${w.map(v=>v.toFixed(3)).join(', ')}].\nResult = ${out.toFixed(3)}`;
        baseline.textContent='Fixed scalar dot-product attention, no learned projections. Baseline m=1 at I=0 gives summary 0 and output 0. This is a numerical skeleton, not a checkpoint.';
      }else if(mode==='neighbors'){
        const candidates=[{id:0,value:0,label:0},{id:2,value:2,label:1},{id:3,value:4,label:1}];
        table(['Row ID','Squared distance','Selected'],candidates.map((r,i)=>[r.id,(r.value-1)**2,i<x?'yes':'no']));
        output.textContent=`Context IDs [${candidates.slice(0,x).map(r=>r.id).join(', ')}]; label mean ${(candidates.slice(0,x).reduce((a,r)=>a+r.label,0)/x).toFixed(3)}.`;
        baseline.textContent='Query x=1; row ID 1 excluded by identity. Fixed geometry and stable ties. Label mean illustrates context composition; it is not a PFN prediction.';
      }else if(mode==='time'){
        const events=[1,2,3],labels=[2,3,4],rows=events.map((v,i)=>[i,v,labels[i]+x,v<=4&&labels[i]+x<=4?'eligible':'blocked']);table(['Row','Event','Label arrival','At cutoff 4'],rows);
        output.textContent=`Delay +${x}: ${rows.filter(r=>r[3]==='eligible').length} eligible labeled context rows.`;
        baseline.textContent='Fixed events, base label arrivals [2,3,4], inclusive cutoff 4. Baseline delay 0 admits all three. Eligibility precedes recent-window selection.';
      }else if(mode==='classes'){
        const loss=(-(5-x)*Math.log(.8)-x*Math.log(1e-12))/5;output.textContent=`${x}/5 labels unsupported.\nAll-row mean log loss = ${loss.toFixed(3)} nats.\nKnown-row loss remains ${(-Math.log(.8)).toFixed(3)}.`;
        baseline.textContent='Fixed known correct-class probability .8 and epsilon 1e−12. Baseline has no unsupported classes and loss .223. Epsilon is a finite diagnostic convention.';
        table(['Population','Rows','Per-row penalty'],[['known',5-x,(-Math.log(.8)).toFixed(3)],['unsupported',x,(-Math.log(1e-12)).toFixed(3)]]);
      }else if(mode==='cost'){
        const q=60,f=10,m=16,n=x+q,axial=n*f*f+f*n*x,col=f*m*(x+n),row=n*f*f,icl=n*x;
        table(['Simplified stage','Score elements/head'],[['Axial feature + sample',axial],['TabICL column',col],['TabICL row',row],['TabICL dataset ICL',icl]]);
        output.textContent=`C=${x}, Q=60, F=10, m=16\nAxial: N F² + F N C = ${axial.toLocaleString()}\nTabICL components total: ${(col+row+icl).toLocaleString()}`;
        baseline.textContent='Fixed Q/F/m; change context only. At C=180, axial has 456,000 scores. These omit target/CLS/grouping constants and are not peak memory or runtime.';
      }else if(mode==='ranks'){
        const rows=[[1,2,3],[3,1,2],[2,3,1]],means=[0,1,2].map(j=>rows.slice(0,x).reduce((a,r)=>a+r[j],0)/x);
        table(['Dataset','A rank','B rank','C rank'],rows.slice(0,x).map((r,i)=>[i+1,...r]));output.textContent=`Mean ranks with ${x} task blocks: A=${means[0].toFixed(2)}, B=${means[1].toFixed(2)}, C=${means[2].toFixed(2)}.`;
        baseline.textContent='Fixed illustrative per-task outcomes. Three-block baseline ties at rank 2. Omitting tasks changes the question without changing a prediction. Actual measured ranks appear below.';
      }else if(mode==='selection'){
        const budgets=[1,4,16,64,256],r=(config.selection||[])[x-1];
        if(!r)throw Error('Missing measured selection state');output.textContent=`Candidates=${budgets[x-1]}\nMean selected validation error=${r.validation_error.toFixed(4)}\nIndependent test error=${r.test_error.toFixed(4)}\nOptimism=${r.optimism.toFixed(4)}`;
        baseline.textContent='Author-measured pure-noise simulation: 200 trials, 80 validation and 2,000 test rows. Fixed generator; increase candidate count. No signal was introduced.';
        table(['Budget','Validation','Test'],config.selection.map(a=>[a.candidates,a.validation_error.toFixed(3),a.test_error.toFixed(3)]));
      }
      root.dataset.state=String(x);
    }
    control.addEventListener('input',draw);reset.addEventListener('click',()=>{control.value=s[3];draw()});draw();
    return {set(value){control.value=Math.max(+control.min,Math.min(+control.max,value));draw()},reset(){control.value=s[3];draw()},read(){return output.textContent}};
  }
  global.FoundationViz={mount};
})(window);
