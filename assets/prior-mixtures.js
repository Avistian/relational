/* States: p=0,.25,.5,.75,1; fixed u=[.1,.4,.6,.9]. One generator per task.
 * Selection exercise: none/development/final. No simulated performance claims. */
(function(){'use strict';
 const warm=document.getElementById('b06-warmup');
 if(warm&&window.RetrievalBank) window.RetrievalBank.mount(warm,{upTo:200.06,count:3});
 const back=document.getElementById('b06-teachback');
 if(back&&window.Teachback) window.Teachback.mount(back,{prompt:'Why do the nine controlled fits fail to establish a benefit from mixed priors?',points:['All arms are near chance and lose to uniform cross-entropy.','Fixed architecture and budget isolate a scoped intervention, not successful learning.','Course generators and scale differ from original Mitra.','Original Table12 requires authenticated checkpoints, splits and evaluator.'],model:'The controlled comparison is complete, but none of its means beats uniform prediction. Small loss gaps do not establish useful prior diversity. The course proxies and training budget differ from Mitra, whose original six-arm reproduction remains source-gated.'});
 document.querySelectorAll('[data-b06="mixture"]').forEach(function(board){
  const select=board.querySelector('select'),out=board.querySelector('output');
  function draw(){const p=Number(select.value),u=[.1,.4,.6,.9],families=u.map(x=>x<p?'scm':'tree');
   board.querySelectorAll('.b06-task').forEach((el,i)=>{el.dataset.family=families[i];el.textContent='u='+u[i].toFixed(1)+' → '+families[i].toUpperCase();});
   const n=families.filter(f=>f==='scm').length;out.dataset.count=String(n);out.textContent='Changed: '+n+' SCM + '+(4-n)+' tree tasks. Baseline p=0.5: 2 SCM + 2 tree. Same four draws, learner and update count. An outer mixture chooses whole tasks; it does not blend their target values.';
  }
  select.addEventListener('change',draw);board.querySelector('button').addEventListener('click',()=>{select.value='.5';draw();});draw();
 });
 document.querySelectorAll('[data-b06="selection"]').forEach(function(board){
  const select=board.querySelector('select'),out=board.querySelector('output');
  function draw(){let v=select.value;out.dataset.status=v;
   out.textContent={none:'Prespecified p=0.5: the final benchmark has not selected the mixture. This does not prove generality.',development:'Choose p=0.7 using development loss 0.38. Lock it before the final test. The development dataset is now selection evidence.',final:'Choosing p=0.5 from the final benchmark loss 0.39 makes that benchmark selection evidence. Its selected score is no longer an untouched test.'}[v];
  }
  select.addEventListener('change',draw);board.querySelector('button').addEventListener('click',()=>{select.value='none';draw();});draw();
 });
 document.querySelectorAll('[data-b06="quiz"]').forEach(function(board){
  const out=board.querySelector('output');board.querySelectorAll('input').forEach(el=>el.addEventListener('change',()=>{const ok=el.value==='prior';out.dataset.correct=String(ok);out.textContent=ok?'Correct: change prior; match architecture, initialization, task count and evaluation.':'Try again: changing the learner or evaluation set confounds a prior comparison.';}));
  board.querySelector('button').addEventListener('click',()=>{board.querySelectorAll('input').forEach(el=>el.checked=false);out.textContent='Choose before revealing feedback.';delete out.dataset.correct;});
 });
})();
