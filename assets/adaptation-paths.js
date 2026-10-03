/* Reusable two-channel label-path worked example. No model-performance claim. */
(function(){'use strict';
 document.querySelectorAll('[data-adaptation-paths]').forEach(function(board){
  const reach=board.querySelector('[name=reach]'),mode=board.querySelector('[name=mode]'),out=board.querySelector('output');
  function render(){
   const hidden=mode.value==='hidden',batch=hidden?.5:(mode.value==='shuffled'?.25:.75);
   const relation=reach.value==='high'?batch:.5,dual=(relation+batch)/2;
   out.dataset.relation=relation;out.dataset.batch=batch;out.dataset.dual=dual;
   out.textContent='Relational reader = '+relation.toFixed(3)+'. Batch reader = '+batch.toFixed(3)+'. Dual average = '+dual.toFixed(3)+'. '+(hidden?'Both channels use the declared 0.5 fallback.':reach.value==='low'?'No labels in the walk; labels can still travel through the batch path.':'Both example labels are reachable in this worked example.')+' Weights stay fixed.';
  }
  reach.addEventListener('change',render);mode.addEventListener('change',render);
  board.querySelector('button').addEventListener('click',()=>{reach.value='low';mode.value='intact';render();});render();
 });
 if(window.RetrievalBank)RetrievalBank.mount(document.getElementById('b12-warmup'),{upTo:200.12,count:3});
 if(window.Predict)Predict.mount(document.getElementById('b12-predict'),{prompt:'Low reachability: after shuffling support labels, which prediction can change?',options:[{label:'Only the dual reader',value:'dual'},{label:'Only the local reader',value:'local'}],correct:'dual',reveal:'The local label path is empty. The dual reader has a batch path. Possible influence does not establish accuracy.'});
 if(window.Teachback)Teachback.mount(document.getElementById('b12-teachback'),{prompt:'Explain how fixed weights can yield new-task predictions, and why label sensitivity does not prove useful adaptation.',points:['Distinguish parameters from activations','Trace an eligible label through a permitted path','Separate sensitivity from accuracy and training evidence','Name a temporal or support-budget confound'],model:'Parameters can stay fixed while support inputs change activations. A permitted path can carry a changed label to the query. Even an untrained kernel can be sensitive; accuracy and learned adaptation need separate evidence. Check label arrival, completed outcome windows and the number of labels actually read.'});
})();
