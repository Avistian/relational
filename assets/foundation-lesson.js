(function(){
  'use strict';
  const element=document.getElementById('foundation-config');if(!element)return;
  const c=JSON.parse(element.textContent);
  RetrievalBank.mount(document.getElementById('warmup'),{upTo:c.lesson,count:3});
  Predict.mount(document.getElementById('prediction'),{prompt:c.quiz[0],
    options:c.quiz[1].map((label,i)=>({label,value:String(i)})),correct:String(c.quiz[2]),reveal:c.quiz[3]});
  Teachback.mount(document.getElementById('teachback'),{prompt:'Explain the load-bearing mechanism and the strongest claim this experiment can support. Include one failed case.',
    points:['Name the information boundary','Connect one intermediate value to the output','Distinguish local evidence from paper reproduction','Propose a falsifiable next experiment'],model:c.answer+' State the version, data and protocol before generalizing the result.'});
  window.foundationExhibits=[];
  document.querySelectorAll('[data-foundation-viz]').forEach(root=>{window.foundationExhibits.push(FoundationViz.mount(root,{...c,mode:root.dataset.foundationViz}))});
})();
