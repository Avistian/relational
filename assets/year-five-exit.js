/* L200: all 27 evidence states; baseline PASS/PENDING/PENDING; reset is exact.
   Local planning only: never persists or changes learner mastery. */
(function(){
  const el=document.getElementById('exit200');
  if(el){
    el.className='exit-board';
    el.innerHTML='<h3>Which evidence is still missing?</h3><p>Baseline: verified reproduction; proposal and defense pending.</p><div class="exit-controls">'+['reproduction','proposal','defense'].map(k=>'<label>'+k+'<select aria-label="'+k+'">'+['PASS','FAIL','PENDING'].map(v=>'<option>'+v+'</option>').join('')+'</select></label>').join('')+'</div><output aria-live="polite"></output><button type="button">Reset to baseline</button><p>Teacher review is required. Selecting PASS does not prove the requirement.</p>';
    const selects=[...el.querySelectorAll('select')],out=el.querySelector('output');
    function update(){const missing=selects.map((s,i)=>s.value==='PASS'?null:['reproduction','proposal','defense'][i]).filter(Boolean);out.textContent=missing.length?'INCOMPLETE — missing: '+missing.join(', '):'READY_FOR_TEACHER_REVIEW — automated checks do not establish mastery';}
    function reset(){selects.forEach((s,i)=>s.value=i===0?'PASS':'PENDING');update();}
    selects.forEach(s=>s.addEventListener('change',update));el.querySelector('button').addEventListener('click',reset);reset();
  }
  if(window.RetrievalBank)RetrievalBank.mount(document.getElementById('warmup'),{upTo:200,count:3});
  if(window.Predict)Predict.mount(document.getElementById('predict200'),{prompt:'Ten paired support draws share one test set. What do they establish?',options:[{label:'Sensitivity to support selection',value:'support'},{label:'Generalization across unseen databases',value:'databases'}],correct:'support',reveal:'These draws measure sensitivity to which 512 training rows are supplied. All draws reuse the same 702 test queries.'});
  if(window.Teachback)Teachback.mount(document.getElementById('teachback200'),{prompt:'Why can all 30 fresh evaluations pass while the Year 5 exit remains incomplete?',points:['The numerical reproduction is one gate.','A learner proposal and defense are separate evidence.','Fresh inference is not fresh pretraining or historical identity.','Future data availability and complete cost must be justified.'],model:'The run can establish a complete selected checkpoint result, but cannot write or defend my hypothesis for me. I must explain the scope, define a controlled test and justify its data and budget. Unresolved historical availability and the blocked RDBLearn experiment remain unresolved.'});
})();
