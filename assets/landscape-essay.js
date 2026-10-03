/* Evidence admission and writing structure are explicit teaching policies. */
(function(g){
'use strict';
const policy={published_table:'published_comparison',saved_predictions:'scoped_pipeline_comparison',source_diagnostic:'implementation_observation'};
function admit(lane,claim,authenticated,complete){return authenticated&&complete&&policy[lane]===claim?'ADMISSIBLE_SCOPED':'NOT_ESTABLISHED';}
function mountAudit(el){
 el.classList.add('landscape-panel');el.innerHTML='<p><strong>Practice: choose the claim this kind of evidence can support.</strong> The source-diagnostic option is hypothetical here; this lesson does not rerun L196.</p><div class="controls"><label>Evidence <select id="essay-lane"><option value="published_table">Published table</option><option value="saved_predictions">Saved predictions</option><option value="source_diagnostic">Source diagnostic</option></select></label><label>Claim <select id="essay-claim"><option value="published_comparison">Published comparison</option><option value="scoped_pipeline_comparison">Scoped pipeline comparison</option><option value="implementation_observation">Implementation observation</option><option value="fresh_model_reproduction">Fresh model reproduction</option><option value="architecture_cause">Architectural cause</option><option value="economic_undervaluation">Economic undervaluation</option></select></label><label>Authenticated <select id="essay-auth"><option value="yes">Yes</option><option value="no">No</option></select></label><label>Complete <select id="essay-complete"><option value="yes">Yes</option><option value="no">No</option></select></label><button type="button">Reset audit</button></div><output aria-live="polite"></output>';
 function update(){let state=admit(el.querySelector('#essay-lane').value,el.querySelector('#essay-claim').value,el.querySelector('#essay-auth').value==='yes',el.querySelector('#essay-complete').value==='yes');el.dataset.state=state;el.querySelector('output').textContent=state==='ADMISSIBLE_SCOPED'?'Admissible within scope. Cite the protocol and limitations; admission is not proof of the claim.':'Not established by this evidence. Narrow the claim or obtain the missing kind of measurement.';}
 el.querySelectorAll('select').forEach(s=>s.addEventListener('change',update));el.querySelector('button').addEventListener('click',()=>{el.querySelectorAll('select').forEach(s=>s.selectedIndex=0);update();});update();
}
const fields=['claim','evidence','warrant','limitation','revision'];
function readiness(values){const missing=fields.filter(k=>!values[k].trim());return {state:missing.length?'DRAFT':'READY_FOR_REVIEW',missing,mastery:'PENDING_WRITTEN_DEFENSE'};}
function mountWriting(el){
 el.classList.add('landscape-panel');el.innerHTML='<h3>Your five-part argument</h3><p>Text stays in this page until you download it. This checks presence, not truth or writing quality.</p>'+fields.map(k=>'<label>'+k[0].toUpperCase()+k.slice(1)+'<textarea name="'+k+'" aria-label="Essay '+k+'"></textarea></label>').join('')+'<button type="button">Download draft</button><output aria-live="polite"></output>';
 function values(){return Object.fromEntries(fields.map(k=>[k,el.querySelector('[name="'+k+'"]').value]));}
 function update(){const r=readiness(values());el.dataset.state=r.state;el.querySelector('output').textContent=(r.missing.length?'Missing: '+r.missing.join(', ')+'.':'All five fields present; ready for human review.')+' Learner: PENDING_WRITTEN_DEFENSE.';}
 el.querySelectorAll('textarea').forEach(x=>x.addEventListener('input',update));el.querySelector('button').addEventListener('click',()=>{const v=values(),url=URL.createObjectURL(new Blob([JSON.stringify({sections:v,review:readiness(v)},null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='l197-argument.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});update();
}
g.LandscapeEssay={admit,readiness,mountAudit,mountWriting};
const id=x=>document.getElementById(x);
if(id('landscape-map'))g.ArchFamilyViz.mountLandscape(id('landscape-map'));
if(id('claim-audit'))mountAudit(id('claim-audit'));
if(id('essay-structure'))mountWriting(id('essay-structure'));
if(id('warmup'))g.RetrievalBank.mount(id('warmup'),{upTo:197,count:3});
if(id('gap-predict'))g.Predict.mount(id('gap-predict'),{prompt:'Include supervised open methods: what happens to the Table 3 gap?',options:[{label:'The reported gap becomes smaller',value:'smaller'},{label:'The reported gap becomes larger',value:'larger'}],correct:'smaller',reveal:'4.0533 becomes 1.5417 AUROC percentage points. The eligible pool changed; Kumo scores did not.'});
if(id('teachback'))g.Teachback.mount(id('teachback'),{prompt:'Why can an API-only model lead a published comparison while local full reproduction remains unfinished?',points:['Published scores are source-reported evidence.','API, code, weights and training data are different access axes.','Replaying tables or predictions does not train a new model.','A source gate leaves fresh scores missing.'],model:'The paper reports a comparison under its protocol. Our local audit can reproduce table arithmetic without access to its training process. A different open-code model can remain blocked by preprocessing. These facts describe different evidence and access questions; neither makes missing fresh scores into wins or losses.'});
})(window);
