/* Evidence selector: four fixed packet kinds, baseline preserved; keyboard native select and reset. */
(function () {
'use strict';
const host=document.getElementById('evidence-explorer');
if(host){
 const labels={PUBLISHED_TABLE:'Published table',SAVED_SOURCE_DIAGNOSTIC:'Saved source diagnostic',UNRUN_BENCHMARK:'Unrun benchmark',PROPOSED_INTERVENTION:'Proposed intervention'};
 const outcomes={PUBLISHED_TABLE:'DESCRIPTIVE_REFERENCE_ONLY — compare printed scores; a cause is not identified.',SAVED_SOURCE_DIAGNOSTIC:'SOURCE_INVARIANT_FAILURE_ONLY — known category codes change; benchmark harm is unmeasured.',UNRUN_BENCHMARK:'NO_PERFORMANCE_CONCLUSION — missing is neither a win nor a loss.',PROPOSED_INTERVENTION:'HYPOTHESIS_NOT_TESTED — a study plan creates no new observations.'};
 host.className='repro-widget';host.innerHTML='<h3>Which sentence can you defend?</h3><p class="baseline">Baseline: 0/21 measured task results, fixed across every selection. This control changes the kind of evidence you inspect, not the experiment.</p><div class="controls"><label for="evidence-kind">Evidence kind</label><select id="evidence-kind">'+Object.entries(labels).map(([k,v])=>'<option value="'+k+'">'+v+'</option>').join('')+'</select></div><output aria-live="polite"></output><button type="button" data-reset>Reset evidence</button>';
 const update=()=>{const kind=host.querySelector('select').value;host.dataset.license=outcomes[kind].split(' — ')[0];host.querySelector('output').textContent=outcomes[kind];};
 host.addEventListener('change',update);host.querySelector('[data-reset]').onclick=()=>{host.querySelector('select').value='PUBLISHED_TABLE';update();};update();
}
if(window.RetrievalBank)RetrievalBank.mount(document.getElementById('warmup'),{upTo:194,count:3});
if(window.Predict)Predict.mount(document.getElementById('prediction'),{prompt:'RDBLearn beats AutoGluon+DFS on one published task. What follows?',options:[{label:'The pipeline scores rank differently',value:'description'},{label:'The structure alone caused improvement',value:'cause'}],correct:'description',reveal:'A pipeline difference is descriptive. Both use relational features; the comparison changes more than the presence of relational structure.',shuffle:false});
if(window.Teachback)Teachback.mount(document.getElementById('teachback'),{prompt:'Write a three-sentence report abstract: what was targeted, what was observed, and what remains unknown?',points:['Name all 21 tasks and the stopped inference scope','Separate source-code evidence from benchmark performance','Name a controlled follow-up and its limits'],model:'We targeted the complete RDBLearn v1 task column with a frozen validation search. Recorded preprocessing interventions violate code consistency, so inference remains unrun; the published comparisons are reference arithmetic. Benchmark occurrence and score effects remain unknown and require a separately specified, valid experiment.'});
})();
