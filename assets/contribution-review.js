/* Reusable review exercise: one claim, one required evidence boundary. */
window.ContributionReview={mount(host){
 if(!host)return;
 host.innerHTML=`<h3>Review a proposed contribution</h3><p>Baseline: intact evidence, all ten fits, historical arrival times unknown, no public URL. Change one assumption and inspect what the claim can support.</p><div class="review-controls"><label>Evidence integrity<select data-integrity><option value="PASS">Hashes pass</option><option value="FAIL">Changed evidence</option></select></label><label>Experiment coverage<select data-coverage><option value="COMPLETE">Ten complete fits</option><option value="INCOMPLETE">One seed missing</option></select></label><label>Proposed claim<select data-claim><option value="selected_reproduction">Selected experiment reproduced</option><option value="historically_leak_free">Historically leak free</option><option value="public_contribution">Public contribution completed</option><option value="whole_paper">Whole paper reproduced</option><option value="upstream_bug">Current upstream bug proved</option></select></label><label>Public link<select data-url><option value="none">No public URL</option><option value="provided">URL supplied, unchecked</option></select></label></div><output aria-live="polite"></output><p data-explain></p><button type="button">Reset baseline</button>`;
 const get=n=>host.querySelector(`[data-${n}]`).value;
 function update(){const c=get('claim');let verdict,explain;
 if(get('integrity')==='FAIL'){verdict='REJECTED';explain='Changed bytes break the evidence chain. Investigate before making any supported claim.';}
 else if(c==='selected_reproduction'){verdict=get('coverage')==='COMPLETE'?'SUPPORTED':'INCOMPLETE';explain='This claim requires every planned full run and the independent audit. A missing seed cannot be replaced by an average of the rest.';}
 else if(c==='public_contribution'){verdict=get('url')==='none'?'PENDING_PUBLICATION':'NOT_CHECKED';explain='A local package is not public. A supplied URL still needs an access and artifact check.';}
 else if(c==='whole_paper'){verdict='NOT_RUN';explain='Ten fits cover one selected task and a course intervention, not every paper experiment.';}
 else{verdict='NOT_ESTABLISHED';explain=c==='upstream_bug'?'The archived source and a course fit-horizon policy do not diagnose current upstream.':'Passing cutoff audits cannot recover absent historical arrival and version data.';}
 host.dataset.status=verdict;host.querySelector('output').textContent=verdict;host.querySelector('[data-explain]').textContent=explain;
 }
 host.querySelectorAll('select').forEach(x=>x.addEventListener('change',update));host.querySelector('button').addEventListener('click',()=>{host.querySelectorAll('select').forEach(x=>x.selectedIndex=0);update();});update();
}};
