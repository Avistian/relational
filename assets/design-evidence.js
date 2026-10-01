/* Five claim gates × nine evidence flags. Hypothetical controls never alter saved evidence.
   Baseline: artifacts and metrics true; all remaining flags false. Reset restores baseline.
   READY_FOR_REVIEW checks prerequisites only; it never certifies a scientific result. */
window.DesignEvidence={
  requirements:{saved_replay:['artifacts','metrics'],full_pipeline:['artifacts','metrics','dfs','temporal'],fresh_pretraining:['artifacts','metrics','fresh_training','exposure'],general_advantage:['artifacts','metrics','matched_pipeline','heldout_databases','temporal','exposure'],exact_repeatability:['artifacts','metrics','repeatability']},
  gate(claim,facts){const missing=this.requirements[claim].filter(key=>!facts[key]);return {status:missing.length?'NOT_ESTABLISHED':'READY_FOR_REVIEW',missing};},
  mount(host,baseline){
    const labels={artifacts:'Authenticated artifacts',metrics:'Independent metrics',dfs:'Complete DFS regeneration',temporal:'Full temporal availability audit',exposure:'Pretraining exposure inventory',fresh_training:'Fresh pretraining executed',matched_pipeline:'Matched complete pipelines',heldout_databases:'Adequate prespecified database evaluation',repeatability:'Exact repeatability'};
    const claims={saved_replay:'Saved-evidence replay',full_pipeline:'Complete temporal pipeline',fresh_pretraining:'Fresh pretraining reproduction',general_advantage:'General paradigm advantage',exact_repeatability:'Exact repeatability'};
    host.className='design-explorer';
    host.innerHTML='<p class="design-eyebrow">What evidence would this claim require?</p><p>Hypothetical declarations only. The authenticated baseline stays visible.</p><div class="design-controls"><label>Claim <select data-claim>'+Object.entries(claims).map(([v,l])=>'<option value="'+v+'">'+l+'</option>').join('')+'</select></label><fieldset><legend>Evidence you declare available</legend>'+Object.entries(labels).map(([k,l])=>'<label><input type="checkbox" data-key="'+k+'"> '+l+'</label>').join('')+'</fieldset><button type="button" data-reset>Reset to authenticated evidence</button></div><p data-baseline></p><output aria-live="polite"></output><p data-missing></p><small>READY_FOR_REVIEW means necessary prerequisites are present. A reviewer still has to assess the result, uncertainty and relevance.</small>';
    const update=()=>{
      const claim=host.querySelector('[data-claim]').value;
      const facts=Object.fromEntries([...host.querySelectorAll('[data-key]')].map(x=>[x.dataset.key,x.checked]));
      const result=this.gate(claim,facts),base=this.gate(claim,baseline);
      host.dataset.verdict=result.status;host.dataset.missing=result.missing.join(',');
      host.querySelector('[data-baseline]').textContent='Authenticated baseline for this claim: '+base.status+(base.missing.length?' — missing '+base.missing.map(k=>labels[k]).join('; '):'');
      host.querySelector('output').textContent='Your hypothetical declaration: '+result.status;
      host.querySelector('[data-missing]').textContent=result.missing.length?'Still required: '+result.missing.map(k=>labels[k]).join('; '):'All listed prerequisites declared. No scientific claim has been automatically proved.';
    };
    const reset=()=>{host.querySelector('[data-claim]').value='general_advantage';host.querySelectorAll('[data-key]').forEach(x=>x.checked=baseline[x.dataset.key]);update();};
    host.addEventListener('change',update);host.querySelector('[data-reset]').addEventListener('click',reset);reset();
  }
};
