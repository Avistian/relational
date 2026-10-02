(function(global){'use strict';
function verdict(state){
 const reasons=[];
 if(state.horizon!=='30')reasons.push('Unmatched horizon_days');
 if(state.access!=='matched')reasons.push('RDBLearn: extra target-history labels');
 if(state.selection!=='validation')reasons.push('RelGNN: test-based selection');
 if(state.audit!=='pass')reasons.push('Complete preprocessing and temporal audits are missing');
 if(state.health!=='pass')reasons.push('RelGNN: nonfinite training gradients');
 return {status:state.health!=='pass'?'INCOMPLETE_TRAINING_HEALTH_GATE':reasons.length?'INCOMPLETE_COMPARABILITY_GATE':'READY_FOR_PILOT',reasons:reasons};
}
function mount(host){
 host.className='cg';host.innerHTML='<p class="cg-note"><strong>Hypothetical exercise.</strong> Observed L178 remains stopped; these controls do not repair it or launch a job.</p><div class="cg-controls"></div><output aria-live="polite"></output><button type="button" data-reset>Reset assumptions</button>';
 const controls=host.querySelector('.cg-controls');
 const fields=[['horizon','RDBLearn label window',[['30','30 days, matched'],['60','60 days, mismatched']]],['access','Explicit target-label access',[['matched','Shared support only'],['extra','Full target-history table']]],['selection','GNN configuration selection',[['validation','Validation only'],['test','Best test score']]],['audit','Complete feature/time audits',[['unknown','Not completed'],['pass','Assume passed']]],['health','GNN training health',[['fail','Observed gradient failure'],['pass','Assume healthy']]]];
 for(const [key,title,choices] of fields){const label=document.createElement('label');label.textContent=title;const select=document.createElement('select');select.dataset[key]='';for(const [value,text] of choices){const option=document.createElement('option');option.value=value;option.textContent=text;select.append(option);}label.append(select);controls.append(label);select.addEventListener('change',render);}
 function render(){const state={};for(const [key] of fields)state[key]=host.querySelector('[data-'+key+']').value;const result=verdict(state);host.dataset.state=fields.map(([k])=>state[k]).join('/');const out=host.querySelector('output');out.dataset.verdict=result.status;out.replaceChildren();const title=document.createElement('strong');title.textContent=result.status;out.append(title,document.createTextNode(result.reasons.length?result.reasons.join('; '):'Contract assumptions match. A separately budgeted pilot is still required; no winner is established.'));}
 host.querySelector('[data-reset]').addEventListener('click',()=>{for(const [key,,choices] of fields)host.querySelector('[data-'+key+']').value=choices[0][0];render();});render();
}
global.ComparisonGate={mount,verdict};
})(window);
