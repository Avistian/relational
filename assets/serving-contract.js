/* Reusable complete-evidence explorer. It displays frozen cells, never interpolated timings. */
window.ServingContract={mount(host,evidence){
 const controls=document.createElement('div');controls.className='sc-controls';
 const fields=[['policy','Serving policy',[['precomputed','Precomputed prediction'],['cached','Cached features'],['request','Request-time features']]],['rate','Offered requests / second',[[10,'10'],[50,'50'],[100,'100']]],['condition','Upstream arrivals',[['normal','Normal'],['delayed','Payments delayed'],['interrupted','Payments interrupted']]],['seed','Paired trace seed',[[0,'0'],[1,'1'],[2,'2']]]];
 for(const [key,label,options] of fields){const el=document.createElement('label');el.textContent=label;const select=document.createElement('select');select.dataset[key]='';for(const [value,text] of options){const option=document.createElement('option');option.value=value;option.textContent=text;select.append(option)}el.append(select);controls.append(el)}
 const reset=document.createElement('button');reset.type='button';reset.dataset.reset='';reset.textContent='Reset baseline';controls.append(reset);host.append(controls);
 const out=document.createElement('output');out.className='sc-result';out.setAttribute('aria-live','polite');host.append(out);
 const meters=document.createElement('div');meters.className='sc-meters';host.append(meters);
 function update(){const state={};for(const [key] of fields)state[key]=host.querySelector('[data-'+key+']').value;
 const r=evidence.results.find(r=>r.policy===state.policy&&String(r.rate)===state.rate&&r.condition===state.condition&&String(r.seed)===state.seed);
 host.dataset.state=fields.map(([key])=>state[key]).join('/');out.dataset.p99=r.p99_ms;out.dataset.stale=r.stale_responses;out.dataset.verdict=(r.deadline_misses||r.stale_responses)?'BREACH':'WITHIN';
 const alert=r.post_onset_alert_delay_ms===null?'No incident detection delay defined':`${r.post_onset_alert_delay_ms} ms to first post-onset alert response${r.active_at_onset?' — alert was already active; this is not new detection':''}`;
 out.innerHTML=`<strong>${r.requests.toLocaleString()} simulated responses · ${state.policy}</strong>p50 / p95 / p99: <b>${r.p50_ms} / ${r.p95_ms} / ${r.p99_ms} ms</b><br>Deadline misses: ${r.deadline_misses} (${(r.deadline_misses/100).toFixed(2)}%)<br>Stale responses: ${r.stale_responses} (${(r.stale_responses/100).toFixed(2)}%)<br>Source-age p95: ${r.source_age_p95_ms} ms; materialization-age p95: ${r.material_age_p95_ms} ms<br>${alert}<br>Labels available at final response: ${r.labels_available_at_last_response}/10000; predictive quality NOT_SCORED.`;
 meters.innerHTML=`<label>Deadline misses <meter min="0" max="10000" value="${r.deadline_misses}"></meter></label><label>Stale responses <meter min="0" max="10000" value="${r.stale_responses}"></meter></label>`;
 }
 controls.addEventListener('change',update);reset.addEventListener('click',()=>{for(const [key,value] of Object.entries({policy:'cached',rate:'50',condition:'normal',seed:'0'}))host.querySelector('[data-'+key+']').value=value;update()});reset.click();
}};
