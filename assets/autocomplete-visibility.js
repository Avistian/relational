/* Synthetic autocomplete visibility: global/seed_only/unmasked × past/future × proxy removal.
   Target and proxies are removed globally in the pinned RelBench baseline.
   Seed-only is the paper-described RelGT-AC policy; this widget is not either model. */
(function(global){'use strict';
function cells(policy,contextTime,drop){
 const rows=[{name:'Query row',time:10,seed:true,grid:3,position:4,points:12},{name:'Context row',time:Number(contextTime),seed:false,grid:5,position:2,points:18}];
 return rows.map(r=>{const admitted=r.time<=10,mask=policy==='global'||(policy==='seed_only'&&r.seed);return {...r,admitted,shown:[r.grid,mask?'MASKED':r.position,mask&&drop?'MASKED':r.points]};});
}
function mount(host){
 host.classList.add('autocomplete-visibility');
 host.innerHTML='<p><strong>Predict first:</strong> which answer-bearing cells survive? Query time stays at 10. Values are synthetic.</p><div class="av-controls"><label>Masking policy<select data-policy><option value="global">Whole target table</option><option value="seed_only">Query row only</option><option value="unmasked">No target masking</option></select></label><label>Context timestamp<select data-time><option value="9">9 — before query</option><option value="11">11 — after query</option></select></label><label>Drop proxy points<select data-drop><option value="yes">Yes</option><option value="no">No</option></select></label></div><p class="av-scroll-note">Scroll the table horizontally on a narrow screen to inspect every column.</p><div class="av-scroll" tabindex="0"><table><thead><tr><th>Row / time</th><th>Grid</th><th>Position target</th><th>Points proxy</th></tr></thead><tbody></tbody></table></div><output aria-live="polite"></output><p class="av-baseline">Fixed reference: whole-table masking + proxy removal + past context exposes zero position/points values.</p><button type="button" data-reset>Reset</button>';
 const policy=host.querySelector('[data-policy]'),time=host.querySelector('[data-time]'),drop=host.querySelector('[data-drop]');
 function update(){const rs=cells(policy.value,time.value,drop.value==='yes');host.querySelector('tbody').innerHTML=rs.map(r=>'<tr><th>'+r.name+' / '+r.time+'</th>'+r.shown.map(x=>'<td>'+(r.admitted?x:'EXCLUDED')+'</td>').join('')+'</tr>').join('');
 const targets=rs.filter(r=>r.admitted&&r.shown[1]!=='MASKED').length,proxies=rs.filter(r=>r.admitted&&r.shown[2]!=='MASKED').length;
 const queryTarget=rs[0].shown[1]!=='MASKED';
 const out=host.querySelector('output');out.dataset.targets=targets;out.dataset.proxies=proxies;out.textContent=targets+' visible target values; '+proxies+' visible proxy values. '+(queryTarget?'Query target exposed: trivial answer access.':proxies?'Query or context proxies remain: check the declared drop policy.':targets?'Past context target retained: different information from whole-table masking.':'Target and proxy values removed. Temporal filtering is a separate check.');
 host.dataset.state=[policy.value,time.value,drop.value].join('/');}
 [policy,time,drop].forEach(x=>x.addEventListener('change',update));host.querySelector('[data-reset]').addEventListener('click',()=>{policy.value='global';time.value='9';drop.value='yes';update();});update();return {update};}
 global.AutocompleteVisibility={mount,cells};})(window);
