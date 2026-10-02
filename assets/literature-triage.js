/* Synthetic evidence-admission exercise: 2×2×4 states. All remain NOT_REPRODUCED. */
(function(global){
'use strict';
function decision(verified,reviewed,relevance){if(!verified||!reviewed)return 'DEFER';return ['sota','failure_mode','baseline'].includes(relevance)?'INCLUDE':'EXCLUDE';}
function mount(host){
 host.classList.add('lit-panel');host.innerHTML=`<p><strong>Predict → change one assumption → explain</strong></p><p>Synthetic paper: a Q3 submission reports a benchmark gain. The log has a written reason. Change its verification and relevance below.</p><div class="lit-controls"><label>Official identity verified<select data-verified><option value="yes">Yes</option><option value="no">No</option></select></label><label>Claim screened by reader<select data-reviewed><option value="yes">Yes</option><option value="no">No</option></select></label><label>Reason for tracking<select data-role><option value="baseline">Required baseline</option><option value="failure_mode">Relevant failure mode</option><option value="sota">Reported new best result</option><option value="incremental">Incremental application</option></select></label></div><output aria-live="polite"></output><p class="lit-baseline">Baseline: verified + reviewed + baseline → INCLUDE for reading. Paper-result reproduction: NOT_RUN in every state.</p><button type="button" data-reset>Reset example</button>`;
 function update(){const d=decision(host.querySelector('[data-verified]').value==='yes',host.querySelector('[data-reviewed]').value==='yes',host.querySelector('[data-role]').value);host.dataset.decision=d;host.querySelector('output').textContent=d+' — '+({INCLUDE:'Keep a traceable reading candidate. A reported gain still needs an aligned reproduction.',DEFER:'Missing verification or screening prevents an admission decision.',EXCLUDE:'The stated relevance does not meet the course admission rule. Retain the reason in the log.'}[d]);}
 host.addEventListener('change',update);host.querySelector('[data-reset]').onclick=()=>{host.querySelectorAll('select').forEach(x=>x.selectedIndex=0);update();};update();
}
global.LiteratureTriage={mount,decision};
})(window);
