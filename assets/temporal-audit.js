/* Synthetic worked trace: each owner keeps its own cutoff; unknown is not pass. */
window.TemporalAudit={mount(host){
 host.innerHTML=`<h3>One row, two clocks, one owner</h3><p>Synthetic example. Query A has cutoff 10; query B has cutoff 20. Change A's dependency while keeping both queries fixed.</p>
 <label>Dependency type <select data-kind><option value="event">Observed event</option><option value="schedule">Published schedule</option></select></label>
 <label>Event time <select data-event><option value="9">9 · before A</option><option value="10">10 · exactly at A</option><option value="12">12 · after A</option></select></label>
 <label>Availability time <select data-arrival><option value="9">9 · available before A</option><option value="12">12 · arrives after A</option><option value="unknown">Unknown · no history</option></select></label>
 <label>Event boundary <select data-rule><option value="inclusive">At or before cutoff (≤)</option><option value="strict">Strictly before cutoff (&lt;)</option></select></label>
 <div class="audit-path"><div class="audit-node">Query A<br><b>owner cutoff = 10</b></div><div class="audit-node" data-dependency></div><div class="audit-node">Query B<br><b>owner cutoff = 20</b></div></div>
 <output aria-live="polite" data-result></output><p data-wrong></p><p>Baseline: event 9, availability 9 → PASS under both event boundaries. A future scheduled event can pass only if its publication was already available.</p><button type="button">Reset example</button>`;
 const el=n=>host.querySelector('[data-'+n+']');
 function update(){const kind=el('kind').value,event=+el('event').value,arrival=el('arrival').value,strict=el('rule').value==='strict';const future=kind==='event'&&(event>10||(strict&&event===10));const late=arrival!=='unknown'&&+arrival>10;const status=future||late?'FAIL':arrival==='unknown'?'NOT_ESTABLISHED':'PASS';
 host.dataset.status=status;el('dependency').textContent=`${kind==='event'?'Event':'Schedule'} time = ${event}; available = ${arrival}`;
 el('result').className=status==='FAIL'?'audit-fail':status==='PASS'?'audit-pass':'audit-unknown';el('result').textContent=status+': '+(future?'The event violates A’s declared boundary. ':late?'The dependency arrives after A’s cutoff. ':arrival==='unknown'?'The archive does not establish when this dependency became available. ':'The stated event/publication conditions hold for query A.');
 const wrong=kind==='event'&&(event>20||(strict&&event===20))||(arrival!=='unknown'&&+arrival>20)?'FAIL':arrival==='unknown'?'NOT_ESTABLISHED':'PASS';el('wrong').textContent=`Incorrect batch-maximum audit (cutoff 20): ${wrong}. Query B cannot lend its later cutoff to A.`;
 }
 host.querySelectorAll('select').forEach(x=>x.addEventListener('change',update));host.querySelector('button').onclick=()=>{el('kind').value='event';el('event').value='9';el('arrival').value='9';el('rule').value='inclusive';update();};update();
}};
