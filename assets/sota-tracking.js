/* Versioned benchmark explorer. Expected: T3 foundation single +4.053333;
   T4/T8 foundation have no eligible row. Pools/rules change comparisons only.
   Every state is checked against the independent Python report by delivery. */
(function () {
  'use strict';
  const host = document.getElementById('sota-explorer');
  if (!host) return;
  const data = JSON.parse(document.getElementById('sota-data').textContent);
  host.className = 'sota-explorer';
  host.innerHTML = '<h3>Change the comparison contract</h3><div class="controls">' +
    '<label>Frozen table<select data-table><option value="3">3 · V1 classification</option><option value="4">4 · V2 classification</option><option value="7">7 · V1 regression</option><option value="8">8 · V2 regression</option></select></label>' +
    '<label>Audited open pool<select data-pool><option value="foundation">Foundation models</option><option value="supervised">Supervised models</option><option value="all">Both families</option></select></label>' +
    '<label>Selection rule<select data-rule><option value="single">One complete method</option><option value="oracle">Taskwise test oracle</option></select></label><button type="button" data-reset>Reset comparison</button></div>' +
    '<output aria-live="polite"></output><p data-explanation></p><div data-table-result></div>' +
    '<p class="baseline">Fixed reference: Table 3 · foundation pool · single RDBLearn · +4.053333 AUROC points. Frozen April 2026 paper; no fresh predictions.</p>';
  const tableControl = host.querySelector('[data-table]'), poolControl = host.querySelector('[data-pool]'), ruleControl = host.querySelector('[data-rule]');
  const mean = x => x.reduce((a,b)=>a+b,0)/x.length;
  const esc = x => String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
  function draw() {
    const t = data.tables.find(t => t.number === Number(tableControl.value));
    const pool = poolControl.value, oracle = ruleControl.value === 'oracle', higher = t.metric === 'AUROC';
    const candidates = t.rows.filter(r => { const p=data.policy[r.method] || {};return p.access==='open_code' && p.protocol==='reported_same_table' && ['foundation','supervised'].includes(p.family) && (pool==='all'||p.family===pool) && r.values.every(Number.isFinite); });
    const out = host.querySelector('output'), expl = host.querySelector('[data-explanation]'), result = host.querySelector('[data-table-result]');
    host.dataset.status = candidates.length ? 'COMPLETE_PUBLISHED_COMPARISON' : 'NO_ELIGIBLE_COMPARATOR';
    if (!candidates.length) {
      host.dataset.gap='';host.dataset.methods='';out.textContent='NO_ELIGIBLE_COMPARATOR';
      expl.textContent='This frozen table contains no audited open foundation row. Missing evidence is neither a zero score nor a model loss.';result.innerHTML='';return;
    }
    const target = t.rows.find(r=>r.method==='KumoRFM-2').values;
    const base = t.rows.find(r=>r.method==='LightGBM').values;
    const score = values => mean(values.map((v,i)=>higher?v:v/base[i]));
    const choose = higher ? Math.max : Math.min;
    const best = choose(...candidates.map(r=>score(r.values)));
    const winners = candidates.filter(r=>Math.abs(score(r.values)-best)<=1e-12).sort((a,b)=>a.method.localeCompare(b.method));
    const values = oracle ? target.map((_,i)=>choose(...candidates.map(r=>r.values[i]))) : winners[0].values;
    const names = target.map((_,i)=>oracle?candidates.filter(r=>r.values[i]===values[i]).map(r=>r.method).sort().join(' / '):winners.map(r=>r.method).join(' / '));
    const gap = higher ? score(target)-score(values) : score(values)-score(target);
    host.dataset.gap=String(gap);host.dataset.methods=oracle?'taskwise oracle':winners.map(r=>r.method).join(',');
    out.textContent=(gap>=0?'+':'')+gap.toFixed(6)+' '+(higher?'AUROC percentage points':'normalized MAE')+' · Kumo advantage';
    expl.textContent=oracle?'Taskwise oracle: the eligible test winner may change in every row. This envelope is not one deployable model.':'Single method: '+winners.map(r=>r.method).join(' / ')+'. Chosen retrospectively using the complete published test table; deployment selection requires validation data.';
    result.innerHTML='<div class="sota-table" tabindex="0"><table><thead><tr><th>Task</th><th>Comparator</th><th>Kumo</th><th>Open</th><th>Task gap</th></tr></thead><tbody>'+target.map((v,i)=>'<tr><td>'+esc(t.tasks[i])+'</td><td>'+esc(names[i])+'</td><td>'+v.toFixed(higher?2:3)+'</td><td>'+values[i].toFixed(higher?2:3)+'</td><td>'+((higher?v-values[i]:values[i]-v)).toFixed(higher?2:3)+'</td></tr>').join('')+'</tbody></table></div><p>Task gaps use '+(higher?'AUROC percentage points':'raw task-specific MAE units; the aggregate above uses LightGBM-normalized ratios')+'.</p>';
  }
  [tableControl,poolControl,ruleControl].forEach(x=>x.addEventListener('change',draw));
  host.querySelector('[data-reset]').addEventListener('click',()=>{tableControl.value='3';poolControl.value='foundation';ruleControl.value='single';draw();});draw();
  if(window.RetrievalBank) RetrievalBank.mount(document.getElementById('warmup'),{upTo:191,count:3});
  if(window.Predict) Predict.mount(document.getElementById('gap-predict'),{
    prompt:'Let supervised open methods enter the Table 3 comparison. What happens to the single-method gap?',
    options:[{label:'The gap becomes smaller',value:'smaller'},{label:'The gap becomes larger',value:'larger'},{label:'The gap stays unchanged',value:'same'}],correct:'smaller',
    reveal:'RelGNN replaces RDBLearn as the strongest eligible single row. The gap shrinks from 4.053 to 1.542 AUROC percentage points; the Kumo task cells did not change.'});
  if(window.Teachback) Teachback.mount(document.getElementById('gap-teachback'),{
    prompt:'Explain why “4.05 points ahead” does not establish current global superiority or fresh reproduction.',
    points:['Name the April 2026 source and bounded foundation pool.','Use AUROC percentage points and complete task coverage.','Separate published arithmetic from fresh inference and current tracking.'],
    model:'The 4.053-point gap comes from the 12 displayed Table 3 task scores and a bounded open foundation pool where RDBLearn is the best single row. It is a retrospective published comparison. Newer methods and fresh model predictions have not been evaluated here.'});
})();
