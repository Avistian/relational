/* Reusable cutoff intervention: features and label-ready support, no model score. */
(function(global){
  'use strict';
  function mount(host){
    host.className='route-widget transfer-widget';
    host.innerHTML='<div class="transfer-controls"><label>Prediction cutoff<select data-cutoff><option>10</option><option>20</option></select></label><label>Day 14 purchase value<select data-value><option>12</option><option>1200</option></select></label><button type="button" data-reset>Reset baseline</button></div><p class="transfer-baseline">Fixed baseline: cutoff 10, final value 12 → count 2, mean 6, support A only.</p><output aria-live="polite"></output><div class="route-scroll" tabindex="0"><table><caption>Customer 7 events</caption><thead><tr><th>Event day</th><th>Value</th><th>Eligibility</th></tr></thead><tbody data-events></tbody></table></div><div class="route-scroll" tabindex="0"><table><caption>Candidate labeled context</caption><thead><tr><th>Support</th><th>Query day</th><th>Label ready</th><th>Eligibility</th></tr></thead><tbody data-support></tbody></table></div>';
    function update(){
      const cutoff=Number(host.querySelector('[data-cutoff]').value),value=Number(host.querySelector('[data-value]').value);
      const events=[[2,4],[8,8],[10,100],[14,value]],support=[['A',2,9],['B',5,12],['C',10,10]];
      const kept=events.filter(r=>r[0]<cutoff),eligible=support.filter(r=>r[1]<cutoff&&r[2]<cutoff);
      const mean=kept.length?kept.reduce((sum,r)=>sum+r[1],0)/kept.length:0;
      host.querySelector('output').textContent='Count '+kept.length+' · Mean '+mean+' · Eligible support: '+eligible.map(r=>r[0]).join(', ');
      host.querySelector('[data-events]').innerHTML=events.map(r=>'<tr class="'+(r[0]<cutoff?'eligible':'excluded')+'"><td>'+r[0]+'</td><td>'+r[1]+'</td><td>'+(r[0]<cutoff?'Included':'At or after cutoff')+'</td></tr>').join('');
      host.querySelector('[data-support]').innerHTML=support.map(r=>'<tr class="'+(r[1]<cutoff&&r[2]<cutoff?'eligible':'excluded')+'"><td>'+r[0]+'</td><td>'+r[1]+'</td><td>'+r[2]+'</td><td>'+(r[1]<cutoff&&r[2]<cutoff?'Included':'Query or label too late')+'</td></tr>').join('');
    }
    host.querySelectorAll('select').forEach(s=>s.addEventListener('change',update));
    host.querySelector('[data-reset]').addEventListener('click',()=>{host.querySelector('[data-cutoff]').value='10';host.querySelector('[data-value]').value='12';update();});update();
  }
  global.TransferMap={mount};
})(window);
