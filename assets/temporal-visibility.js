/* Query-time visibility. Default t=5: legal mean=3; event-only=14/3; static=6.
   Slider 0..10 changes the SAME query cutoff; reset restores t=5/event-only.
   Empty neighborhoods use the declared zero fallback. */
(function(global){'use strict';
const rows=[['Past purchase',3,3,2],['Late correction',4,7,8],['Boundary purchase',5,5,4],['Future outcome',8,8,10]];
function mount(el){
 el.innerHTML='<div class="temporal-controls"><label>Prediction day <input aria-label="Prediction day" type="range" min="0" max="10" value="5"></label><label>Compare with <select aria-label="Visibility rule"><option value="event">Event date only</option><option value="static">All stored rows</option></select></label><button type="button">Reset</button></div><p class="temporal-readout" aria-live="polite"></p><p>On narrow screens, scroll the table horizontally.</p><div class="temporal-scroll" tabindex="0" role="region" aria-label="Row availability table"><table><thead><tr><th>Record</th><th>Event day</th><th>Arrival day</th><th>Value</th><th>Legal</th><th>Compared rule</th></tr></thead><tbody></tbody></table></div><p>Held fixed: records, values and the mean aggregator. Zero is the declared fallback for no visible rows.</p>';
 const slider=el.querySelector('input'),select=el.querySelector('select');
 function draw(){const t=Number(slider.value), legal=rows.filter(r=>r[1]<=t&&r[2]<=t),other=rows.filter(r=>select.value==='static'||r[1]<=t);const mean=a=>a.length?a.reduce((s,r)=>s+r[3],0)/a.length:0;
 el.querySelector('tbody').innerHTML=rows.map(r=>'<tr><td>'+r[0]+'</td><td>'+r[1]+'</td><td>'+r[2]+'</td><td>'+r[3]+'</td><td>'+(legal.includes(r)?'Include':'Exclude')+'</td><td>'+(other.includes(r)?'Include':'Exclude')+'</td></tr>').join('');
 el.querySelector('.temporal-readout').textContent='Day '+t+' · Legal mean: '+mean(legal).toFixed(3)+' · Compared mean: '+mean(other).toFixed(3)+' · Illegally included rows: '+other.filter(r=>!legal.includes(r)).length;
 }
 slider.addEventListener('input',draw);select.addEventListener('change',draw);el.querySelector('button').addEventListener('click',()=>{slider.value=5;select.value='event';draw();});draw();return {draw};
}global.TemporalVisibility={mount};})(window);
