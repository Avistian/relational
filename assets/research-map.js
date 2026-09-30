/* Reusable research planning map. Values are explicit judgments, never model scores. */
(function(g){'use strict';
function rank(rows,hours,usd){return rows.filter(r=>r.ready&&r.hours<=hours&&r.usd<=usd).sort((a,b)=>b.evidence-a.evidence||a.hours-b.hours||a.id.localeCompare(b.id)).map(r=>r.id);}
function mount(el,rows){
 el.classList.add('research-map');
 el.innerHTML='<p><strong>Baseline: 4 hours, $10 per candidate; all other assumptions fixed.</strong></p><label>Investigation time <select aria-label="Investigation time"><option value="1">1 hour</option><option value="4" selected>4 hours</option><option value="8">8 hours</option><option value="40">40 hours</option></select></label><label>Research question <select aria-label="Research question"></select></label><button type="button">Reset map</button><output aria-live="polite"></output><div class="map-card"></div>';
 const selects=el.querySelectorAll('select'),time=selects[0],question=selects[1],out=el.querySelector('output'),card=el.querySelector('.map-card');
 rows.forEach(r=>{const o=document.createElement('option');o.value=r.id;o.textContent=r.title;question.appendChild(o);});question.value='selection';
 function update(){const ids=rank(rows,Number(time.value),10),r=rows.find(r=>r.id===question.value);el.dataset.ranking=ids.join(',');el.dataset.question=r.id;
 out.textContent='Eligible order: '+(ids.join(' → ')||'none')+'. Same evidence; only the time constraint changed.';
 card.replaceChildren();const heading=document.createElement('h3');heading.textContent=r.title;card.appendChild(heading);
 for(const [label,value] of [['Problem',r.problem],['Mechanism',r.mechanism],['Unresolved',r.gap],['Minimal test',r.test],['Held fixed',r.held_fixed],['Falsifier',r.falsifier],['Planning basis',r.basis],['Feasibility',!r.ready?'BLOCKED: required inputs/protocol unavailable.':ids.includes(r.id)?'Eligible under current assumptions.':'Exceeds current planning limit.']]){const p=document.createElement('p'),b=document.createElement('strong');b.textContent=label+': ';p.append(b,document.createTextNode(value));card.appendChild(p);}
 const a=document.createElement('a');a.href=r.source;a.textContent='Primary-source context';card.appendChild(a);}
 time.addEventListener('change',update);question.addEventListener('change',update);el.querySelector('button').addEventListener('click',()=>{time.value='4';question.value='selection';update();});update();
}
g.ResearchMap={rank,mount};})(window);
