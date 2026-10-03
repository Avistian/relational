/* Research choice explorer. Form completion never grants authorization or mastery. */
(() => {
  'use strict';
  const names={availability:'Availability',composite:'Composite structure',constraints:'Joint constraints'};
  const fields=['direction','hypothesis','contrast','baselines','metric','threshold','uncertainty','cost','stop','deferred','evidence','revision'];
  const labels={direction:'Provisional direction',hypothesis:'Falsifiable hypothesis',contrast:'Exact contrast',baselines:'Matched baselines',metric:'Metric and units',threshold:'Useful-effect threshold',uncertainty:'Uncertainty and sampling unit',cost:'Aggregate cost and cutoff',stop:'Stop conditions',deferred:'Deferred alternative and opportunity cost',evidence:'Evidence and its limits',revision:'What would change your choice'};
  const ranking=document.getElementById('direction-ranking');
  if(ranking){
    ranking.className='direction-widget';
    ranking.innerHTML='<h3>Change a judgment, then a weight</h3><p>The other eleven ratings remain frozen. First predict the winner.</p><div class="direction-controls"></div><ul class="direction-results"></ul><output aria-live="polite"></output><button type="button">Reset ranking</button>';
    const inputs=[['impact','Availability impact',[1,2,3,4,5],4],['data','Data weight',[1,2,3],1],['implementation','Implementation weight',[1,2,3],1],['compute','Compute weight',[1,2,3],1]];
    inputs.forEach(([key,label,options,initial])=>{const host=document.createElement('label');host.textContent=label;const select=document.createElement('select');select.id='rank-'+key;options.forEach(v=>select.add(new Option(String(v),String(v),v===initial,v===initial)));host.append(select);ranking.querySelector('.direction-controls').append(host);select.addEventListener('change',update);});
    function update(){const get=k=>Number(ranking.querySelector('#rank-'+k).value);const w=['data','implementation','compute'].map(get);const impact=get('impact');const den=w.reduce((a,b)=>a+b,0);const numerators={availability:impact*(5*w[0]+4*w[1]+5*w[2]),composite:5*(3*w[0]+2*w[1]+4*w[2]),constraints:4*(3*w[0]+3*w[1]+2*w[2])};const best=Math.max(...Object.values(numerators));const leaders=Object.keys(numerators).filter(k=>numerators[k]===best).sort();ranking.dataset.leaders=leaders.join(',');ranking.querySelector('ul').innerHTML=Object.entries(numerators).map(([k,n])=>'<li><strong>'+ (n/den).toFixed(3)+'</strong> '+names[k]+'</li>').join('');ranking.querySelector('output').textContent=(leaders.length>1?'Tie: ':'Priority: ')+leaders.map(k=>names[k]).join(' / ')+'. This does not change any launch requirement.';}
    ranking.querySelector('button').addEventListener('click',()=>{inputs.forEach(([key,, ,initial])=>ranking.querySelector('#rank-'+key).value=String(initial));update();});update();
  }
  const gates=document.getElementById('direction-gates');
  if(gates){
    gates.className='direction-widget';gates.innerHTML='<h3>Can the proposed study launch?</h3><p>Explore hypothetical evidence states. Current evidence is UNKNOWN for all four.</p><div class="direction-controls"></div><output aria-live="polite"></output><button type="button">Reset to current evidence</button>';
    ['data','baseline','design','budget'].forEach(key=>{const label=document.createElement('label');label.textContent=key[0].toUpperCase()+key.slice(1)+' requirement';const select=document.createElement('select');select.id='gate-'+key;['UNKNOWN','PASS','FAIL'].forEach(v=>select.add(new Option(v,v)));select.addEventListener('change',update);label.append(select);gates.querySelector('.direction-controls').append(label);});
    function update(){const blocked=['data','baseline','design','budget'].filter(k=>gates.querySelector('#gate-'+k).value!=='PASS');gates.dataset.state=blocked.length?'DO_NOT_LAUNCH':'READY_FOR_REVIEW';gates.querySelector('output').textContent=gates.dataset.state+(blocked.length?' · unresolved: '+blocked.join(', '):' · all requirements marked PASS in this hypothetical scenario')+'. Authorization NOT_GRANTED.';}
    gates.querySelector('button').addEventListener('click',()=>{gates.querySelectorAll('select').forEach(s=>s.value='UNKNOWN');update();});update();
  }
  const memo=document.getElementById('direction-memo');
  if(memo){
    memo.className='direction-widget';memo.innerHTML='<h3>Your selection memo</h3><p>Use “unknown” candidly where a requirement is unresolved; justify how you will resolve it. This form checks presence only.</p><div class="direction-memo-grid"></div><output aria-live="polite"></output><button type="button">Download memo</button><button type="button">Clear memo</button><p class="direction-disclaimer">Kept in this page until downloaded; nothing is sent or saved to an account. Download before leaving.</p>';
    fields.forEach(key=>{const label=document.createElement('label');label.textContent=labels[key];const area=document.createElement('textarea');area.name=key;area.addEventListener('input',update);label.append(area);memo.querySelector('.direction-memo-grid').append(label);});
    function sections(){return Object.fromEntries(fields.map(k=>[k,memo.querySelector('[name="'+k+'"]').value]));}
    function update(){const values=sections();const missing=fields.filter(k=>!values[k].trim());memo.dataset.state=missing.length?'DRAFT':'READY_FOR_REVIEW';memo.querySelector('output').textContent=memo.dataset.state+' · '+missing.length+' empty fields · PENDING_WRITTEN_DEFENSE. Presence does not establish scientific quality.';return {state:memo.dataset.state,missing,mastery:'PENDING_WRITTEN_DEFENSE'};}
    memo.querySelectorAll('button')[0].addEventListener('click',()=>{const review=update();const data={experiment:'L199-DIRECTION-SELECTION-AUDIT',sections:sections(),review,authorization:'NOT_GRANTED'};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='l199-selection-memo.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
    memo.querySelectorAll('button')[1].addEventListener('click',()=>{memo.querySelectorAll('textarea').forEach(a=>a.value='');update();});update();
  }
})();
