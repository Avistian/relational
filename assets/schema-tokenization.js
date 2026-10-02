/* States: known/unknown/missing/masked; fit future values 30, 300, 999 with/without leakage.
   Baselines remain visible. This synthetic arithmetic illustrates the course contract. */
window.SchemaTokenization={
 states(host){
  host.className='token-explorer';
  host.innerHTML='<div class="token-controls"><label>Cell condition<select><option value="known">Known: b</option><option value="unknown">Unseen: future</option><option value="missing">Missing: null</option><option value="masked">Masked: b hidden</option></select></label><button type="button" data-reset>Reset</button></div><div class="token-panels"><div class="token-panel" data-baseline><strong>Fixed baseline</strong><p>Training vocabulary: a → 1, b → 2.</p><p>Observed b → VALUE, payload 2.</p></div><div class="token-panel changed"><strong>Changed cell</strong><output aria-live="polite"></output><p data-reason></p></div></div>';
  const select=host.querySelector('select');
  function draw(){const states={known:['VALUE',2,'The training vocabulary contains b.'],unknown:['UNKNOWN',0,'The value is observed, but absent from the frozen vocabulary.'],missing:['MISSING',0,'The source did not supply a value.'],masked:['MASKED',0,'We deliberately hide the value; its payload is erased.']};const [s,p,reason]=states[select.value];host.dataset.state=s;host.dataset.payload=p;host.querySelector('output').textContent=s+' · payload '+p;host.querySelector('[data-reason]').textContent=reason+' Schema descriptor stays fixed.';}
  select.addEventListener('change',draw);host.querySelector('button').onclick=()=>{select.value='known';draw();};draw();
 },
 fit(host){
  host.className='token-explorer';host.innerHTML='<div class="token-controls"><label>Held-out value<select><option value="30">30</option><option value="300">300</option><option value="999" selected>999</option></select></label><label><span><input type="checkbox"> Incorrectly include held-out row in fit</span></label><button type="button" data-reset>Reset</button></div><div class="token-panels"><div class="token-panel" data-baseline><strong>Correct fixed fit</strong><p>Admitted values: 10, 20, 30.</p><p>Mean = 20; population SD = 8.164966.</p><p>Probe value 20 → 0.000000.</p></div><div class="token-panel changed"><strong>Current fit</strong><output aria-live="polite"></output><p data-reason></p></div></div>';
  const select=host.querySelector('select'),check=host.querySelector('input');
  function draw(){const values=check.checked?[10,20,30,Number(select.value)]:[10,20,30];const mean=values.reduce((a,b)=>a+b,0)/values.length;const sd=Math.sqrt(values.reduce((a,b)=>a+(b-mean)**2,0)/values.length)||1;const z=(20-mean)/sd;host.dataset.mean=mean;host.dataset.probe=z;host.querySelector('output').textContent='Mean '+mean.toFixed(6)+' · SD '+sd.toFixed(6)+' · probe '+z.toFixed(6);host.querySelector('[data-reason]').textContent=check.checked?'LEAKED FIT: changing a held-out row can change the encoding of an unchanged training value.':'FROZEN FIT: changing the held-out value cannot change fitted statistics or the probe encoding.';}
  select.onchange=check.onchange=draw;host.querySelector('button').onclick=()=>{select.value='999';check.checked=false;draw();};draw();
 }
};
