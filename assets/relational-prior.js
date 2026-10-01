/* Measured checkpoint trace viewer. Data are built from the canonical Python model. */
window.RelationalPrior={mount(host,packet){
 if(!host)return;
 host.className='prior-explorer';
 host.innerHTML='<div class="prior-controls"><label>Dependency strength<select data-strength><option>0</option><option selected>1</option><option>2</option></select></label><label>Support rows<select data-support><option>2</option><option selected>4</option></select></label><label>Support labels<select data-flip><option value="0">Original</option><option value="1">Flipped</option></select></label><button type="button" data-reset>Reset baseline</button></div><output aria-live="polite"></output><p class="prior-small" data-baseline></p><div class="route-scroll" tabindex="0"><table><thead><tr><th>Row / role</th><th>Count</th><th>Mean value</th><th>Supplied label</th><th>Query P(class 1)</th></tr></thead><tbody></tbody></table></div><p class="prior-small" data-mask></p><p class="prior-small">Scroll the table horizontally to see every column. Measured on six invented parents. The weights stay fixed. No benchmark accuracy is displayed.</p>';
 const baseline=packet.traces.find(t=>t.strength===1&&t.support===4&&t.flip===0);
 function update(){
  const strength=Number(host.querySelector('[data-strength]').value),support=Number(host.querySelector('[data-support]').value),flip=Number(host.querySelector('[data-flip]').value);
  const t=packet.traces.find(t=>t.strength===strength&&t.support===support&&t.flip===flip),prob=t.query_probability.at(-1),base=baseline.query_probability.at(-1);
  host.querySelector('output').textContent=`Common query row 5: P(class 1) = ${prob.toFixed(4)}; change from baseline = ${(prob-base>=0?'+':'')+(prob-base).toFixed(4)}.`;
  host.querySelector('[data-baseline]').textContent=`Fixed baseline: strength 1, four support rows, original labels; row 5 probability ${base.toFixed(4)}.`;
  host.querySelector('tbody').innerHTML=t.features.map((f,i)=>`<tr class="${i>=support?'query-row':''}"><td>${i} / ${i<support?'support':'query'}</td><td>${f[0]}</td><td>${f[1].toFixed(4)}</td><td>${i<support?t.labels[i]:'hidden'}</td><td>${i<support?'—':t.query_probability[i-support].toFixed(4)}</td></tr>`).join('');
  host.querySelector('[data-mask]').textContent=`Every row may read key/value rows ${Array.from({length:support},(_,i)=>i).join(', ')} only. Query rows never become keys or values.`;
 }
 host.querySelectorAll('select').forEach(s=>s.addEventListener('change',update));
 host.querySelector('[data-reset]').addEventListener('click',()=>{host.querySelector('[data-strength]').value='1';host.querySelector('[data-support]').value='4';host.querySelector('[data-flip]').value='0';update();});update();
}};
