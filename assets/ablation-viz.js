/* Reusable teaching controls. Baseline365day window; toy combined MAE6.5. */
window.AblationViz={
 history(host){
  host.innerHTML='<label>History window <select aria-label="History window"><option value="30">30 days</option><option value="365" selected>365 days</option><option value="730">730 days</option></select></label> <button type="button">Reset</button><p>Fixed query cutoffs: A = 400; B = 500. Dated non-root rows only. Baseline: 365 days.</p><div class="ablation-scroll"><table><thead><tr><th>Row day</th><th>A baseline</th><th>A current</th><th>B baseline</th><th>B current</th></tr></thead><tbody></tbody></table></div><p role="status"></p>';
  const select=host.querySelector('select');const verdict=(t,c,w)=>t>c?'Future':t<c-w?'Too old':'Keep';
  function draw(){let w=Number(select.value);host.dataset.window=String(w);host.querySelector('tbody').innerHTML=[34,100,399,450,501].map(t=>`<tr><td>${t}</td><td>${verdict(t,400,365)}</td><td>${verdict(t,400,w)}</td><td>${verdict(t,500,365)}</td><td>${verdict(t,500,w)}</td></tr>`).join('');host.querySelector('[role=status]').textContent=`Window ${w} days. Future rows never become legal when the window grows.`;}
  select.addEventListener('change',draw);host.querySelector('button').onclick=()=>{select.value='365';draw();};draw();
 },
 interaction(host){
  host.innerHTML='<label>Hypothetical combined MAE <input aria-label="Combined MAE" type="range" min="5" max="9" step="0.5" value="6.5"></label> <button type="button">Reset</button><p>Fixed: full F = 4, encoder E = 5, messages M = 6. Additive prediction = 7. Baseline combined = 6.5.</p><p role="status"></p>';
  const control=host.querySelector('input');function draw(){let v=Number(control.value),i=v-7;host.dataset.interaction=String(i);host.querySelector('[role=status]').textContent=`Combined EM = ${v.toFixed(1)}. Interaction = ${v.toFixed(1)} − 5 − 6 + 4 = ${i.toFixed(1)} MAE. ${i<0?'Smaller combined penalty than additive prediction.':i>0?'Larger combined penalty than additive prediction.':'Additive on this error scale.'}`;}
  control.oninput=draw;host.querySelector('button').onclick=()=>{control.value='6.5';draw();};draw();
 }
};
