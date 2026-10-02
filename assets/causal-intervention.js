/* Four exact population states: do(B=0/1):.5; do(A=0):.475; do(A=1):.525.
 * Fixed 50/50 demand, assignment propensity .1/.9, no spillovers; not fitted estimates. */
window.CausalIntervention = {mount(host) {
  host.classList.add('causal-explorer');
  host.innerHTML = `<p><strong>Change the assignment rule.</strong> Keep demand, outcome rules and customer population fixed.</p>
  <div class="controls"><label>Intervene on<select data-target><option value="B">Company badge B</option><option value="A">Customer action A</option></select></label>
  <label>Set everyone to<select data-value><option value="1">1 — present</option><option value="0">0 — absent</option></select></label><button type="button" data-reset>Reset</button></div>
  <p class="baseline">Observed baseline: 50.0% purchase probability. Demand stays 50/50.</p><output aria-live="polite"></output><div class="bar" aria-hidden="true"></div><p data-explanation></p>`;
  function update() {
    const target=host.querySelector('[data-target]').value,value=Number(host.querySelector('[data-value]').value);
    const risk=target==='B'?.5:.475+.05*value;
    host.querySelector('output').textContent=`Expected purchases: ${(100*risk).toFixed(1)}% · change: ${(100*(risk-.5)).toFixed(1)} percentage points`;
    host.querySelector('output').dataset.risk=risk;
    host.querySelector('.bar').style.width=`${100*risk}%`;
    host.querySelector('[data-explanation]').textContent=target==='B'?'Cut U → B. Badge changes, but it has no causal route to Y in this simulator. U and A remain unchanged.':'Cut U → A. The A → Y route remains. Risk is 0.05 + 0.85 × 0.5 + 0.05 × '+value+'.';
  }
  host.addEventListener('change',update);
  host.querySelector('[data-reset]').addEventListener('click',()=>{host.querySelector('[data-target]').value='B';host.querySelector('[data-value]').value='1';update();});update();
}};
