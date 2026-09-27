/* L125: the same row under frozen versus contaminated preprocessing. */
window.FrameContractViz={mount(root){
 if(!root)return;
 root.className='stream-widget task-table-widget';
 root.innerHTML=`<h3>Predict the token before changing the controls</h3>
 <p>Fitting rows: 10, 20, 30. A future row has value 1000. Weight [2, −1], bias [0.5, 0.5]. Population standard deviation + 10⁻⁶ matches Frame.</p>
 <label>Row value <select aria-label="Row value"><option value="30">30</option><option value="10">10</option><option value="missing">Missing</option></select></label>
 <label><input type="checkbox"> Include future row in fitted statistics</label>
 <button type="button">Reset</button><output aria-live="polite"></output>
 <div class="stream-scroll" tabindex="0"><table><thead><tr><th>State</th><th>Mean</th><th>Scale</th><th>z</th><th>Token</th></tr></thead><tbody></tbody></table></div>
 <p>Baseline: fitted state uses only 10, 20, 30. The checkbox changes the fitting population, while the queried row stays fixed.</p>`;
 const select=root.querySelector('select'),box=root.querySelector('input'),out=root.querySelector('output'),body=root.querySelector('tbody');
 function calc(values,raw){const mean=values.reduce((a,b)=>a+b,0)/values.length;const scale=Math.sqrt(values.reduce((s,v)=>s+(v-mean)**2,0)/values.length)+1e-6;const z=raw==='missing'?0:(Number(raw)-mean)/scale;return [mean,scale,z,[2*z+.5,-z+.5]];}
 function update(){const baseline=calc([10,20,30],select.value),active=calc(box.checked?[10,20,30,1000]:[10,20,30],select.value);body.innerHTML=[['Baseline',baseline],['Active',active]].map(([n,r])=>`<tr><td>${n}</td>${r.slice(0,3).map(x=>`<td>${x.toFixed(3)}</td>`).join('')}<td>[${r[3].map(x=>x.toFixed(3)).join(', ')}]</td></tr>`).join('');out.textContent=box.checked?'Fitted state changed: the future row altered the coordinate system.':'Frozen fitting state: changing query rows cannot change these statistics.';if(select.value==='missing')out.textContent+=' Missing → mean → z = 0 → bias [0.5, 0.5].';}
 select.addEventListener('change',update);box.addEventListener('change',update);root.querySelector('button').addEventListener('click',()=>{select.value='30';box.checked=false;update();});update();
}};
