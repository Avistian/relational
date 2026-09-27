/* A query changes both its input cutoff and its label window. */
window.TaskTableViz={mount(root){
 root.className='stream-widget task-table-widget';
 root.innerHTML='<h3>Which rows answer this question?</h3><label>Query day <select aria-label="Query day"><option value="0">0</option><option value="1">1</option><option value="2">2</option></select></label> <label><input type="checkbox" checked> Require past activity</label> <button type="button">Reset</button><div class="stream-scroll" tabindex="0"><table><thead><tr><th>Driver</th><th>Past results</th><th>Future positions</th><th>Target</th></tr></thead><tbody></tbody></table></div><output aria-live="polite"></output><p>Fixed baseline: day 0, past activity required → Driver 0 only, label 4. No future races means no observed label.</p>';
 const select=root.querySelector('select'),check=root.querySelector('input'),out=root.querySelector('output');
 const events=[{id:0,t:0,p:99},{id:0,t:1,p:2},{id:0,t:60,p:6},{id:1,t:2,p:3}];
 function update(){const t=Number(select.value);let n=0;root.querySelector('tbody').innerHTML=[0,1].map(id=>{
 const past=events.filter(x=>x.id===id&&x.t<=t),future=events.filter(x=>x.id===id&&x.t>t&&x.t<=t+60),eligible=!check.checked||past.length>0;
 const y=future.length&&eligible?future.reduce((s,x)=>s+x.p,0)/future.length:null;if(y!==null)n++;
 return `<tr><td>Driver ${id}</td><td>${past.map(x=>`day ${x.t}: ${x.p}`).join(', ')||'None'}</td><td>${future.map(x=>x.p).join(', ')||'None'}</td><td>${y===null?'No task row':y}</td></tr>`;
 }).join('');out.textContent=`${n} labeled ${n===1?'query':'queries'}. Input: event day ≤ ${t}. Target: ${t} < event day ≤ ${t+60}. Labels mature no earlier than day ${t+60}. ${check.checked?'Past eligibility enforced.':'Released-style cohort includes future-only entrants.'}`;}
 select.addEventListener('change',update);check.addEventListener('change',update);root.querySelector('button').addEventListener('click',()=>{select.value='0';check.checked=true;update();});update();
}};
