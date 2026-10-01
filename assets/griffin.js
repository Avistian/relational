/* Synthetic two-dimensional teaching states, not trained predictions.
   Cell board: task A/B × paired/reversed columns × keep/hide experience =8 states.
   Relation board: repetitions1/2/4 × cutoff 5/7 =6 states. All retain baseline.
*/
(function(global){
'use strict';
const fmt=v=>'['+v.map(x=>x.toFixed(3)).join(', ')+']';
function attention(task,order,hide){
 const q=task==='A'?[Math.SQRT2,0]:[0,Math.SQRT2];
 let rows=[{name:'recent form',k:[1,0],v:[2,0],hidden:false},{name:'experience',k:[0,1],v:[0,4],hidden:hide}];
 if(order==='reverse')rows.reverse();
 const logits=rows.map(r=>(q[0]*r.k[0]+q[1]*r.k[1])/Math.SQRT2);
 const exp=logits.map((x,i)=>rows[i].hidden?0:Math.exp(x));const sum=exp.reduce((a,b)=>a+b,0);const w=exp.map(x=>x/sum);
 return {q,rows,logits,w,out:[0,1].map(j=>rows.reduce((a,r,i)=>a+w[i]*r.v[j],0))};
}
function relations(repetitions,cutoff){
 const rows=[];for(let i=0;i<repetitions;i++){rows.push({r:0,x:[2,4],t:4});rows.push({r:0,x:[4,2],t:4});}
 rows.push({r:1,x:[8,1],t:4});rows.push({r:1,x:[20,10],t:6});
 const active=rows.filter(x=>x.t<cutoff);const weights=[[1,1],[.5,2]];
 const messages=[0,1].map(r=>{let ns=active.filter(x=>x.r===r);return[0,1].map(j=>ns.reduce((a,n)=>a+n.x[j],0)/ns.length*weights[r][j]);});
 const out=[0,1].map(j=>Math.max(...messages.map(m=>m[j])));
 const flat=[0,1].map(j=>active.reduce((a,n)=>a+n.x[j]*weights[n.r][j],0)/active.length);
 return {messages,out,flat,count:active.length,excluded:rows.length-active.length};
}
function mountCells(host){
 host.className='griffin-board';host.innerHTML='<h3>Which cells does this task read?</h3><p>Synthetic projected Q/K/V; learned projections and output gate are held out of this arithmetic view.</p><div class="griffin-controls"><label>Task query <select data-task><option value="A">A · recent form</option><option value="B">B · experience</option></select></label><label>Column order <select data-order><option value="same">Original pairs</option><option value="reverse">Reversed pairs</option></select></label><label>Experience cell <select data-mask><option value="keep">Visible</option><option value="hide">Masked</option></select></label><button type="button" data-reset>Reset</button></div><div class="griffin-grid"><div class="griffin-card"><h3>Query → scores → weights</h3><div data-query></div><table><thead><tr><th>Cell</th><th>Score</th><th>Weight</th></tr></thead><tbody data-rows></tbody></table></div><div class="griffin-card"><h3>Weighted cell values</h3><p>recent form [2,0]<br>experience [0,4]</p><output aria-live="polite"></output><p class="baseline">Baseline A, visible: [1.462, 1.076]. Values and metadata remain paired when reordered.</p></div></div>';
 function draw(){const s=attention(host.querySelector('[data-task]').value,host.querySelector('[data-order]').value,host.querySelector('[data-mask]').value==='hide');host.querySelector('[data-query]').textContent='Q = '+fmt(s.q)+'; divide dot products by √2';host.querySelector('[data-rows]').innerHTML=s.rows.map((r,i)=>'<tr><td>'+r.name+'</td><td>'+ (r.hidden?'masked':s.logits[i].toFixed(3))+'</td><td>'+s.w[i].toFixed(3)+'</td></tr>').join('');host.querySelector('output').textContent='Cell readout '+fmt(s.out);host.dataset.result=JSON.stringify(s.out);}
 host.querySelectorAll('select').forEach(x=>x.addEventListener('change',draw));host.querySelector('[data-reset]').onclick=()=>{host.querySelector('[data-task]').value='A';host.querySelector('[data-order]').value='same';host.querySelector('[data-mask]').value='keep';draw();};draw();
}
function mountRelations(host){
 host.className='griffin-board';host.innerHTML='<h3>Can one crowded relation drown out another?</h3><p>Repeat both results neighbours together; then move the owner cutoff. Embeddings and relation weights stay fixed.</p><div class="griffin-controls"><label>Results repetitions <select data-repeat><option>1</option><option>2</option><option>4</option></select></label><label>Owner cutoff <select data-cutoff><option>5</option><option>7</option></select></label><button type="button" data-reset>Reset</button></div><div class="griffin-grid"><div class="griffin-card"><h3>Eligibility → within-relation mean</h3><p>Results: [2,4], [4,2] at time 4.<br>Teammate: [8,1] at time 4.<br>Later teammate: [20,10] at time 6.</p><p data-count></p></div><div class="griffin-card"><h3>Relation weighting → coordinate-wise max</h3><p data-messages class="equation"></p><output aria-live="polite"></output><p data-flat></p></div></div><p class="baseline">Baseline: cutoff 5, one copy → Griffin [4.000, 3.000], flat mean [3.333, 2.667]. Strict time &lt; cutoff; equality is excluded.</p>';
 function draw(){const s=relations(+host.querySelector('[data-repeat]').value,+host.querySelector('[data-cutoff]').value);host.querySelector('[data-count]').textContent=s.count+' eligible neighbours; '+s.excluded+' excluded by time.';host.querySelector('[data-messages]').textContent='Results '+fmt(s.messages[0])+'; teammate '+fmt(s.messages[1]);host.querySelector('output').textContent='Relation max '+fmt(s.out);host.querySelector('[data-flat]').textContent='Flat neighbour mean '+fmt(s.flat);host.dataset.result=JSON.stringify(s);}
 host.querySelectorAll('select').forEach(x=>x.addEventListener('change',draw));host.querySelector('[data-reset]').onclick=()=>{host.querySelector('[data-repeat]').value='1';host.querySelector('[data-cutoff]').value='5';draw();};draw();
}
global.GriffinTrace={attention,relations,mountCells,mountRelations};
})(window);
