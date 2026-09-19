/* Companion-recognition and subset-aggregation traces, independent of any lesson. */
(function(global){
'use strict';
function lossTrace(root){
 root.classList.add('contrastive-widget');
 root.innerHTML='<h3>Same vectors, different candidate sets</h3><p>Illustrative vectors: a₁=b₁=(1,0); a₂=b₂=(0,1). Positive cells are marked +; × means excluded.</p><label>Candidate layout <select aria-label="Candidate layout"><option value="scarf">SCARF: 2 clean anchors × 2 corrupt candidates</option><option value="subtab">SubTab: 4 anchors × 4 candidates, self excluded</option></select></label><label>Temperature τ <input aria-label="Temperature" type="range" min="0.1" max="2" step="0.1" value="1"></label><div class="result-scroll matrix"></div><output aria-live="polite"></output><button type="button">Reset to SCARF, τ=1</button>';
 const mode=root.querySelector('select'), slider=root.querySelector('input'), out=root.querySelector('output');
 function update(){
  const tau=Number(slider.value), sub=mode.value==='subtab', n=sub?4:2, names=sub?['a₁','a₂','b₁','b₂']:['b₁','b₂'];
  let table='<table><caption>Cosine / τ before softmax</caption><thead><tr><th>anchor</th>'+names.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>';
  for(let i=0;i<n;i++){table+='<tr><th>'+(sub?names[i]:['a₁','a₂'][i])+'</th>';for(let j=0;j<n;j++){const self=sub&&i===j, pos=sub?j===(i+2)%4:i===j;table+='<td class="'+(self?'masked':pos?'positive':'')+'">'+(self?'×':((i%2===j%2?1:0)/tau).toFixed(2)+(pos?' +':''))+'</td>';}table+='</tr>';}
  root.querySelector('.matrix').innerHTML=table+'</tbody></table>';
  const neg=sub?2:1, p=1/(1+neg*Math.exp(-1/tau)), loss=-Math.log(p);
  out.textContent='τ = '+tau.toFixed(1)+' · 1 positive + '+neg+' negative'+(neg>1?'s':'')+' per anchor. P(companion) = '+p.toFixed(4)+'; cross-entropy = '+loss.toFixed(4)+'. Baseline SCARF at τ=1: 0.3133. '+(sub?'Self is excluded; companion remains.':'The diagonal is the companion, so keep it.');
 }
 mode.addEventListener('change',update);slider.addEventListener('input',update);root.querySelector('button').onclick=()=>{mode.value='scarf';slider.value='1';update();};update();
}
function subsetTrace(root){
 root.classList.add('contrastive-widget');root.innerHTML='<h3>Follow one row through aggregation</h3><p>Fixed illustrative latent vectors. Toggle availability; at least one view must remain.</p>';
 const vectors=[[1,3],[5,7],[3,2]];
 vectors.forEach((v,i)=>{const label=document.createElement('label');label.innerHTML='<input type="checkbox" checked> View '+(i+1)+': h = ('+v.join(', ')+')';root.appendChild(label);});
 const out=document.createElement('output');out.setAttribute('aria-live','polite');root.appendChild(out);
 const inputs=Array.from(root.querySelectorAll('input'));
 function update(event){let selected=inputs.map((x,i)=>x.checked?i:-1).filter(i=>i>=0);if(!selected.length){event.target.checked=true;selected=[inputs.indexOf(event.target)];}
 const sums=[0,1].map(j=>selected.reduce((s,i)=>s+vectors[i][j],0));out.textContent='Sum = ('+sums.join(', ')+'); divide by '+selected.length+' view(s) → representation = ('+sums.map(v=>(v/selected.length).toFixed(2)).join(', ')+'). Baseline with all three: (3.00, 4.00). Shape stays [rows, hidden]; no row mixing.';}
 inputs.forEach(x=>x.addEventListener('change',update));update();
}
global.ContrastiveViews={lossTrace,subsetTrace};
})(window);
