/* Target-only interventions in a six-row, standardized L2 neighborhood. */
(()=>{'use strict';
const board=document.querySelector('[data-b05="retrieval"]');
if(board){const policy=board.querySelector('[name=policy]'),target=board.querySelector('[name=target]'),count=board.querySelector('[name=count]'),out=board.querySelector('output');
function update(){const xs=[0,.1,.2,.3,1,2],ys=target.value==='original'?[0,100,0,100,0,100]:[0,0,100,0,100,100];
const sd=a=>{const m=a.reduce((s,v)=>s+v,0)/a.length;return Math.sqrt(a.reduce((s,v)=>s+(v-m)**2,0)/a.length)||1;};
const sx=sd(xs),sy=sd(ys),ds=xs.map((x,i)=>(x/sx)**2+(policy.value==='included'?((ys[i]-ys[0])/sy)**2:0));
const picked=xs.map((_,i)=>i).sort((a,b)=>ds[a]-ds[b]||a-b).slice(0,+count.value);out.dataset.ids=JSON.stringify(picked);
out.textContent=`Selected row IDs: [${picked.join(', ')}]. ${policy.value==='excluded'?'Target removed: changing target values cannot change this distance.':'Target included: selection can carry information from the target.'} Baseline (target removed, k=3): [0, 1, 2].`;
board.querySelector('tbody').innerHTML=xs.map((x,i)=>`<tr data-picked="${picked.includes(i)}"><td>${i}${i===0?' · anchor':''}</td><td>${x}</td><td>${ys[i]}</td><td>${ds[i].toFixed(4)}</td></tr>`).join('');}
for(const el of [policy,target,count])el.addEventListener('change',update);board.querySelector('button').addEventListener('click',()=>{policy.value='excluded';target.value='original';count.value='3';update();});update();}
const quiz=document.querySelector('[data-b05="quiz"]');if(quiz){const out=quiz.querySelector('output');quiz.querySelectorAll('input').forEach(el=>el.addEventListener('change',()=>{out.dataset.correct=String(el.value==='remove');out.textContent=el.value==='remove'?'Correct. Remove the target before fitting the retrieval representation or index. Deleting it after selection cannot undo selection information.':'Try tracing the distance calculation: a target can influence chosen rows even if it never reaches the transformer.';}));quiz.querySelector('button').addEventListener('click',()=>{quiz.querySelectorAll('input').forEach(el=>el.checked=false);out.removeAttribute('data-correct');out.textContent='Choose before revealing feedback.';});}
})();
