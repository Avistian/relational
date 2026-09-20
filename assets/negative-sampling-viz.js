/* A deterministic intervention on a typed candidate distribution. */
window.NegativeSamplingViz={mount(root){
 if(!root)return;
 root.innerHTML=`<div class="viz-card"><h3>One user, three candidate destinations</h3><p>Observed: items 0 and 1. Candidates: 2, 3, 4. Scores: .2, 1.4, .7.</p><label>Proposal <select aria-label="Proposal"><option value="uniform">Uniform</option><option value="degree">Degree weighted</option><option value="hard">Hard: fixed pool [2,3,3,4]</option></select></label><p><label><input type="checkbox"> Exclude item 3 as an observed fitting link</label></p><button type="button">Reset</button><output aria-live="polite"></output><p>Baseline: each eligible item has probability 1/3. Hard mode shows one fixed draw, not its full probability law.</p></div>`;
 const select=root.querySelector('select'),check=root.querySelector('input'),out=root.querySelector('output');
 function update(){let ids=check.checked?[2,4]:[2,3,4],weights=ids.map(i=>select.value==='degree'?Math.pow(({2:0,3:8,4:1}[i])+1,.75):1),sum=weights.reduce((a,b)=>a+b,0);
 if(select.value==='hard'){out.textContent=`Selected item ${check.checked?4:3}; largest score in the eligible fixed pool. This is a training assumption, not a verified dislike.`;return;}
 out.textContent=ids.map((id,i)=>`item ${id}: ${(100*weights[i]/sum).toFixed(1)}%`).join(' · ');}
 select.addEventListener('change',update);check.addEventListener('change',update);root.querySelector('button').addEventListener('click',()=>{select.value='uniform';check.checked=false;update();});update();
}};
