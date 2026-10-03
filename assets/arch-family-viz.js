/* Reusable HIN route explorer. Exact integer matrices; no model-score claim. */
(function(g){
'use strict';
function transpose(a){return a[0].map((_,j)=>a.map(r=>r[j]));}
function multiply(a,b){return a.map(r=>b[0].map((_,j)=>r.reduce((s,v,k)=>s+v*b[k][j],0)));}
function calculate(route,remove){const ap=[[1,1,0,0],[0,remove?0:1,1,0],[0,0,0,1],[0,0,0,0]],pv=[[1,0],[0,1],[0,1],[1,0]];const paper=multiply(ap,transpose(ap)),av=multiply(ap,pv),venue=multiply(av,transpose(av));return route==='paper'?paper:route==='venue'?venue:paper.map((r,i)=>r.map((v,j)=>Number(v>0&&venue[i][j]>0)));}
function mount(el){
 el.classList.add('hin-explorer');el.innerHTML='<p><strong>Change the route; keep the author set fixed.</strong></p><label>Meaning <select aria-label="Route meaning"><option value="paper">Shared paper: A–P–A</option><option value="venue">Shared venue: A–P–V–P–A</option><option value="both">Both predicates: branching pattern</option></select></label><p><label><input type="checkbox"> Remove Bo’s link to P1</label> <button type="button">Reset</button></p><output aria-live="polite"></output><div class="route-tables"></div>';
 const select=el.querySelector('select'),check=el.querySelector('input'),out=el.querySelector('output'),tables=el.querySelector('.route-tables');
 function table(m,title){const names=['Ada','Bo','Cy','Dee'];return '<table><caption>'+title+'</caption><thead><tr><th scope="col">Author</th>'+names.map(n=>'<th scope="col">'+n+'</th>').join('')+'</tr></thead><tbody>'+m.map((r,i)=>'<tr><th scope="row">'+names[i]+'</th>'+r.map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('')+'</tbody></table>';}
 function update(){let m=calculate(select.value,check.checked);out.textContent='Ada–Bo: '+m[0][1]+'; Ada–Cy: '+m[0][2]+'. '+(select.value==='both'?'Binary conjunction of two paths; not their sum.':'Counts retain distinct path instances.');tables.innerHTML=table(calculate('paper',false),'Fixed baseline: shared-paper counts')+table(m,'Current route: '+select.options[select.selectedIndex].text);}
 select.addEventListener('change',update);check.addEventListener('change',update);el.querySelector('button').addEventListener('click',()=>{select.value='paper';check.checked=false;update();});update();
 return {calculate,reset:()=>el.querySelector('button').click()};
}
g.ArchFamilyViz={mount,calculate};
})(window);

/* Three overlapping FM strategies. Separate mount preserves the HIN explorer above. */
(function(g){
'use strict';
const strategies={
 graph:{name:'Graph-native learning',steps:['Eligible linked rows','Cell / row encoder','Messages or attention on links','Task prediction'],prior:'Shared representations learned before the target task.',adapt:'Griffin fine-tunes; relational ICL systems can instead use support examples.',trace:'Ada remains connected to eligible order rows. Learned messages can retain distinctions absent from a chosen summary.',risk:'A performance difference alone cannot isolate architecture from data, tuning or pretraining.'},
 synthetic:{name:'Synthetic relational prior',steps:['Generate relational tasks','DFS feature construction','Pretrain context predictor','New support + query → prediction'],prior:'RDB-PFN learns from synthetic relational tasks.',adapt:'Frozen predictor consumes DFS-linearized support and query rows.',trace:'Ada becomes a feature row. Synthetic relational experience is in the predictor weights, not explicit graph messages at inference.',risk:'A generated prior may fail to cover a new database or target.'},
 reuse:{name:'Reuse a tabular foundation model',steps:['Eligible linked rows','Relational aggregation','Existing tabular model','Support + query → prediction'],prior:'Backend tabular pretraining already happened.',adapt:'No new relational pretraining; aggregation and context inference still cost work.',trace:'Ada becomes a feature row for an existing predictor. Summaries can carry useful relational signal.',risk:'Information discarded by the selected feature map cannot be recovered from that map alone.'}
};
function mountLandscape(el){
 el.classList.add('landscape-panel');
 el.innerHTML='<div class="controls"><label>Follow one strategy <select aria-label="Foundation model strategy"><option value="graph">Graph-native learning</option><option value="synthetic">Synthetic relational prior</option><option value="reuse">Reuse a tabular foundation model</option></select></label><button type="button">Reset map</button></div><div class="landscape-route" aria-live="polite"></div>';
 const select=el.querySelector('select'),out=el.querySelector('.landscape-route');
 function update(){const s=strategies[select.value];el.dataset.strategy=select.value;out.innerHTML='<h3>'+s.name+'</h3><ol class="route-steps">'+s.steps.map(x=>'<li>'+x+'</li>').join('')+'</ol><p><strong>Prior:</strong> '+s.prior+'</p><p><strong>Adaptation:</strong> '+s.adapt+'</p><p><strong>Trace Ada:</strong> '+s.trace+'</p><p><strong>Test next:</strong> '+s.risk+'</p>';}
 select.addEventListener('change',update);el.querySelector('button').addEventListener('click',()=>{select.value='graph';update();});update();
}
g.ArchFamilyViz.mountLandscape=mountLandscape;
g.ArchFamilyViz.landscapeStrategies=strategies;
})(window);
