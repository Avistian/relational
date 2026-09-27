/* Synthetic defaults: FE second prediction 14 => mean gap 0.
   Cluster draw A+B => pooled mean 2; A+A=>0; B+B=>6. */
(() => {
 const pair=document.getElementById('l137-pair');
 if(pair){
 pair.innerHTML='<label for="l137-pred">FE prediction on query B <output>14</output></label><input id="l137-pred" type="range" min="6" max="14" value="14" step="1"><p>Fixed targets [5,10], GNN [8,8], first FE prediction 6. Baseline: FE second prediction 14 → mean gap 0.</p><p class="audit-result" aria-live="polite"></p><button type="button">Reset example</button>';
 const input=pair.querySelector('input');function update(){const loss=Math.abs(+input.value-10),d=2-loss;pair.querySelector('output').textContent=input.value;pair.querySelector('.audit-result').textContent=`GNN losses [3,2]; FE losses [1,${loss}]. Paired differences [+2,${d}]. Mean GNN − FE = ${((2+d)/2).toFixed(2)}.`;}
 input.addEventListener('input',update);pair.querySelector('button').addEventListener('click',()=>{input.value=14;update();});update();
 }
 const cluster=document.getElementById('l137-cluster');
 if(cluster){
 cluster.innerHTML='<label for="l137-draw">Resampled drivers</label><select id="l137-draw"><option value="AA">A + A</option><option value="AB" selected>A + B</option><option value="BB">B + B</option></select><p>Fixed driver A: [0, 0], driver B: [6]. Baseline A+B → query-weighted mean 2.</p><p class="audit-result" aria-live="polite"></p><button type="button">Reset draw</button>';
 const input=cluster.querySelector('select');function update(){const map={AA:[0,4,0],AB:[6,3,2],BB:[12,2,6]},v=map[input.value];cluster.querySelector('.audit-result').textContent=`Whole-driver draw ${input.value}: loss sum ${v[0]} / ${v[1]} queries = ${v[2]}.`;}
 input.addEventListener('change',update);cluster.querySelector('button').addEventListener('click',()=>{input.value='AB';update();});update();
 }
 if(window.RetrievalBank)RetrievalBank.mount(document.getElementById('warmup'),{upTo:137,count:3});
 if(window.Teachback)Teachback.mount(document.getElementById('l137-teachback'),{prompt:'Why does a large validation slice gap not identify a broken GNN mechanism?',points:['Selection makes validation gaps optimistic','Test support can vanish','Repeated queries share drivers and races','Pipelines differ beyond their architecture','An intervention should isolate the proposed cause'],model:'A validation-selected group locates an association. Its errors do not isolate architecture, optimization or feature access. I must inspect test support, preserve paired query identities, state conditional uncertainty and run a controlled repair experiment before claiming a mechanism.'});
})();
