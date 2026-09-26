/* Fixed synthetic embeddings expose task readouts, not benchmark predictions. */
(function(g){'use strict';
const z=[[1,2],[3,5],[7,11]];
function compute(task,reverse,reduction){
 if(task==='node')return {shape:'1 × 2',value:z[reverse?1:0],meaning:'Selected node vector before its learned class head.'};
 if(task==='link'){const u=z[reverse?1:0],v=z[reverse?0:1];return {shape:reduction==='ordered'?'1 × 4':'1 × 2',value:reduction==='ordered'?u.concat(v):u.map((x,i)=>x*v[i]),meaning:reduction==='ordered'?'Ordered endpoints permit asymmetric scores.':'Elementwise product forces endpoint-swap invariance.'};}
 const groups=reverse?[[0,1,2]]:[[0,2],[1]];
 return {shape:groups.length+' × 2',value:groups.map(ids=>[0,1].map(d=>ids.reduce((s,i)=>s+z[i][d],0)/(reduction==='sum'?1:ids.length))),meaning:reverse?'Incorrect for two independent graphs: graph IDs were discarded.':'One pooled vector per graph; graph 0 contains nodes 0 and 2.'};
}
function mount(el){
 el.className='repro-widget';el.innerHTML='<p><b>Predict first:</b> Which change alters meaning while keeping a plausible tensor shape?</p><p>Fixed vectors: z₀=[1,2], z₁=[3,5], z₂=[7,11]. These are synthetic.</p><label>Prediction unit <select data-task aria-label="Prediction unit"><option value="node">Node</option><option value="link">Link</option><option value="graph">Graph</option></select></label> <label>Readout <select data-readout aria-label="Readout"></select></label><p><label><input type="checkbox" data-change> <span data-change-label></span></label></p><p data-baseline></p><pre><output aria-live="polite"></output></pre><button type="button">Reset example</button>';
 const task=el.querySelector('[data-task]'),read=el.querySelector('[data-readout]'),change=el.querySelector('[data-change]'),out=el.querySelector('output');
 function draw(){const r=compute(task.value,change.checked,read.value);out.textContent='Shape: '+r.shape+'\nValue: '+JSON.stringify(r.value)+'\n'+r.meaning;}
 function setup(){change.checked=false;read.innerHTML=task.value==='link'?'<option value="product">Undirected product</option><option value="ordered">Directed concatenation</option>':task.value==='graph'?'<option value="mean">Within-graph mean</option><option value="sum">Within-graph sum</option>':'<option value="select">Select target</option>';el.querySelector('[data-change-label]').textContent=task.value==='node'?'Select node 1 instead of node 0':task.value==='link'?'Reverse the candidate endpoints':'Discard graph IDs (deliberate error)';el.querySelector('[data-baseline]').textContent='Baseline: '+(task.value==='node'?'node 0 → [1,2].':task.value==='link'?'0 → 1, product → [3,10].':'IDs [0,1,0], mean → [[4,6.5],[3,5]].');draw();}
 task.addEventListener('change',setup);read.addEventListener('change',draw);change.addEventListener('change',draw);el.querySelector('button').onclick=()=>{task.value='node';setup();};setup();
}
g.GraphPatterns={compute,mount};document.querySelectorAll('[data-graph-patterns]').forEach(mount);
})(window);
