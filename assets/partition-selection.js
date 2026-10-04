/* Finite course evidence, not Table 1: partition 8 subsets x 3 lambdas;
   search 3 worlds x 3 lambdas x 3 methods; graph 2 value identities x 2 row features.
   Every output is derived from the same archived Python experiment. */
(function(global){
'use strict';
const E=global.B16_EVIDENCE,fmt=x=>Number(x).toFixed(3),name=s=>s.length?s.join(' + '):'∅';
if(!E)return;
function getScore(world,penalty,cols){return E.subsets.find(r=>r.world===world&&r.penalty===penalty&&JSON.stringify(r.cols)===JSON.stringify(cols));}
function partition(el){
 const sub=el.querySelector('[name=subset]'),lam=el.querySelector('[name=penalty]'),out=el.querySelector('output');
 function draw(){const cols=sub.value?sub.value.split(','):[],p=Number(lam.value),r=getScore('signal',p,cols),rows=E.datasets.signal.train;
 const groups=new Map();rows.forEach((row,i)=>{const k=JSON.stringify(cols.map(c=>row[c]));if(!groups.has(k))groups.set(k,[]);groups.get(k).push(i);});
 el.querySelector('[data-blocks]').innerHTML=[...groups.values()].map((ids,b)=>'<div class="ps-block"><strong>Block '+(b+1)+': rows '+ids.join(', ')+'</strong><small>n = '+ids.length+'; training P(1) = '+fmt(ids.reduce((a,i)=>a+rows[i].y,0)/ids.length)+'</small></div>').join('');
 out.dataset.j=r.J;out.textContent='S = '+name(cols)+'. Validation probabilities: '+r.probabilities.map(x=>x.toFixed(1)).join(', ')+'. Risk '+fmt(r.risk)+' + '+p+' × occupancy '+fmt(r.omega)+' = J '+fmt(r.J)+'.';
 el.querySelector('[data-baseline]').textContent='Fixed baseline: empty subset has risk 0.250 and occupancy 0.354; at λ = '+p+', J = '+fmt(getScore('signal',p,[]).J)+'. Same training and validation rows throughout.';
 }
 sub.addEventListener('change',draw);lam.addEventListener('change',draw);el.querySelector('button').onclick=()=>{sub.value='A';lam.value='0.5';draw();};draw();
}
function search(el){
 const world=el.querySelector('[name=world]'),lam=el.querySelector('[name=penalty]'),method=el.querySelector('[name=method]'),out=el.querySelector('output');
 function draw(){const w=world.value,p=Number(lam.value),m=method.value,r=E.selections.find(x=>x.world===w&&x.penalty===p&&x.method===m),best=E.selections.find(x=>x.world===w&&x.penalty===p&&x.method==='exhaustive');
 el.querySelector('[data-lattice]').innerHTML=E.subsets.filter(x=>x.world===w&&x.penalty===p).map(x=>'<div class="ps-subset" data-selected="'+(JSON.stringify(x.cols)===JSON.stringify(r.selected))+'"><strong>'+name(x.cols)+'</strong>J = '+fmt(x.J)+'<small>risk '+fmt(x.risk)+'; Ω '+fmt(x.omega)+'</small></div>').join('');
 const path=m==='exhaustive'?'All eight subsets → '+name(r.selected):r.trace.filter(x=>x.action==='init'||x.action==='accept').map(x=>name(x.selected||x.proposed)).join(' → ');
 el.querySelector('[data-path]').textContent=path+(m==='exhaustive'?'':' → stop');out.dataset.selected=JSON.stringify(r.selected);out.dataset.gap=r.J-best.J;
 out.textContent='Selected '+name(r.selected)+'; J = '+fmt(r.J)+'. Exhaustive best '+name(best.selected)+'; J = '+fmt(best.J)+'. Search gap = '+fmt(r.J-best.J)+'. Held-out test Brier = '+fmt(r.test_brier)+'.';
 }
 [world,lam,method].forEach(x=>x.addEventListener('change',draw));el.querySelector('button').onclick=()=>{world.value='xor';lam.value='0.5';method.value='forward';draw();};draw();
}
function graph(el){
 const typed=el.querySelector('[name=identity]'),features=el.querySelector('[name=features]'),out=el.querySelector('output');
 const palette=['#246879','#956312','#7751a1','#296d44'];
 function draw(){const has=typed.value==='typed',full=features.value==='B',key=(has?'typed':'anonymous')+(full?'-B':''),blocks=E.graph_states[key],colors=Array(8);
 blocks.forEach((b,c)=>b.forEach(i=>colors[i]=c));const pts=Array.from({length:8},(_,i)=>[40+(i%4)*86.5,i<4?45:285]);
 let svg='<svg viewBox="0 0 340 320" role="img" aria-label="Eight rows connected to their A value node. Colours show stable row classes.">';
 pts.forEach(([x,y],i)=>{svg+='<line x1="'+x+'" y1="'+y+'" x2="170" y2="'+(i<4?100:220)+'" stroke="#91a8af" stroke-width="2"/>';});
 [0,1].forEach(a=>{const y=a?220:100;svg+='<rect x="112" y="'+(y-18)+'" width="116" height="36" rx="7" fill="#fff" stroke="#236775" stroke-width="2"/><text x="170" y="'+(y+5)+'" text-anchor="middle" font-size="14" fill="#172d38">'+(has?'(A, '+a+')':'A type; x = 0')+'</text>';});
 pts.forEach(([x,y],i)=>{svg+='<circle cx="'+x+'" cy="'+y+'" r="20" fill="'+palette[colors[i]]+'"/><text x="'+x+'" y="'+(y+5)+'" text-anchor="middle" fill="white" font-size="14">r'+i+'</text><text x="'+x+'" y="'+(i<4?12:314)+'" text-anchor="middle" fill="#172d38" font-size="12">'+(full?'B='+Math.floor(i/2)%2:'same row x')+'</text>';});
 svg+='<text x="170" y="153" text-anchor="middle" fill="#172d38" font-size="14">All edges have column type A</text><text x="170" y="175" text-anchor="middle" fill="#172d38" font-size="13">Node IDs are labels for us, not features</text></svg>';
 el.querySelector('[data-graph]').innerHTML=svg;out.dataset.blocks=blocks.length;out.textContent=blocks.length+' stable row class'+(blocks.length===1?'':'es')+': '+blocks.map(b=>'{'+b.join(', ')+'}').join(' · ')+'. '+(has&&!full?'Lemma 1 assumptions: typed values, erased row features.':'Feature assumptions changed; do not reuse the projection equality automatically.');
 }
 [typed,features].forEach(x=>x.addEventListener('change',draw));el.querySelector('button').onclick=()=>{typed.value='typed';features.value='none';draw();};draw();
}
document.querySelectorAll('[data-partition-selection]').forEach(partition);document.querySelectorAll('[data-greedy-search]').forEach(search);document.querySelectorAll('[data-incidence-boundary]').forEach(graph);
if(global.RetrievalBank)RetrievalBank.mount(document.getElementById('b16-warmup'),{upTo:200.16,count:3});
if(global.Predict)Predict.mount(document.getElementById('b16-predict'),{prompt:'A unique key makes every training block pure. What happens on entirely new validation keys?',options:[{label:'Predict the training marginal',value:'marginal'},{label:'Predict the memorized labels',value:'memory'}],correct:'marginal',reveal:'No validation key has a fitted block. Every prediction falls back to the training marginal, 0.5 here, giving Brier loss 0.25.'});
if(global.Teachback)Teachback.mount(document.getElementById('b16-teachback'),{prompt:'Defend the XOR decision at λ=0.5: why does forward search fail, when is abstention justified, and which graph assumptions limit the partition claim?',points:['Counts fit on train; selection uses validation; test labels stay hidden','Forward misses the informative pair; exhaustive finds a lower J','Empty can be the exact penalized optimum, not refusal to predict','Lemma 1 erases row features and exposes typed value identities','Finite verification and source parity do not complete Table 1'],model:'For XOR, a single column increases J, so forward search stops before reaching the better pair. This is an optimization failure at λ=0.5; at λ=1 the empty set has the best penalized score. An empty graph still permits row-local prediction. Projection blocks equal stable row colours only under the lemma’s feature assumptions. Our complete finite test and narrow source comparisons do not fill the missing Table 1 protocol.'});
})(window);
