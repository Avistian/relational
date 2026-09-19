/* Joint color dictionaries are essential: independent color IDs are incomparable. */
(function(global){
'use strict';
const colors=['#2563eb','#d97706','#059669','#9333ea','#db2777','#475569'];
const pairs={
 path:{names:['Path: four nodes','Star: four nodes'],edges:[[[0,1],[1,2],[2,3]],[[0,1],[0,2],[0,3]]],n:4},
 collision:{names:['Cycle: six nodes','Two disjoint triangles'],edges:[[[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]],[[0,1],[1,2],[2,0],[3,4],[4,5],[5,3]]],n:6}
};
function trace(mode,round){
 const p=pairs[mode]||pairs.path;round=Math.max(0,Math.min(4,Math.trunc(Number(round)||0)));
 let state=[Array(p.n).fill(0),Array(p.n).fill(0)],signatures=state.map(a=>a.map(()=> 'initial label'));
 for(let k=0;k<round;k++){
  signatures=state.map((s,g)=>s.map((c,v)=>JSON.stringify([c,p.edges[g].flatMap(([a,b])=>a===v?[s[b]]:b===v?[s[a]]:[]).sort((a,b)=>a-b)])));
  const keys=[...new Set(signatures.flat())].sort();state=signatures.map(a=>a.map(s=>keys.indexOf(s)));
 }
 const hist=state.map(s=>{let h={};s.forEach(c=>h[c]=(h[c]||0)+1);return h;});
 return {state,signatures,hist,round,distinguished:JSON.stringify(hist[0])!==JSON.stringify(hist[1])};
}
function graphSVG(mode,g,state){
 const p=pairs[mode];let pos=Array.from({length:p.n},(_,i)=>[140+74*Math.cos(2*Math.PI*i/p.n-Math.PI/2),100+74*Math.sin(2*Math.PI*i/p.n-Math.PI/2)]);
 if(mode==='path'){pos=g===0?[[38,100],[105,100],[172,100],[239,100]]:[[140,100],[60,44],[220,44],[140,174]];}
 if(mode==='collision'&&g===1){pos=[[64,36],[25,126],[103,126],[216,36],[177,126],[255,126]];}
 return '<svg viewBox="0 0 280 210" role="img" aria-label="'+p.names[g]+'">'+p.edges[g].map(([a,b])=>`<line x1="${pos[a][0]}" y1="${pos[a][1]}" x2="${pos[b][0]}" y2="${pos[b][1]}" stroke="#94a3b8" stroke-width="3"/>`).join('')+pos.map(([x,y],i)=>`<circle cx="${x}" cy="${y}" r="15" fill="${colors[state[i]%colors.length]}"/><text x="${x}" y="${y+5}" text-anchor="middle" fill="white" font-size="13">${state[i]}</text><text x="${x}" y="${y+30}" text-anchor="middle" font-size="11">node ${i}</text>`).join('')+'</svg>';
}
function mount(el){
 if(!el)return;el.classList.add('wl-panel');
 el.innerHTML='<label>Graph pair <select><option value="path">Path versus star</option><option value="collision">Cycle versus two triangles</option></select></label><label>Refinement round <input type="range" min="0" max="4" value="0" step="1"></label><button type="button">Reset</button><p class="wl-baseline">Baseline round 0: all nodes have color 0; the two graph histograms agree.</p><div class="wl-graphs"></div><output aria-live="polite"></output><details><summary>Inspect node signatures</summary><pre></pre></details>';
 const select=el.querySelector('select'),slider=el.querySelector('input');
 function draw(){const mode=select.value,r=trace(mode,slider.value);slider.value=r.round;
  el.querySelector('.wl-graphs').innerHTML=r.state.map((s,g)=>'<section><strong>'+pairs[mode].names[g]+'</strong>'+graphSVG(mode,g,s)+'<p>Color counts: '+Object.entries(r.hist[g]).map(([c,n])=>`${c} × ${n}`).join(', ')+'</p></section>').join('');
  el.querySelector('output').textContent=`Round ${r.round}: ${r.distinguished?'different histograms — distinguished':'same histograms — inconclusive'}.`;
  el.querySelector('pre').textContent=r.signatures.map((ss,g)=>pairs[mode].names[g]+'\n'+ss.map((s,i)=>`node ${i}: ${s} → color ${r.state[g][i]}`).join('\n')).join('\n\n');
 }
 select.addEventListener('change',draw);slider.addEventListener('input',draw);el.querySelector('button').onclick=()=>{select.value='path';slider.value=0;draw();};
 const api={setState:(mode,round)=>{select.value=pairs[mode]?mode:'path';slider.value=Math.max(0,Math.min(4,round));draw();}};el.wlAPI=api;draw();return api;
}
global.WLViz={trace,mount};
})(window);
