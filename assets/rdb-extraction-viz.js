/* Six-row worked example. FK edges: 1->0,0->2,3->2,4->3,5->1.
   Algorithm 1 selects [0,1,2,5]; two-hop selects [0,1,2,3,5].
   Available times: Order A=5, Line A=9, Order B=11. Cutoff extension is COURSE_ONLY. */
(function(global){'use strict';
const names=['Customer A','Order A','Country','Customer B','Order B','Line A'];
const edges=[[1,0],[0,2],[3,2],[4,3],[5,1]],times=[null,5,null,null,11,9];
function compute(mode,cutoff,filter){
 const eligible=i=>!filter||times[i]===null||times[i]<=cutoff;
 let chosen=new Set([0]);
 if(mode==='paper'){
  for(const reverse of [true,false]){let todo=[...chosen];while(todo.length){let u=todo.pop();for(const [a,b] of edges){let from=reverse?b:a,to=reverse?a:b;if(from===u&&!chosen.has(to)&&eligible(to)){chosen.add(to);todo.push(to);}}}}
 }else{let front=[0];for(let hop=0;hop<2;hop++){let next=[];for(const u of front)for(const [a,b] of edges){let v=a===u?b:b===u?a:null;if(v!==null&&!chosen.has(v)&&eligible(v)){chosen.add(v);next.push(v);}}front=next;}}
 return [...chosen].sort((a,b)=>a-b);
}
function mount(root){
 root.classList.add('rdb-extraction');root.innerHTML='<p><strong>Predict:</strong> will sharing Country bring Customer B into the graph?</p><label>Extraction <select data-mode><option value="paper">Algorithm 1: incoming then outgoing</option><option value="radius">Released Home Credit: two hops</option></select></label><label><input type="checkbox" data-filter> Apply course availability filter</label><label>Query time <input data-time type="range" min="0" max="12" value="7"> <span data-time-value>7</span></label><p>Fixed baseline: Algorithm 1 without filtering selects Customer A, Order A, Country, Line A.</p><div class="stream-scroll" tabindex="0"><table><thead><tr><th>Row</th><th>Available</th><th>Baseline</th><th>Selected now</th></tr></thead><tbody></tbody></table></div><output aria-live="polite"></output><p>The availability filter is a course extension. It is not part of the published Home Credit query.</p><button type="button">Reset trace</button>';
 const mode=root.querySelector('[data-mode]'),filter=root.querySelector('[data-filter]'),time=root.querySelector('[data-time]');
 function render(){const selected=compute(mode.value,+time.value,filter.checked),baseline=compute('paper',7,false);root.querySelector('[data-time-value]').textContent=time.value;root.querySelector('tbody').innerHTML=names.map((name,i)=>`<tr><th>${name}</th><td>${times[i]===null?'Timeless':times[i]}</td><td>${baseline.includes(i)?'Included':'Excluded'}</td><td>${selected.includes(i)?'Included':'Excluded'}</td></tr>`).join('');root.querySelector('output').textContent=`${selected.length} nodes: ${selected.map(i=>names[i]).join(', ')}.`;}
 for(const control of [mode,filter,time])control.addEventListener('input',render);
 root.querySelector('button').addEventListener('click',()=>{mode.value='paper';filter.checked=false;time.value='7';render();});render();
}
global.RDBExtractionViz={compute,mount};
})(window);
