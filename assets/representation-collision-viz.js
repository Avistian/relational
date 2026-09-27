/* Fixed dated amounts Ada [10,30,50], Bo [50,30,10].
   Default coarse features collide; adding time-weighted sum gives 220/140.
   The finite two-row deterministic accuracy ceiling changes .5 -> 1.
   This is a representation calculation, not trained generalization. */
(function(global){'use strict';
function compute(augmented){
 const rows=[[10,30,50],[50,30,10]].map(a=>{
  const x=[a.length,a.reduce((s,v)=>s+v,0),a.reduce((s,v)=>s+v,0)/a.length,Math.max(...a)];
  if(augmented)x.push(a.reduce((s,v,i)=>s+(i+1)*v,0));return x;
 });
 return {rows,ceiling:JSON.stringify(rows[0])===JSON.stringify(rows[1])?.5:1};
}
function mount(el){
 el.classList.add('stream-widget');
 el.innerHTML='<p><strong>Predict first:</strong> will a richer flat feature separate these same two rows?</p><label><input type="checkbox" data-augment> Add sum(time × amount)</label><div class="stream-scroll" tabindex="0"><table><thead></thead><tbody></tbody></table></div><output aria-live="polite"></output><p>Fixed baseline: the four-feature ceiling is 50%. Labels are authored rising/falling categories, not customer outcomes.</p><button type="button">Reset representation</button>';
 const control=el.querySelector('input');
 function draw(){const state=compute(control.checked),names=['Customer','Count','Total','Mean','Max'];if(control.checked)names.push('Time × amount');
  el.querySelector('thead').innerHTML='<tr>'+names.map(x=>'<th scope="col">'+x+'</th>').join('')+'</tr>';
  el.querySelector('tbody').innerHTML=state.rows.map((x,i)=>'<tr><th scope="row">'+['Ada','Bo'][i]+'</th>'+x.map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('');
  el.querySelector('output').textContent='Observed representation ceiling: '+(100*state.ceiling)+'%. '+(control.checked?'220 and 140 distinguish the rows; training/generalization is untested.':'Identical features force the same prediction; one of two labels must be wrong.');
 }
 control.addEventListener('change',draw);el.querySelector('button').addEventListener('click',()=>{control.checked=false;draw();});draw();
}
global.RepresentationCollisionViz={mount,compute};
})(window);
