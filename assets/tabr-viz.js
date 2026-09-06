/* Three independent mechanisms. Defaults: query=1, m=2, correction=.5,
 * cutoff=4. Test query endpoints 0/3, self exclusion, correction -1/1,
 * and label-availability cutoff 2/8. All values are synthetic, not scores.
 */
(function(g){'use strict';
const rows=[{id:'A',k:0,y:0},{id:'SELF',k:1,y:1},{id:'B',k:2,y:1},{id:'C',k:3,y:0}];
function compute(q=1,exclude=true,beta=.5){
 const legal=rows.filter(r=>!exclude||r.id!=='SELF').map(r=>({...r,s:-((q-r.k)**2)})).sort((a,b)=>b.s-a.s).slice(0,2);
 const z=legal.reduce((s,r)=>s+Math.exp(r.s),0);
 return legal.map(r=>({...r,w:Math.exp(r.s)/z,v:r.y+beta*(q-r.k)}));
}
function mount(node,mode){
 node.classList.add('tabr-viz');let q=1,exclude=true,beta=.5,cutoff=4;
 const controls=document.createElement('div'),body=document.createElement('div');node.append(controls,body);
 function slider(title,min,max,step,value,change){const label=document.createElement('label');label.textContent=title+' ';const input=document.createElement('input');input.type='range';Object.assign(input,{min,max,step,value});const out=document.createElement('span');out.textContent=value;input.addEventListener('input',()=>{out.textContent=input.value;change(+input.value);render();});label.append(input,out);controls.append(label);return input;}
 if(mode==='neighbors'){
  slider('Query key',0,3,.25,q,v=>q=v);
  const label=document.createElement('label'),check=document.createElement('input');check.type='checkbox';check.checked=true;check.addEventListener('change',()=>{exclude=check.checked;render();});label.append(check,document.createTextNode(' Exclude SELF by row identity'));controls.append(label);
 }else if(mode==='values'){slider('Correction coefficient β',-1,1,.1,beta,v=>beta=v);}
 else{slider('Prediction cutoff (day)',2,8,1,cutoff,v=>cutoff=v);}
 const reset=document.createElement('button');reset.textContent='Reset';reset.addEventListener('click',()=>{node.innerHTML='';mount(node,mode);});controls.append(reset);
 function render(){
  const baseline=compute(1,true,.5),current=compute(q,exclude,beta);
  if(mode==='neighbors'){
   body.innerHTML='<p>Predict which two rows survive. Distance ranking changes; stored labels stay fixed.</p><table><thead><tr><th>Row / key</th><th>−distance²</th><th>Weight</th></tr></thead><tbody>'+rows.map(r=>{const selected=current.find(x=>x.id===r.id);return '<tr class="'+(selected?'selected':r.id==='SELF'&&exclude?'excluded':'')+'"><td>'+r.id+' / '+r.k+'</td><td>'+(-((q-r.k)**2)).toFixed(2)+'</td><td>'+(selected?selected.w.toFixed(3):r.id==='SELF'&&exclude?'excluded':'outside top-2')+'</td></tr>';}).join('')+'</tbody></table><p>Selected label-only mean: <output>'+current.reduce((s,r)=>s+r.w*r.y,0).toFixed(3)+'</output>. This is an illustration, not TabR’s final probability.</p><p class="baseline">Baseline q=1, self excluded: A and B, weights 0.500 each, label-only mean 0.500. Including SELF gives it direct access to its target.</p>';
  }else if(mode==='values'){
   body.innerHTML='<p>Keys, neighbors and weights fixed: q=1; A key=0, y=0; B key=2, y=1. Scalar stand-in Wᵧ(y)=y, T(Δk)=βΔk.</p><table><thead><tr><th>Row</th><th>Δkey</th><th>Label + correction</th><th>Weighted value</th></tr></thead><tbody>'+current.map(r=>'<tr><td>'+r.id+'</td><td>'+(1-r.k)+'</td><td>'+r.y+' + '+(beta*(1-r.k)).toFixed(2)+' = '+r.v.toFixed(2)+'</td><td>'+(r.w*r.v).toFixed(3)+'</td></tr>').join('')+'</tbody></table><p>Retrieval sum: <output>'+current.reduce((s,r)=>s+r.w*r.v,0).toFixed(3)+'</output>.</p><p class="baseline">Baseline β=0.5: both corrected values are 0.500. In this symmetric example the sum stays 0.500 for every β: the opposite corrections cancel. Individual contributions still change. Actual T is a learned vector-valued nonlinear network.</p>';
  }else{
   const events=[['A',1,3],['B',2,6],['C',5,7]];
   body.innerHTML='<p>Each row has an event day and a later day when its label becomes known. Eligibility requires both to precede the prediction cutoff.</p><table><thead><tr><th>Row</th><th>Event</th><th>Label known</th><th>Eligible?</th></tr></thead><tbody>'+events.map(r=>'<tr class="'+(r[1]<cutoff&&r[2]<cutoff?'selected':'excluded')+'"><td>'+r[0]+'</td><td>'+r[1]+'</td><td>'+r[2]+'</td><td>'+(r[1]<cutoff&&r[2]<cutoff?'yes':'no')+'</td></tr>').join('')+'</tbody></table><p class="baseline">Baseline cutoff=4: only A is legal. B already happened, but its label is unavailable. This illustrates a deployment filter; the measured lab uses the authors’ IID splits.</p>';
  }
 }
 render();return {render};
}
g.TabRViz={mount,compute};
})(typeof window!=='undefined'?window:globalThis);
