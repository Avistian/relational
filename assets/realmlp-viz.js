/* Expected states: preprocess z=2 →1.664; NTP s=2 → preactivation=-.5;
   schedule t=.25 uses factors 6/1/.1, with exact valleys at 0,1/15,3/15,7/15,1.
   Sliders clamp finite inputs; reset restores the pictured state. All values
   recompute from one pure function, exported for arithmetic verification. */
(function(g){'use strict';
function compute(mode,v){
 if(mode==='preprocess'){v=Math.max(-12,Math.min(12,v));return {v,raw:v,smooth:v/Math.sqrt(1+(v/3)**2),hard:Math.max(-3,Math.min(3,v))};}
 if(mode==='ntp'){v=Math.max(0,Math.min(4,v));return {v,dot:v-4,scaled:(v-4)/2,output:(v-4)/2+.5,baseline:-1};}
 v=Math.max(0,Math.min(1,v));const f=.5*(1-Math.cos(2*Math.PI*Math.log2(1+15*v)));return {v,f,scale:.04*6*f,weight:.04*f,bias:.004*f};
}
function mount(el,mode){
 el.className='realmlp-viz';const initial=mode==='schedule'?.25:2;
 const text=mode==='preprocess'?'Robust-scaled coordinate z':mode==='ntp'?'First learned feature scale s₁':'Fraction of training steps t';
 const label=document.createElement('label');label.textContent=text;
 const input=document.createElement('input');input.type='range';input.min=mode==='preprocess'?-12:0;input.max=mode==='schedule'?1:mode==='ntp'?4:12;input.step=mode==='schedule'?.001:.1;input.value=initial;
 label.appendChild(input);el.appendChild(label);
 const out=document.createElement('output');out.setAttribute('aria-live','polite');el.appendChild(out);
 const chart=document.createElement('div');chart.className='rv-bars';el.appendChild(chart);
 function render(){const r=compute(mode,Number(input.value));
  if(mode==='preprocess'){out.textContent=`z = ${r.v.toFixed(2)} → smooth = ${r.smooth.toFixed(3)}. Hard clip = ${r.hard.toFixed(3)}; no clip = ${r.raw.toFixed(2)}. Median and scale stay fixed.`;chart.innerHTML='';}
  else if(mode==='ntp'){out.textContent=`h = [${r.v.toFixed(1)}, 2, −1, 0]; W[:,j] = [1, −1, 2, 1]; bias = 0.5. Dot = ${r.dot.toFixed(2)} → divide by √4 = ${r.scaled.toFixed(2)} → add bias = ${r.output.toFixed(2)}. Identity-scale baseline = ${r.baseline.toFixed(2)}.`;chart.innerHTML='';}
  else {out.textContent=`t = ${r.v.toFixed(3)}; multiplier = ${r.f.toFixed(4)}. Base classification learning rate stays 0.04. These are step sizes, not measured parameter changes.`;chart.innerHTML='';[['Feature scale ×6',r.scale],['Weights ×1',r.weight],['Biases ×0.1',r.bias]].forEach(([name,v])=>{const b=document.createElement('div');b.className='rv-bar';b.textContent=name+': '+v.toFixed(5);b.style.background=`linear-gradient(to right,#d8e8ed ${100*v/.24}%,transparent ${100*v/.24}%)`;chart.appendChild(b);});}
 }
 input.addEventListener('input',render);const reset=document.createElement('button');reset.type='button';reset.textContent='Reset';reset.addEventListener('click',()=>{input.value=initial;render();});el.appendChild(reset);render();
 return {set(v){input.value=compute(mode,Number.isFinite(v)?v:initial).v;render();},input};
}
g.RealMLPViz={compute,mount};
})(typeof window==='undefined'?globalThis:window);
