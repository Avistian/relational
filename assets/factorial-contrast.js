/** Synthetic MAE explorer. Defaults: gains .4/.7, interaction +.3.
 * Equal-gain preset: interaction 0. GT-best preset: interaction -.5.
 * All controls describe invented numbers; measured evidence never changes. */
(function(global){
'use strict';
function contrast(values){
 if(values.length!==4||values.some(x=>typeof x!=='number'||!Number.isFinite(x)||x<0||x>6))throw new Error('Four finite scores in [0,6] required');
 const mp=values[0]-values[1],gt=values[2]-values[3];return {mp:mp,gt:gt,interaction:gt-mp};
}
function mount(host){
 host.classList.add('factorial-contrast');
 host.innerHTML='<p class="fc-scope"><strong>Synthetic scores only.</strong> Predict the interaction before moving a control. No trained-model results change.</p><label>Example <select data-preset><option value="extra">Extra transformer benefit</option><option value="equal">Equal pretraining benefit</option><option value="best">Transformer best; smaller benefit</option></select></label><div class="fc-controls"></div><div class="fc-bars"></div><output aria-live="polite"></output><button type="button" data-reset>Reset example</button>';
 const names=['MP scratch','MP pretrained','GT scratch','GT pretrained'];const presets={extra:[4,3.6,3.9,3.2],equal:[4,3.6,3.9,3.5],best:[5,4,3,2.5]};
 const controls=host.querySelector('.fc-controls');
 names.forEach((name,i)=>{const label=document.createElement('label');label.innerHTML=name+' MAE: <span data-score="'+i+'"></span><input type="range" min="0" max="6" step="0.1" value="'+presets.extra[i]+'" data-index="'+i+'" aria-label="'+name+' MAE">';controls.appendChild(label);});
 const sliders=[...host.querySelectorAll('input')];
 function update(){
  const values=sliders.map(x=>Number(x.value)),c=contrast(values);
  values.forEach((v,i)=>host.querySelector('[data-score="'+i+'"]').textContent=v.toFixed(1));
  host.querySelector('.fc-bars').innerHTML=names.map((n,i)=>'<div><span>'+n+' · '+values[i].toFixed(1)+'</span><div class="fc-track"><div style="width:'+(values[i]/6*100)+'%" class="fc-bar fc-arm-'+Math.floor(i/2)+'"></div></div></div>').join('');
  const signed=x=>(x>1e-9?'+':'')+(Math.abs(x)<1e-9?0:x).toFixed(2);const out=host.querySelector('output');out.dataset.interaction=c.interaction.toFixed(6);
  out.textContent='MP gain: '+signed(c.mp)+' MAE; GT gain: '+signed(c.gt)+' MAE. Extra GT benefit: '+signed(c.interaction)+' MAE. '+(Math.abs(c.interaction)<1e-9?'Both backbones gain equally.':c.interaction>0?'The transformer gains more from pretraining.':'Message passing gains more from pretraining.')+' Lower absolute MAE is better; compare the four scores as well.';
 }
 sliders.forEach(x=>x.addEventListener('input',()=>{host.querySelector('[data-preset]').selectedIndex=-1;update();}));
 function preset(name){sliders.forEach((x,i)=>x.value=presets[name][i]);host.querySelector('[data-preset]').value=name;update();}
 host.querySelector('[data-preset]').addEventListener('change',e=>preset(e.target.value));host.querySelector('[data-reset]').addEventListener('click',()=>preset('extra'));update();
 return {setScores:values=>{contrast(values);sliders.forEach((x,i)=>x.value=values[i]);update();}};
}
global.FactorialContrast={contrast:contrast,mount:mount};
})(window);
