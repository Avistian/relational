/** Reusable subjective-priority explorer. Default weights 1/1/1, impact 4.
 * Expected: temporal 18.67 > composite 15 > transfer 10. Impact 3 makes composite lead.
 * Weights express preferences; changing them never authorizes a model run. */
(function(global){'use strict';
function mount(host,cases){
 host.classList.add('rp-widget');
 const controls=document.createElement('div');controls.className='rp-controls';host.appendChild(controls);
 const inputs=[];
 ['Data access weight','Implementation weight','Compute weight','Temporal impact judgment'].forEach((title,i)=>{
  const label=document.createElement('label'),span=document.createElement('span'),input=document.createElement('input'),value=document.createElement('output');
  span.textContent=title;input.type='range';input.min=1;input.max=i===3?5:3;input.step=1;input.value=i===3?4:1;input.id='rp-'+i;input.setAttribute('aria-label',title);input.dataset.control=String(i);
  label.htmlFor=input.id;label.append(span,value,input);controls.append(label);inputs.push(input);
  input.addEventListener('input',update);
 });
 const readout=document.createElement('div');readout.className='rp-readout';readout.setAttribute('aria-live','polite');host.append(readout);
 const reset=document.createElement('button');reset.textContent='Reset judgments';reset.type='button';reset.dataset.reset='';host.append(reset);
 reset.addEventListener('click',()=>{inputs.forEach((input,i)=>input.value=i===3?4:1);update();});
 function update(){
  const w=inputs.slice(0,3).map(x=>Number(x.value)),impact=Number(inputs[3].value);
  inputs.forEach(input=>input.previousSibling.value=input.value);
  const scores=cases.map(c=>({id:c.id,title:c.title,score:(c.id==='temporal'?impact:c.impact)*c.feasibility.reduce((sum,v,i)=>sum+v*w[i],0)/w.reduce((a,b)=>a+b,0)}));
  scores.sort((a,b)=>b.score-a.score||a.id.localeCompare(b.id));
  readout.replaceChildren();
  scores.forEach(c=>{const row=document.createElement('div');row.className='rp-score';const label=document.createElement('span');label.textContent=c.title;const number=document.createElement('strong');number.textContent=c.score.toFixed(2);const bar=document.createElement('meter');bar.min=0;bar.max=25;bar.value=c.score;bar.setAttribute('aria-label',c.title+' priority');row.append(label,number,bar);readout.append(row);});
  const leaders=scores.filter(c=>c.score===scores[0].score).map(c=>c.id).sort();
  const feedback=document.createElement('p');feedback.textContent=(leaders.length>1?'Tie: ':'First: ')+leaders.join(', ')+'. These are priorities for investigation. All full-run cost bounds remain unverified.';readout.append(feedback);
  host.dataset.result=JSON.stringify({weights:w,impact,scores,leaders});
 }
 update();
}
global.ResearchPriority={mount};
})(window);
