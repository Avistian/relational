/* Context explorer: 2 databases × 3 models × 5 contexts. Fixed64 baseline.
   Claim explorer: 4 axes × 2 control states × 2 level counts × 2 extrapolation states. */
(function(global){'use strict';
const KS=[64,128,256,512,1024];
function claim(axis,controlled,levels,extrapolating){
 if(!controlled)return 'CONFOUNDED';if(levels<2)return 'INSUFFICIENT_LEVELS';
 if(extrapolating)return 'EXTRAPOLATION_NOT_ESTABLISHED';
 return axis==='context'?'CONTEXT_RESPONSE_ONLY':'CONTROLLED_SWEEP_NOT_LAW';
}
function mountCurve(host,data){
 host.className='scaling-widget';host.innerHTML=`<div class="scaling-controls"><label>Database<select data-db><option value="rel-f1">F1 / driver-dnf</option><option value="rel-trial">Trial / study-outcome</option></select></label><label>Model<select data-model><option>RDBPFN</option><option>RDBPFN_single</option><option>TabICLv1.1</option></select></label><label>Labeled context<select data-context>${KS.map(k=>`<option>${k}</option>`).join('')}</select></label><button type="button" data-reset>Reset</button></div><p class="scaling-scroll-hint">On a narrow screen, scroll the curve horizontally to see all five sizes.</p><div class="scaling-scroll" tabindex="0" aria-label="Context curve; scroll horizontally on narrow screens"><svg viewBox="0 0 640 300" role="img" aria-label="Measured mean AUROC with sample standard deviation"></svg></div><output aria-live="polite"></output><p class="scaling-baseline"></p>`;
 const db=host.querySelector('[data-db]'),model=host.querySelector('[data-model]'),ctx=host.querySelector('[data-context]');ctx.value='512';
 function draw(){
  const c=data.curves[db.value+'/'+model.value],i=KS.indexOf(+ctx.value),a=c.levels[i],b=c.levels[0];
  host.dataset.mean=a.mean;host.dataset.context=ctx.value;host.dataset.evidence=i===3?'REUSED':'FRESH';
  const x=j=>70+j*125,y=v=>250-(v-.45)/.4*220;
  const svg=host.querySelector('svg');let s='';
  [.45,.55,.65,.75,.85].forEach(v=>s+=`<line x1="65" y1="${y(v)}" x2="575" y2="${y(v)}" stroke="#d6e1e8"/><text x="53" y="${y(v)+4}" text-anchor="end">${v.toFixed(2)}</text>`);
  s+=`<text x="12" y="19">AUROC</text><polyline points="${c.levels.map((r,j)=>`${x(j)},${y(r.mean)}`).join(' ')}" fill="none" stroke="#167d80" stroke-width="3"/>`;
  c.levels.forEach((r,j)=>{s+=`<line x1="${x(j)}" x2="${x(j)}" y1="${y(r.mean-r.sample_sd)}" y2="${y(r.mean+r.sample_sd)}" stroke="#167d80" stroke-width="2"/><circle cx="${x(j)}" cy="${y(r.mean)}" r="${j===i?8:5}" fill="${j===3?'white':'#167d80'}" stroke="${j===i?'#9e4b11':'#167d80'}" stroke-width="${j===i?3:2}"/><text x="${x(j)}" y="274" text-anchor="middle">${r.context}</text>`;});
  s+='<text x="325" y="297" text-anchor="middle">Labeled supports K · equal spacing means doubling</text>';svg.innerHTML=s;
  host.querySelector('output').textContent=`${db.value} · ${model.value} · K=${a.context}: AUROC ${a.mean.toFixed(6)} ± ${a.sample_sd.toFixed(6)} sample SD. Change from 64: ${(a.mean-b.mean).toFixed(6)}. ${i===3?'REUSED 512 evidence':'FRESH L169 evidence'}.`;
  host.querySelector('.scaling-baseline').textContent=`Fixed baseline: same task/model at K=64, mean ${b.mean.toFixed(6)}. Weights, features and test queries stay fixed. Support composition changes with K. Bars describe ten support draws, not database uncertainty.`;
 }
 [db,model,ctx].forEach(e=>e.addEventListener('change',draw));host.querySelector('[data-reset]').onclick=()=>{db.value='rel-f1';model.value='RDBPFN';ctx.value='512';draw();};draw();
}
function mountClaim(host){
 host.className='scaling-widget';host.innerHTML=`<div class="scaling-controls"><label>Axis varied<select data-axis><option value="context">Inference context</option><option value="parameters">Pretrained parameters</option><option value="pretraining_data">Pretraining examples</option><option value="schema_diversity">Schema diversity</option></select></label><label>Other axes controlled?<select data-controlled><option value="true">Yes, documented</option><option value="false">No, confounded</option></select></label><label>Observed levels<select data-levels><option value="5">Five</option><option value="1">One</option></select></label><label>Predict beyond observed range?<select data-extrapolate><option value="false">No</option><option value="true">Yes</option></select></label><button type="button" data-reset>Reset</button></div><output aria-live="polite"></output><p class="scaling-baseline">Fixed baseline: controlled five-level context sweep → CONTEXT_RESPONSE_ONLY. Other axes are hypothetical declarations, not additional measured experiments.</p>`;
 const a=host.querySelector('[data-axis]'),c=host.querySelector('[data-controlled]'),l=host.querySelector('[data-levels]'),e=host.querySelector('[data-extrapolate]');
 const explanations={CONFOUNDED:'Several factors changed. The intended axis cannot explain the result by itself.',INSUFFICIENT_LEVELS:'One observed level cannot define a response curve.',EXTRAPOLATION_NOT_ESTABLISHED:'No held-out-scale validation is supplied; the declaration cannot establish extrapolation.',CONTEXT_RESPONSE_ONLY:'The observations concern target-time context. Pretraining parameters and data stayed fixed.',CONTROLLED_SWEEP_NOT_LAW:'A controlled sweep is useful evidence. A fitted and validated quantitative law still needs its own checks.'};
 function draw(){const result=claim(a.value,c.value==='true',+l.value,e.value==='true');host.dataset.claim=result;host.querySelector('output').textContent=result+' — '+explanations[result];}
 [a,c,l,e].forEach(x=>x.addEventListener('change',draw));host.querySelector('[data-reset]').onclick=()=>{a.value='context';c.value='true';l.value='5';e.value='false';draw();};draw();
}
global.ScalingContext={mountCurve,mountClaim,claim};
})(window);
