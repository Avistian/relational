/* Educational protocol classifier. No model inference or benchmark scores are simulated. */
window.CrossDatabase={mount(host){
 if(!host)return;
 host.className='cross-widget';
 host.innerHTML=`<div class="cross-controls">
 <label>Target exposure in pretraining<select data-exposure><option value="unknown">Lineage incomplete</option><option value="excluded">Verified exclusion</option><option value="seen">Target database included</option></select></label>
 <label>Target-task labels<select data-labels><option value="512">512 labeled supports</option><option value="0">No labeled supports</option></select></label>
 <label>Target adaptation<select data-updates><option value="0">Frozen weights</option><option value="5">Five supervised updates</option></select></label>
 <button type="button" data-reset>Reset</button></div>
 <div class="cross-path"><div data-source></div><div data-context></div><div data-weights></div></div>
 <output aria-live="polite"></output>
 <p class="cross-baseline">Fixed baseline: lineage incomplete; 512 target labels; zero gradient steps → database holdout NOT_ESTABLISHED; FEW_SHOT_ICL. This classifier changes the permitted claim, not an AUROC.</p>`;
 const controls=['exposure','labels','updates'].map(k=>host.querySelector('[data-'+k+']'));
 function update(){const[e,l,u]=controls.map(x=>x.value),labels=Number(l),steps=Number(u);
 const holdout=e==='seen'?'SEEN':e==='excluded'?'HELD_OUT':'NOT_ESTABLISHED';
 const adaptation=steps?(labels?'SUPERVISED_ADAPTATION':'INVALID_SUPERVISED_PROTOCOL'):(labels?'FEW_SHOT_ICL':'ZERO_LABEL_ZERO_GRADIENT');
 host.querySelector('[data-source]').innerHTML='<strong>1 · Pretraining evidence</strong>'+({unknown:'Training lineage is incomplete.',excluded:'Assume audited target-data exclusion.',seen:'Target database was included.'}[e]);
 host.querySelector('[data-context]').innerHTML='<strong>2 · Inference context</strong>'+labels+' target-task labels supplied.';
 host.querySelector('[data-weights]').innerHTML='<strong>3 · Adaptation</strong>'+steps+' supervised weight updates.';
 host.querySelector('[data-source]').classList.toggle('attention',e!=='excluded');
 host.querySelector('output').textContent='Database holdout: '+holdout+' · Adaptation: '+adaptation+'. '+(labels===0?'A protocol category is not evidence that RDB-PFN supports zero-label prediction.':'No gradient updates does not mean no target labels.');
 host.dataset.holdout=holdout;host.dataset.adaptation=adaptation;
 }
 controls.forEach(x=>x.addEventListener('change',update));host.querySelector('[data-reset]').addEventListener('click',()=>{controls[0].value='unknown';controls[1].value='512';controls[2].value='0';update()});update();
}};
