/* Reusable measured nested-context explorer. Every display derives from per-seed evidence. */
window.FewShotEvaluation={mount(host,evidence){
 const dbs=['rel-f1','rel-trial'],arms=['RDBPFN','RDBPFN_single','TabICLv1.1'],ks=[64,128,256,512,1024];
 host.innerHTML='<div class="fs-controls"><label>Task<select data-db>'+dbs.map(x=>`<option>${x}</option>`).join('')+'</select></label><label>Frozen model<select data-arm>'+arms.map(x=>`<option>${x}</option>`).join('')+'</select></label><label>Labeled examples<select data-k>'+ks.map(x=>`<option>${x}</option>`).join('')+'</select></label><label>Support seed<select data-seed>'+Array.from({length:10},(_,i)=>`<option>${i}</option>`).join('')+'</select></label><button type="button" data-reset>Reset baseline</button></div><p class="fs-baseline" data-baseline></p><output class="fs-output" aria-live="polite"></output><div class="fs-support" aria-hidden="true"></div><p class="fs-legend">Each block represents 16 support rows. Blue: retained baseline rows. Green: additional rows. Counts are exact; block positions do not encode feature values.</p>';
 function update(){
  const db=host.querySelector('[data-db]').value,arm=host.querySelector('[data-arm]').value,k=+host.querySelector('[data-k]').value,seed=+host.querySelector('[data-seed]').value;
  const curve=evidence.curves[db+'/'+arm],base=curve.levels[0].per_seed[seed],level=curve.levels.find(x=>x.context===k),value=level.per_seed[seed],delta=value-base;
  host.dataset.state=[db,arm,k,seed].join('/');
  host.querySelector('[data-baseline]').textContent=`Fixed baseline: k=64, same task/model/seed; AUROC ${base.toFixed(6)}.`;
  host.querySelector('output').textContent=`k=${k}: AUROC ${value.toFixed(6)}; paired change ${delta>=0?'+':''}${delta.toFixed(6)}. Retained 64 of 64 baseline rows; added ${k-64}. Across ten support draws: mean ${level.mean.toFixed(6)}, sample SD ${level.sample_sd.toFixed(6)}. These are support draws on one fixed task.`;
  host.querySelector('.fs-support').innerHTML=Array.from({length:k/16},(_,i)=>`<span class="${i<4?'old':'new'}"></span>`).join('');
 }
 host.querySelectorAll('select').forEach(x=>x.addEventListener('change',update));
 host.querySelector('[data-reset]').addEventListener('click',()=>{host.querySelectorAll('select').forEach(x=>x.selectedIndex=0);update();});update();
}};
