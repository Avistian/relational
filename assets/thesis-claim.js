/* Synthetic scope intervention: same numerical evidence, different requested claim. */
(() => {
 const host=document.getElementById('thesis-claim');if(!host)return;
 host.innerHTML=`<fieldset><legend>What can this evidence support?</legend><p>Fixed measured evidence: one matched F1 task; benefit −0.0642 MAE; conditional interval [−0.3123, +0.1730].</p><label>Requested claim <select data-claim><option value="quality">Local predictive quality</option><option value="portfolio">Portfolio superiority</option><option value="effort">Human effort saving</option><option value="validity">Historical information legality</option></select></label><p><label><input type="checkbox" data-repeat> Cite the same result in five reports</label></p><button type="button" data-reset>Reset</button><p role="status" aria-live="polite" data-answer></p><p data-count></p></fieldset>`;
 const q=host.querySelector('[data-claim]'),repeat=host.querySelector('[data-repeat]');
 const answers={quality:'Point estimate favors FE; superiority and equivalence remain unestablished.',portfolio:'NOT_ESTABLISHED: one matched task cannot establish the portfolio claim.',effort:'NOT_OBSERVED: no local human-effort ratio can be calculated.',validity:'NOT_ESTABLISHED: historical arrival information is missing.'};
 function update(){host.querySelector('[data-answer]').textContent=answers[q.value];host.querySelector('[data-count]').textContent=`${repeat.checked?5:1} report view(s) → 1 underlying comparison → 1 task → 1 database. Report count changes; evidence does not.`;}
 q.addEventListener('change',update);repeat.addEventListener('change',update);host.querySelector('[data-reset]').addEventListener('click',()=>{q.value='quality';repeat.checked=false;update();});update();
})();
