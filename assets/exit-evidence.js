/* Evidence-gate exercise. Hypothetical changes never alter the frozen report. */
(() => {
 const host=document.getElementById('exit-evidence');if(!host)return;
 host.innerHTML=`<fieldset><legend>Which observation changes the exit decision?</legend>
 <p><strong>Measured baseline:</strong> 2/3 completed tasks · 1/3 FE comparisons · 0/3 effort ratios · 0/3 temporal sign-offs. Failure cases documented; defense pending.</p>
 <p>Hypothetical additions only. Keep the original scores fixed.</p>
 <p><label><input type="checkbox" data-gate="task"> Complete the third test experiment</label></p>
 <p><label><input type="checkbox" data-gate="fe"> Complete both missing matched FE comparisons</label></p>
 <p><label><input type="checkbox" data-gate="effort"> Observe comparable human effort for all three tasks</label></p>
 <p><label><input type="checkbox" data-gate="temporal"> Establish reviewed temporal sign-offs for all three tasks</label></p>
 <p><label><input type="checkbox" data-gate="review"> Receive defense review ≥8/10 with no zero</label></p>
 <button type="button" data-reset>Reset to measured baseline</button>
 <p role="status" aria-live="polite" data-verdict></p><p data-trace></p>
 <p><small>Scenario flags represent evidence that would need to be produced and reviewed. Clicking cannot certify validity or mastery.</small></p></fieldset>`;
 const boxes=Object.fromEntries([...host.querySelectorAll('[data-gate]')].map(x=>[x.dataset.gate,x]));
 function update(){
  const v=Object.fromEntries(Object.entries(boxes).map(([k,x])=>[k,x.checked]));
  const complete=v.task?3:2,fe=v.fe?complete:1,effort=v.effort?fe:0,temporal=v.temporal?complete:0;
  const ready=complete===3&&fe===3&&effort===3&&temporal===3;
  host.querySelector('[data-verdict]').textContent='Hypothetical exit: '+(ready?(v.review?'PASS':'PENDING_WRITTEN_DEFENSE'):'INCOMPLETE');
  host.querySelector('[data-trace]').textContent=`Eligible counts: ${complete}/3 tasks; ${fe}/3 matched FE; ${effort}/3 effort ratios; ${temporal}/3 temporal sign-offs. Failure disclosure: PASS. Defense: ${v.review?'review passed':'pending'}.`;
 }
 Object.values(boxes).forEach(x=>x.addEventListener('change',update));host.querySelector('[data-reset]').addEventListener('click',()=>{Object.values(boxes).forEach(x=>x.checked=false);update();});update();
})();
