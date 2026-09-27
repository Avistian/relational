/* Synthetic worked examples only. Scale baseline: MAE1 / SD2=.5.
 * Coverage baseline: [.1,.5] -> .3; one task -> diagnostic .1, reject.
 * Reusable native controls retain fixed values and expose reset/keyboard states. */
(() => {
 const scale=document.getElementById('l136-scale');
 if(scale){
  scale.innerHTML='<p><strong>Predict:</strong> can a score improve while every prediction stays fixed?</p><label for="l136-denominator">Training scale <output>2.0</output></label><input id="l136-denominator" type="range" min="1" max="4" step="0.5" value="2"><p>Fixed MAE = 1. Baseline: scale 2 → NMAE 0.500. This is an illustration, not permission to change a published denominator.</p><p class="audit-result" aria-live="polite"></p><button type="button">Reset scale</button>';
  const input=scale.querySelector('input');const update=()=>{scale.querySelector('output').textContent=Number(input.value).toFixed(1);scale.querySelector('.audit-result').textContent=`NMAE = 1 / ${input.value} = ${(1/Number(input.value)).toFixed(3)}. Predictions unchanged.`;};
  input.addEventListener('input',update);scale.querySelector('button').addEventListener('click',()=>{input.value=2;update();});update();
 }
 const board=document.getElementById('l136-coverage');
 if(board){
  board.innerHTML='<p><strong>Predict:</strong> is a lower partial mean an improved board result?</p><label for="l136-present">Tasks present <output>2</output></label><input id="l136-present" type="range" min="1" max="2" step="1" value="2"><p>Fixed canonical tasks: A=0.1, B=0.5. Baseline: both present → mean 0.300.</p><p class="audit-result" aria-live="polite"></p><button type="button">Reset coverage</button>';
  const input=board.querySelector('input');const update=()=>{board.querySelector('output').textContent=input.value;board.querySelector('.audit-result').textContent=input.value==='2'?'COMPLETE: 2/2 tasks; board mean 0.300.':'REJECT: 1/2 tasks; diagnostic partial mean 0.100 is not a board score.';};
  input.addEventListener('input',update);board.querySelector('button').addEventListener('click',()=>{input.value=2;update();});update();
 }
 if(window.RetrievalBank)RetrievalBank.mount(document.getElementById('warmup'),{upTo:136,count:3});
 if(window.Teachback)Teachback.mount(document.getElementById('l136-teachback'),{prompt:'Why can exact reproduction of every leaderboard score still leave a model-superiority claim unresolved?',points:['Submitted predictions do not recover their entire generating procedure','Metrics and all canonical tasks must match','Temporal inputs and search budgets can differ','Fresh selected training is distinct from archive replay'],model:'We can verify that submitted answers produce the reported scores. That does not identify all training data, feature-time access, tuning effort or selection decisions. A controlled model comparison must hold those conditions fixed or explicitly measure their effects. Our historical five-seed RDL experiment and current prediction replay establish different claims.'});
})();
