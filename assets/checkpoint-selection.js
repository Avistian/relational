/* Reusable selection experiment: scores supplied as data, no network or storage. */
(() => {
  document.querySelectorAll('[data-checkpoint-selection]').forEach(root => {
    const valid=JSON.parse(root.dataset.valid), test=JSON.parse(root.dataset.test);
    root.innerHTML='<label>Selection rule <select aria-label="Checkpoint selection rule"><option value="valid">First validation maximum</option><option value="test">Best test score (invalid)</option><option value="last">Final epoch</option></select></label><p role="status" aria-live="polite"></p><button type="button">Reset to baseline</button>';
    const select=root.querySelector('select'), result=root.querySelector('[role=status]');
    function update(){const metric=select.value==='test'?test:valid;const i=select.value==='last'?valid.length-1:metric.indexOf(Math.max(...metric));result.textContent=`Epoch ${i+1}: validation ${valid[i].toFixed(1)}%, reported test ${test[i].toFixed(1)}%. ${select.value==='test'?'Invalid: test labels selected the predictor.':select.value==='last'?'Different protocol: final epoch ignores validation selection.':'Baseline: first validation maximum; restore its complete state.'}`;}
    select.addEventListener('change',update);root.querySelector('button').addEventListener('click',()=>{select.value='valid';update();});update();
  });
})();
