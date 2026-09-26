/* Exact scalar update and label-dependence probes; course illustrations, no training claims. */
(function (global) {
  'use strict';
  function update(lr, missing) {
    lr = Math.max(0, Math.min(1, Number(lr)));
    let theta = 1;
    const rows = [];
    for (let k = 0; k < 4; k++) {
      const gradient = theta, before = theta;
      if (!missing) theta -= lr * gradient;
      rows.push({step:k+1, loss:before*before/2, gradient, delta:theta-before, theta});
    }
    return rows;
  }
  function leakage(scope, perturb) {
    const probabilities = [[.9,.1],[.2,.8],[.7,.3],[.4,.6]];
    const labels = perturb ? [0,1,0,1] : [0,1,1,0];
    const ids = scope === 'all' ? [0,1,2,3] : [0,1];
    const terms = ids.map(i => -Math.log(probabilities[i][labels[i]]));
    return {labels,ids,terms,loss:terms.reduce((a,b)=>a+b,0)/ids.length};
  }
  function mountUpdate(el) {
    el.innerHTML = '<fieldset><legend>Does the parameter actually move?</legend><p>Scalar example: θ starts at 1; target 0; loss θ²/2; gradient θ. Baseline always executes the step at the same learning rate.</p><label>Learning rate <input type="range" min="0" max="1" step="0.05" value="0.2"></label><label><input type="checkbox" checked> Omit optimizer step</label><button type="button">Reset update</button><output aria-live="polite"></output><div class="stream-scroll" tabindex="0"><table><thead><tr><th>Step</th><th>Observed gradient</th><th>Observed Δθ</th><th>Observed θ</th><th>Repaired θ</th></tr></thead><tbody></tbody></table></div></fieldset>';
    const lr=el.querySelector('input[type=range]'), missing=el.querySelector('input[type=checkbox]');
    function draw() {
      const observed=update(lr.value,missing.checked), baseline=update(lr.value,false);
      el.querySelector('output').textContent='Learning rate '+Number(lr.value).toFixed(2)+'. '+(Number(lr.value)===0?'Both paths are frozen at learning rate zero; zero movement alone cannot identify a missing step.':missing.checked?'Nonzero gradient; zero parameter movement. Restore the update before tuning the model.':'The step changes θ. This verifies execution, not generalization.');
      el.querySelector('tbody').innerHTML=observed.map((r,i)=>'<tr><td>'+r.step+'</td><td>'+r.gradient.toFixed(4)+'</td><td>'+r.delta.toFixed(4)+'</td><td>'+r.theta.toFixed(4)+'</td><td>'+baseline[i].theta.toFixed(4)+'</td></tr>').join('');
    }
    lr.addEventListener('input',draw);missing.addEventListener('change',draw);el.querySelector('button').addEventListener('click',()=>{lr.value=.2;missing.checked=true;draw();});draw();
  }
  function mountLeak(el) {
    el.innerHTML='<fieldset><legend>Can held-out labels change training loss?</legend><p>Fixed class probabilities by row: [0.9,0.1], [0.2,0.8], [0.7,0.3], [0.4,0.6]. Training IDs are [0,1]. Only labels at rows 2 and 3 are changed.</p><label>Supervised rows <select><option value="train">Training IDs only</option><option value="all">All node IDs — faulty</option></select></label><label><input type="checkbox"> Flip held-out labels</label><button type="button">Reset label probe</button><output aria-live="polite"></output></fieldset>';
    const scope=el.querySelector('select'),flip=el.querySelector('input');
    function draw() {
      const current=leakage(scope.value,flip.checked),baseline=leakage(scope.value,false);
      el.querySelector('output').textContent='Labels ['+current.labels.join(', ')+']; supervised IDs ['+current.ids.join(', ')+']. Baseline loss '+baseline.loss.toFixed(6)+'; current loss '+current.loss.toFixed(6)+'; change '+(current.loss-baseline.loss).toFixed(6)+'. '+(scope.value==='train'?'Held-out perturbation has no effect on this loss.':'The loss reads held-out labels; flip them to expose the dependence.');
    }
    scope.addEventListener('change',draw);flip.addEventListener('change',draw);el.querySelector('button').addEventListener('click',()=>{scope.value='train';flip.checked=false;draw();});draw();
  }
  global.GNNDiagnostics={update,leakage,mountUpdate,mountLeak};
})(window);
