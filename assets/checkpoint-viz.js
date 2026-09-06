/** L050 reusable experiment-audit widgets. All numbers are synthetic.
 * fit: held-out range changes; training-only fitted mean stays 1.
 * selection: default validation picks B; peeking picks A and invalidates report.
 * No layout-fixed SVG; native tables/controls reflow to 375px. */
(function(global){
'use strict';
function fitState(held,leak){held=Math.max(2,Math.min(200,Number(held)||2));var mean=leak?(0+2+held)/3:1;return {held:held,mean:mean,centered:2-mean};}
function choice(peek){return peek?{id:'A',valid:.80,test:.91}:{id:'B',valid:.86,test:.82};}
function fitMount(root){
 root.className='claim-viz';
 root.innerHTML='<h3>Change a held-out value</h3><p>Synthetic train = [0, 2]. Center the training value 2.</p><label>Held-out value <input type="range" min="2" max="200" value="100" step="1"></label><output></output><div class="panels"><section class="panel"><h4>Fit on train only</h4><p class="safe"></p></section><section class="panel"><h4>Fit on all rows</h4><p class="leak"></p></section></div><p class="readout" aria-live="polite"></p><button type="button">Reset</button>';
 var input=root.querySelector('input');
 function draw(){var a=fitState(input.value,false),b=fitState(input.value,true);root.querySelector('output').textContent='Held-out value: '+a.held;root.querySelector('.safe').textContent='Mean = '+a.mean.toFixed(2)+'; centered 2 = '+a.centered.toFixed(2);root.querySelector('.leak').textContent='Mean = '+b.mean.toFixed(2)+'; centered 2 = '+b.centered.toFixed(2);root.querySelector('.readout').textContent='Training rows are fixed. The all-row fit lets held-out information change the learned preprocessing. This is contamination even without labels; it need not inflate every measured score.';}
 input.addEventListener('input',draw);root.querySelector('button').addEventListener('click',function(){input.value=100;draw();});draw();
 return {setValue:function(x){input.value=fitState(x,false).held;draw();}};
}
function selectMount(root){
 root.className='claim-viz';root.innerHTML='<h3>Which candidate may cross the test boundary?</h3><table><thead><tr><th>Candidate</th><th>Validation</th><th>Test (audit only)</th></tr></thead><tbody><tr><td>A</td><td>0.80</td><td>0.91</td></tr><tr><td>B</td><td>0.86</td><td>0.82</td></tr></tbody></table><label><input type="checkbox"> Select using the test column</label><div class="panels"><section class="panel"><h4>Baseline: validation</h4><p>B → test AUROC 0.82</p></section><section class="panel"><h4>Current selection</h4><p class="selected"></p></section></div><p class="readout" aria-live="polite"></p>';
 function draw(){var peek=root.querySelector('input').checked,s=choice(peek);root.querySelector('.selected').textContent=s.id+' → test AUROC '+s.test.toFixed(2);root.querySelector('.readout').textContent=peek?'A larger number, but the test set has become selection data. A new untouched evaluation is required.':'B was selected by validation. Test evaluates the frozen decision. Both test scores are shown only for this synthetic audit.';}
 root.querySelector('input').addEventListener('change',draw);draw();
 return {setPeek:function(x){root.querySelector('input').checked=!!x;draw();}};
}
global.CheckpointViz={fitState:fitState,choice:choice,fit:fitMount,selection:selectMount};
})(typeof window!=='undefined'?window:globalThis);
