/* Two fixed task losses (1 and 5); vary A's population and the averaging policy.
   Baseline: A=9, B=1, cell mean=1.4, macro mean=3. Keyboard-native controls. */
window.MultitaskLoss={mount:function(host){
 host.classList.add('multitask-explorer');
 host.innerHTML='<p><strong>Same predictions, different objective</strong></p><div class="multitask-controls"><label>A cells <select aria-label="A cells"><option>1</option><option selected>9</option><option>99</option></select></label><label>Averaging <select aria-label="Averaging"><option value="cell">Every cell equally</option><option value="task">Every task equally</option></select></label><button type="button" data-reset>Reset baseline</button></div><p data-baseline>Fixed baseline: 9 A cells, 1 B cell; losses 1 and 5. Cell mean 1.4; task mean 3.</p><div class="multitask-bars"></div><output aria-live="polite"></output>';
 const selectors=host.querySelectorAll('select');
 function draw(){const n=Number(selectors[0].value),arm=selectors[1].value;const wa=arm==='cell'?n/(n+1):.5,wb=1-wa;const loss=wa+5*wb;
 host.dataset.loss=loss;host.dataset.a=wa;host.dataset.b=wb;
 host.querySelector('.multitask-bars').innerHTML=[['A',wa,1],['B',wb,5]].map(([label,w,l])=>'<p>'+label+' · total weight '+(w*100).toFixed(1)+'% · mean loss '+l+'<span style="display:block;width:'+Math.max(1,w*100)+'%;height:12px;background:'+(label==='A'?'#157f78':'#a34832')+'"></span></p>').join('');
 host.querySelector('output').textContent=wa.toFixed(3)+' × 1 + '+wb.toFixed(3)+' × 5 = '+loss.toFixed(3)+'. B per-cell weight '+(arm==='task'?(n+1)/2:1).toFixed(3)+' before the minibatch mean. '+(arm==='task'?'Changing A’s population leaves the task objective fixed.':'Adding A cells reduces B’s influence without changing any prediction.');}
 selectors.forEach(s=>s.addEventListener('change',draw));host.querySelector('[data-reset]').addEventListener('click',()=>{selectors[0].value='9';selectors[1].value='cell';draw();});draw();
}};
