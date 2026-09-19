/* VIME worked trace. Default: x=[.2,.8,.8], donor=[.9,.8,.1], mask=[1,1,0].
   Actual-change target=[1,0,0]. Every toggle updates corruption and loss targets.
   A separate consistency control contrasts clean-anchor MSE with view variance. */
(function () {
  'use strict';
  const mount = document.getElementById('mask-trace');
  if (mount) {
    mount.innerHTML = '<p><strong>Predict:</strong> does selecting a cell always change its value?</p><div class="vime-controls"></div><div class="vime-table"></div><p class="vime-readout" aria-live="polite"></p><button type="button">Reset example</button>';
    const x=[.2,.8,.8], donor=[.9,.8,.1], selected=[true,true,false];
    const controls=mount.querySelector('.vime-controls');
    x.forEach((_,j)=>{const label=document.createElement('label');const input=document.createElement('input');input.type='checkbox';input.checked=selected[j];input.dataset.column=j;input.addEventListener('change',()=>{selected[j]=input.checked;render();});label.append(input,document.createTextNode(' Replace column '+(j+1)));controls.append(label);});
    function render(){
      const xt=x.map((v,j)=>selected[j]?donor[j]:v), target=x.map((v,j)=>+(v!==xt[j]));
      mount.querySelector('.vime-table').innerHTML='<table><thead><tr><th>Quantity</th><th>Col. 1</th><th>Col. 2</th><th>Col. 3</th></tr></thead><tbody>'+[['Original x',x],['Donor values',donor],['Selected m',selected.map(Number)],['Corrupted x̃',xt],['Changed target',target]].map(([label,row])=>'<tr><th>'+label+'</th>'+row.map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
      mount.querySelector('.vime-readout').textContent='Unchanged baseline: ['+x.join(', ')+']. Selected '+selected.filter(Boolean).length+'/3; actually changed '+target.reduce((a,b)=>a+b,0)+'/3. Column 2 has a donor collision whenever selected. The reconstruction target stays ['+x.join(', ')+'].';
    }
    mount.querySelector('button').addEventListener('click',()=>{selected.splice(0,3,true,true,false);controls.querySelectorAll('input').forEach((el,j)=>el.checked=selected[j]);render();});render();
  }
  const consistency=document.getElementById('consistency-trace');
  if(consistency){
    consistency.innerHTML='<label>Clean reference logit <input type="range" min="0" max="1" step=".1" value=".8"></label><p>Fixed augmented logits: 0.2 and 0.6. Their mean is 0.4.</p><output aria-live="polite"></output>';
    const slider=consistency.querySelector('input');
    function render(){const c=+slider.value, variance=.04, anchor=((.2-c)**2+(.6-c)**2)/2;consistency.querySelector('output').textContent='Clean logit = '+c.toFixed(1)+'. Released variance = '+variance.toFixed(2)+'. Clean-anchor MSE = '+anchor.toFixed(2)+' = 0.04 + '+((.4-c)**2).toFixed(2)+'. Baseline at clean=0.8: anchor MSE 0.20.';}
    slider.addEventListener('input',render);render();
  }
  if(window.RetrievalBank) RetrievalBank.mount(document.getElementById('warmup'),{upTo:71,count:3});
  if(window.Predict) Predict.mount(document.getElementById('prediction'),{prompt:'Holding label budget fixed, must dual-task pretraining beat training from scratch?',options:[{label:'It must improve',value:'yes'},{label:'It may decline',value:'no'}],correct:'no',reveal:'The pretext task learns feature structure. Downstream benefit depends on whether that structure helps predict the target, and on optimization and regularization.'});
  if(window.Teachback) Teachback.mount(document.getElementById('teachback'),{prompt:'Explain why a sampled replacement mask can disagree with the released mask target, and how you would test label efficiency fairly.',points:['A donor may repeat the original value','The release targets actual changes','Reconstruction includes all coordinates','Validation labels count toward the budget','Paired arms share splits and evaluation'],model:'A selected coordinate can receive the same value, so the release labels that coordinate unchanged. The encoder predicts this change mask and reconstructs the full original row. To measure downstream benefit, compare paired arms at equal total labeled budgets, including validation, without test rows entering pretraining. Report seed differences and protocol limits.'});
})();
