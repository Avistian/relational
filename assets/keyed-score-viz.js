/* Synthetic fixture. Ordered: positional/keyed MAE=.5. Reversed: positional=2.5,
   keyed=.5. Duplicate: keyed refuses. Controls keep labels and predictions fixed. */
window.KeyedScoreViz={mount(root){
 if(!root)return;
 root.innerHTML='<p><strong>Predict first:</strong> can changing only row order change a valid metric?</p><label>Prediction rows <select aria-label="Prediction rows"><option value="ordered">Task order</option><option value="reversed">Reverse order</option><option value="duplicate">Duplicate one key</option></select></label><div class="stream-scroll" tabindex="0"><table><thead><tr><th>Task key → target</th><th>Received key → prediction</th></tr></thead><tbody></tbody></table></div><p><output aria-live="polite"></output></p><button type="button">Reset</button>';
 const control=root.querySelector('select'),body=root.querySelector('tbody'),out=root.querySelector('output');
 function render(){
 const rows=control.value==='ordered'?[[10,2],[20,4]]:control.value==='reversed'?[[20,4],[10,2]]:[[20,4],[20,4]];
 body.innerHTML=rows.map((r,i)=>'<tr><td>(driver 7, t='+[10,20][i]+') → '+[2,5][i]+'</td><td>(driver 7, t='+r[0]+') → '+r[1]+'</td></tr>').join('');
 const positional=(Math.abs(rows[0][1]-2)+Math.abs(rows[1][1]-5))/2;
 out.textContent='Fixed correct baseline: MAE 0.5. Positional MAE: '+positional.toFixed(1)+'. '+(control.value==='duplicate'?'Keyed scorer: REJECTED — duplicate key and missing t=10.':'Keyed MAE: 0.5 — complete one-to-one alignment.');
 }
 control.addEventListener('change',render);root.querySelector('button').addEventListener('click',()=>{control.value='ordered';render();});render();
}};
