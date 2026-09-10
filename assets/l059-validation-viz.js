/* Fixed-prediction information-flow toy, not the measured KRR fit.
Default held fold1, labels unchanged: internal A, global B.
Flip held fold1 labels: internal A and its held predictions unchanged; global A.
Every state recomputes per-row losses, selection masks, candidate means and test scores.
*/
(function(){'use strict';const root=document.getElementById('nested-boundary');if(!root)return;
root.innerHTML='<h3>Which targets can change this choice?</h3><p>Fixed candidate predictions isolate selection. Predict what flipping only the held targets will change. This eight-row toy is separate from the fitted KRR experiment.</p><label>Held fold <select aria-label="Held fold"><option value="1">Rows 4–7</option><option value="0">Rows 0–3</option></select></label><label><input type="checkbox"> Flip the held fold’s targets</label><button type="button">Reset</button><p>Scroll the table horizontally to inspect row eligibility.</p><div class="validation-table" role="region" tabindex="0" aria-label="Scrollable row losses"></div><output aria-live="polite"></output><p class="detail"></p>';
const pick=root.querySelector('select'),flip=root.querySelector('input'),table=root.querySelector('.validation-table'),out=root.querySelector('output'),detail=root.querySelector('.detail');
const pa=[.8,.6,-.8,-.6,.1,.2,-.1,-.2],pb=[.2,.1,-.2,-.1,.9,.8,-.9,-.8],base=[1,1,-1,-1,1,1,-1,-1];
function render(){const fold=Number(pick.value),held=base.map((_,i)=>Math.floor(i/4)===fold),y=base.map((v,i)=>flip.checked&&held[i]?-v:v),loss=[pa,pb].map(p=>p.map((v,i)=>(v-y[i])**2));
const mean=(v,mask)=>v.filter((_,i)=>mask[i]).reduce((a,b)=>a+b,0)/mask.filter(Boolean).length;
const inner=loss.map(v=>mean(v,held.map(v=>!v))),all=loss.map(v=>mean(v,held.map(()=>true))),ci=inner[0]<=inner[1]?0:1,ce=all[0]<=all[1]?0:1;
table.innerHTML='<table><thead><tr><th>Row</th><th>Target</th><th>A loss</th><th>B loss</th><th>Internal selector sees?</th></tr></thead><tbody>'+y.map((v,i)=>'<tr><td>'+i+'</td><td>'+v+'</td><td>'+loss[0][i].toFixed(2)+'</td><td>'+loss[1][i].toFixed(2)+'</td><td>'+(held[i]?'No · held':'Yes · fit')+'</td></tr>').join('')+'</tbody></table>';
out.textContent='Internal: '+['A','B'][ci]+' from ['+inner.map(v=>v.toFixed(3)).join(', ')+']. External: '+['A','B'][ce]+' from ['+all.map(v=>v.toFixed(3)).join(', ')+'].';
detail.textContent='Held predictions from the internal choice: ['+[pa,pb][ci].filter((_,i)=>held[i]).join(', ')+']. Held MSE: internal '+mean(loss[ci],held).toFixed(3)+'; external '+mean(loss[ce],held).toFixed(3)+'. Flipping held labels may change evaluation losses; it must not change the internal choice or predictions.';
root.dataset.internal=['A','B'][ci];root.dataset.external=['A','B'][ce];}
pick.addEventListener('change',render);flip.addEventListener('change',render);root.querySelector('button').addEventListener('click',()=>{pick.value='1';flip.checked=false;render()});render();
})();
