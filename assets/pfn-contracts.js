/* Permutation values come from a frozen course diagnostic, never a simulated score. */
(()=>{'use strict';const root=document.querySelector('.b03-widget');if(!root)return;
const data=JSON.parse(root.querySelector('script[type="application/json"]').textContent),select=root.querySelector('select'),out=root.querySelector('output');
function update(){const r=data.rows[Number(select.value)];out.textContent='Label map ['+r.permutation.join(', ')+'] → restored-column max probability change '+r.max_abs_delta.toFixed(6)+'. Accuracy '+(100*r.accuracy).toFixed(2)+'%; log loss '+r.log_loss.toFixed(6)+'. '+(r.within_tolerance?'WITHIN TOLERANCE':'OUTSIDE TOLERANCE')+' (atol 0.000001).';out.dataset.state=r.within_tolerance?'within':'outside';}
select.addEventListener('change',update);root.querySelector('button').addEventListener('click',()=>{select.value='0';update();});update();})();
