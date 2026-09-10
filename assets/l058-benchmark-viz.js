/* L058: fixed six-task rank fixture; budgets 1,2,3 choose best seen-MAE.
 * Expected: first subset [0,1] => seen .5, unseen 0; budget>=2 chooses
 * [0,2] => seen 0, unseen .5. Hidden-method values NEVER select a subset.
 */
(function(){'use strict';
 const rankTable=document.querySelector('#ranks-viz table');
 if(rankTable){
  const region=rankTable.parentElement;
  region.classList.add('lesson-table-scroll');region.tabIndex=0;
  region.setAttribute('role','region');region.setAttribute('aria-label','Synthetic task ranks');
  const hint=document.createElement('p');hint.textContent='Scroll horizontally to compare all method columns.';
  region.before(hint);
 }
 const root=document.getElementById('tiny-selector');if(!root)return;
 const seen=[[1,2],[1,2],[2,1],[2,1],[1,2],[2,1]],unseen=[[1,2],[2,1],[1,2],[2,1],[1,2],[2,1]],candidates=[[0,1],[0,2],[1,3]];
 const mean=(x,ids)=>[0,1].map(j=>ids.reduce((s,i)=>s+x[i][j],0)/ids.length);
 const loss=(x,ids)=>mean(x,ids).reduce((s,v)=>s+Math.abs(v-1.5),0)/2;
 root.innerHTML='<p><strong>Predict: does a better seen-method match require a better unseen-method match?</strong></p><label for="tiny-budget">Candidate subsets evaluated</label><input id="tiny-budget" type="range" min="1" max="3" step="1" value="1"><output aria-live="polite"></output><div class="lesson-table-scroll" role="region" tabindex="0" aria-label="Candidate subset scores"><table><thead><tr><th>Task IDs</th><th>Seen MAE</th><th>Unseen MAE</th><th>Decision</th></tr></thead><tbody></tbody></table></div><p>Fixed full means: [1.5, 1.5] in each two-method pool. Only seen MAE selects. First equal minimum wins; unseen MAE is diagnostic after selection.</p><button type="button">Reset</button>';
 const slider=root.querySelector('input'),out=root.querySelector('output'),body=root.querySelector('tbody');
 function draw(){const n=Number(slider.value);let winner=0;for(let i=1;i<n;i++)if(loss(seen,candidates[i])<loss(seen,candidates[winner]))winner=i;
 body.innerHTML=candidates.map((ids,i)=>'<tr><td>['+ids.join(', ')+']</td><td>'+loss(seen,ids).toFixed(2)+'</td><td>'+loss(unseen,ids).toFixed(2)+'</td><td>'+(i>=n?'Not proposed':i===winner?'Selected':'Rejected')+'</td></tr>').join('');
 out.textContent=n+' candidate(s): rows ['+candidates[winner].join(', ')+']; seen MAE '+loss(seen,candidates[winner]).toFixed(2)+', unseen MAE '+loss(unseen,candidates[winner]).toFixed(2)+'. Baseline first candidate: seen 0.50, unseen 0.00.';root.dataset.state=String(n);}
 slider.addEventListener('input',draw);root.querySelector('button').addEventListener('click',()=>{slider.value=1;draw()});draw();
 window.tinySelector={set(v){slider.value=Math.max(1,Math.min(3,Math.round(v)));draw()},read(){return out.textContent}};
})();
