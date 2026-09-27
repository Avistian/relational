/* Reusable binary AP trace: equal scores define one threshold. */
window.AveragePrecisionViz={mount(root){
 if(!root)return;
 root.className='stream-widget';
 root.innerHTML=`<h3>Predict before changing a tie</h3><p>Five fixed labels contain three positives. Higher scores rank first. Does moving a positive ahead of a negative with the same score improve AP?</p>
 <label>Score pattern <select aria-label="Score pattern"><option value="groups">0.9, 0.5, 0.5, 0.1, 0.1</option><option value="constant">All scores 0.5</option></select></label>
 <label><input type="checkbox"> Put positives first within each tie</label><button type="button">Reset</button>
 <output aria-live="polite"></output><div class="stream-scroll" tabindex="0"><table><thead><tr><th>Threshold</th><th>Labels admitted</th><th>Precision</th><th>Recall gain</th><th>AP contribution</th></tr></thead><tbody></tbody></table></div>
 <p class="ap-diagnostic"></p><p>Baseline stays fixed: AP = 0.755556 for the original three score groups. Changing only order within a tie must preserve AP.</p>`;
 const select=root.querySelector('select'),box=root.querySelector('input'),out=root.querySelector('output'),body=root.querySelector('tbody'),diag=root.querySelector('.ap-diagnostic');
 function update(){
  let rows=[1,0,1,0,1].map((y,i)=>({y,s:select.value==='constant'?.5:[.9,.5,.5,.1,.1][i]}));
  rows.sort((a,b)=>b.s-a.s||(box.checked?b.y-a.y:0));
  let tp=0,prev=0,ap=0,wrong=0,html='';
  rows.forEach((r,i)=>{tp+=r.y;if(r.y)wrong+=tp/(i+1)/3;
   if(i===rows.length-1||rows[i+1].s!==r.s){const p=tp/(i+1),recall=tp/3,gain=recall-prev;ap+=p*gain;prev=recall;html+=`<tr><td>≥ ${r.s.toFixed(1)}</td><td>[${rows.slice(0,i+1).map(x=>x.y).join(', ')}]</td><td>${p.toFixed(6)}</td><td>${gain.toFixed(6)}</td><td>${(p*gain).toFixed(6)}</td></tr>`;}});
  body.innerHTML=html;out.textContent=`Correct grouped AP: ${ap.toFixed(6)}. ${box.checked?'Tie order changed; grouped AP is unchanged.':'Rows with equal scores enter together.'}`;
  diag.textContent=`Incorrect row-by-row calculation: ${wrong.toFixed(6)}. It may coincide by accident; dependence on arbitrary tie order makes it invalid.`;
 }
 select.addEventListener('change',update);box.addEventListener('change',update);root.querySelector('button').addEventListener('click',()=>{select.value='groups';box.checked=false;update();});update();
}};
