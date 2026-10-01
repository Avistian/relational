/* Worked query only. Removing candidate B changes the evaluation problem. */
(function(root){'use strict';
function compute(position,k,mode){
 const ranked=['A','B','D'];ranked.splice(Number(position)-1,0,'C');const catalog=mode==='sampled'?ranked.filter(x=>x!=='B'):ranked;
 const top=catalog.slice(0,Number(k));let hits=0,sum=0;const trace=top.map((id,i)=>{const positive=['A','C'].includes(id);if(positive){hits++;sum+=hits/(i+1)}return {id,positive,rank:i+1,term:positive?hits/(i+1):0}});
 return {ranked:catalog,trace,ap:sum/Math.min(Number(k),2),hit:Number(hits>0),recall:hits/2};
}
function mount(host){
 host.innerHTML='<p><strong>Predict:</strong> can removing one distractor improve AP without improving the model?</p><label>Move relevant candidate C to rank <input type="range" min="1" max="4" value="3" step="1" aria-label="Relevant candidate rank"></label><label>Cutoff k <select aria-label="Ranking cutoff"><option value="2">2</option><option value="3">3</option></select></label><label>Candidate protocol <select aria-label="Candidate protocol"><option value="full">Full catalog A, B, C, D</option><option value="sampled">Remove distractor B</option></select></label><button type="button">Reset</button><p class="route-readout" aria-live="polite"></p><div class="route-list"></div><p>Fixed truth: {A,C}. Baseline ranking A,B,C,D with k=2 gives AP=.50, Hit=1, Recall=.50. This is a synthetic intervention, not a benchmark score.</p>';
 const range=host.querySelector('input'),selects=host.querySelectorAll('select');
 function draw(){const r=compute(range.value,selects[0].value,selects[1].value);host.dataset.ap=r.ap;host.dataset.hit=r.hit;host.dataset.recall=r.recall;
 host.querySelector('.route-readout').textContent=`Ranking ${r.ranked.join(' → ')} · AP@${selects[0].value}=${r.ap.toFixed(3)} · Hit=${r.hit} · Recall=${r.recall.toFixed(3)}${selects[1].value==='sampled'?' · CHANGED CANDIDATE PROTOCOL':''}`;
 host.querySelector('.route-list').innerHTML=r.trace.map(x=>`<div>Rank ${x.rank}: ${x.id} · ${x.positive?'relevant':'irrelevant'}<br>AP numerator term ${x.term.toFixed(3)}</div>`).join('');}
 range.addEventListener('input',draw);selects.forEach(x=>x.addEventListener('change',draw));host.querySelector('button').addEventListener('click',()=>{range.value=3;selects[0].value='2';selects[1].value='full';draw()});draw();
}
root.RecommendationRankingViz={compute,mount};if(typeof module!=='undefined')module.exports={compute,mount};
})(typeof window==='undefined'?globalThis:window);
