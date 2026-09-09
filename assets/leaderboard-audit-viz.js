/* Synthetic repeat-weight trace: n=1 gives a tie under both summaries;
   n=3 gives pooled ranks A=1.75/B=1.25, macro ranks always 1.5/1.5.
   Elo trace: gaps 0/400/-400 yield p=.5/10:11/1:11. No ratings are fitted. */
(function(root){
  function weighting(n){n=Math.max(1,Math.min(30,Math.round(Number(n)||1)));return {n,a:(1+2*n)/(1+n),b:(2+n)/(1+n),macro:1.5};}
  function elo(gap){gap=Math.max(-800,Math.min(800,Number(gap)||0));return {gap,odds:Math.pow(10,gap/400),p:1/(1+Math.pow(10,-gap/400))};}
  function mountWeight(host){
    host.innerHTML='<label>Small-dataset repetitions: <input type="range" min="1" max="30" value="3" step="1" aria-label="Small-dataset repetitions"></label><output aria-live="polite"></output><table><thead><tr><th>Summary</th><th>A rank</th><th>B rank</th></tr></thead><tbody></tbody></table><button type="button">Reset</button>';
    const input=host.querySelector('input'),out=host.querySelector('output'),body=host.querySelector('tbody');
    function update(){const v=weighting(input.value);out.textContent='Large dataset: A=1, B=2 in one split. Small dataset: A=2, B=1 in '+v.n+' identical splits. Errors and dataset identities stay fixed.';body.innerHTML='<tr><td>Equal datasets</td><td>1.500</td><td>1.500</td></tr><tr><td>Pooled splits</td><td>'+v.a.toFixed(3)+'</td><td>'+v.b.toFixed(3)+'</td></tr>';}
    input.addEventListener('input',update);host.querySelector('button').addEventListener('click',()=>{input.value=3;update();});update();
  }
  function mountElo(host){
    host.innerHTML='<label>Elo gap A − B: <input type="range" min="-800" max="800" value="400" step="25" aria-label="Elo rating gap"></label><output aria-live="polite"></output><p>Fixed baseline: gap 0 → odds 1 → probability 50%. This is a rating-model expectation, not classification accuracy.</p><button type="button">Reset</button>';
    const input=host.querySelector('input'),out=host.querySelector('output');
    function update(){const v=elo(input.value);out.textContent='Gap '+v.gap+' → odds 10^('+v.gap+'/400) = '+v.odds.toFixed(3)+' → p = odds/(1+odds) = '+(100*v.p).toFixed(2)+'%.';}
    input.addEventListener('input',update);host.querySelector('button').addEventListener('click',()=>{input.value=400;update();});update();
  }
  root.LeaderboardAudit={weighting,elo,mountWeight,mountElo};
})(typeof window==='undefined'?globalThis:window);
