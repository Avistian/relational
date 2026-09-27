/* Fixed illustrative fixture. Threshold changes accuracy, never AUROC=.875.
 * MAP@3 baseline 5/6; move the missed item first -> 7/12. Two separate mounts. */
(function(g){
 function mount(el,kind){
  el.className='rdl-viz'; const auc=kind==='auc';
  el.innerHTML=auc?'<p><strong>Fixed:</strong> labels [0,1,0,1], scores [.1,.4,.4,.8]. Baseline threshold .5: accuracy .75; AUROC .875.</p><label>Decision threshold <input type="range" min="0" max="1" value="0.5" step="0.1"></label><output aria-live="polite"></output><button type="button">Reset</button>':'<p><strong>Fixed:</strong> relevant items {2,4}, k=3. Baseline [2,7,4]: AP=5/6.</p><label>Candidate ordering <select><option value="baseline">2, 7, 4</option><option value="miss-first">7, 2, 4</option><option value="best">2, 4, 7</option></select></label><output aria-live="polite"></output><button type="button">Reset</button>';
  const c=el.querySelector(auc?'input':'select'),out=el.querySelector('output');
  function draw(){if(auc){const t=Number(c.value),ys=[0,1,0,1],p=[.1,.4,.4,.8],dec=p.map(x=>Number(x>t)),acc=dec.filter((x,i)=>x===ys[i]).length/4;out.textContent='Threshold '+t.toFixed(1)+'; predicted classes ['+dec+']; accuracy '+acc.toFixed(2)+'; AUROC 0.875 (unchanged).';}else{const row=c.value==='baseline'?[2,7,4]:c.value==='best'?[2,4,7]:[7,2,4];let h=0;const terms=row.map((v,i)=>{if(v===2||v===4){h++;return h/(i+1);}return 0;});out.textContent='Ranked IDs ['+row+']; hit precision contributions ['+terms.map(v=>v.toFixed(3))+']; divide their sum by 2 -> AP@3 '+(terms.reduce((a,b)=>a+b,0)/2).toFixed(6)+'.';}}
  c.addEventListener('input',draw);c.addEventListener('change',draw);el.querySelector('button').onclick=()=>{c.value=auc?'0.5':'baseline';draw();};draw();
 }
 g.TaskMetricsViz={mount};
})(window);
