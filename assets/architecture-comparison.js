/* Hand-checkable incoming-edge intervention, separate from trained-model evidence. */
(() => {
 const root=document.getElementById('architecture-comparison');if(!root)return;
 root.innerHTML=`<div class="comparison-controls"><label>Subject score <select id="subject-score"><option value="0">0</option><option value="0.6931471805599453" selected>ln 2 (baseline)</option><option value="2.0794415416798357">ln 8</option></select></label><label>Copies of author A <select id="author-copies"><option value="1">1 (baseline)</option><option value="2">2</option><option value="3">3</option></select></label><button type="button">Reset baseline</button></div><p>Author messages: A = 2, B = 8. Subject message = −4. Author attention scores: 0 and ln 3.</p><output aria-live="polite"></output><p class="muted">Only scalar aggregation changes here. Learned transforms, self paths, residuals and normalization are omitted. This trace cannot predict the trained models' ranking.</p>`;
 const selects=root.querySelectorAll('select');const output=root.querySelector('output');
 function render(){const n=Number(selects[1].value),w=Math.exp(Number(selects[0].value)),total=n+3+w;
 const rgcn=(2*n+8)/(n+1)-4,uniform=(2*n+8-4)/(n+2),attention=(2*n+24-4*w)/total;
 output.textContent=`R-GCN sum of relation means: ${rgcn.toFixed(3)} · Uniform incoming mean: ${uniform.toFixed(3)} · HGT weighted sum: ${attention.toFixed(3)}. Attention mass: author A copies ${(n/total).toFixed(3)}, author B ${(3/total).toFixed(3)}, subject ${(w/total).toFixed(3)}.`;}
 selects.forEach(s=>s.addEventListener('change',render));root.querySelector('button').addEventListener('click',()=>{selects[0].value='0.6931471805599453';selects[1].value='1';render();});render();
})();
