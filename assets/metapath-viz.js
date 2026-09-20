/* Two-node semantic reduction intervention. Baseline scores [[2,0],[0,0]].
 * Global beta = softmax([1,0]); local row0 = softmax([2,0]). */
(function(g){'use strict';g.MetapathViz={mount:function(el){
 el.classList.add('hetero-viz');
 el.innerHTML='<p><strong>Change only node 1’s PAP score.</strong> Node 0 stays at [2, 0]. All PSP scores stay at 0.</p><label>Node 1 PAP score <input type="range" min="-4" max="4" step="1" value="0" aria-label="Node 1 PAP score"></label> <button type="button">Reset</button><output aria-live="polite"></output><div class="semantic-bars"></div><p>Baseline: paper-global node 0 PAP weight = 0.731; released node 0 PAP weight = 0.881. The remote score changes only the global mixture for node 0.</p>';
 const input=el.querySelector('input'),out=el.querySelector('output'),bars=el.querySelector('.semantic-bars');
 function update(){const t=Number(input.value),global=1/(1+Math.exp(-(2+t)/2)),local=1/(1+Math.exp(-2));
 out.textContent='Node 1 score = '+t+'. Node 0 PAP weights: global '+global.toFixed(3)+'; per-node '+local.toFixed(3)+'.';
 bars.innerHTML=[['Paper-global',global],['Release per-node',local]].map(([name,b])=>'<p>'+name+' · PAP '+b.toFixed(3)+' / PSP '+(1-b).toFixed(3)+'</p><div class="semantic-track"><span style="width:'+(100*b)+'%">PAP</span><span style="width:'+(100*(1-b))+'%">PSP</span></div>').join('');
 }
 input.addEventListener('input',update);el.querySelector('button').addEventListener('click',()=>{input.value=0;update();});update();
}};})(window);
