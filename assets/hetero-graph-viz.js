/* Typed-edge intervention. Default means: buys=5, returns=8 => 3.
 * Moving A to returns: buys=8, returns=5 => 12. Features/weights fixed. */
(function(g){'use strict';g.HeteroGraphViz={mount:function(el){
 el.classList.add('hetero-viz');
 el.innerHTML='<p><strong>Change one edge role.</strong> Features: A=2, C=8, T=1. Weights: buys=2, returns=−1, self=1.</p><label>A → T relation <select aria-label="A to T relation"><option value="buys">buys (baseline)</option><option value="returns">returns (intervention)</option></select></label> <button type="button">Reset</button><div class="figure-scroll" tabindex="0" role="region" aria-label="Typed graph, horizontally scrollable"><svg viewBox="0 0 740 260" role="img" aria-label="Two source nodes send typed messages to target T"></svg></div><output aria-live="polite"></output><p>Fixed baseline: 10 − 8 + 1 = <strong>3</strong>. Only A’s relation changes; recompute both relation degrees.</p>';
 const select=el.querySelector('select'),svg=el.querySelector('svg'),out=el.querySelector('output');
 function update(){const changed=select.value==='returns',color=changed?'#a34430':'#087f8c';
 svg.innerHTML='<defs><marker id="typed-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker></defs>'+
 '<path d="M130 62 L555 128" stroke="'+color+'" stroke-width="3" fill="none" marker-end="url(#typed-arrow)"/>'+
 '<path d="M130 193 Q350 65 555 142" stroke="#087f8c" stroke-width="3" fill="none" marker-end="url(#typed-arrow)"/>'+
 '<path d="M130 206 Q360 263 555 163" stroke="#a34430" stroke-width="3" fill="none" marker-end="url(#typed-arrow)"/>'+
 '<circle cx="90" cy="58" r="38" fill="#e4f2f5"/><circle cx="90" cy="201" r="38" fill="#e4f2f5"/><rect x="560" y="110" width="145" height="75" rx="14" fill="#f5edcc"/>'+
 '<g fill="#193248" font-size="19" font-family="system-ui" text-anchor="middle"><text x="90" y="64">A = 2</text><text x="90" y="207">C = 8</text><text x="632" y="140">T = 1</text><text x="632" y="168">self: +1</text></g>'+
 '<g font-size="17" font-family="system-ui"><text x="272" y="68" fill="'+color+'">'+select.value+'</text><text x="185" y="130" fill="#087f8c">buys × 2</text><text x="283" y="250" fill="#a34430">returns × (−1)</text></g>';
 out.textContent=changed?'buys: 8 × 2 = 16; returns: (2 + 8)/2 × (−1) = −5; self: 1. ReLU(12) = 12. Change: +9.':'buys: (2 + 8)/2 × 2 = 10; returns: 8 × (−1) = −8; self: 1. ReLU(3) = 3. Change: 0.';
 }
 select.addEventListener('change',update);el.querySelector('button').addEventListener('click',()=>{select.value='buys';update();});update();
 return {reset:()=>{select.value='buys';update();}};
}};})(window);
