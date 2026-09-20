/* Reusable identity-preserving four-table path intervention; no dependencies. */
(function(global){'use strict';
 global.SchemaGraphViz={mount:function(root){
 root.classList.add('hetero-viz');
 root.innerHTML='<p><strong>Change one FK, keep the line and quantity.</strong></p><label>Product referenced by line (100,2) <select aria-label="Product for line 100,2"><option value="10">Product 10 (baseline)</option><option value="50">Product 50</option></select></label> <button type="button">Reset</button><div class="figure-scroll" tabindex="0" role="region" aria-label="Scrollable FK paths"><svg viewBox="0 0 780 250" role="img" aria-label="Two line nodes connect customer 10 through order 100 to product nodes"></svg></div><output aria-live="polite"></output><p>On a narrow screen, scroll the diagram horizontally; the live text gives the totals.</p><p>Baseline: product 10 totals 5; product 50 totals 1 for customer 10. Four line nodes and twelve forward FK links remain in the complete fixture.</p>';
 const select=root.querySelector('select'),svg=root.querySelector('svg'),output=root.querySelector('output');
 function update(){const changed=select.value==='50';
 svg.innerHTML='<g stroke="#087f8c" stroke-width="3" fill="none"><path d="M270 115 L150 115"/><path d="M480 55 L350 115"/><path d="M480 185 L350 115"/><path d="M570 55 L670 55"/><path d="M570 185 L670 '+(changed?'185':'55')+'" stroke="#bd762c" stroke-width="4"/></g>'+
 '<g fill="#e1efec" stroke="#74928e"><rect x="15" y="85" width="140" height="60" rx="10"/><rect x="235" y="85" width="125" height="60" rx="10"/><rect x="440" y="25" width="150" height="60" rx="10"/><rect x="440" y="155" width="150" height="60" rx="10"/><rect x="665" y="25" width="105" height="60" rx="10"/><rect x="665" y="155" width="105" height="60" rx="10"/></g>'+
 '<g fill="#193248" font-family="system-ui" font-size="16" text-anchor="middle"><text x="85" y="120">customer 10</text><text x="297" y="120">order 100</text><text x="515" y="49">line (100,1)</text><text x="515" y="72">quantity 2</text><text x="515" y="178">line (100,2)</text><text x="515" y="201">quantity 3</text><text x="717" y="60">product 10</text><text x="717" y="190">product 50</text><text x="195" y="100">buyer</text><text x="395" y="70">order</text><text x="395" y="185">order</text></g>';
 output.textContent=changed?'Changed: customer 10 → product 10 = 2 units; → product 50 = 4 units. Two highlighted line nodes retained.':'Baseline: customer 10 → product 10 = 5 units; → product 50 = 1 unit. Two highlighted line nodes retained.';
 }
 select.addEventListener('change',update);root.querySelector('button').addEventListener('click',()=>{select.value='10';update();});update();
 }};
})(window);
