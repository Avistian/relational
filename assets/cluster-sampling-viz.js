/* Six-node path, fixed scalar states [1,2,4,8,16,32]. q=1: one edge and node1 Sh=2.5;
 * q=2: three edges and Sh=3; q=3: five edges and Sh=3. Controls change an induced
 * union, not a concatenation of blocks. This is a later-layer synthetic trace. */
(function(global){'use strict';
 global.ClusterSamplingViz={mount:function(el){
  el.classList.add('cluster-sampling');
  el.innerHTML='<label>Selected clusters, starting at C0: <select aria-label="Selected clusters"><option value="1">1 cluster</option><option value="2">2 clusters</option><option value="3">3 clusters</option></select></label> <button type="button">Reset</button><p class="cluster-scroll-hint">On narrow screens, scroll the graph sideways to see all three clusters.</p><div class="cluster-graph" tabindex="0" aria-label="Scrollable graph diagram"></div><output aria-live="polite"></output><p class="cluster-baseline">Full-graph baseline: 6 nodes, 5 undirected edges, node 1 enhanced message = 3.</p>';
  const sel=el.querySelector('select'),panel=el.querySelector('.cluster-graph'),out=el.querySelector('output');
  function setState(value){let q=Math.max(1,Math.min(3,Math.round(Number(value)||1)));sel.value=String(q);let count=2*q;
   let svg='<svg viewBox="0 0 600 125" role="img" aria-label="Six-node path grouped into three clusters">';
   for(let i=0;i<5;i++){let kept=i+1<count;svg+='<line x1="'+(50+i*100)+'" y1="48" x2="'+(150+i*100)+'" y2="48" stroke="'+(kept?'#203347':'#aaa')+'" stroke-width="'+(kept?4:2)+'" '+(kept?'':'stroke-dasharray="5 5"')+'/>';}
   for(let i=0;i<6;i++){let active=i<count;svg+='<circle cx="'+(50+i*100)+'" cy="48" r="23" fill="'+(active?['#176b83','#a65a24','#65539b'][Math.floor(i/2)]:'#eee')+'"/><text x="'+(50+i*100)+'" y="54" text-anchor="middle" fill="'+(active?'white':'#444')+'">'+i+'</text>';}
   for(let i=0;i<3;i++)svg+='<text x="'+(100+i*200)+'" y="104" text-anchor="middle">C'+i+(i<q?' selected':' excluded')+'</text>';
   panel.innerHTML=svg+'</svg>';
   out.textContent=count+' nodes · '+(count-1)+' retained edges · '+(q>1?'edge 1–2 restored':'edge 1–2 absent')+'. Node 1: '+(q===1?'½ × 1 + 1 × 2 = 2.5':'⅓ × 1 + ⅔ × 2 + ⅓ × 4 = 3')+'.';
  }
  sel.addEventListener('change',()=>setState(sel.value));el.querySelector('button').addEventListener('click',()=>setState(1));setState(1);el.clusterAPI={setState};return el.clusterAPI;
 }};
})(window);
