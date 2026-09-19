/* Reusable, dependency-free row graph with explicit illustrative embeddings. */
(function(global){'use strict';
 global.CellGraph={mount:function(root){
 root.classList.add('contrastive-widget');
 root.innerHTML='<p><strong>Predict:</strong> Which messages remain if volume is missing?</p><label>Transformed volume <input type="range" min="0" max="4" step="1" value="2"></label><label><input type="checkbox"> Omit volume</label><label>Relation vector <select><option value="3,1">volume [3, 1]</option><option value="1,3">alternative [1, 3]</option></select></label><p>On a narrow screen, scroll the graph horizontally. The live text below lists all values.</p><div class="figure-scroll" tabindex="0" role="region" aria-label="Scrollable row graph"><svg viewBox="0 0 660 230" role="img" aria-label="Two cells send column-conditioned messages to the row center" style="width:100%;min-width:480px"><line x1="130" y1="65" x2="465" y2="115" stroke="#087e83" stroke-width="3"/><g class="numeric"><line x1="130" y1="180" x2="465" y2="115" stroke="#b77522" stroke-width="3"/><circle cx="100" cy="180" r="43" fill="#fae6c8"/><text x="100" y="184" text-anchor="middle" class="value"></text><text x="235" y="178" class="edge"></text></g><circle cx="100" cy="65" r="43" fill="#d7ece7"/><text x="100" y="70" text-anchor="middle">red [2, −1]</text><text x="230" y="65">colour [1, 2]</text><circle cx="500" cy="115" r="62" fill="#d7ece7"/><text x="500" y="108" text-anchor="middle">center</text><text x="500" y="133" text-anchor="middle" class="center"></text></svg></div><output aria-live="polite"></output><button type="button">Reset example</button><p>Baseline: volume 2, edge [3, 1], center [10, 0]. Toy vectors, not FastText measurements.</p>';
 const range=root.querySelector('input[type=range]'),missing=root.querySelector('input[type=checkbox]'),select=root.querySelector('select');
 function update(){const v=+range.value,e=select.value.split(',').map(Number),m=[v*e[0]*e[0],v*e[1]*e[1]],c=missing.checked?[2,-2]:[(2+m[0])/2,(-2+m[1])/2];
 root.querySelector('.numeric').style.display=missing.checked?'none':'';
 root.querySelector('.value').textContent='v = '+v;root.querySelector('.edge').textContent='edge ['+e.join(', ')+']';root.querySelector('.center').textContent='['+c.join(', ')+']';
 root.querySelector('output').textContent=(missing.checked?'One observed cell. ':'Two observed cells. Text message [2, −2]; numerical message ['+m.join(', ')+']. ')+'Released center ['+c.join(', ')+']. Missing region has no node.';
 }
 [range,missing,select].forEach(e=>e.addEventListener('input',update));root.querySelector('button').addEventListener('click',()=>{range.value=2;missing.checked=false;select.selectedIndex=0;update();});update();
 }};
})(window);
