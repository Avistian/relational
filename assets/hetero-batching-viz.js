/* Reusable dependency-route widget; deterministic arithmetic, not random sampling. */
(function(g){'use strict';g.HeteroBatchingViz={mount:function(el){
 el.classList.add('hetero-viz');
 el.innerHTML='<p><strong>Predict the customer mean.</strong> Product scalars [2, 8, −4] travel through orders at days [1, 2, 3]. Root terms and nonlinearities are omitted here.</p><label>Cutoff <select aria-label="Query cutoff"><option value="3">Day 3 (baseline)</option><option value="2">Day 2</option><option value="1">Day 1</option></select></label> <label>Fanout <select aria-label="Fanout"><option value="3">All neighbors</option><option value="2">Two neighbors</option><option value="1">One neighbor</option></select></label> <button type="button">Reset</button><div class="figure-scroll" tabindex="0" role="region" aria-label="Sampled dependency route"><svg viewBox="0 0 700 280" role="img" aria-label="Products through legal orders to a customer"></svg></div><output aria-live="polite"></output><p>Frozen baseline: cutoff 3, all neighbors, mean 2. Prefix selection is only a worked example; the notebook uses random uniform sampling.</p>';
 const selects=el.querySelectorAll('select'),svg=el.querySelector('svg'),out=el.querySelector('output');
 function update(){const legal=+selects[0].value,n=Math.min(legal,+selects[1].value),values=[2,8,-4];let s='<text x="35" y="22">Product features</text><text x="280" y="22">Order events</text><text x="535" y="22">Seed customer</text>';
 for(let i=0;i<3;i++){const y=65+85*i,on=i<n,color=on?'#087f8c':'#aaa',opacity=on?1:.35;
 s+='<g opacity="'+opacity+'"><path d="M115 '+y+' L295 '+y+' M350 '+y+' L570 145" stroke="'+color+'" stroke-width="3" fill="none"/><circle cx="80" cy="'+y+'" r="29" fill="#e1efec" stroke="'+color+'"/><text x="80" y="'+(y+5)+'" text-anchor="middle">'+values[i]+'</text><rect x="295" y="'+(y-23)+'" width="70" height="46" rx="7" fill="#f6e7cb" stroke="'+color+'"/><text x="330" y="'+(y+5)+'" text-anchor="middle">day '+(i+1)+'</text></g>';}
 const mean=values.slice(0,n).reduce((a,b)=>a+b,0)/n;
 s+='<circle cx="605" cy="145" r="35" fill="#e8e8f2" stroke="#193248"/><text x="605" y="151" text-anchor="middle">'+mean.toFixed(2)+'</text>';svg.innerHTML=s;
 out.textContent='Legal orders: '+legal+'; selected: '+n+'; customer mean: '+mean.toFixed(2)+'. Loss targets: 1 seed, not '+(2*n+1)+' sampled nodes.';
 }
 selects.forEach(s=>s.addEventListener('change',update));el.querySelector('button').addEventListener('click',()=>{selects[0].value='3';selects[1].value='3';update();});update();
}};const el=document.getElementById('batching-intervention');if(el)g.HeteroBatchingViz.mount(el);})(window);
