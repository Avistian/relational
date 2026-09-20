/* Bipartite intervention: baseline B=[[1,1,0],[0,1,1]].
 * Remove shared i1 edge u1-i1: overlap 1→0 and walk u0→i2 1/8→0.
 * Boundary baseline hides target u0-i2 in both directions; reverse-only leak fails. */
(function(g){'use strict';
g.BipartiteViz={mount:function(el,mode){
 el.classList.add('bipartite-viz');
 if(mode==='boundary'){
  el.innerHTML='<p><strong>Predict first:</strong> the forward target is removed. Can its reverse still reveal it?</p><label><input type="checkbox"> Keep held-out i2 → u0 in the reverse store</label> <button type="button">Reset</button><output aria-live="polite"></output><p>Fixed: target u0 → i2 is absent from the forward store. A reverse edge represents the same held-out interaction.</p>';
  const c=el.querySelector('input'),o=el.querySelector('output');function draw(){o.textContent=c.checked?'FAIL: reverse contains (i2,u0). The held-out interaction is visible; forward-only checks miss it.':'PASS: target absent in both directions. Forward and reversed reverse-store pair sets agree.';}c.addEventListener('change',draw);el.querySelector('button').onclick=()=>{c.checked=false;draw();};draw();return;
 }
 el.innerHTML='<p><strong>Predict first:</strong> remove the shared-item bridge. Does the unseen item remain reachable?</p><label><input type="checkbox"> Remove u1–i1 from fitting likes</label> <button type="button">Reset</button><p><small>On narrow screens, scroll the graph horizontally.</small></p><div class="figure-scroll" tabindex="0" role="region" aria-label="Bipartite graph, scroll horizontally"><svg viewBox="0 0 700 300" role="img" aria-label="Two users and three items; only cross-type edges"></svg></div><output aria-live="polite"></output><p>Fixed baseline: shared-item count = 1; three-step probability u0 → i2 = 0.125. IDs, remaining edges and scoring rule stay fixed.</p>';
 const c=el.querySelector('input'),o=el.querySelector('output'),svg=el.querySelector('svg');
 function draw(){const b=[[1,1,0],[0,c.checked?0:1,1]],uy=[90,220],iy=[45,145,245];let paths='';for(let u=0;u<2;u++)for(let i=0;i<3;i++)if(b[u][i])paths+=`<line x1="160" y1="${uy[u]}" x2="520" y2="${iy[i]}" stroke="#087f8c" stroke-width="3"/>`;
 svg.innerHTML=paths+uy.map((y,u)=>`<circle cx="120" cy="${y}" r="36" fill="#264c78"/><text x="120" y="${y+6}" text-anchor="middle" fill="white" font-size="20">u${u}</text>`).join('')+iy.map((y,i)=>`<rect x="520" y="${y-28}" width="100" height="56" rx="6" fill="#087f8c"/><text x="570" y="${y+7}" text-anchor="middle" fill="white" font-size="20">i${i}</text>`).join('')+'<text x="260" y="285" font-size="17" fill="#17324a">No user–user or item–item edges</text>';
 let probability=0;for(let i=0;i<3;i++)for(let v=0;v<2;v++){const du=b[0].reduce((x,y)=>x+y,0),di=b[0][i]+b[1][i],dv=b[v].reduce((x,y)=>x+y,0);if(du&&di&&dv)probability+=b[0][i]/du*b[v][i]/di*b[v][2]/dv;}
 const overlap=b[0].reduce((s,x,i)=>s+x*b[1][i],0);o.textContent=`Shared-item count (BBᵀ)[u0,u1] = ${overlap}; walk probability S[u0,i2] = ${probability.toFixed(3)}. Change = ${(probability-.125).toFixed(3)}.`;
 }c.addEventListener('change',draw);el.querySelector('button').onclick=()=>{c.checked=false;draw();};draw();
}};
})(window);
