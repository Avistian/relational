/* Two independent mechanisms: FK value access (four states), schema identity
   under four topology choices and renaming (eight states). No fitted result here. */
(function(){
  document.querySelectorAll('[data-fk-trace]').forEach(host=>{
    const a=host.querySelector('[name=a]'),b=host.querySelector('[name=b]'),out=host.querySelector('output');
    function update(){const av=[2,6][Number(a.value)],bv=[10,14][Number(b.value)],mean=(av+bv)/2,pred=2+mean;
      out.dataset.mean=mean;out.dataset.prediction=pred;
      out.textContent=`Read A=${av}, B=${bv}. Parent mean = (${av} + ${bv}) / 2 = ${mean}. Relational rule = 0.5 × 4 + ${mean} = ${pred}. Feature-only component stays 2. Baseline relational rule: 10; change: ${pred-10}.`;}
    a.addEventListener('change',update);b.addEventListener('change',update);host.querySelector('button').addEventListener('click',()=>{a.value='0';b.value='1';update();});update();
  });
  const graphs={chain:[[0,1],[1,2],[2,3]],'out-star':[[0,1],[0,2],[0,3]],'in-star':[[0,3],[1,3],[2,3]],diamond:[[0,1],[0,2],[1,3],[2,3]]};
  document.querySelectorAll('[data-schema-identity]').forEach(host=>{
    const family=host.querySelector('[name=family]'),rename=host.querySelector('[name=rename]'),out=host.querySelector('output');
    function update(){const f=family.value,edges=graphs[f],names=rename.value==='yes'?['Z','X','W','Y']:['A','B','C','D'];
      const positions=[[55,55],[210,35],[55,190],[210,170]];
      let svg='<svg viewBox="0 0 280 240" role="img" aria-label="Directed '+f+' schema"><defs><marker id="b13-arrow" markerWidth="7" markerHeight="7" refX="7" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="#526879"/></marker></defs>';
      edges.forEach(([a,b])=>{const [x,y]=positions[a],[u,v]=positions[b],d=Math.hypot(u-x,v-y),dx=(u-x)/d,dy=(v-y)/d;svg+=`<line x1="${x+24*dx}" y1="${y+24*dy}" x2="${u-27*dx}" y2="${v-27*dy}" stroke="#526879" stroke-width="2" marker-end="url(#b13-arrow)"/>`;});
      names.forEach((n,i)=>{const[x,y]=positions[i];svg+=`<circle cx="${x}" cy="${y}" r="22" fill="#e0eee9" stroke="#16877c"/><text x="${x}" y="${y+6}" text-anchor="middle" font-size="18" fill="#163a50">${n}</text>`;});svg+='</svg>';
      host.querySelector('[data-graph]').innerHTML=svg;out.dataset.family=f;out.dataset.heldout=String(f==='diamond');
      out.textContent=`${f}: ${edges.map(([a,b])=>names[a]+' → '+names[b]).join('; ')}. ${f==='diamond'?'Held-out topology: four edges.':'Training topology: three edges.'} Renaming does not change directed graph identity. Every training arm still contains 12 databases and 12,288 numeric feature cells.`;
    }
    family.addEventListener('change',update);rename.addEventListener('change',update);host.querySelector('button').addEventListener('click',()=>{family.value='chain';rename.value='no';update();});update();
  });
  if(window.RetrievalBank)RetrievalBank.mount(document.getElementById('b13-warmup'),{upTo:200.13,count:3});
  if(window.Predict)Predict.mount(document.getElementById('b13-predict'),{prompt:'With the correct parent-mean feature already provided, must three training-schema families beat one?',options:[{label:'Improvement is always guaranteed',value:'yes'},{label:'Improvement is not guaranteed',value:'no'}],correct:'no',reveal:'The same sufficient statistic and target equation apply to both. B13 shows mixed tiny changes across the three relational seeds; the course result does not reproduce or refute published scaling laws.'});
  if(window.Teachback)Teachback.mount(document.getElementById('b13-teachback'),{prompt:'Why can a relational synthetic prior feed a tabular predictor, and why does our held-out-schema result not establish learned relational transfer?',points:['Generator, representation and learner are separate.','RDB-PFN linearizes; PluRel trains RT on cell contexts.','Canonical topology holdout removes renamed copies.','Our feature readout already encodes the target statistic.','Replay, fresh inference and full pretraining are different evidence.'],model:'A generator controls training worlds, not the form of inference. RDB-PFN receives relationally derived feature vectors; PluRel supplies databases from which RT samples relational cell contexts. Our course ridge model transfers to a new topology because an engineered readout supplies the same parent statistic. It does not learn the graph computation. The paper-level claims need the full source-aligned experiments and historical identity checks.'});
})();
