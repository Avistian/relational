/* Fixed predictions. Default: test/homophily. Controls never select model weights.
   Empty slices stay visible. Bar widths use a fixed 0–100% scale. */
(function(){'use strict';
 function mount(root){
  const data=JSON.parse(root.querySelector('script[type="application/json"]').textContent);
  const controls=document.createElement('div');controls.className='error-controls';
  function select(label,choices,value){const wrap=document.createElement('label');wrap.textContent=label+' ';const el=document.createElement('select');for(const choice of choices){const op=document.createElement('option');op.value=choice;op.textContent=choice;el.append(op);}el.value=value;wrap.append(el);controls.append(wrap);return el;}
  const population=select('Population',['valid','test'],'test');population.setAttribute('aria-label','Population');
  const family=select('Slice family',['degree','homophily','class','year'],'homophily');family.setAttribute('aria-label','Slice family');
  const reset=document.createElement('button');reset.type='button';reset.textContent='Reset view';controls.append(reset);root.append(controls);
  const status=document.createElement('p');status.setAttribute('aria-live','polite');root.append(status);
  const chart=document.createElement('div');chart.className='error-chart';root.append(chart);
  const scroll=document.createElement('div');scroll.className='stream-scroll';scroll.tabIndex=0;scroll.setAttribute('aria-label','Slice metric table');root.append(scroll);
  function draw(){
   const rows=data.rows.filter(r=>r.population===population.value&&r.family===family.value);
   const n=rows.reduce((s,r)=>s+r.n,0);status.textContent=population.value+': '+n.toLocaleString()+' nodes. '+(family.value==='homophily'||family.value==='class'?'Uses true labels: retrospective analysis only.':'Fixed model predictions; descriptive comparison.');
   chart.replaceChildren();
   for(const r of rows){if(!r.n)continue;const block=document.createElement('div');block.className='error-row';const name=document.createElement('strong');name.textContent=r.slice+' · n='+r.n.toLocaleString();block.append(name);
    for(const model of ['gcn','mlp']){const line=document.createElement('div');line.className='error-line';const value=r[model+'_mean_percent'];const label=document.createElement('span');label.textContent=model.toUpperCase()+' '+value.toFixed(1)+'%';const track=document.createElement('div');track.className='error-track';const bar=document.createElement('div');bar.className='error-bar '+model;bar.style.width=Math.max(0,Math.min(100,value))+'%';track.append(bar);line.append(label,track);block.append(line);}chart.append(block);}
   const table=document.createElement('table');const caption=document.createElement('caption');caption.textContent='Accuracy mean ± sample seed SD; Δ = GCN − MLP (pp).';table.append(caption);
   const header=document.createElement('tr');for(const title of ['Slice','Nodes','GCN (%)','MLP (%)','Δ (pp)']){const th=document.createElement('th');th.scope='col';th.textContent=title;header.append(th);}table.append(header);
   for(const r of rows){const tr=document.createElement('tr');const values=[r.slice,r.n.toLocaleString(),r.n?r.gcn_mean_percent.toFixed(2)+' ± '+r.gcn_sd_pp.toFixed(2):'no nodes',r.n?r.mlp_mean_percent.toFixed(2)+' ± '+r.mlp_sd_pp.toFixed(2):'no nodes',r.n?r.mean_delta_pp.toFixed(2):'undefined'];for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}table.append(tr);}scroll.replaceChildren(table);
  }
  population.addEventListener('change',draw);family.addEventListener('change',draw);reset.addEventListener('click',()=>{population.value='test';family.value='homophily';draw();});draw();return {draw};
 }
 document.querySelectorAll('[data-error-slices]').forEach(mount);
})();
