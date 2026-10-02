/* Expected states: 7 holdouts × clean/contaminated. Baseline has 6 candidates;
   synthetic intervention quarantines copy, bridge and tail by transitive closure. */
window.CorpusHoldout={mount(host,records){
  const names=records.map(r=>r.database).sort();
  host.innerHTML='<div class="corpus-controls"><label>Held-out database <select data-holdout></select></label><label><input type="checkbox" data-contaminate> Add synthetic copy and bridge</label><button type="button" data-reset>Reset</button></div><p class="corpus-baseline" data-baseline></p><output class="corpus-readout" aria-live="polite"></output><div class="corpus-list"></div><p class="corpus-note">Candidate means admitted by this declared-lineage split only. Data rights, row integrity and availability require separate evidence.</p>';
  const select=host.querySelector('select');names.forEach(n=>{const o=document.createElement('option');o.value=n;o.textContent=n;select.append(o)});select.value='rel-f1';
  function update(){
    const held=select.value,rows=structuredClone(records),original=rows.find(r=>r.database===held),extra=host.querySelector('input').checked;
    if(extra){rows.push({database:'synthetic-copy',source_families:['synthetic-copy-family'],archive_sha256:original.archive_sha256},{database:'synthetic-bridge',source_families:['synthetic-copy-family','synthetic-tail-family'],archive_sha256:'1'.repeat(64)},{database:'synthetic-tail',source_families:['synthetic-tail-family'],archive_sha256:'2'.repeat(64)});}
    const excluded=new Set([held]);let changed=true;
    while(changed){changed=false;for(const row of rows){if(excluded.has(row.database))continue;const linked=rows.some(other=>excluded.has(other.database)&&(other.archive_sha256===row.archive_sha256||other.source_families.some(f=>row.source_families.includes(f))));if(linked){excluded.add(row.database);changed=true}}}
    const train=rows.filter(r=>!excluded.has(r.database)).map(r=>r.database).sort(),quarantine=[...excluded].filter(n=>n!==held).sort();
    host.dataset.train=JSON.stringify(train);host.dataset.quarantine=JSON.stringify(quarantine);host.dataset.heldout=held;
    host.querySelector('[data-baseline]').textContent='Clean baseline: '+held+' held out · 6 declared training candidates · 0 quarantined relatives.';
    host.querySelector('output').textContent=(extra?'Synthetic intervention':'Clean declared inventory')+': '+train.length+' training candidates · 1 held-out database · '+quarantine.length+' quarantined relatives.';
    const list=host.querySelector('.corpus-list');list.replaceChildren();
    for(const [label,items] of [['Training candidates',train],['Evaluation holdout',[held]],['Quarantine',quarantine]]){const section=document.createElement('section'),heading=document.createElement('h3');heading.textContent=label;section.append(heading);for(const name of items.length?items:['None']){const p=document.createElement('p');p.textContent=name;section.append(p)}list.append(section)}
  }
  host.addEventListener('change',update);host.querySelector('[data-reset]').onclick=()=>{select.value='rel-f1';host.querySelector('input').checked=false;update()};update();
}};
