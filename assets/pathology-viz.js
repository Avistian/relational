/* Exact scalar diagnostics; these are not fitted neural predictions. */
(() => {
  'use strict';
  function mount(id,hub){
    const host=document.getElementById(id);if(!host)return;
    const labels=hub?[['s','Source',2],['f','Fact',1],['d','Destination',3],['n','Third role',8]]:[['s','Source',2],['f','Fact',1],['d','Destination',3]];
    host.innerHTML='<h3>'+(hub?'Change one role, then swap source and third role':'Count each original input')+'</h3>'+labels.map(([key,label,value])=>`<label for="${id}-${key}">${label} <input id="${id}-${key}" data-key="${key}" type="range" min="0" max="10" step="1" value="${value}"><output for="${id}-${key}">${value}</output></label>`).join('')+'<p class="path-readout" aria-live="polite"></p>'+(hub?'<button type="button" data-swap>Swap source and third role</button> ':'')+'<button type="button" data-reset>Reset baseline</button>';
    const get=k=>Number(host.querySelector(`[data-key="${k}"]`)?.value||0);
    function update(){host.querySelectorAll('input').forEach(x=>x.nextElementSibling.value=x.value);const s=get('s'),f=get('f'),d=get('d'),n=get('n');host.querySelector('.path-readout').textContent=`Fact after ordinary layer 1: ${s+f+d+n}. Destination after layer 1: ${d+f}. Ordinary after layer 2: ${s+2*f+2*d+n}. Source-route composite: ${s+f+d}. ${hub?'Ordinary sees the sum of source and third role; the source route excludes the third role.':'Baseline: ordinary 10; composite 6.'}`;}
    host.addEventListener('input',update);host.querySelector('[data-reset]').onclick=()=>{labels.forEach(([key,,value])=>host.querySelector(`[data-key="${key}"]`).value=value);update();};
    if(hub)host.querySelector('[data-swap]').onclick=()=>{const s=get('s'),n=get('n');host.querySelector('[data-key="s"]').value=n;host.querySelector('[data-key="n"]').value=s;update();};update();
  }
  mount('l142-walks',false);mount('l142-hub',true);
  if(window.RetrievalBank&&document.getElementById('warmup'))RetrievalBank.mount(document.getElementById('warmup'),{upTo:142,count:3});
  if(window.Teachback&&document.getElementById('l142-teachback'))Teachback.mount(document.getElementById('l142-teachback'),{prompt:'Derive the return path, construct a harmful collision, and explain the limits of our benchmark comparison.',points:['Derive d₂=s+2f+2d+n under the stated update.','Use swapped source/third-role values with a source-dependent target.','Distinguish a scalar collision from a theorem about all learned GNNs.','Name capacity, RNG, preprocessing and gradient-health limitations.'],model:'The destination can travel d→fact→d while its root path also survives. Equal source-plus-third-role sums collide under the scalar diagnostic. Source-specific fusion separates those roles before attention. The trained networks have different capacity and nonlinear operations, so their error difference does not identify that collision as its cause.'});
})();
