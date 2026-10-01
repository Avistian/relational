/* Course-only trace: 0–3 hops, star edge on/off, planet table on/off.
   Token audit: 2 rows, 100 rows, wide row; limits 4/8/1024. No model training. */
(function(global){
 'use strict';
 function context(host){
  host.className='fm-explorer';
  host.innerHTML='<p class="fm-eyebrow">Which rows can reach moon m1?</p><div class="fm-controls"><label>Message-passing hops<select data-hops><option>0</option><option>1</option><option selected>2</option><option>3</option></select></label><label><input type="checkbox" data-edge checked> Retain p1—s1 edge</label><label><input type="checkbox" data-planet checked> Retain planet table</label><button data-reset>Reset</button></div><div class="fm-flow"><div><small>Root · 0 hops</small><strong>m1 · moon</strong><span>Always retained</span></div><div><small>Bridge · 1 hop</small><strong data-bridge></strong><span data-path></span></div><div><small>Context · 2 hops</small><strong data-context></strong><span>z1 stays disconnected</span></div></div><p class="fm-baseline">Fixed baseline: 2 hops, all tables, all edges → m1, m2, p1, s1 (4 rows).</p><output aria-live="polite"></output><p>Access is possible influence, not measured usefulness. The graph is a course example.</p>';
  function draw(){
   const h=Number(host.querySelector('[data-hops]').value),edge=host.querySelector('[data-edge]').checked,planet=host.querySelector('[data-planet]').checked;
   const rows=['m1'];if(h>=1&&planet)rows.push('p1');if(h>=2&&planet){rows.push('m2');if(edge)rows.push('s1');}rows.sort();
   host.querySelector('[data-bridge]').textContent=planet?'p1 · planet':'p1 removed';
   host.querySelector('[data-path]').textContent=planet?'m1 — p1 — m2':'No path out of m1';
   host.querySelector('[data-context]').textContent=planet&&edge?'s1 · star accessible at 2 hops':'Star path blocked';
   host.querySelector('output').textContent=rows.join(', ')+' · '+rows.length+' reachable row'+(rows.length===1?'':'s')+' within '+h+' hops';
  }
  host.addEventListener('change',draw);host.querySelector('[data-reset]').onclick=()=>{host.querySelector('[data-hops]').value='2';host.querySelector('[data-edge]').checked=true;host.querySelector('[data-planet]').checked=true;draw();};draw();
 }
 function tokens(host){
  host.className='fm-explorer';
  host.innerHTML='<p class="fm-eyebrow">Row count is not row width</p><div class="fm-controls"><label>Supplied token lengths<select data-lengths><option value="two">2 rows: 3, 5</option><option value="many">100 rows: 8 each</option><option value="wide">3 rows: 1100, 18, 22</option></select></label><label>Per-row token limit<select data-limit><option>4</option><option>8</option><option>1024</option></select></label><button data-reset>Reset</button></div><div class="fm-flow"><div><small>Whole-table attention</small><strong data-whole></strong><span>Hypothetical; no truncation</span></div><div><small>Independent row attention</small><strong data-rows></strong><span>No truncation</span></div><div><small>Rows after truncation</small><strong data-kept></strong><span>Deletes over-limit tokens</span></div></div><p class="fm-baseline">Fixed baseline: lengths 3/5, limit 4 → 64 whole-table pairs; 34 row pairs; 25 retained pairs; 1 dropped token.</p><output aria-live="polite"></output><p>Ordered token-pair proxy only. Excludes decoder, graph, padding and other model operations; not runtime or total memory.</p>';
  function draw(){
   const kind=host.querySelector('[data-lengths]').value,limit=Number(host.querySelector('[data-limit]').value),ns=kind==='two'?[3,5]:kind==='many'?Array(100).fill(8):[1100,18,22];
   const sum=ns.reduce((a,b)=>a+b,0),rows=ns.reduce((a,b)=>a+b*b,0),kept=ns.reduce((a,b)=>a+Math.min(b,limit)**2,0),dropped=ns.reduce((a,b)=>a+Math.max(0,b-limit),0),over=ns.filter(n=>n>limit).length;
   host.querySelector('[data-whole]').textContent=sum*sum+' pairs';host.querySelector('[data-rows]').textContent=rows+' pairs';host.querySelector('[data-kept]').textContent=kept+' pairs';
   host.querySelector('output').textContent=ns.length+' rows · '+over+' over limit · '+dropped+' dropped tokens';
  }
  host.addEventListener('change',draw);host.querySelector('[data-reset]').onclick=()=>{host.querySelector('[data-lengths]').value='two';host.querySelector('[data-limit]').value='4';draw();};draw();
 }
 global.RelationalVision={context,tokens};
})(window);
