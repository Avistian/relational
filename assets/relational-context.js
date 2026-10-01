/* Course temporal contract explorer. Not a KumoRFM inference client. */
(function(global){
'use strict';
function mount(host){
 host.className='fm-explorer';
 host.innerHTML='<p class="fm-eyebrow">When may this example enter the prompt?</p><div class="fm-controls"><label>Query day<select data-query><option>8</option><option selected>10</option><option>12</option><option>13</option><option>16</option></select></label><label>Context horizon<select data-horizon><option>0</option><option>2</option><option selected>4</option></select></label><label>Arrival delay<select data-delay><option>0</option><option selected>1</option><option>3</option></select></label><label>Hidden query label<select data-hidden><option>0</option><option>1</option></select></label><button data-reset>Reset</button></div><div class="fm-flow"><div><small>Context graph</small><strong>Cutoff = day 8</strong><span>Never advances with query time</span></div><div><small>Outcome window</small><strong data-end></strong><span data-arrival></span></div><div><small>Query graph</small><strong data-now></strong><span>Target label stays outside input</span></div></div><p class="fm-baseline">Fixed baseline: anchor 8, horizon 4, delay 1, query 10 → EXCLUDE; the label arrives on day 13.</p><output aria-live="polite"></output><p data-reasons></p><p>Course contract only. No model prediction or proprietary mask is executed.</p>';
 function draw(){
  const q=Number(host.querySelector('[data-query]').value),h=Number(host.querySelector('[data-horizon]').value),delay=Number(host.querySelector('[data-delay]').value),end=8+h,arrival=end+delay;
  const reasons=[];if(8>=q)reasons.push('anchor is not earlier');if(end>q)reasons.push('outcome window is unfinished');if(arrival>q)reasons.push('label has not arrived');
  host.querySelector('[data-end]').textContent='Ends on day '+end;
  host.querySelector('[data-arrival]').textContent='Available on day '+arrival;
  host.querySelector('[data-now]').textContent='Cutoff = day '+q;
  host.querySelector('output').textContent=(reasons.length?'EXCLUDE':'INCLUDE')+' · context graph cutoff 8 · query graph cutoff '+q;
  host.querySelector('[data-reasons]').textContent=reasons.length?reasons.join('; ')+'.':'Earlier anchor, complete window and arrived label. Input packet is independent of the hidden query label.';
 }
 host.addEventListener('change',draw);host.querySelector('[data-reset]').onclick=()=>{for(const [key,value] of Object.entries({query:10,horizon:4,delay:1,hidden:0}))host.querySelector('[data-'+key+']').value=String(value);draw();};draw();
}
global.RelationalContext={mount};
})(window);
