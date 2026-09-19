/* Link prediction arithmetic. Leak default H=[0,1,0], intervention [0,1,1].
   Rank default score .5 versus [.6,.5,.1]: rank2.5, reciprocal .4.
   Native controls, explicit baseline, reset; no stochastic or trained-model claims. */
(function(global){
'use strict';
function rank(p,neg){return 1+neg.filter(x=>x>p).length+.5*neg.filter(x=>x===p).length;}
function leak(el){
 if(!el)return;
 el.classList.add('lp-viz');
 el.innerHTML='<h3>Does a hidden reverse edge still send a message?</h3><label><input type="checkbox"> Include forbidden A → C message</label><p>Input stays X=[1,0,0]. Allowed messages follow A—B—C.</p><div class="lp-pair"><p><strong>Safe baseline</strong><br>H = [0,1,0]</p><p><strong>Current graph</strong><br><output aria-live="polite"></output></p></div><button type="button">Reset</button>';
 const c=el.querySelector('input'),o=el.querySelector('output');
 const draw=()=>{o.textContent=c.checked?'H = [0,1,1]; C receives 1 directly from A.':'H = [0,1,0]; C receives 0 from B.';};
 c.addEventListener('change',draw);el.querySelector('button').onclick=()=>{c.checked=false;draw();};draw();
}
function ranking(el){
 if(!el)return;
 el.classList.add('lp-viz');
 el.innerHTML='<h3>Move the positive score; keep candidates fixed</h3><label>Positive score <input type="range" min="0" max="8" step="1" value="5"></label><p>Negative scores remain [.6,.5,.1]. The .5 baseline has rank 2.5 and reciprocal rank .400.</p><output aria-live="polite"></output><p class="lp-order"></p><button type="button">Reset</button>';
 const c=el.querySelector('input'),o=el.querySelector('output');
 const draw=()=>{const p=Number(c.value)/10,n=[.6,.5,.1],r=rank(p,n);o.textContent=`Positive ${p.toFixed(1)} · rank ${r} · reciprocal ${(1/r).toFixed(3)} · Hits@1 ${r<=1?1:0}`;el.querySelector('.lp-order').textContent=[{s:p,t:'positive'},...n.map(s=>({s,t:'negative'}))].sort((a,b)=>b.s-a.s).map(x=>`${x.s.toFixed(1)} (${x.t})`).join(' → ');};
 c.addEventListener('input',draw);el.querySelector('button').onclick=()=>{c.value='5';draw();};draw();
}
global.LinkPredictionViz={rank,mountLeak:leak,mountRanking:ranking};
})(window);
