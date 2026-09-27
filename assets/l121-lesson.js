/* Baseline: B puts lines 2 and 8 in one order => 100; split => 68.
   Flat [count,sum,max] stays [2,10,8]. Native checkbox and reset are keyboard usable. */
(() => {
 const host=document.getElementById('l121-group');
 const compute=shared=>({flat:[2,10,8],orders:shared?[10]:[2,8],signal:shared?100:68});
 if(host){
  host.className='stream-widget';host.innerHTML='<label><input type="checkbox" checked> Put customer B’s two lines in one order</label><button type="button">Reset grouping</button><p><output aria-live="polite"></output></p><p>The flat feature recipe sees only line count, total, and maximum. The nested recipe keeps order ownership before summing into the customer.</p>';
  const input=host.querySelector('input'),out=host.querySelector('output');
  function update(){const r=compute(input.checked);out.textContent=`Customer A: 2² + 8² = 68. Customer B order totals: [${r.orders.join(', ')}]; sum of squared totals = ${r.signal}. Both flat vectors: [2, 10, 8]. Baseline: one order gives 100.`;}
  input.addEventListener('change',update);host.querySelector('button').addEventListener('click',()=>{input.checked=true;update();});update();
 }
 window.L121Grouping={compute};
 if(window.RetrievalBank)RetrievalBank.mount(document.getElementById('l121-warmup'),{upTo:121,count:3});
 if(window.Teachback)Teachback.mount(document.getElementById('l121-teachback'),{prompt:'Why does the 68 versus 100 example support neither a universal GNN win nor a claim that relational learning began in 2019?',points:['The selected flat recipe loses order ownership.','A nested engineered feature repairs the collision.','ILP and automated relational features predate this GNN.','Author checks are distinct from five-fold predictive evidence.'],model:'The flat vector preserves only count, total and maximum. Grouping by order before squaring produces 68 versus 100, and both explicit feature engineering and a suitable graph computation can retain that distinction. Earlier relational learning already used relationships. The actual model comparison still needs aligned data, protocols and measured scores.'});
})();
