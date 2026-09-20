/* Reusable mean-of-means intervention. Fixed individual losses; only batch partition changes.
   Default size3: weighted .600, naive .625. Size1/5: both .600; size4: naive .525. */
window.BatchWeighting={mount:function(root){
 const losses=[.2,.8,.5,1.1,.4];
 root.classList.add('batch-weighting');
 root.innerHTML='<p><strong>Predict:</strong> does changing only the batch size change the correct five-seed mean?</p><div class="batch-controls"><label>Seeds per batch <select aria-label="Seeds per batch">'+[1,2,3,4,5].map(n=>'<option '+(n===3?'selected':'')+'>'+n+'</option>').join('')+'</select></label><button type="button">Reset</button></div><p>Fixed individual losses: [0.20, 0.80, 0.50, 1.10, 0.40]</p><div class="batch-rows"></div><output aria-live="polite"></output>';
 const select=root.querySelector('select');
 function draw(){let n=+select.value,groups=[];for(let i=0;i<losses.length;i+=n)groups.push(losses.slice(i,i+n));let means=groups.map(g=>g.reduce((a,b)=>a+b,0)/g.length),correct=groups.reduce((s,g,i)=>s+g.length/5*means[i],0),naive=means.reduce((a,b)=>a+b,0)/means.length;
 root.querySelector('.batch-rows').innerHTML=groups.map((g,i)=>'<p><strong>Batch '+(i+1)+'</strong> · '+g.length+' seeds · mean '+means[i].toFixed(3)+' · weight '+g.length+'/5 → contribution '+(means[i]*g.length/5).toFixed(3)+'</p>').join('');
 root.querySelector('output').textContent='Weighted mean: '+correct.toFixed(3)+' | Naive batch mean: '+naive.toFixed(3)+' | Fixed full mean: 0.600';}
 select.addEventListener('change',draw);root.querySelector('button').addEventListener('click',()=>{select.value='3';draw();});draw();
}};
