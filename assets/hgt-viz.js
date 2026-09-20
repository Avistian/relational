/* Synthetic HGT edge trace. Baseline prior1 -> weights1/3,2/3 ->14/3.
   Prior2 -> weights.2,.8 ->5.2. Normalization is shared across relations. */
(function(global){'use strict';global.HGTViz={mount:function(el){
 if(!el)return;
 el.className='hgt-viz';el.innerHTML='<h3>Change one relation prior</h3><p>Hold the query, keys and messages fixed. The author score stays 0; the citation score is μ × ln2.</p><label>Citation prior μ <input type="range" min="0" max="3" step="0.25" value="1" aria-label="Citation prior"></label><button type="button">Reset</button><p>Baseline: author 1/3; citation 2/3; output 4.667.</p><div class="hgt-bars" aria-hidden="true"><span></span><span></span></div><output aria-live="polite"></output><p>The displayed output is the first message coordinate, before GELU, residual mixing and normalization.</p>';
 var input=el.querySelector('input'),out=el.querySelector('output'),bars=el.querySelectorAll('.hgt-bars span');
 function render(){var mu=Number(input.value),b=Math.pow(2,mu),author=1/(1+b),citation=b/(1+b),value=2*author+6*citation;out.textContent='μ '+mu.toFixed(2)+' · author '+author.toFixed(3)+' · citation '+citation.toFixed(3)+' · output '+value.toFixed(3);bars[0].style.width=(100*author)+'%';bars[1].style.width=(100*citation)+'%';bars[0].textContent='Author';bars[1].textContent='Citation';}
 input.addEventListener('input',render);el.querySelector('button').addEventListener('click',function(){input.value=1;render();});render();
}};})(window);
