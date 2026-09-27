/* Typed incoming expansion: the same finite-fanout contract as the lab. */
(() => {
 const host=document.querySelector('#l134-budget');if(!host)return;
 host.innerHTML=`<h3>Build the sampled frontier</h3><p>Schema: orders → users; items → orders; users → orders. Start from user queries. Baseline B=2, fanouts=[3,2].</p><label>Seed queries <input aria-label="Seed queries" type="range" min="1" max="8" value="2"></label><label>First-hop fanout <input aria-label="First-hop fanout" type="range" min="0" max="8" value="3"></label><label>Second-hop fanout <input aria-label="Second-hop fanout" type="range" min="0" max="8" value="2"></label><output aria-live="polite"></output><button type="button">Reset</button><p>Counts are no-collision occurrence bounds. The memory number covers one width-128 float32 matrix only.</p>`;
 const inputs=[...host.querySelectorAll('input')],out=host.querySelector('output');
 function render(){const [b,f,g]=inputs.map(x=>Number(x.value)),one=b*f,two=one*g,n=b+one+2*two;out.textContent=`B=${b}, fanouts=[${f},${g}]: users ${b} → orders ${one} → items ${two} + users ${two}. Total ${n} occurrences; ${n-b} edges; one hidden matrix ${(n*128*4/1024).toFixed(1)} KiB. Baseline: 32 occurrences, 16.0 KiB.`;}
 inputs.forEach(x=>x.addEventListener('input',render));host.querySelector('button').addEventListener('click',()=>{[2,3,2].forEach((v,i)=>inputs[i].value=v);render();});render();
})();
