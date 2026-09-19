/* Reusable budget accounting: no synthetic model accuracy is implied. */
(function(g){g.LabelBudget={mount:function(root){
 root.innerHTML='<label>Training rows labeled: <input type="range" min="10" max="100" step="10" value="10" aria-label="Training label fraction"></label> <button type="button">Reset to 10%</button><p><output aria-live="polite"></output></p><p>Fixed baseline: 70 training + 100 validation = 170 labels (21.25% of development rows). Test annotations: 200, counted separately.</p>';
 const input=root.querySelector('input'),output=root.querySelector('output');
 function update(){const pct=Number(input.value),n=Math.floor(700*pct/100),total=n+100;output.textContent=pct+'% training fraction → '+n+' training + 100 validation = '+total+' development labels ('+(100*total/800).toFixed(2)+'%). Change from baseline: '+(total-170)+' labels.';}
 input.addEventListener('input',update);root.querySelector('button').addEventListener('click',()=>{input.value=10;update();});update();
}};})(window);
