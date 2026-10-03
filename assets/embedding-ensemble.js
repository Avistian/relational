/* Fixed-pool selection exercise; no claim to implement TabPack online training. */
(function(){'use strict';document.querySelectorAll('.b02-widget').forEach(function(root){
const predictions=[[0,2],[2,0],[5,5]],select=root.querySelector('select'),out=root.querySelector('output');
function update(){const leak=select.value==='test',y=leak?[5,5]:[1,1],ids=[];let sum=[0,0],best=Infinity;
for(let n=0;n<4;n++){let win=0,loss=Infinity;predictions.forEach((p,k)=>{const v=p.reduce((a,x,i)=>a+((sum[i]+x)/(n+1)-y[i])**2,0)/2;if(v<loss){win=k;loss=v;}});if(loss>=best)break;ids.push(win);sum=sum.map((v,i)=>v+predictions[win][i]);best=loss;}
out.dataset.ids=ids.join(',');out.dataset.valid=String(!leak);out.textContent=(leak?'INVALID EVALUATION — test labels chose the predictor. ':'VALID SELECTION — validation labels only. ')+'Chosen: '+ids.map(k=>'ABC'[k]).join(' + ')+'. Selection MSE: '+best.toFixed(2)+'. Freeze these indices before test scoring.';}
select.addEventListener('change',update);root.querySelector('button').addEventListener('click',()=>{select.value='validation';update();});update();
});})();
