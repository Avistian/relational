/* L051: three independently mounted interventions. Synthetic examples, no model scores.
 * States: smoothing h=0/.5/1/1.5; rotation 0/45/90 degrees; noise candidates 1/10/40.
 * Controls retain a baseline, recompute actual arithmetic, clamp values and reset.
 */
(function(g){'use strict';
const blue='#276899',red='#ad4c35',green='#276e58';
const clamp=(x,a,b)=>Math.max(a,Math.min(b,Number(x)));
const fmt=x=>Number(x).toFixed(3);
const text=(x,y,s)=>`<text x="${x}" y="${y}">${s}</text>`;
const svg=s=>`<svg viewBox="0 0 340 240" role="img">${s}</svg>`;
function smoothing(h){h=clamp(h,0,1.5);let w=h===0?[0,1,0]:[Math.exp(-.5/(h*h)),1,Math.exp(-.5/(h*h))];let total=w.reduce((a,b)=>a+b);let p=1/total;return {h,w,p,label:p>.5?1:0};}
function rotation(degrees){degrees=clamp(degrees,0,90);const a=degrees*Math.PI/180,c=Math.cos(a),s=Math.sin(a);return {degrees,c,s,point:[c+2*s,-s+2*c],weight:[c,-s]};}
function gains(count){count=clamp(Math.round(count),1,40);let state=5151;const random=()=>{state=(Math.imul(1664525,state)+1013904223)>>>0;return state/4294967296;};
 const y=Array.from({length:32},(_,i)=>i%2),all=[];
 function best(col){let ix=col.map((v,i)=>i).sort((a,b)=>col[a]-col[b]),sum=0,top=0;
  for(let n=1;n<32;n++){sum+=y[ix[n-1]];const pl=sum/n,pr=(16-sum)/(32-n);top=Math.max(top,.5-n/32*2*pl*(1-pl)-(32-n)/32*2*pr*(1-pr));}return top;}
 for(let j=0;j<40;j++)all.push(best(y.map(()=>random())));
 return {count,all:all.slice(0,count),max:Math.max(...all.slice(0,count)),first:all[0],parameters:64*count};}
function shell(root,label,min,max,step,value){root.className='intervention-viz';root.innerHTML=`<div class="iv-controls"><label>${label} <input aria-label="${label}" type="range" min="${min}" max="${max}" step="${step}" value="${value}"></label><output></output><button type="button">Reset</button></div><div class="iv-pair"><div class="iv-left"></div><div class="iv-right"></div></div><p class="iv-value" aria-live="polite"></p>`;return {input:root.querySelector('input'),out:root.querySelector('output'),left:root.querySelector('.iv-left'),right:root.querySelector('.iv-right'),note:root.querySelector('.iv-value'),reset:root.querySelector('button')};}
function bind(ui,draw,value){ui.input.addEventListener('input',()=>draw(ui.input.value));ui.reset.addEventListener('click',()=>draw(value));draw(value);return {setValue:draw};}
function smooth(root){const ui=shell(root,'Lengthscale h',0,1.5,.05,1);
 function draw(value){const z=smoothing(value);ui.input.value=z.h;ui.out.textContent=z.h.toFixed(2);
  ui.left.innerHTML=svg(text(16,24,'Fixed training rows; query x = 1')+[0,1,2].map((x,i)=>`<circle cx="${65+100*i}" cy="${170-90*(i===1)}" r="7" fill="${blue}"/>`+text(48+100*i,205,`x=${x}`)+text(42+100*i,225,`y=${i===1?1:0}`)).join('')+text(16,50,'Original query label: 1'));
  ui.right.innerHTML=svg(text(16,24,'Gaussian weights into the query')+z.w.map((w,i)=>`<rect x="${38+100*i}" y="${175-110*w}" width="50" height="${110*w}" fill="${i===1?blue:'#c2d2d8'}"/>`+text(37+100*i,195,fmt(w))+text(36+100*i,220,`× y=${i===1?1:0}`)).join(''));
  ui.note.textContent=`Weighted numerator = 1.000; denominator = ${fmt(z.w.reduce((a,b)=>a+b))}. Probability = ${fmt(z.p)} → label ${z.label} using p > 0.5. Validation and test targets stay fixed. Self-weight remains 1.`;}
 return bind(ui,draw,1);}
function board(deg,title){const z=rotation(deg),cx=170,cy=132,scale=32;let body=text(15,23,title)+`<line x1="40" x2="300" y1="132" y2="132" stroke="#b7c5c8"/><line x1="170" x2="170" y1="36" y2="226" stroke="#b7c5c8"/>`;
 for(let i=-2;i<=2;i++)for(let j=-2;j<=2;j++){let a=i*z.c+j*z.s,b=-i*z.s+j*z.c;body+=`<circle cx="${cx+scale*a}" cy="${cy-scale*b}" r="4" fill="${i>.4?green:blue}"/>`;}
 const a=[.4*z.c-2*z.s,-.4*z.s-2*z.c],b=[.4*z.c+2*z.s,-.4*z.s+2*z.c];
 body+=`<line x1="${cx+scale*a[0]}" y1="${cy-scale*a[1]}" x2="${cx+scale*b[0]}" y2="${cy-scale*b[1]}" stroke="${red}" stroke-width="3"/>`;return svg(body);}
function rotate(root){const ui=shell(root,'Rotation degrees',0,90,5,45);function draw(v){const z=rotation(v);ui.input.value=z.degrees;ui.out.textContent=z.degrees+'°';ui.left.innerHTML=board(0,'Original: class 1 if x₀ > 0.4');ui.right.innerHTML=board(z.degrees,'Same rows and labels, rotated axes');ui.note.textContent=`Row (1, 2) → (${fmt(z.point[0])}, ${fmt(z.point[1])}). Weight (1, 0) → (${fmt(z.weight[0])}, ${fmt(z.weight[1])}). Their dot product stays ${fmt(z.point[0]*z.weight[0]+z.point[1]*z.weight[1])}. Distances and information survive; one-column split geometry changes. This transports weights; it does not retrain Adam.`;}return bind(ui,draw,45);}
function noise(root){const ui=shell(root,'Independent noise columns',1,40,1,10);function draw(v){const z=gains(v);ui.input.value=z.count;ui.out.textContent=z.count;ui.left.innerHTML=svg(text(16,25,'Fixed labels; first noise candidate')+text(16,54,'32 balanced synthetic training rows')+`<rect x="50" y="${190-z.first*450}" width="70" height="${z.first*450}" fill="${blue}"/>`+text(45,214,`Gain ${fmt(z.first)}`));ui.right.innerHTML=svg(text(16,25,'More candidates; keep the largest')+z.all.map((a,i)=>`<rect x="${22+i*7.3}" y="${190-a*450}" width="5" height="${a*450}" fill="${a===z.max?red:'#91acb6'}"/>`).join('')+text(16,220,`Maximum training gain: ${fmt(z.max)}`));ui.note.textContent=`Each bar is the best Gini reduction over thresholds for one independent noise column. The maximum cannot decrease when more candidates are offered. A tree can overfit noise too. A width-64 MLP gains ${z.parameters} first-layer weights; parameter count alone does not prove worse test performance. No test scores are shown.`;}return bind(ui,draw,10);}
g.InterventionViz={smoothing,rotation,gains,smooth,rotate,noise};
})(typeof window==='undefined'?globalThis:window);
