/* DCNv2 Eqs.1–4. Synthetic numerical illustrations; no benchmark numbers here.
 * Each control recomputes the pictured tensors. Narrow frames use vertical flow.
 */
(function(g){
'use strict';
const NS='http://www.w3.org/2000/svg';
const C={ink:'#203c50',blue:'#286da3',teal:'#117d71',gold:'#b16d21',line:'#9ab0bf',soft:'#edf4f8',purple:'#78559c'};
const clamp=(v,a,b)=>Number.isFinite(+v)?Math.max(a,Math.min(b,+v)):a;
const fmt=v=>Math.abs(v)<1e-9?'0':Number(v.toFixed(2)).toString();
const matvec=(w,x)=>w.map(row=>row.reduce((s,v,j)=>s+v*x[j],0));
function cross(t){const x=[2,3],w=[[1,t],[0,1]],b=[1,-1],z=matvec(w,x).map((v,i)=>v+b[i]);return {x,w,b,z,product:z.map((v,i)=>v*x[i]),out:z.map((v,i)=>x[i]+v*x[i])};}
function lowrank(r){r=Math.round(clamp(r,1,4));const h=[[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]].map(a=>a.map(v=>v/2)),s=[4,2,1,.25];const full=h.map(row=>row.map((v,j)=>v*s[j])),approx=full.map(row=>row.map((v,j)=>j<r?v:0));return {r,h,s,full,approx,error:Math.sqrt(s.slice(r).reduce((a,v)=>a+v*v,0)),cost:8*r};}
function mix(t){const gate=1/(1+Math.exp(-t)),x=[2,3],e1=[2*Math.tanh(Math.tanh(2)),0],e2=[0,3*Math.tanh(Math.tanh(3))];return {gate,e1,e2,out:x.map((v,i)=>v+gate*e1[i]+(1-gate)*e2[i])};}
g.DcnMath={cross,lowrank,mix};
function el(t,text,p){const n=document.createElement(t);if(text!=null)n.textContent=text;if(p)p.appendChild(n);return n;}
function sv(t,a,p,text){const n=document.createElementNS(NS,t);Object.keys(a||{}).forEach(k=>n.setAttribute(k,a[k]));if(text!=null)n.textContent=text;if(p)p.appendChild(n);return n;}
function label(s,x,y,text,size=14,color=C.ink,anchor='middle'){return sv('text',{x,y,'font-family':'DejaVu Sans, sans-serif','font-size':size,fill:color,'text-anchor':anchor},s,text);}
function arrow(s,x1,y1,x2,y2,color=C.line){sv('line',{x1,y1,x2,y2,stroke:color,'stroke-width':2},s);const a=Math.atan2(y2-y1,x2-x1),l=7;sv('path',{d:`M ${x2-l*Math.cos(a-.45)} ${y2-l*Math.sin(a-.45)} L ${x2} ${y2} L ${x2-l*Math.cos(a+.45)} ${y2-l*Math.sin(a+.45)}`,fill:'none',stroke:color,'stroke-width':2},s);}
function box(s,x,y,w,h,title,lines,color=C.blue){const n=sv('g',{'data-box':'1'},s);sv('rect',{x,y,width:w,height:h,rx:7,fill:'#fff',stroke:color,'stroke-width':1.5},n);label(n,x+w/2,y+24,title,14,color);lines.forEach((line,i)=>label(n,x+w/2,y+47+20*i,line,13));return n;}
function matrix(s,x,y,a,title,cell=40){const w=a[0].length*cell;label(s,x+w/2,y-13,title,14);a.forEach((row,i)=>row.forEach((v,j)=>{sv('rect',{x:x+j*cell,y:y+i*cell,width:cell,height:cell,fill:v===0?'#fff':v>0?'#d9eee8':'#fae8d8',stroke:'#b9cbd4'},s);label(s,x+(j+.5)*cell,y+(i+.63)*cell,fmt(v),13);}));}
const titles={cross:'01 · Read one cross layer, coordinate by coordinate',degree:'02 · Keep the original input anchored',rank:'03 · Trade matrix freedom for a smaller subspace',architecture:'04 · Place the deep network in the actual data path',mix:'05 · Let two nonlinear experts contribute'};
function mount(root,options){
 const kind=options.kind;while(root.firstChild)root.removeChild(root.firstChild);root.className='dcn-viz';el('h3',titles[kind],root);
 const controls=el('div',null,root);controls.className='dcn-controls';
 const svg=sv('svg',{xmlns:NS,role:'img','aria-label':titles[kind]},root);
 const output=el('output',null,root);output.setAttribute('aria-live','polite');
 const small=root.clientWidth<500,W=small?340:600;let value=kind==='degree'?2:kind==='rank'?2:kind==='architecture'?0:1;
 let input;
 function slider(name,min,max,step){const lab=el('label',null,controls);el('span',name,lab);input=el('input',null,lab);input.type='range';input.min=min;input.max=max;input.step=step;input.value=value;input.setAttribute('aria-label',name);input.addEventListener('input',()=>set(+input.value));}
 if(kind==='cross')slider('Weight W₁₂',-1,2,.25);
 if(kind==='degree')slider('Cross layers L',1,4,1);
 if(kind==='rank')slider('Retained rank r',1,4,1);
 if(kind==='mix')slider('Expert 1 gate logit',-4,4,.5);
 if(kind==='architecture')['Parallel','Stacked'].forEach((name,i)=>{const b=el('button',name,controls);b.type='button';b.addEventListener('click',()=>set(i));});
 function draw(){
  while(svg.firstChild)svg.removeChild(svg.firstChild);
  let caption='',H=500;
  if(kind==='cross'){
   const q=cross(value);H=small?665:450;
   label(svg,W/2,23,'Illustration: x₀ = xₗ = [2, 3]',15,C.ink);
   if(small){
    matrix(svg,30,62,q.w,'W');matrix(svg,160,62,[[2],[3]],'xₗ');matrix(svg,250,62,[[1],[-1]],'b');
    label(svg,170,173,'W xₗ + b = ['+q.z.map(fmt).join(', ')+']',16,C.blue);
    arrow(svg,170,190,170,218);
    box(svg,35,228,270,95,'Multiply by the ORIGINAL input',['[2, 3] ⊙ ['+q.z.map(fmt).join(', ')+']','= ['+q.product.map(fmt).join(', ')+']'],C.teal);
    arrow(svg,170,323,170,355);
    box(svg,35,365,270,95,'Add the CURRENT residual',['[2, 3] + ['+q.product.map(fmt).join(', ')+']','= ['+q.out.map(fmt).join(', ')+']'],C.gold);
    box(svg,15,500,310,130,'Where did x₁x₂ appear?',['First coordinate:','2 + 2 × (1 × 2 + '+fmt(value)+' × 3 + 1)','The 2 × '+fmt(value)+' × 3 term is W₁₂x₁x₂.'],C.purple);
   }else{
    matrix(svg,45,73,q.w,'W');matrix(svg,170,73,[[2],[3]],'xₗ');matrix(svg,260,73,[[1],[-1]],'b');
    arrow(svg,318,112,358,112);
    box(svg,370,65,200,100,'Affine mixture',['W xₗ + b','['+q.z.map(fmt).join(', ')+']']);
    arrow(svg,470,165,470,205);
    box(svg,350,215,240,90,'Original × mixture',['[2, 3] ⊙ ['+q.z.map(fmt).join(', ')+']','= ['+q.product.map(fmt).join(', ')+']'],C.teal);
    arrow(svg,350,260,292,260);
    box(svg,20,215,260,90,'Residual + crossed update',['[2, 3] + ['+q.product.map(fmt).join(', ')+']','= ['+q.out.map(fmt).join(', ')+']'],C.gold);
    label(svg,300,356,'First coordinate: 2 + 2 × (1 × 2 + '+fmt(value)+' × 3 + 1)',16);
    label(svg,300,390,'W₁₂ = '+fmt(value)+' controls the x₁x₂ contribution in coordinate 1.',14,C.purple);
   }
   caption='Synthetic values, Eq.1. Move W₁₂: only the first coordinate changes. Bias is multiplied by x₀; the residual is xₗ. Output = ['+q.out.map(fmt).join(', ')+'].';
  }else if(kind==='degree'){
   const L=value;H=small?650:420;
   label(svg,W/2,26,'Scalar case: every w = 1, every b = 0',14);
   if(small){
    box(svg,55,55,230,65,'Original x₀ = x',['Fixed multiplier at every layer'],C.teal);
    for(let i=0;i<4;i++){const y=155+i*102;box(svg,70,y,255,73,'Layer '+(i+1)+(i<L?' · active':' · beyond L'),['x × (1 + x)'+['¹','²','³','⁴'][i]],i<L?C.blue:C.line);arrow(svg,35,90,35,y+35,C.teal);arrow(svg,35,y+35,67,y+35,C.teal);if(i<3)arrow(svg,198,y+73,198,y+99);}
   }else{
    box(svg,190,50,220,65,'Original x₀ = x',['Reuse; never replace by xₗ'],C.teal);
    for(let i=0;i<4;i++){const x=8+i*150;box(svg,x,180,136,83,'Layer '+(i+1),['x (1+x)'+['¹','²','³','⁴'][i],i<L?'active':'beyond L'],i<L?C.blue:C.line);arrow(svg,300,115,x+68,172,C.teal);if(i<3)arrow(svg,x+136,225,x+146,225);}
   }
   const expansions=['x + x²','x + 2x² + x³','x + 3x² + 3x³ + x⁴','x + 4x² + 6x³ + 4x⁴ + x⁵'];
   label(svg,W/2,H-62,'At L = '+L+': '+expansions[L-1],small?12:16,C.purple);
   label(svg,W/2,H-29,'At x = 2, output = '+(2*3**L)+'; degree ≤ '+(L+1)+'.',14);
   caption='Each layer multiplies a degree-≤L state by degree-1 x₀, then preserves the residual. At '+L+' layers the bound is '+(L+1)+'. This statement is about the linear cross stack in x₀, before an MLP or sigmoid.';
  }else if(kind==='rank'){
   const q=lowrank(value);H=small?650:425;
   const cx=small?90:30,cy=65;matrix(svg,cx,cy,q.full,'Full W = H diag(4, 2, 1, .25)',small?40:38);
   const ax=small?90:385,ay=small?310:65;matrix(svg,ax,ay,q.approx,'Rank '+q.r+' reconstruction',small?40:38);
   if(small)arrow(svg,170,243,170,267);else arrow(svg,220,142,345,142);
   const by=small?500:278;
   box(svg,small?20:25,by,small?300:550,85,'Project down → project back',['Vᵀ: 4 → '+q.r+'; U: '+q.r+' → 4',small?'Weights: '+q.cost+' factored versus 16 dense':'W ≈ UVᵀ; '+q.cost+' weights versus 16 dense weights'],C.teal);
   label(svg,W/2,H-30,'Reconstruction error ‖W − UVᵀ‖F = '+fmt(q.error),14,C.purple);
   caption='Constructed SVD example: H has orthonormal columns; V = identity, singular values 4, 2, 1, .25. Rank '+q.r+' uses '+q.cost+' weights (bias excluded). Strict savings need r < d/2. Error is '+fmt(q.error)+'. This is matrix error, not measured prediction loss.';
  }else if(kind==='architecture'){
   H=small?670:490;const stacked=value===1;
   box(svg,W/2-135,25,270,75,'Categoricals + numeric values',['Embeddings concatenate into x₀'],C.teal);
   if(stacked){
    arrow(svg,W/2,100,W/2,133);
    box(svg,W/2-125,145,250,90,'Cross stack',['x ← x + x₀ ⊙ (Wx + b)','Repeat L times; x₀ stays fixed']);
    arrow(svg,W/2,235,W/2,268);
    box(svg,W/2-125,280,250,70,'Deep network',['MLP reads xL'],C.purple);
    arrow(svg,W/2,350,W/2,385);
    box(svg,W/2-125,395,250,65,'Linear head → sigmoid',['One click / positive probability'],C.gold);
   }else if(small){
    box(svg,40,145,260,80,'Cross branch',['xL = Cross(x₀)']);arrow(svg,170,100,170,135);
    box(svg,40,280,260,80,'Deep branch',['h = MLP(x₀), independently'],C.purple);arrow(svg,20,65,20,320,C.teal);arrow(svg,20,320,35,320,C.teal);arrow(svg,35,65,20,65,C.teal);
    box(svg,40,420,260,75,'Concatenate [xL ; h]',['Both branches reach the head'],C.gold);arrow(svg,312,185,312,450);arrow(svg,305,185,312,185);arrow(svg,312,450,304,450);arrow(svg,170,360,170,410);
    box(svg,40,555,260,65,'Linear head → sigmoid',['One positive probability'],C.gold);arrow(svg,170,495,170,543);
   }else{
    arrow(svg,230,100,155,153);arrow(svg,370,100,445,153);
    box(svg,25,165,260,95,'Cross branch',['xL = Cross(x₀)','Keep x₀ anchored']);
    box(svg,315,165,260,95,'Deep branch',['h = MLP(x₀)','Read the ORIGINAL input'],C.purple);
    arrow(svg,155,260,230,305);arrow(svg,445,260,370,305);
    box(svg,135,318,330,65,'Concatenate [xL ; h]',['Linear head reads BOTH branches'],C.gold);
    arrow(svg,300,383,300,420);label(svg,300,445,'Sigmoid → one positive probability',16,C.gold);
   }
   caption=(stacked?'Stacked: the MLP consumes the crossed state.':'Parallel: the MLP consumes x₀; the head consumes the concatenation of crossed and deep states.')+' Every arrow belongs to a single row. No other examples enter this computation.';
  }else if(kind==='mix'){
   const q=mix(value);H=small?650:465;
   label(svg,W/2,25,'Synthetic K = 2, rank = 1, shared bias = 0',14);
   const bx=small?35:20,bw=small?270:265,y1=65,y2=small?235:65,x2=small?35:315;
   box(svg,bx,y1,bw,125,'Expert 1 · coordinate 1',['V₁ᵀx = 2 → tanh → C₁ = 1','→ tanh → U₁: expand to [v, 0]','E₁ = ['+q.e1.map(fmt).join(', ')+']'],C.blue);
   box(svg,x2,y2,bw,125,'Expert 2 · coordinate 2',['V₂ᵀx = 3 → tanh → C₂ = 1','→ tanh → U₂: expand to [0, v]','E₂ = ['+q.e2.map(fmt).join(', ')+']'],C.purple);
   const gy=small?405:250;const left=small?25:80,total=small?290:440;
   label(svg,W/2,gy-25,'softmax(['+fmt(value)+', 0])',15);
   sv('rect',{x:left,y:gy,width:total*q.gate,height:30,fill:C.blue},svg);
   sv('rect',{x:left+total*q.gate,y:gy,width:total*(1-q.gate),height:30,fill:C.purple},svg);
   label(svg,W/2,gy+57,'g₁ = '+fmt(q.gate)+'; g₂ = '+fmt(1-q.gate),14);
   box(svg,small?20:60,gy+85,small?300:480,85,'Add residual ONCE',['[2,3] + g₁E₁ + g₂E₂','= ['+q.out.map(fmt).join(', ')+']'],C.teal);
   caption='Both expert outputs already contain x₀ multiplication. Gates mix those updates, then add xₗ once. Tanh and input-dependent softmax make this a nonlinear mixture; the L+1 polynomial bound no longer describes it.';
  }
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);output.textContent=caption;
 }
 function set(v){const bounds={cross:[-1,2],degree:[1,4],rank:[1,4],architecture:[0,1],mix:[-4,4]}[kind];value=clamp(v,...bounds);if(['degree','rank','architecture'].includes(kind))value=Math.round(value);if(input)input.value=value;draw();return value;}
 set(value);return {set,svg,output};
}
g.DcnViz={mount};
})(window);
