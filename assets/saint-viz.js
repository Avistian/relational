/**
 * SAINT visual explanations, §3/Figs.1–2/Algorithm 1 and §4/Eqs.3–5.
 * Drawings follow the in-repo supervised stage; synthetic values are labeled.
 * Expected states:
 * - Axes: tokens → feature attention → entire-row attention; matrix axes change.
 * - Architecture: inspect embedding, feature block, row block, or CLS readout.
 * - Context: fixed q=1, self k=1/v=2, companion k slider/v=8; exact softmax.
 * - Views: CutMix swaps a raw category; mixup moves along an embedding segment.
 * - Contrast: clean/augmented identity pairs stay diagonal; temperature changes loss.
 * Every drawing switches to a vertical layout below 440px. No raster dependencies.
 */
(function (global) {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';
  var C = {ink:'#243b4a',muted:'#657582',line:'#bbc9d0',blue:'#2874a6',teal:'#147d70',gold:'#a97819',purple:'#8056a3',paper:'#f5f8fa'};
  function el(tag,text,parent,cls) { var n=document.createElement(tag); if(text!=null)n.textContent=text;if(cls)n.className=cls;if(parent)parent.appendChild(n);return n; }
  function sv(tag,attrs,parent) {var n=document.createElementNS(NS,tag);Object.keys(attrs||{}).forEach(function(k){n.setAttribute(k,attrs[k]);});if(parent)parent.appendChild(n);return n;}
  function clear(n) {while(n.firstChild)n.removeChild(n.firstChild);}
  function clamp(x,a,b) {x=Number(x);return Number.isFinite(x)?Math.max(a,Math.min(b,x)):a;}
  function weights(logits) {var m=Math.max.apply(null,logits),v=logits.map(function(x){return Math.exp(x-m);}),s=v.reduce(function(a,b){return a+b;},0);return v.map(function(x){return x/s;});}
  function context(key) {var a=weights([1,key]);return {weights:a,value:a[0]*2+a[1]*8};}
  function mix(a,b,alpha) {return a.map(function(v,i){return alpha*v+(1-alpha)*b[i];});}
  function contrast(temp) {var a=weights([1/temp,0,0]);return {weights:a,loss:-Math.log(a[0])};}
  global.SaintMath={weights:weights,context:context,mix:mix,contrast:contrast};
  function frame(root,number,title,sub) {
    if(root._saintDispose)root._saintDispose();clear(root);root.className='saint-viz';
    var header=el('div',null,root,'saint-heading');el('span',number,header,'saint-figure-no');el('h3',title,header);
    el('p',sub,root,'saint-intro');var controls=el('div',null,root,'saint-controls');
    var svg=sv('svg',{xmlns:NS,role:'img',class:'saint-svg'},root);
    var out=el('output',null,root,'saint-readout');out.setAttribute('aria-live','polite');
    return {root:root,controls:controls,svg:svg,out:out};
  }
  function narrow(f) {return (f.root.clientWidth||640)<440;}
  function canvas(f,height,title) {clear(f.svg);var w=narrow(f)?320:600;f.svg.setAttribute('viewBox','0 0 '+w+' '+height);f.svg.setAttribute('width','100%');sv('title',{},f.svg).textContent=title;return w;}
  function text(svg,x,y,value,opts) {var a={x:x,y:y,fill:C.ink,'font-family':'Arial, sans-serif','font-size':13,'text-anchor':'middle'};Object.assign(a,opts||{});var n=sv('text',a,svg);n.textContent=value;return n;}
  function rect(svg,x,y,w,h,fill,stroke,extra) {return sv('rect',Object.assign({x:x,y:y,width:w,height:h,rx:7,fill:fill||'#fff',stroke:stroke||C.line,'stroke-width':1.2},extra||{}),svg);}
  function card(svg,x,y,w,h,lines,color,active) {var g=sv('g',{'data-box':'true'},svg);rect(g,x,y,w,h,active===false?'#fafbfc':'#fff',active===false?C.line:color);rect(g,x,y,4,h,active===false?C.line:color,'none');lines.forEach(function(l,i){text(g,x+w/2,y+h/2+(i-(lines.length-1)/2)*19+4,l,{fill:active===false?C.muted:(i===0?color:C.ink),'font-size':i===0?14:12,'font-weight':i===0?600:400});});return g;}
  function arrow(svg,x1,y1,x2,y2,color,width,dash) {color=color||C.line;sv('line',{x1:x1,y1:y1,x2:x2,y2:y2,stroke:color,'stroke-width':width||2,'stroke-dasharray':dash?'5 4':'none'},svg);var a=Math.atan2(y2-y1,x2-x1),r=7;sv('path',{d:'M '+(x2-r*Math.cos(a-.5))+' '+(y2-r*Math.sin(a-.5))+' L '+x2+' '+y2+' L '+(x2-r*Math.cos(a+.5))+' '+(y2-r*Math.sin(a+.5)),fill:'none',stroke:color,'stroke-width':width||2},svg);}
  function buttons(f,labels,action) {return labels.map(function(label,i){var b=el('button',label,f.controls);b.type='button';b.addEventListener('click',function(){action(i);});return b;});}
  function selected(bs,i) {bs.forEach(function(b,j){b.setAttribute('aria-pressed',String(i===j));});}
  function range(f,label,min,max,step,value,action) {var wrap=el('label',null,f.controls,'saint-slider'),line=el('span',null,wrap,'saint-slider-label');el('span',label,line);var val=el('strong',String(value),line);var input=el('input',null,wrap);input.type='range';input.min=min;input.max=max;input.step=step;input.value=value;input.addEventListener('input',function(){action(Number(input.value));});return {input:input,label:val};}
  function responsive(f,draw) {if(global.ResizeObserver){var previous=narrow(f),observer=new global.ResizeObserver(function(){var n=narrow(f);if(n!==previous){previous=n;draw();}});observer.observe(f.root);f.root._saintDispose=function(){observer.disconnect();};}}
  function matrix(svg,x,y,labels,color,focus,values,cell) {cell=cell||38;
    labels.forEach(function(l,i){text(svg,x+(i+.5)*cell,y-12,l,{'font-size':12});text(svg,x-12,y+(i+.5)*cell+4,l,{'font-size':12,'text-anchor':'end'});});
    for(var i=0;i<labels.length;i++)for(var j=0;j<labels.length;j++){
      var opacity=values?(.1+.8*values[i][j]):(i===focus?.65:.13);
      rect(svg,x+j*cell+2,y+i*cell+2,cell-4,cell-4,color,'none',{'fill-opacity':opacity,rx:4});
      if(values)text(svg,x+(j+.5)*cell,y+(i+.5)*cell+4,values[i][j].toFixed(2),{'font-size':12,fill:values[i][j]>.55?'#fff':C.ink});
    }
  }
  global.SaintAxesViz={mount:function(root){
    var f=frame(root,'01','Which things can attend?','Follow one query. The sequence changes from features of a row to entire rows in a batch.');var stage=1;
    var bs=buttons(f,['Tokens','Feature attention','Row attention'],function(i){draw(i);});
    function draw(next){stage=Math.round(clamp(next,0,2));selected(bs,stage);var small=narrow(f),w=canvas(f,small?520:300,'Token layout and permitted attention connections');var mx=small?109:434,my=small?346:85;
      text(f.svg,small?160:153,25,stage===2?'Flatten every complete row':'Three rows · three tokens each',{'font-weight':600});
      var names=['CLS','age','job'],colors=[C.gold,C.blue,C.purple];
      for(var i=0;i<3;i++){
        var y=64+i*56; text(f.svg,24,y+23,['A','B','C'][i],{'font-weight':600});
        if(stage===2)rect(f.svg,41,y-3,239,49,i===0?'#edf7f4':'#f8fafb',i===0?C.teal:C.line);
        if(stage===1&&i===0)rect(f.svg,41,y-3,239,49,'#edf5fa',C.blue);
        for(var j=0;j<3;j++){
          var x=48+j*77;rect(f.svg,x,y,69,42,'#fff',colors[j]);text(f.svg,x+34.5,y+17,names[j],{'font-size':12,fill:colors[j]});
          for(var d=0;d<4;d++)rect(f.svg,x+11+d*12,y+26,8,7,colors[j],'none',{rx:1,'fill-opacity':.3+d*.16});
        }
      }
      text(f.svg,160,253,stage===2?'[1, B, T·d] = [1, 3, 3d]':'[B, T, d] = [3, 3, d]',{'font-size':13,'font-family':'monospace',fill:stage===2?C.teal:C.blue});
      if(!small)arrow(f.svg,300,143,372,143,stage===2?C.teal:C.blue,2);
      else arrow(f.svg,160,267,160,285,stage===2?C.teal:C.blue,2);
      text(f.svg,mx+57,my-43,stage===2?'One B × B matrix':'A’s T × T matrix',{'font-weight':600});
      matrix(f.svg,mx,my,stage===2?['A','B','C']:['CLS','age','job'],stage===2?C.teal:C.blue,stage===0?-1:0);
      text(f.svg,mx+57,my+139,stage===2?'query A → rows A, B, C':'query CLS → CLS, age, job',{'font-size':12});
      if(stage===0)text(f.svg,mx+57,my+161,'Connectivity only; no learned weights.',{'font-size':10,fill:C.muted});
      f.out.textContent=[
        'Each colored token contains d coordinates. Color identifies the feature; A, B, C identify samples. The matrix is a schematic of available connections, not trained attention weights.',
        'The blue matrix belongs to row A. Its highlighted query is A’s CLS token; its keys are A’s CLS, age, and job. Rows B and C have their own separate matrices. Shading is schematic connectivity, not measured weights.',
        'The green matrix belongs to the whole batch. Query A is now the flattened vector [CLS | age | job]. Its keys are complete rows A, B, C. Restore the feature axes after the row block. Shading is schematic connectivity, not measured weights.'
      ][stage];return stage;
    }
    draw(stage);responsive(f,function(){draw(stage);});return {setStage:draw};
  }};
  global.SaintArchitectureViz={mount:function(root){
    var f=frame(root,'02','One complete SAINT stage','The shape on each connection is the contract between the blocks. Click a stage to trace it.');var stage=4;
    var bs=buttons(f,['Embed','Features','Rows','Readout','Whole model'],function(i){draw(i);});
    function draw(i){stage=Math.round(clamp(i,0,4));selected(bs,stage);var w=canvas(f,555,'Supervised SAINT architecture including both feed-forward blocks'),cx=w/2,bw=narrow(f)?252:376,x=cx-bw/2;
      var nodes=[{y:18,h:58,c:C.gold,t:['Embed + prepend CLS','categories: lookup','numbers: 1 → 100 → d'],s:0},
        {y:123,h:65,c:C.blue,t:['Feature attention → feature FF','T tokens interact within each row'],s:1},
        {y:250,h:76,c:C.teal,t:['Row attention → row FF','B complete rows interact','head width 64 · flattened width T·d'],s:2},
        {y:386,h:48,c:C.purple,t:['Restore [B, T, d]','Repeat the stage if depth > 1'],s:2},
        {y:484,h:55,c:C.gold,t:['Final CLS → prediction head','d → 1000 → 2 class logits'],s:3}];
      for(var n=0;n<nodes.length-1;n++)arrow(f.svg,cx,nodes[n].y+nodes[n].h+3,cx,nodes[n+1].y-6,C.line);
      text(f.svg,cx+12,104,'[B, T, d]',{'text-anchor':'start','font-family':'monospace','font-size':12});
      text(f.svg,cx,214,'PACK: [B, T, d] → [1, B, T·d]',{'font-family':'monospace','font-size':12,fill:C.teal});
      text(f.svg,cx,356,'UNPACK: [1, B, T·d] → [B, T, d]',{'font-family':'monospace','font-size':12,fill:C.purple});
      text(f.svg,cx+12,461,'take token 0',{'text-anchor':'start','font-size':12});
      nodes.forEach(function(b){card(f.svg,x,b.y,bw,b.h,b.t,b.c,stage===4||stage===b.s);});
      // Shape labels sit on a white backing so connector lines never run through text.
      [214,356].forEach(function(y){var labels=Array.prototype.filter.call(f.svg.children,function(n){return n.tagName==='text'&&String(n.getAttribute('y'))===String(y);});labels.forEach(function(label){var backing=rect(f.svg,cx-145,y-14,290,20,'#f5f8fa','none');f.svg.appendChild(label);});});
      f.out.textContent=[
        'Each feature gets its own embedding; CLS is a learned summary slot and never stores the target label. Numeric embeddings follow the released implementation.',
        'Feature attention exchanges information across the T tokens of each sample. The feature feed-forward then transforms each token independently.',
        'Packing changes the meaning of a sequence element. Row attention exchanges information across B samples; the following feed-forward operates on all T·d coordinates of a row.',
        'Unpack before selecting CLS. The head reads the final contextual CLS vector, then produces two class logits. Softmax converts those logits into class probabilities.',
        'Both attention blocks and both feed-forward blocks are present. Each sublayer follows the released wrapper u = LN(x), output = u + F(u); see the paper/code audit below.'
      ][stage];return stage;
    }
    draw(stage);responsive(f,function(){draw(stage);});return {setStage:draw};
  }};
  global.SaintContextViz={mount:function(root){
    var f=frame(root,'03','A fixed query, a changing companion','Line thickness shows how much information each row contributes. All numbers are a one-head toy.');var key=0;
    var presets=buttons(f,['Weak match','Initial example','Strong match'],function(i){draw([-3,0,3][i]);});
    var slider=range(f,'Companion key k',-3,3,.1,0,draw);
    function draw(v){key=clamp(v,-3,3);slider.input.value=key;slider.label.textContent=key.toFixed(1);selected(presets,[-3,0,3].indexOf(key));var r=context(key),small=narrow(f),w=canvas(f,small?442:336,'Softmax weights and their weighted contributions to the fixed query');var cx=w/2,left=small?12:55,bw=small?138:205,right=w-left-bw;
      arrow(f.svg,left+bw/2,108,cx-24,194,C.blue,1+7*r.weights[0]);arrow(f.svg,right+bw/2,108,cx+24,194,C.teal,1+7*r.weights[1]);
      card(f.svg,left,22,bw,83,['Self row','key = 1','value = 2'],C.blue,true);
      card(f.svg,right,22,bw,83,['Companion','key = '+key.toFixed(1),'value = 8'],C.teal,true);
      text(f.svg,left+bw/2-3,140,(100*r.weights[0]).toFixed(1)+'%',{fill:C.blue,'font-weight':600});
      text(f.svg,right+bw/2+3,140,(100*r.weights[1]).toFixed(1)+'%',{fill:C.teal,'font-weight':600});
      card(f.svg,cx-89,196,178,73,['Fixed query q = 1','updated value = '+r.value.toFixed(3)],C.ink,true);
      var bx=small?20:60,by=small?318:293,total=w-2*bx,b1=total*r.weights[0];
      text(f.svg,cx,by-13,'Softmax probability mass (sum = 1)',{'font-size':12,fill:C.muted});
      rect(f.svg,bx,by,b1,17,C.blue,'none',{rx:0});rect(f.svg,bx+b1,by,total-b1,17,C.teal,'none',{rx:0});
      if(small){text(f.svg,cx,371,r.weights[0].toFixed(3)+' × 2  +  '+r.weights[1].toFixed(3)+' × 8',{'font-family':'monospace','font-size':13});text(f.svg,cx,401,'= '+r.value.toFixed(3),{'font-size':23,'font-weight':600});}
      f.out.textContent='Scores [1, '+key.toFixed(1)+'] → softmax ['+r.weights.map(function(x){return x.toFixed(3);}).join(', ')+']. Contributions: '+(2*r.weights[0]).toFixed(3)+' from self + '+(8*r.weights[1]).toFixed(3)+' from companion = '+r.value.toFixed(3)+'. The query, values, and model weights stay fixed; only the companion key changes.';return r;
    }
    draw(0);responsive(f,function(){draw(key);});return {setKey:draw};
  }};
  global.SaintViewsViz={mount:function(root){
    var f=frame(root,'04','From a raw row to an augmented view','CutMix chooses actual feature values. Mixup interpolates vectors after embedding.');var cut=true,alpha=.2;
    var btn=el('button','CutMix: replace job',f.controls);btn.type='button';btn.addEventListener('click',function(){cut=!cut;draw(alpha);});
    var slider=range(f,'Weight α on A’s augmented embedding',0,1,.1,.2,draw);
    function draw(value){alpha=clamp(value,0,1);slider.input.value=alpha;slider.label.textContent=alpha.toFixed(1);btn.setAttribute('aria-pressed',String(cut));var a=cut?[1,3]:[1,0],r=mix(a,[3,1],alpha),small=narrow(f),w=canvas(f,small?565:480,'CutMix categorical replacement followed by embedding-space interpolation');var cx=w/2,bw=small?138:200,left=small?12:55,right=w-left-bw;
      arrow(f.svg,left+bw/2,88,cx-24,122,C.blue,2);arrow(f.svg,right+bw/2,88,cx+24,122,cut?C.purple:C.line,2,!cut);
      card(f.svg,left,12,bw,74,['Original A','age 40','job teacher'],C.blue,true);card(f.svg,right,12,bw,74,['CutMix donor B','age 60','job nurse'],C.purple,true);
      card(f.svg,cx-124,125,248,54,['A’s raw view','age 40 · job '+(cut?'nurse':'teacher')],cut?C.purple:C.blue,true);
      text(f.svg,cx,204,'Embed, then mix with a NEW partner',{'font-size':small?12:13,'font-weight':600});
      var px=small?66:175,py=small?438:394,scale=small?51:48;
      arrow(f.svg,px,py,px+3.6*scale,py,C.line,1);arrow(f.svg,px,py,px,py-3.5*scale,C.line,1);
      for(var tick=0;tick<=3;tick++){text(f.svg,px+tick*scale,py+19,String(tick),{'font-size':11,fill:C.muted});text(f.svg,px-15,py-tick*scale+4,String(tick),{'font-size':11,fill:C.muted});}
      var ax=px+a[0]*scale,ay=py-a[1]*scale,bx=px+3*scale,by=py-scale,rx=px+r[0]*scale,ry=py-r[1]*scale;
      sv('line',{x1:ax,y1:ay,x2:bx,y2:by,stroke:C.purple,'stroke-width':3,'stroke-dasharray':'5 4'},f.svg);
      [[ax,ay,C.blue],[bx,by,C.purple],[rx,ry,C.teal]].forEach(function(p,i){sv('circle',{cx:p[0],cy:p[1],r:i===2?7:5,fill:p[2],stroke:'#fff','stroke-width':2},f.svg);});
      text(f.svg,ax-10,ay-15,'A view ['+a.join(',')+']',{'font-size':12,'text-anchor':small?'middle':'end',fill:C.blue});
      text(f.svg,bx+3,by+25,'donor [3,1]',{'font-size':12,fill:C.purple});
      text(f.svg,cx,small?489:441,'Mixture = ['+r.map(function(v){return v.toFixed(2);}).join(', ')+']',{'font-size':19,'font-weight':600,fill:C.teal});
      if(small)text(f.svg,cx,520,'green point = α · blue + (1−α) · purple',{'font-size':12});
      f.out.textContent='The plotted coordinates are illustrative embeddings, not raw age/job values. With α = '+alpha.toFixed(1)+', the augmented vector is '+alpha.toFixed(1)+' × ['+a+'] + '+(1-alpha).toFixed(1)+' × [3,1]. Its positive partner is still clean A; denoising reconstructs age 40 and job teacher.';return r;
    }
    draw(alpha);responsive(f,function(){draw(alpha);});return {setAlpha:draw,setCutMix:function(v){cut=!!v;return draw(alpha);}};
  }};
  global.SaintContrastViz={mount:function(root){
    var f=frame(root,'05','Match identities; reconstruct features','Two pretraining objectives use the augmented row for different tasks. The matrix below is a synthetic example.');var temp=.7;
    var slider=range(f,'Contrastive temperature τ',.2,2,.1,.7,draw);
    function draw(v){temp=clamp(v,.2,2);slider.input.value=temp;slider.label.textContent=temp.toFixed(1);var r=contrast(temp),small=narrow(f),w=canvas(f,small?627:420,'Identity-pair probability matrix and separate denoising objective');var mx=small?116:101,my=small?260:192;
      card(f.svg,small?28:22,12,small?264:250,61,['Clean A → encoder → g₁','projection zA'],C.blue,true);
      card(f.svg,small?28:328,small?106:12,small?264:250,61,['A view → encoder → g₂','projection z′A'],C.teal,true);
      text(f.svg,small?160:179,small?216:127,'Contrastive: find the same row',{'font-weight':600,'font-size':13});
      var vals=[0,1,2].map(function(i){return [0,1,2].map(function(j){return i===j?r.weights[0]:r.weights[1];});});
      matrix(f.svg,mx,my,['A′','B′','C′'],C.teal,-1,vals,small?36:42);
      // Matrix rows are CLEAN identities; columns are augmented candidates.
      var rowTexts=Array.prototype.filter.call(f.svg.children,function(n){return n.tagName==='text'&&n.getAttribute('text-anchor')==='end';});rowTexts.forEach(function(n,i){if(i<3)n.textContent=['A','B','C'][i];});
      text(f.svg,mx+54,my-30,'augmented candidates',{'font-size':12,fill:C.muted});
      text(f.svg,small?160:164,my+147,'Positive = matching diagonal',{'font-size':12});
      text(f.svg,small?160:164,my+174,'mean loss = '+r.loss.toFixed(3),{'font-size':17,'font-weight':600,fill:C.teal});
      var dx=small?28:348,dy=small?473:159,dw=small?264:230;
      card(f.svg,dx,dy,dw,105,['Denoising decoders','reconstruct original A','age → squared error','job → cross-entropy'],C.purple,true);
      text(f.svg,dx+dw/2,dy+137,'Original features are the targets.',{'font-size':12});
      f.out.textContent='For this illustration, a matching pair has dot product 1 and other pairs have 0. At τ = '+temp.toFixed(1)+', matching-view probability is '+r.weights[0].toFixed(3)+' and mean contrastive loss is '+r.loss.toFixed(3)+'. Denoising predicts original features separately; class labels are not used by either pretraining objective.';return r;
    }
    draw(temp);responsive(f,function(){draw(temp);});return {setTemperature:draw};
  }};
})(window);
