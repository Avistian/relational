/* Mechanisms from SAINT §3.2/§4. No trained values in these toy visuals. */
(function(global){
  'use strict';
  function el(tag,text,parent){var n=document.createElement(tag);if(text!==null)n.textContent=text;if(parent)parent.appendChild(n);return n;}
  function box(root){root.className='saint-viz';root.innerHTML='';return root;}
  function weights(logits){var max=Math.max.apply(null,logits),v=logits.map(function(x){return Math.exp(x-max);}),s=v.reduce(function(a,b){return a+b;},0);return v.map(function(x){return x/s;});}
  function context(key){var a=weights([1,key]);return {weights:a,value:a[0]*2+a[1]*8};}
  function mix(a,b,alpha){return a.map(function(v,i){return alpha*v+(1-alpha)*b[i];});}
  global.SaintMath={weights:weights,context:context,mix:mix};
  global.SaintAxesViz={mount:function(root){
    box(root);var ctl=el('div',null,root);ctl.className='controls';var canvas=el('div',null,root),out=el('output',null,root),buttons=[];
    out.setAttribute('aria-live','polite');
    var labels=['1 · Tokens','2 · Within row','3 · Between rows'];
    function draw(stage){stage=Math.max(0,Math.min(2,stage));canvas.innerHTML='';buttons.forEach(function(b,i){b.setAttribute('aria-pressed',String(i===stage));});
      if(stage<2){var grid=el('div',null,canvas);grid.className='saint-grid';['A: CLS','A: age','A: job','B: CLS','B: age','B: job','C: CLS','C: age','C: job'].forEach(function(t,i){var n=el('div',t,grid);n.className='saint-cell'+(stage===1&&i<3?' active':'');});}
      else {['A','B','C'].forEach(function(row){var n=el('div',row+' = [CLS vector | age vector | job vector]',canvas);n.className='saint-row';});}
      out.textContent=[
        'B=3 rows, T=3 tokens (including CLS), d numbers per token: shape [3,3,d]. The same feature can have different values in each row.',
        'One attention sequence per row: [B,T,d]. A’s CLS can read A’s age and job. Each head has B matrices of shape T×T; no other row enters.',
        'Flatten each WHOLE row: [1,B,T·d]. Now A, B, C are the sequence. Each head has one B×B matrix. Restore [B,T,d] after attention and the row feed-forward block.'
      ][stage];return stage;}
    labels.forEach(function(t,i){var b=el('button',t,ctl);b.type='button';b.addEventListener('click',function(){draw(i);});buttons.push(b);});draw(0);return {setStage:draw};
  }};
  global.SaintContextViz={mount:function(root){
    box(root);el('p','Fixed query q=1. Self: key=1, value=2. Companion: value=8.',root);
    var label=el('label','Change the companion key (the query stays fixed)',root),slider=el('input',null,label);
    slider.type='range';slider.min='-3';slider.max='3';slider.step='.1';slider.value='0';
    var text=el('output',null,root),bar=el('div',null,root);bar.className='bar';text.setAttribute('aria-live','polite');
    function draw(v){v=Math.max(-3,Math.min(3,Number(v)));slider.value=String(v);var r=context(v);bar.style.width=(100*r.weights[1])+'%';text.textContent='Scores [1, '+v.toFixed(1)+']; softmax weights ['+r.weights.map(function(x){return x.toFixed(3);}).join(', ')+']. Updated query value = '+r.weights[0].toFixed(3)+' × 2 + '+r.weights[1].toFixed(3)+' × 8 = '+r.value.toFixed(3)+'. Bar: companion weight. This is a toy attention value, not a probability.';return r;}
    slider.addEventListener('input',function(){draw(slider.value);});draw(0);return {setKey:draw};
  }};
  global.SaintViewsViz={mount:function(root){
    box(root);el('p','Raw row A = [age 40, job teacher]; donor B = [age 60, job nurse].',root);
    var btn=el('button','CutMix: replace job',root);btn.type='button';var cut=false;
    var raw=el('div',null,root),label=el('label','Mixup: weight α on A’s augmented embedding',root),slider=el('input',null,label);
    slider.type='range';slider.min='0';slider.max='1';slider.step='.1';slider.value='.2';var out=el('output',null,root);out.setAttribute('aria-live','polite');
    function draw(alpha){alpha=Math.max(0,Math.min(1,Number(alpha)));slider.value=String(alpha);btn.setAttribute('aria-pressed',String(cut));raw.className='saint-row';raw.textContent=cut?'CutMix output: [age 40, job nurse]':'Original raw values: [age 40, job teacher]';
      var a=cut?[1,3]:[1,0],r=mix(a,[3,1],alpha);out.textContent='Illustrative learned vectors: A view ['+a+'], second donor [3,1]. Mixup = '+alpha.toFixed(1)+' × ['+a+'] + '+(1-alpha).toFixed(1)+' × [3,1] = ['+r.map(function(v){return v.toFixed(2);})+']. Pair its encoded projection with clean A; reconstruct original A. The donor is not the positive identity.';return r;}
    btn.addEventListener('click',function(){cut=!cut;draw(slider.value);});slider.addEventListener('input',function(){draw(slider.value);});draw(.2);return {setAlpha:draw,setCutMix:function(v){cut=!!v;return draw(slider.value);}};
  }};
})(window);
