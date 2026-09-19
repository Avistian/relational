/* Reusable L078/Y3 arithmetic trace. Default means [4,5,4,0], updates [3,4.5,6,5].
   C=20: A one-hop remains 3; A two-hop becomes 5.25 (baseline 3.75).
   GCN B baseline 5.415816; self-inclusive ordinary mean 4.666667. */
(function(global){
  const edges=[[0,1],[1,0],[1,2],[2,1]];
  function step(x){return [.5*x[0]+.5*x[1],.5*x[1]+.25*(x[0]+x[2]),.5*x[2]+.5*x[1],.5*x[3]];}
  function mount(el,{mode='messages'}={}){
    el.classList.add('message-passing');
    el.innerHTML='<p><strong>Predict first:</strong> change C while keeping the graph and update fixed.</p><label>C’s initial state <input type="range" min="0" max="20" step="1" value="8"> <span class="value">8</span></label><button type="button">Reset</button><div class="mp-stages"></div><output aria-live="polite"></output>';
    const input=el.querySelector('input'),out=el.querySelector('output'),body=el.querySelector('.mp-stages');
    function draw(){const c=Number(input.value),x=[2,4,c,10],y=step(x),z=step(y);el.querySelector('.value').textContent=c;
      if(mode==='messages'){
        body.innerHTML='<div><b>1 · Send old states</b><p>A → B: 2<br>C → B: '+c+'</p></div><div><b>2 · Reduce at B</b><p>(2 + '+c+') / 2 = '+((2+c)/2).toFixed(2)+'</p></div><div><b>3 · Update B</b><p>0.5 × 4 + 0.5 × '+((2+c)/2).toFixed(2)+' = '+y[1].toFixed(2)+'</p></div>';
        out.textContent='Updated [A, B, C, D] = ['+y.map(v=>v.toFixed(2)).join(', ')+']. Baseline C=8: [3, 4.5, 6, 5].';
      } else if(mode==='reach'){
        body.innerHTML='<div><b>C → B → A</b><p>Round 0: C = '+c+'</p></div><div><b>One round</b><p>B = '+y[1].toFixed(2)+'<br>A = '+y[0].toFixed(2)+'</p></div><div><b>Two rounds</b><p>A = 0.5 × '+y[0]+' + 0.5 × '+y[1]+' = '+z[0].toFixed(2)+'</p></div>';
        out.textContent='A one-hop '+y[0].toFixed(2)+' (baseline 3.00); A two-hop '+z[0].toFixed(2)+' (baseline 3.75).';
      } else {
        const a=2/Math.sqrt(6),b=4/3,d=c/Math.sqrt(6);
        body.innerHTML='<div><b>A → B</b><p>2 / √(2×3) = '+a.toFixed(6)+'</p></div><div><b>B → B</b><p>4 / √(3×3) = '+b.toFixed(6)+'</p></div><div><b>C → B</b><p>'+c+' / √(2×3) = '+d.toFixed(6)+'</p></div>';
        out.textContent='GCN B = '+(a+b+d).toFixed(6)+' (baseline 5.415816). Ordinary self-inclusive mean = '+((6+c)/3).toFixed(6)+'. Degrees fixed at [2,3,2,1]; W=1.';
      }
    }
    input.addEventListener('input',draw);el.querySelector('button').addEventListener('click',()=>{input.value=8;draw();});draw();
    return {setValue(v){input.value=Math.min(20,Math.max(0,v));draw();}};
  }
  global.MessagePassingViz={mount,step};
})(window);
