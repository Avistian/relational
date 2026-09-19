/* Fixed propagation only: no learned weights, nonlinearities or training. */
(function(global){
  function compute(depth,mode,cut){
    const a=[[1,1,0],[1,1,cut?0:1],[0,cut?0:1,1]],d=a.map(r=>r.reduce((x,y)=>x+y,0));
    const s=a.map((r,i)=>r.map((v,j)=>v/(mode==='mean'?d[i]:Math.sqrt(d[i]*d[j]))));
    let h=[2,4,8];for(let k=0;k<depth;k++)h=s.map(r=>r.reduce((v,w,j)=>v+w*h[j],0));
    return {h,d,s};
  }
  function mount(el){
    el.innerHTML='<fieldset><legend>Predict, then change the propagation</legend><label>Depth <input type="range" min="0" max="100" value="1" aria-label="Propagation depth"></label> <label>Operator <select aria-label="Propagation operator"><option value="symmetric">Symmetric GCN</option><option value="mean">Row mean</option></select></label> <label><input type="checkbox"> Remove B—C</label> <button type="button">Reset</button><div class="smooth-output" aria-live="polite"></div></fieldset>';
    const slider=el.querySelector('input[type=range]'),mode=el.querySelector('select'),cut=el.querySelector('input[type=checkbox]'),out=el.querySelector('.smooth-output');
    function draw(){
      const k=Number(slider.value),r=compute(k,mode.value,cut.checked);
      out.innerHTML='<p><strong>Depth '+k+'</strong> · baseline inputs stay [2, 4, 8]. '+(cut.checked?'C is an isolated component.':'One connected component.')+'</p><div class="smooth-nodes">'+r.h.map((v,i)=>'<div style="border-top:7px solid hsl('+(220-18*v)+',65%,48%)"><strong>'+['A','B','C'][i]+'</strong><p>'+v.toFixed(3)+'</p><small>start '+[2,4,8][i]+' · degree '+r.d[i]+'</small></div>').join('')+'</div><p>Degree-corrected H/√d: ['+r.h.map((v,i)=>(v/Math.sqrt(r.d[i])).toFixed(3)).join(', ')+']</p><p>Receiver B: '+r.s[1].map(v=>v.toFixed(3)).join(' × A + ').replace(' × A + ',' × previous A + ')+' × previous C. Each step uses the previous state simultaneously.</p>';
      out.lastElementChild.textContent='Receiver B coefficients on previous [A, B, C]: ['+r.s[1].map(v=>v.toFixed(3)).join(', ')+']. Each step updates all nodes simultaneously.';
    }
    slider.addEventListener('input',draw);mode.addEventListener('change',draw);cut.addEventListener('change',draw);
    el.querySelector('button').addEventListener('click',()=>{slider.value=1;mode.value='symmetric';cut.checked=false;draw();});draw();return {draw};
  }
  global.OversmoothingViz={compute,mount};
})(window);
