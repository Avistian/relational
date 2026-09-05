/** L049 reusable computed examples. Pure compute is exported for checks.
 * spa: equal logits; values [2,4,9], highest importance first. Mask keeps j<=i.
 * prompt: two columns [1,0]/[0,1], third [0,0]; query [t,0]; values [2,4,9].
 * mix: importance [6,3,1]; donor labels 1/0; default retains first feature.
 * Each change retains the default for comparison. Reset restores all controls.
 */
(function(root){
  'use strict';
  const softmax=a=>{const m=Math.max(...a),e=a.map(x=>Math.exp(x-m)),s=e.reduce((x,y)=>x+y,0);return e.map(x=>x/s);};
  const fmt=x=>Number(x).toFixed(3);
  function spa(noise,masked=true){
    const v=[2,4,noise];
    const weights=v.map((_,i)=>v.map((_,j)=>masked?(j<=i?1/(i+1):0):1/3));
    return {weights,output:weights.map(row=>row.reduce((sum,a,j)=>sum+a*v[j],0))};
  }
  function prompt(t){const w=softmax([t,0,0]);return {weights:w,output:w.reduce((s,a,j)=>s+a*[2,4,9][j],0)};}
  function mix(mask){const importance=[6,3,1];return mask.reduce((s,v,i)=>s+(v?importance[i]:0),0)/10;}
  function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
  function mount(container,{kind}){
    container.classList.add('claim-viz');container.innerHTML='';
    let state=kind==='spa'?9:kind==='prompt'?0:[true,false,false];
    let masked=true;
    const title=el('h3',kind==='spa'?'Can the weak feature change the strong feature?':kind==='prompt'?'A changed query reallocates the feature weights':'A feature count is not an information share');
    container.appendChild(title);
    const controls=el('div');container.appendChild(controls);
    const panels=el('div',undefined,'panels');container.appendChild(panels);
    const readout=el('div',undefined,'readout');readout.setAttribute('aria-live','polite');container.appendChild(readout);
    let range,check;
    if(kind==='mix'){
      ['Strong feature · importance 6','Middle feature · importance 3','Weak feature · importance 1'].forEach((text,i)=>{
        const label=el('label');const box=el('input');box.type='checkbox';box.checked=state[i];
        box.addEventListener('change',()=>{state[i]=box.checked;render();});label.append(box,document.createTextNode(' Retain '+text));controls.appendChild(label);
      });
    }else{
      const label=el('label',kind==='spa'?'Weak feature value (other values stay 2 and 4)':'First query coordinate t (columns and values stay fixed)');
      range=el('input');range.type='range';range.min=kind==='spa'?'0':'-4';range.max=kind==='spa'?'18':'4';range.step=kind==='spa'?'1':'.25';range.value=state;
      range.addEventListener('input',()=>{state=Number(range.value);render();});label.appendChild(range);controls.appendChild(label);
      if(kind==='spa'){
        const l=el('label');check=el('input');check.type='checkbox';check.checked=true;
        check.addEventListener('change',()=>{masked=check.checked;render();});l.append(check,document.createTextNode(' Apply the semi-permeable mask'));controls.appendChild(l);
      }
    }
    const reset=el('button','Reset');reset.type='button';reset.addEventListener('click',()=>{
      state=kind==='spa'?9:kind==='prompt'?0:[true,false,false];masked=true;
      if(range)range.value=state;if(check)check.checked=true;
      if(kind==='mix')controls.querySelectorAll('input').forEach((c,i)=>c.checked=state[i]);render();
    });controls.appendChild(reset);
    function panel(label){const p=el('div',undefined,'panel');p.appendChild(el('strong',label));panels.appendChild(p);return p;}
    function table(p,weights){const t=el('table');const cap=el('caption','Sender → strong / middle / weak');t.appendChild(cap);
      weights.forEach((row,i)=>{const tr=el('tr');tr.appendChild(el('th',['Strong receives','Middle receives','Weak receives'][i]));row.forEach(a=>tr.appendChild(el('td',fmt(a),a===0?'blocked':'allowed')));t.appendChild(tr);});p.appendChild(t);}
    function render(){
      panels.innerHTML='';
      if(kind==='spa'){
        const base=spa(9),now=spa(state,masked);
        [[base,'Baseline · weak value 9'],[now,'Changed · weak value '+state]].forEach(([r,t])=>{const p=panel(t);table(p,r.weights);p.appendChild(el('p','Output: ['+r.output.map(fmt).join(', ')+']','formula'));});
        readout.textContent='Strong output change: '+fmt(now.output[0]-base.output[0])+'. '+(masked?'The blocked weak→strong route has zero weight. The weak receiver can still collect all three values.':'Without the mask, weak value enters the strong receiver. We held the logits at zero, so all allowed weights are uniform.');
      }else if(kind==='prompt'){
        const base=prompt(0),now=prompt(state);
        [[base,'Baseline · query [0, 0]'],[now,'Changed · query ['+state+', 0]']].forEach(([r,t])=>{
          const p=panel(t);p.appendChild(el('p','Column vectors: [1,0], [0,1], [0,0]'));
          p.appendChild(el('p','Weights: ['+r.weights.map(fmt).join(', ')+']','formula'));
          p.appendChild(el('p',r.weights.map((w,j)=>fmt(w)+' × '+[2,4,9][j]).join(' + ')+' = '+fmt(r.output),'formula'));
        });
        readout.textContent='Query t = '+state+'; weighted output change = '+fmt(now.output-base.output)+'. Softmax sums across columns. This illustrates one prompt after fusion, not an entire trained Trompt cell.';
      }else{
        const p=panel('Fixed donors and importance');p.appendChild(el('p','Donor A label = 1; donor B label = 0. Importance = [6,3,1].'));
        const n=panel('Your retained coordinates');n.appendChild(el('p',state.map((v,i)=>'Feature '+(i+1)+': '+(v?'A':'B')).join(' · ')));
        n.appendChild(el('p','Information share = '+fmt(mix(state))+'; count share = '+fmt(state.filter(Boolean).length/3),'formula'));
        readout.textContent='Mixed target = '+fmt(mix(state))+' × 1 + '+fmt(1-mix(state))+' × 0 = '+fmt(mix(state))+'. Baseline (first feature only) is 0.600. These training-only importance estimates are a heuristic, not a proof that a synthetic label is correct.';
      }
    }
    render();return {render};
  }
  root.ClaimAuditViz={mount,spa,prompt,mix};
})(typeof window!=='undefined'?window:globalThis);
