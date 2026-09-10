/* TabM BatchEnsemble viz (paper 2410.24210, §3.2).
   One shared weight matrix W; a member gets adapters r,s (±1) so that
       W_i = W ⊙ (s_i r_i^T)   and   l_i(x) = s_i ⊙ (W (r_i ⊙ x)) + b_i.
   Expected states:
   - init (r=s=1): W_i == W exactly, no entries flipped -> adapters have no effect.
   - r=[1,-1], s=[1,1]: column 2 of W_i is negated (2 flipped entries).
   - The two routes (adapter route vs explicit W_i x) always match (equivalence).
   compute() is pure and exported for arithmetic verification. */
(function(g){'use strict';
  var W=[[1,3],[2,4]];                    // W[out][in]: transpose of the atlas row-vector W
  var X=[2,3];                            // same fixed input as the portable diagram

  function compute(state){
    var r=state.r, s=state.s;            // length-2 arrays of +1/-1
    var Wi=[[0,0],[0,0]], flips=[[false,false],[false,false]];
    for(var o=0;o<2;o++)for(var i=0;i<2;i++){
      var m=s[o]*r[i];
      Wi[o][i]=m*W[o][i];
      flips[o][i]=m<0;                    // sign flipped vs shared W
    }
    // adapter route: s ⊙ (W (r ⊙ x))
    var rx=[r[0]*X[0], r[1]*X[1]];
    var Wrx=[W[0][0]*rx[0]+W[0][1]*rx[1], W[1][0]*rx[0]+W[1][1]*rx[1]];
    var outA=[s[0]*Wrx[0], s[1]*Wrx[1]];
    // explicit route: W_i x
    var outB=[Wi[0][0]*X[0]+Wi[0][1]*X[1], Wi[1][0]*X[0]+Wi[1][1]*X[1]];
    var nflip=flips[0][0]+flips[0][1]+flips[1][0]+flips[1][1];
    var match=Math.abs(outA[0]-outB[0])<1e-9 && Math.abs(outA[1]-outB[1])<1e-9;
    return {W:W,Wi:Wi,flips:flips,r:r,s:s,x:X,outA:outA,outB:outB,nflip:nflip,match:match,
            atInit:r.concat(s).every(function(v){return v===1;})};
  }

  function mat(title,M,flips){
    var f=flips||[[false,false],[false,false]];
    var rows='';
    for(var o=0;o<2;o++){
      rows+='<tr>';
      for(var i=0;i<2;i++) rows+='<td class="'+(f[o][i]?'tv-flip':'')+'">'+M[o][i].toFixed(2)+'</td>';
      rows+='</tr>';
    }
    return '<div class="tv-mat"><strong>'+title+'</strong><table>'+rows+'</table></div>';
  }

  function mount(el){
    el.className='tabm-viz';
    var state={r:[1,1],s:[1,1]};
    el.innerHTML='<h4>One shared W, cheap per-member adapters</h4>';
    var ctr=document.createElement('div'); ctr.className='tv-controls'; el.appendChild(ctr);
    var toggles=[];
    [['r',0,'r₁'],['r',1,'r₂'],['s',0,'s₁'],['s',1,'s₂']].forEach(function(spec){
      var b=document.createElement('button'); b.type='button';
      b.dataset.vec=spec[0]; b.dataset.idx=spec[1]; b.dataset.name=spec[2];
      toggles.push(b); ctr.appendChild(b);
      b.addEventListener('click',function(){
        var v=state[spec[0]]; v[spec[1]]=-v[spec[1]]; render();
      });
    });
    var presets=document.createElement('div'); presets.className='tv-controls'; el.appendChild(presets);
    function preset(label,r,s){
      var b=document.createElement('button'); b.type='button'; b.className='tv-preset'; b.textContent=label;
      b.addEventListener('click',function(){state.r=r.slice();state.s=s.slice();render();});
      presets.appendChild(b);
    }
    preset('Init (r=s=1)',[1,1],[1,1]);
    preset('Member A (r=[1,−1])',[1,-1],[1,1]);
    preset('Member B (s=[−1,1])',[1,1],[-1,1]);
    var grid=document.createElement('div'); grid.className='tv-grid'; el.appendChild(grid);
    var out=document.createElement('output'); out.setAttribute('aria-live','polite'); el.appendChild(out);

    function render(){
      var r=compute(state);
      toggles.forEach(function(b){
        var v=state[b.dataset.vec][+b.dataset.idx];
        b.textContent=b.dataset.name+' = '+(v>0?'+1':'−1');
        b.classList.toggle('tv-on', v>0);
      });
      grid.innerHTML=mat('Shared W (all members)',r.W)+mat('This member W_i = W ⊙ (s rᵀ)',r.Wi,r.flips);
      out.innerHTML='<div class="tv-eq">Same input x = ['+r.x[0]+', '+r.x[1]+']. '+
        'Adapter route s ⊙ (W (r ⊙ x)) = ['+r.outA[0].toFixed(2)+', '+r.outA[1].toFixed(2)+']; '+
        'explicit W_i·x = ['+r.outB[0].toFixed(2)+', '+r.outB[1].toFixed(2)+']. '+
        'The two routes '+(r.match?'match — they are the same layer.':'DIFFER (bug).')+'</div>'+
        (r.atInit
          ? '<p>At initialisation the non-first adapters use r = s = 1, so W_i = W exactly: '+
            'the adapter adds no effect yet. Training is free to move it. This is TabM’s init trick.</p>'
          : r.nflip===0
          ? '<p>The input and output signs cancel, so W_i = W even though the adapters are not all one. Different parameter values can represent the same function.</p>'
          : '<p>'+r.nflip+' entr'+(r.nflip===1?'y is':'ies are')+' sign-flipped. This member has a different effective matrix while sharing W. A full BatchEnsemble layer adds 3d parameters per member, including its bias. The bias is zero in this fixture.</p>');
    }
    render();
    return {compute:compute, setState:function(st){state=st; render();}, get state(){return state;}};
  }

  /* Ensemble-averaging intuition. For k predictors each with error variance 1 and
     average pairwise correlation rho, the variance of their mean is
        var_mean(k, rho) = rho + (1 - rho)/k.
     Expected states: k=1 -> 1 for any rho; rho=1 -> 1 for any k (no benefit);
     rho=0,k=32 -> 1/32. We report relative RMSE = sqrt(var_mean). */
  function varMean(k, rho){ return rho + (1-rho)/k; }

  function mountEnsemble(el){
    el.className='tabm-viz';
    var state={k:32, rho:0.3};
    el.innerHTML='<h4>How correlated errors limit averaging</h4><p>Synthetic equal-variance, zero-mean errors. Individual RMSE is 1; correlation and variance are held fixed while k changes. This is an illustrative calculation, not a fitted TabM result.</p>';
    function slider(name,label,min,max,step){
      var wrap=document.createElement('label'); wrap.style.display='block'; wrap.style.margin='.5rem 0';
      wrap.textContent=label+' ';
      var inp=document.createElement('input'); inp.type='range'; inp.min=min; inp.max=max; inp.step=step;
      inp.value=state[name]; inp.style.width='100%'; inp.style.accentColor='#225b78';
      inp.addEventListener('input',function(){state[name]=Number(inp.value); render();});
      wrap.appendChild(inp); el.appendChild(wrap); return inp;
    }
    var ki=slider('k','Number of submodels k',1,32,1);
    var ri=slider('rho','Pairwise error correlation ρ (0 = uncorrelated, 1 = perfectly correlated)',0,1,0.01);
    var out=document.createElement('output'); out.setAttribute('aria-live','polite'); el.appendChild(out);
    function render(){
      var vm=varMean(state.k, state.rho);
      var rmseInd=1, rmseCol=Math.sqrt(vm);
      var reduction=(1-rmseCol/rmseInd)*100;
      out.innerHTML='<div class="tv-eq">Individual relative RMSE = 1.00 &nbsp;·&nbsp; '+
        'collective relative RMSE = √('+state.rho.toFixed(2)+' + (1−'+state.rho.toFixed(2)+')/'+state.k+') = '+
        rmseCol.toFixed(3)+'</div>'+
        '<p>In this calculation, averaging reduces RMSE by <strong>'+reduction.toFixed(1)+'%</strong>. '+
        (state.rho===1? 'Perfectly correlated equal-variance errors give no variance reduction.'
         : state.k===1? 'With one member, there is nothing to average. Its training parameterization can still differ from a plain MLP.'
         : 'The floor as k→∞ is √ρ = '+Math.sqrt(state.rho).toFixed(2)+', set by correlation, not by k. '+
           'Real submodels can also have shared bias, unequal variances and correlations that change with k. Measure their errors before applying this explanation.')+'</p>';
    }
    render();
    return {varMean:varMean, setState:function(st){state=st; render();}, get state(){return state;}};
  }

  g.TabMViz={compute:compute, mount:mount, mountEnsemble:mountEnsemble, varMean:varMean, W:W, X:X};
})(typeof window==='undefined'?globalThis:window);
