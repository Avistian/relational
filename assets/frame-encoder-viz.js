/** L075 reusable arithmetic widgets. Defaults: held-out=1000 -> all-row mean265;
 * x=30 -> z=1 -> token[2.5,-0.5]. Native keyboard controls; reset preserves baseline.
 * Numeric parameters are illustrative, not the actual fitted fixture standard deviation. */
(function(global){
'use strict';
const fmt=x=>Number(x.toFixed(3)).toString();
function scope(el){
 el.className='contrastive-widget';
 el.innerHTML='<p><strong>Predict:</strong> can a held-out amount change training state?</p><label>Held-out amount <input type="range" min="0" max="1000" step="10" value="1000"></label><label><input type="checkbox"> Fit on all rows (leaky)</label><output aria-live="polite"></output><button type="button">Reset</button>';
 const slider=el.querySelector('input[type=range]'),check=el.querySelector('input[type=checkbox]'),out=el.querySelector('output');
 function draw(){let q=+slider.value,m=check.checked?(60+q)/4:20;out.textContent=`Query value: ${q}. Training-only baseline: 20. Current fitted mean: ${fmt(m)}. Change from baseline: ${fmt(m-20)}. ${check.checked?'Query entered the fit.':'Query is converted with frozen training state.'}`;}
 slider.oninput=draw;check.onchange=draw;el.querySelector('button').onclick=()=>{slider.value=1000;check.checked=false;draw();};draw();
}
function numeric(el){
 el.className='contrastive-widget';el.innerHTML='<p><strong>Predict:</strong> how does decreasing x move each coordinate?</p><label>Numeric value <input type="range" min="0" max="40" step="5" value="30"></label><label><input type="checkbox"> Value is missing (mean-impute)</label><output aria-live="polite"></output><button type="button">Reset</button>';
 const slider=el.querySelector('input[type=range]'),check=el.querySelector('input[type=checkbox]'),out=el.querySelector('output');
 function draw(){let x=check.checked?20:+slider.value,z=(x-20)/10;out.textContent=`Baseline x=30 → [2.5, -0.5]. Current x=${check.checked?'missing → 20':x}; z=(${x}−20)/10=${fmt(z)}; z×[2,−1]=[${fmt(2*z)}, ${fmt(-z)}]; +[0.5,0.5] → [${fmt(2*z+.5)}, ${fmt(-z+.5)}].`;}
 slider.oninput=draw;check.onchange=draw;el.querySelector('button').onclick=()=>{slider.value=30;check.checked=false;draw();};draw();
}
global.FrameEncoderViz={scope,numeric};
})(window);
