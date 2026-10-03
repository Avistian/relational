/** Reusable interval classifier. Values are illustrative AUROC contrasts.
 * Default benefit [-.005,.025], margin .01 => INCONCLUSIVE.
 * Sensitivity [-.005,.005] => BELOW_USEFUL_MARGIN; boundary touches stay inconclusive. */
(function(global){'use strict';
function decide(low,high,margin,mode){
 if(![low,high,margin].every(x=>typeof x==='number'&&Number.isFinite(x))||low>high||margin<=0||!['benefit','sensitivity'].includes(mode))throw Error('Invalid interval');
 if(mode==='benefit'){if(low>margin)return 'USEFUL_BENEFIT';if(high<margin)return 'BELOW_USEFUL_MARGIN';}
 else {if(low>margin||high<-margin)return 'MATERIAL_SENSITIVITY';if(low>-margin&&high<margin)return 'BELOW_USEFUL_MARGIN';}
 return 'INCONCLUSIVE';
}
function mount(host){
 host.classList.add('proposal-panel');
 host.innerHTML='<p><strong>Illustrative intervals, not measured results.</strong> Predict the verdict, then change an endpoint. The useful margin stays at 0.01 AUROC.</p><div class="pd-controls"><label>Question<select data-mode aria-label="Interval question"><option value="benefit">Useful positive benefit?</option><option value="sensitivity">Material change either way?</option></select></label><label>Lower endpoint<input data-low aria-label="Lower endpoint" type="number" min="-0.05" max="0.05" step="0.005" value="-0.005"></label><label>Upper endpoint<input data-high aria-label="Upper endpoint" type="number" min="-0.05" max="0.05" step="0.005" value="0.025"></label></div><div class="pd-plot"></div><output aria-live="polite"></output><button type="button">Reset interval</button>';
 const low=host.querySelector('[data-low]'),high=host.querySelector('[data-high]'),mode=host.querySelector('select');
 const explain={USEFUL_BENEFIT:'The entire interval exceeds the useful positive margin.',BELOW_USEFUL_MARGIN:'The interval rules out the specified useful effect under the declared uncertainty model.',MATERIAL_SENSITIVITY:'The entire interval is outside the negligible-change band.',INCONCLUSIVE:'The interval touches or crosses a decision boundary. More precision or a revised study may be needed.'};
 function update(){
  const l=Number(low.value),h=Number(high.value),out=host.querySelector('output');
  if(low.value===''||high.value===''||l>h||l<-.05||h>.05){host.dataset.state='INVALID_INTERVAL';out.textContent='Enter ordered endpoints within −0.05 and +0.05.';host.querySelector('.pd-plot').replaceChildren();return;}
  const state=decide(l,h,.01,mode.value);host.dataset.state=state;out.textContent=state+' — '+explain[state];
  const x=v=>50+(v+.05)*5000;
  host.querySelector('.pd-plot').innerHTML='<svg viewBox="0 0 600 160" role="img" aria-label="Interval '+l+' to '+h+' and useful margin 0.01 AUROC"><rect x="'+(mode.value==='sensitivity'?x(-.01):x(-.05))+'" y="20" width="'+(mode.value==='sensitivity'?100:300)+'" height="85" fill="#e0ece8"/><line x1="50" x2="550" y1="100" y2="100" stroke="#859999"/><line x1="300" x2="300" y1="15" y2="110" stroke="#849292" stroke-dasharray="3 3"/><line x1="350" x2="350" y1="15" y2="110" stroke="#aa5c2c"/>'+(mode.value==='sensitivity'?'<line x1="250" x2="250" y1="15" y2="110" stroke="#aa5c2c"/>':'')+'<line x1="'+x(l)+'" x2="'+x(h)+'" y1="62" y2="62" stroke="#146b64" stroke-width="7"/><circle cx="'+x(l)+'" cy="62" r="7" fill="#146b64"/><circle cx="'+x(h)+'" cy="62" r="7" fill="#146b64"/><text x="50" y="132">−0.05</text><text x="300" y="132" text-anchor="middle">0</text><text x="350" y="132" text-anchor="middle">+.01</text><text x="550" y="132" text-anchor="end">+.05</text><text x="300" y="156" text-anchor="middle">AUROC contrast · shaded: below useful margin</text></svg>';
 }
 [low,high].forEach(el=>el.addEventListener('input',update));mode.addEventListener('change',update);
 host.querySelector('button').addEventListener('click',()=>{low.value='-0.005';high.value='0.025';mode.value='benefit';update();});update();
}
global.ProposalDecisions={decide,mount};
})(window);
