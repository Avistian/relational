/* Scalar gate trace. Defaults: initial=.2, candidates=[.8,.1,.6], gate=.25.
   Both panels preserve a fixed baseline. Node mode uses a retention gate;
   weight mode uses EvolveGCN's replacement gate. Supplied gates isolate arithmetic. */
window.SnapshotStateViz={mount(root,{mode='weight'}={}){
 const name=mode==='node'?'Node state: retention gate':'Weight state: replacement gate';
 root.classList.add('stream-widget');
 root.innerHTML=`<h3>${name}</h3><p>Predict the direction before changing a control. Initial state is 0.20; later candidates stay 0.10 and 0.60.</p><label>Gate <output class="gate-value">0.25</output><input class="gate" aria-label="${name}" type="range" min="0" max="1" step="0.05" value="0.25"></label><label>First candidate <output class="candidate-value">0.80</output><input class="candidate" aria-label="First candidate for ${mode} state" type="range" min="-1" max="1" step="0.1" value="0.8"></label><p class="formula"></p><div class="stream-scroll" tabindex="0"><table><thead><tr><th>Step</th><th>Candidate</th><th>Baseline state</th><th>Changed state</th><th>Difference</th></tr></thead><tbody></tbody></table></div><p class="readout" aria-live="polite"></p><button type="button">Reset baseline</button><p>Scalar illustration only: a fitted model learns its gates and candidates. Baseline remains gate 0.25, first candidate 0.80.</p>`;
 const gate=root.querySelector('.gate'),candidate=root.querySelector('.candidate');
 const trace=(g,c)=>{let h=.2;return [c,.1,.6].map(x=>{h=mode==='node'?g*h+(1-g)*x:(1-g)*h+g*x;return h;});};
 function render(){const g=+gate.value,c=+candidate.value,base=trace(.25,.8),changed=trace(g,c);
 root.querySelector('.gate-value').textContent=g.toFixed(2);root.querySelector('.candidate-value').textContent=c.toFixed(2);
 root.querySelector('.formula').textContent=mode==='node'?`new = ${g.toFixed(2)} × old + ${(1-g).toFixed(2)} × candidate`:`new = ${(1-g).toFixed(2)} × old + ${g.toFixed(2)} × candidate`;
 root.querySelector('tbody').innerHTML=changed.map((v,i)=>`<tr><td>${i+1}</td><td>${[c,.1,.6][i].toFixed(2)}</td><td>${base[i].toFixed(4)}</td><td class="state">${v.toFixed(4)}</td><td>${(v-base[i]).toFixed(4)}</td></tr>`).join('');
 root.querySelector('.readout').textContent=`Final state ${changed[2].toFixed(4)}; change from baseline ${(changed[2]-base[2]).toFixed(4)}. ${mode==='node'?'Gate 1 retains the old state.':'Gate 1 selects the supplied candidate.'}`;
 }
 gate.addEventListener('input',render);candidate.addEventListener('input',render);root.querySelector('button').addEventListener('click',()=>{gate.value='.25';candidate.value='.8';render();});render();
 return {setGate(x){gate.value=Math.max(0,Math.min(1,x));render();},setCandidate(x){candidate.value=Math.max(-1,Math.min(1,x));render();},trace};
}};
