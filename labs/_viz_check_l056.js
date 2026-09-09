/* Arithmetic and actual event callbacks under a minimal DOM, not a browser test. */
const assert=require('assert');
require('../assets/leaderboard-audit-viz.js');
const api=globalThis.LeaderboardAudit;
for(let n=1;n<=30;n++){
  const v=api.weighting(n);assert(Math.abs(v.a+v.b-3)<1e-12);assert.equal(v.macro,1.5);
  if(n>1)assert(v.b<v.a);
}
assert.equal(api.weighting(99).n,30);assert.equal(api.weighting(-3).n,1);
assert.equal(api.elo(0).p,.5);assert(Math.abs(api.elo(400).p-10/11)<1e-12);
assert(Math.abs(api.elo(-400).p-1/11)<1e-12);
for(let g=-800;g<=800;g+=25)assert(Math.abs(api.elo(g).p+api.elo(-g).p-1)<1e-12);
function host(value){
  const nodes={};for(const name of ['input','output','tbody','button'])nodes[name]={value,events:{},addEventListener(k,fn){this.events[k]=fn;},textContent:'',innerHTML:''};
  return {nodes,querySelector(k){return nodes[k];},innerHTML:''};
}
const w=host(3);api.mountWeight(w);assert(w.nodes.tbody.innerHTML.includes('1.750'));
w.nodes.input.value=1;w.nodes.input.events.input();assert(w.nodes.tbody.innerHTML.includes('1.500'));
w.nodes.button.events.click();assert.equal(w.nodes.input.value,3);
const e=host(400);api.mountElo(e);assert(e.nodes.output.textContent.includes('90.91%'));
e.nodes.input.value=-400;e.nodes.input.events.input();assert(e.nodes.output.textContent.includes('9.09%'));
e.nodes.button.events.click();assert.equal(e.nodes.input.value,400);
console.log('PASS: 30 repetition states, 65 rating gaps, bounds, live input and reset callbacks');
