// Arithmetic + interaction checks; not a browser-rendering claim.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const context={window:{}};vm.createContext(context);
vm.runInContext(fs.readFileSync('assets/temporal-audit-viz.js','utf8'),context);
const viz=context.window.TemporalAuditViz;
for(let delay=0;delay<=4;delay++){
  const rows=viz.compute(delay);assert.equal(rows.length,12);
  assert.equal(rows.filter(r=>r.part==='train').length,7);
  assert.equal(rows.filter(r=>r.part==='validation').length,3);
  for(const r of rows){assert.equal(r.available,r.t+delay);assert.equal(r.eligible,r.part==='test'?null:r.available<=(r.part==='train'?8:11));}
}
assert.equal(viz.compute(3).filter(r=>r.part==='train'&&r.eligible).length,5);
assert.equal(viz.compute(3).filter(r=>r.part==='validation'&&r.eligible).length,1);
const nodes={};for(const tag of ['input','tbody','output','.delay','button'])nodes[tag]={value:3,addEventListener(event,fn){this[event]=fn;}};
const el={querySelector(s){return nodes[s];}};const api=viz.mount(el);
assert(nodes.output.textContent.includes('5 / 7'));
api.setDelay(0);assert(nodes.output.textContent.includes('7 / 7'));assert(nodes.output.textContent.includes('3 / 3'));
api.setDelay(99);assert.equal(nodes.input.value,4);assert(nodes.output.textContent.includes('0 / 3'));
nodes.button.click();assert.equal(nodes.input.value,3);
console.log('PASS: all delay states, fixed partitions, clamping, rendered readouts and reset (headless)');
