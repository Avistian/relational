/* Real arithmetic and interaction state checks. No browser-rendering claim. */
const fs=require('fs'),vm=require('vm'),assert=require('assert');
let context={};vm.createContext(context);vm.runInContext(fs.readFileSync('assets/checkpoint-viz.js','utf8'),context);
const v=context.CheckpointViz;
for(const n of [2,100,200]){assert.equal(v.fitState(n,false).mean,1);assert.equal(v.fitState(n,false).centered,1);assert.equal(v.fitState(n,true).mean,(2+n)/3);}
assert.equal(v.fitState(-100,false).held,2);assert.equal(v.fitState(999,false).held,200);
assert.equal(v.choice(false).id,'B');assert.equal(v.choice(true).id,'A');
function root(){const nodes={};return {className:'',innerHTML:'',querySelector(k){return nodes[k]||(nodes[k]={value:100,checked:false,textContent:'',events:{},addEventListener(type,fn){this.events[type]=fn;}});}};}
let a=root(),api=v.fit(a);assert(a.querySelector('.safe').textContent.includes('1.00'));api.setValue(200);assert(a.querySelector('.leak').textContent.includes('67.33'));a.querySelector('button').events.click();assert.equal(a.querySelector('input').value,100);
let b=root(),choice=v.selection(b);assert(b.querySelector('.selected').textContent.startsWith('B'));choice.setPeek(true);assert(b.querySelector('.readout').textContent.includes('new untouched'));choice.setPeek(false);assert(b.querySelector('.selected').textContent.startsWith('B'));
for(const file of ['assets/checkpoint-viz.js','assets/checkpoint-lesson.js','assets/checkpoint-results.js'])new vm.Script(fs.readFileSync(file,'utf8'));
console.log('PASS: synthetic arithmetic, clamping, input/reset events, validation/test selection states, JS parsing. Browser layout NOT checked.');
