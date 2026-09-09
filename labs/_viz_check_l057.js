const fs=require('fs'),vm=require('vm'),assert=require('assert');const ctx={window:{}};vm.createContext(ctx);vm.runInContext(fs.readFileSync('assets/cross-ensemble-viz.js','utf8'),ctx);const f=ctx.window.CrossEnsembleViz.compute;
assert(Math.abs(f(.5,false).mixed+Math.log(.55))<1e-12);
for(const w of [0,.25,.5,.75,1]){let r=f(w,true);assert(Math.abs(r.mixed-r.baseline)<1e-12);}
assert(f(-1,false).w===0);assert(f(2,false).w===1);assert(f(.5,false).mixed<f(.5,false).baseline);
console.log('PASS: mixture arithmetic, duplicate invariance, endpoints and clamping; browser rendering NOT_CHECKED');
const nodes={};for(const k of ['input[type=range]','input[type=checkbox]','output','tbody','.ensemble-readout','button'])nodes[k]={value:k.includes('range')?'.5':'',checked:false,events:{},addEventListener(k,fn){this.events[k]=fn;},textContent:'',innerHTML:''};
const host={innerHTML:'',querySelector(k){return nodes[k];}};ctx.window.CrossEnsembleViz.mount(host);
assert(nodes['.ensemble-readout'].textContent.includes('0.5978'));
nodes['input[type=checkbox]'].checked=true;nodes['input[type=checkbox]'].events.change();assert(nodes['.ensemble-readout'].textContent.includes('change: 0.0000'));
nodes['input[type=range]'].value='0';nodes['input[type=range]'].events.input();assert(nodes.output.textContent==='0.00');
nodes.button.events.click();assert(nodes.output.textContent==='0.50');assert(nodes['input[type=checkbox]'].checked===false);
console.log('PASS: native-control callbacks and reset under minimal DOM; not browser rendering');
