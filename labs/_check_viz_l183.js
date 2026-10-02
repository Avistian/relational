const fs=require('fs'),vm=require('vm'),assert=require('assert');
const context={window:{}};vm.createContext(context);vm.runInContext(fs.readFileSync('assets/factorial-contrast.js','utf8'),context);
const contrast=context.window.FactorialContrast.contrast;
assert(Math.abs(contrast([4,3.6,3.9,3.2]).interaction-.3)<1e-12);
assert(Math.abs(contrast([4,3.6,3.9,3.5]).interaction)<1e-12);
assert(contrast([5,4,3,2.5]).interaction<0);
assert.throws(()=>contrast([NaN,3,2,1]));
console.log('PASS: synthetic contrast arithmetic');
