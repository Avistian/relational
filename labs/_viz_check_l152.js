const assert=require('assert');const {compute}=require('../assets/regression-calibration-viz.js');
let r=compute(1,'ties');assert.deepEqual([r.below,r.equal,r.above],[.25,.5,.25]);assert.equal(r.violation,0);assert.equal(r.mae,2.25);
r=compute(1,'shift');assert.equal(r.violation,.25);assert.equal(r.below,.75);
r=compute(2.75,'ties');assert.equal(r.violation,.25);assert.equal(r.mae,3.125);
for(let p=0;p<=10;p+=.25)for(const mode of ['ties','shift']){r=compute(p,mode);assert(Math.abs(r.below+r.equal+r.above-1)<1e-12);assert(r.rmse>=r.mae-1e-12)}
console.log('PASS 82 intervention states and worked median arithmetic');
