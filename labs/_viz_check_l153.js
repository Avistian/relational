const assert=require('assert');const {compute}=require('../assets/recommendation-ranking-viz.js');
assert.equal(compute(3,2,'full').ap,.5);assert.equal(compute(3,2,'sampled').ap,1);assert(Math.abs(compute(3,3,'full').ap-5/6)<1e-12);
for(const pos of [1,2,3,4])for(const k of [2,3])for(const mode of ['full','sampled']){let r=compute(pos,k,mode);assert(r.ap>=0&&r.ap<=1&&r.recall>=0&&r.recall<=1);assert.equal(new Set(r.ranked).size,r.ranked.length);}
console.log('PASS:16 ranking states, AP denominators and changed-candidate contrast');
