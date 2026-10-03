const assert=require('assert'),api=require('../assets/reproduction-contract.js');
assert.deepEqual(api.encoding('a','released').codes,[1,2,3]);
assert.deepEqual(api.encoding('z','released').codes,[0,1,2]);
assert.deepEqual(api.encoding('a','frozen').codes,[0,1,2]);
assert.equal(api.clock(1).released,true);assert.equal(api.clock(1).available,false);
assert.equal(api.clock(365).available,true);assert.equal(api.clock(0).released,false);
console.log('L192 widget contracts PASS');
