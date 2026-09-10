// Arithmetic/state checks; real layout and keyboard behavior use Chromium separately.
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const ctx = {};
vm.runInNewContext(fs.readFileSync('assets/tabm-viz.js', 'utf8'), ctx);
const viz = ctx.TabMViz;
assert.deepStrictEqual(Array.from(viz.compute({r:[1,1],s:[1,1]}).outA), [11,16]);
assert.deepStrictEqual(Array.from(viz.compute({r:[1,-1],s:[1,1]}).outA), [-7,-8]);
let cases = 0;
for (const a of [-1, 1]) for (const b of [-1, 1])
for (const c of [-1, 1]) for (const d of [-1, 1]) {
  const r = [a, b], s = [c, d], result = viz.compute({r, s});
  for (let o = 0; o < 2; o++) {
    const expected = s[o] * viz.W[o].reduce((sum, w, i) => sum + w * r[i] * viz.X[i], 0);
    assert(Math.abs(expected - result.outA[o]) < 1e-12);
    assert(Math.abs(expected - result.outB[o]) < 1e-12);
  }
  // Equal effective matrices do not imply equal adapter parameters.
  assert.strictEqual(result.atInit, [...r, ...s].every(x => x === 1));
  assert(result.match);
  cases++;
}
for (const k of [1, 2, 4, 16, 32]) for (const rho of [0, .3, .75, 1]) {
  // Expand k variances and k(k-1) covariances directly.
  const expanded = (k + k * (k - 1) * rho) / (k * k);
  assert(Math.abs(viz.varMean(k, rho) - expanded) < 1e-12);
  cases++;
}
console.log(`PASS: ${cases} TabM arithmetic and adapter-state cases; browser checked separately`);
