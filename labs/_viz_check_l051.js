/* Computation + DOM state checks; SVG export is not a live browser pass. */
const fs=require('fs'),vm=require('vm'),assert=require('assert');let context={};vm.createContext(context);vm.runInContext(fs.readFileSync('assets/intervention-viz.js','utf8'),context);let v=context.InterventionViz;
assert.equal(v.smoothing(0).label,1);assert.equal(v.smoothing(1).label,0);assert(Math.abs(v.smoothing(1).p-1/(1+2*Math.exp(-.5)))<1e-12);
assert.equal(v.smoothing(100).h,1.5);assert.equal(v.rotation(-5).degrees,0);
for(let a=0;a<=90;a+=5){let r=v.rotation(a);assert(Math.abs(r.point[0]*r.weight[0]+r.point[1]*r.weight[1]-1)<1e-12);assert(Math.abs(r.point[0]**2+r.point[1]**2-5)<1e-12);}
for(let n=2;n<=40;n++)assert(v.gains(n).max>=v.gains(n-1).max);
fs.writeFileSync('labs/_noise_l051_example.json',JSON.stringify(v.gains(40),null,2));
function root(){let nodes={};return {innerHTML:'',querySelector(k){return nodes[k]||(nodes[k]={innerHTML:'',value:0,textContent:'',events:{},addEventListener(t,f){this.events[t]=f;}});}};}
fs.mkdirSync('/tmp/l051-svg',{recursive:true});
for(let [name,values] of [['smooth',[0,.5,1,1.5]],['rotate',[0,45,90]],['noise',[1,10,40]]]){
 let r=root(),api=v[name](r);for(let value of values){api.setValue(value);assert(r.querySelector('.iv-value').textContent.length>80);
  for(let side of ['left','right']){let s=r.querySelector('.iv-'+side).innerHTML;assert(s.includes('<svg'));assert(!s.includes('NaN'));assert(!s.includes('undefined'));
   // All circles, bars and label anchor points stay inside each 340x240 panel.
   for(let match of s.matchAll(/(?:x|cx)="([\d.\-e]+)"/g))assert(+match[1]>=0 && +match[1]<=340);
   for(let match of s.matchAll(/(?:y|cy)="([\d.\-e]+)"/g))assert(+match[1]>=0 && +match[1]<=240);
   s=s.replace('<svg ','<svg xmlns="http://www.w3.org/2000/svg" ').replace('role="img"','width="340" height="240"') .replace('>','><style>text {font:13px sans-serif;fill:#263a40}</style>');
   fs.writeFileSync(`/tmp/l051-svg/${name}-${value}-${side}.svg`,s);}}
 r.querySelector('input').value=values[0];r.querySelector('input').events.input();r.querySelector('button').events.click();
}
for(let file of ['intervention-lesson.js','intervention-viz.js'])new vm.Script(fs.readFileSync('assets/'+file,'utf8'));
console.log('PASS: arithmetic, monotonic candidate search, state changes, reset/input events, SVG anchors and exports. Browser NOT_CHECKED.');
