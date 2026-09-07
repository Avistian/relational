// Synthetic DOM check; no claim of real browser rendering.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
class Element{
 constructor(){this.children=[];this.style={};this.handlers={};this._html='';this.value='';}
 appendChild(v){this.children.push(v);return v;}setAttribute(k,v){this[k]=v;}
 addEventListener(k,f){this.handlers[k]=f;}set innerHTML(v){this._html=v;this.children=[];}get innerHTML(){return this._html;}
}
const ctx={document:{createElement:()=>new Element()}};ctx.window=ctx;
vm.runInNewContext(fs.readFileSync('assets/realmlp-viz.js','utf8'),ctx);
const viz=ctx.RealMLPViz;let checks=0;
for(const mode of ['preprocess','ntp','schedule']){
 const el=new Element(),api=viz.mount(el,mode);assert(el.children.length>=3);
 for(const x of [-100,0,.25,1,2,12,100]){api.set(x);assert(el.children[1].textContent.length>50);checks++;}
 el.children.at(-1).handlers.click();assert(+api.input.value===(mode==='schedule'?.25:2));
}
assert(Math.abs(viz.compute('preprocess',2).smooth-6/Math.sqrt(13))<1e-12);
assert.equal(viz.compute('ntp',2).output,-.5);assert.equal(viz.compute('ntp',1).output,-1);
for(let j=0;j<=4;j++){assert(Math.abs(viz.compute('schedule',(2**j-1)/15).f)<1e-12);checks++;}
for(let j=0;j<4;j++){assert(Math.abs(viz.compute('schedule',(2**(j+.5)-1)/15).f-1)<1e-12);checks++;}
for(let i=0;i<=100;i++){const r=viz.compute('schedule',i/100);assert(Math.abs(r.scale-6*r.weight)<1e-12);assert(Math.abs(r.bias-.1*r.weight)<1e-12);checks++;}
// Mount lesson pedagogy and verify all quiz answer word counts are equal.
const ids=new Set([...fs.readFileSync('lessons/0053-realmlp-strong-defaults.html','utf8').matchAll(/id="([^"]+)"/g)].map(m=>m[1]));
ctx.document.getElementById=id=>{assert(ids.has(id),'missing mount '+id);return new Element();};
for(const name of ['RetrievalBank','Quiz','Predict','Teachback'])ctx[name]={mount:(el,c)=>{if(c.options){const n=c.options.map(v=>v.label.split(/\s+/).length);assert(n.every(x=>x===n[0]));}checks++;}};
vm.runInNewContext(fs.readFileSync('assets/realmlp-lesson.js','utf8'),ctx);
console.log(`PASS: ${checks} synthetic arithmetic, control and lesson-mount cases; browser NOT_CHECKED`);
