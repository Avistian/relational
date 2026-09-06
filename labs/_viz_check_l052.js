const fs=require('fs'),vm=require('vm'),assert=require('assert');
class Node{constructor(tag='div'){this.tag=tag;this.children=[];this.events={};this.classList={add(){}};this._html='';}append(...x){this.children.push(...x)}addEventListener(e,f){this.events[e]=f}set innerHTML(x){this._html=x;this.children=[]}get innerHTML(){return this._html}}
const document={createElement:t=>new Node(t),createTextNode:t=>({textContent:t})};const ctx={document};vm.createContext(ctx);vm.runInContext(fs.readFileSync('assets/tabr-viz.js','utf8'),ctx);
let checks=0;
for(let q=0;q<=3;q+=.25)for(const mask of [false,true])for(const beta of [-1,0,.5,1]){
 const rows=ctx.TabRViz.compute(q,mask,beta);assert.equal(rows.length,2);assert(Math.abs(rows.reduce((s,r)=>s+r.w,0)-1)<1e-12);
 if(mask)assert(rows.every(r=>r.id!=='SELF'));
 rows.forEach(r=>{assert.equal(r.v,r.y+beta*(q-r.k));assert.equal(r.s,-((q-r.k)**2));});checks++;
}
const baseline=ctx.TabRViz.compute();assert.equal(baseline[0].w,.5);assert.equal(baseline[1].w,.5);assert.equal(baseline[0].v,.5);
function descendants(n){return [n,...(n.children||[]).flatMap(descendants)]}
for(const mode of ['neighbors','values','availability']){
 const node=new Node();ctx.TabRViz.mount(node,mode);assert(node.children[1].innerHTML.includes('<table>'));
 const inputs=descendants(node).filter(n=>n.tag==='input');
 for(const input of inputs){if(input.type==='range'){for(const v of [input.min,input.max]){input.value=v;input.events.input();assert(!node.children[1].innerHTML.includes('NaN'));checks++;}}else{input.checked=false;input.events.change();checks++;}}
 const reset=descendants(node).find(n=>n.tag==='button');reset.events.click();assert.equal(node.children.length,2);checks++;
}
console.log(JSON.stringify({status:'PASS',checks,scope:'Arithmetic and DOM interactions; not browser layout'}));
