/* Real widget handlers and arithmetic in a minimal DOM. Not a browser pass. */
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
let checks=0;const ok=(condition,message)=>{assert(condition,message);checks++;};
class Element{
 constructor(tag){this.tagName=tag;this.children=[];this.events={};this.attrs={};this.textContent='';this.classList={add:()=>{}};}
 appendChild(n){this.children.push(n);return n;}
 append(...ns){ns.forEach(n=>this.appendChild(n));}
 set innerHTML(v){this.children=[];this.textContent=v;}
 setAttribute(k,v){this.attrs[k]=v;}
 addEventListener(k,f){this.events[k]=f;}
 querySelectorAll(tag){return nodes(this).filter(n=>n!==this&&n.tagName===tag);}
}
function nodes(root){return [root,...root.children.flatMap(nodes)];}
const document={createElement:t=>new Element(t),createTextNode:t=>{const n=new Element('#text');n.textContent=t;return n;}};
const window={};vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../assets/claim-audit-viz.js'),'utf8'),{window,document});
const v=window.ClaimAuditViz;
for(let noise=0;noise<=18;noise++){
 const a=v.spa(noise),b=v.spa(noise,false);
 ok(a.output[0]===2&&a.output[1]===3,'Blocked routes protect stronger receivers');
 ok(Math.abs(a.output[2]-(6+noise)/3)<1e-12,'Weak receiver weighted sum');
 ok(b.output.every(x=>Math.abs(x-(6+noise)/3)<1e-12),'Unmasked intervention reaches all receivers');
}
for(let t=-4;t<=4;t+=.25){
 const a=v.prompt(t);ok(Math.abs(a.weights.reduce((x,y)=>x+y,0)-1)<1e-12,'Column normalization');
 ok(Math.abs(a.output-(2*Math.exp(t)+13)/(Math.exp(t)+2))<1e-12,'Prompt weighted sum');
}
for(let mask=0;mask<8;mask++){
 const m=[0,1,2].map(i=>Boolean(mask&(1<<i)));ok(v.mix(m)===m.reduce((s,b,i)=>s+(b?[6,3,1][i]:0),0)/10,'All feature masks');
}
let states=0;
for(const kind of ['spa','prompt','mix']){
 const root=new Element('div');v.mount(root,{kind});
 const text=()=>nodes(root).map(n=>n.textContent).join(' ');
 ok(text().length>250,'Explanatory default');
 for(const input of root.querySelectorAll('input')){
  if(input.type==='range')for(const value of [input.min,input.max]){input.value=value;input.events.input();ok(text().length>250,'Range intervention');states++;}
  else for(const value of [false,true]){input.checked=value;input.events.change();ok(text().length>250,'Checkbox intervention');states++;}
 }
 const button=root.querySelectorAll('button')[0];button.events.click();ok(text().includes(kind==='mix'?'0.600':kind==='prompt'?'query [0, 0]':'weak value 9'),'Reset restores baseline');
 ok(nodes(root).some(n=>n.attrs['aria-live']==='polite'),'Accessible live feedback');
}
console.log(JSON.stringify({checks,interaction_states:states,status:'PASS',browser:'NOT_CHECKED'}));
