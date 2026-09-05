/* Headless logic/DOM check. Does NOT verify rendered typography or geometry. */
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
let count=0;function ok(x){assert(x);count++;}
class Element{
 constructor(tag){this.tagName=tag;this.children=[];this.attrs={};this.style={};this.events={};this.textContent='';this.value='';}
 appendChild(x){this.children.push(x);return x;}
 setAttribute(k,v){this.attrs[k]=v;}
 addEventListener(k,v){this.events[k]=v;}
 set innerHTML(v){this.children=[];this._html=v;}
 get innerHTML(){return this._html||'';}
}
const global={},document={createElement:t=>new Element(t)};
vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../assets/saint-viz.js'),'utf8'),{window:global,document,Math,Number,String});
const m=global.SaintMath;
for(const x of [-3,-1,0,1,3]){let r=m.context(x);ok(Math.abs(r.weights.reduce((a,b)=>a+b,0)-1)<1e-12);ok(r.value>=2&&r.value<=8);}
ok(Math.abs(m.context(0).value-3.613648528219971)<1e-12);
ok(Math.abs(m.context(3).value-7.284782467867294)<1e-12);
ok(m.weights([1000,1000]).every(v=>v===.5));
const aroot=new Element('div'),axes=global.SaintAxesViz.mount(aroot);
for(let stage=0;stage<3;stage++){ok(axes.setStage(stage)===stage);ok(aroot.children[2].textContent.length>50);ok(aroot.children[0].children[stage].attrs['aria-pressed']==='true');}
ok(axes.setStage(99)===2);ok(axes.setStage(-2)===0);
aroot.children[0].children[1].events.click();ok(aroot.children[2].textContent.includes('Within')||aroot.children[2].textContent.includes('One attention'));
const croot=new Element('div'),context=global.SaintContextViz.mount(croot);
for(const x of [-10,-3,0,3,10]){let r=context.setKey(x);ok(Number.isFinite(r.value));ok(parseFloat(croot.children[3].style.width)>=0);ok(parseFloat(croot.children[3].style.width)<=100);}
const vroot=new Element('div'),views=global.SaintViewsViz.mount(vroot);
for(const cut of [false,true]){views.setCutMix(cut);for(const alpha of [-1,0,.2,1,2]){let r=views.setAlpha(alpha);ok(r.length===2);ok(r.every(Number.isFinite));}}
views.setCutMix(true);ok(JSON.stringify(views.setAlpha(1))==='[1,3]');ok(JSON.stringify(views.setAlpha(0))==='[3,1]');
console.log(count+' SAINT math/DOM checks PASS. Browser rendering remains unverified.');
// The reused CD diagram must not assert significance at p=.368 or clip long labels.
Element.prototype.removeChild=function(x){this.children.splice(this.children.indexOf(x),1);};
Object.defineProperty(Element.prototype,'firstChild',{get(){return this.children[0];}});
document.createElementNS=(_,tag)=>new Element(tag);
vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../assets/cd-diagram-viz.js'),'utf8'),{window:global,document,Math,Number});
for(const p of ['0.368','0.001']){
 const root=new Element('div');
 const api=global.CdDiagramViz.mount(root,{models:[{name:'Feature-only',rank:5/3},{name:'Feature + row',rank:5/3},{name:'CatBoost',rank:8/3}],cd:1.9136235155,N:3,friedmanP:p});
 ok(root.children[1].innerHTML.includes(p==='0.368'?'no overall rank difference':'an overall rank difference'));
 const svg=root.children[0];
 for(const label of svg.children.filter(x=>x.tagName==='text')){
  const x=Number(label.attrs.x);ok(x>=0&&x<=640);
  if(label.attrs.class==='cdv-name'){ok(x-label.textContent.length*6.7>=0);}
 }
 api.select('Feature + row');ok(root.children[1].innerHTML.includes('Feature + row'));
 api.select(null);ok(root.children[1].innerHTML.includes('Friedman'));
}
console.log(count+' total checks PASS, including CD significance text and label-space bounds.');
