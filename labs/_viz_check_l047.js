/* Logic, interactions, SVG bounds, and optional exported states for visual inspection.
 * Node mocks are not a browser rendering test. Export: node labs/_viz_check_l047.js /tmp/l047-viz
 */
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
let count=0;function ok(x,message){assert(x,message);count++;}
class Element {
 constructor(tag){this.tagName=tag;this.children=[];this.attrs={};this.style={};this.events={};this.textContent='';this.value='';this.clientWidth=640;}
 appendChild(x){if(this.children.includes(x))this.removeChild(x);this.children.push(x);return x;}
 removeChild(x){this.children.splice(this.children.indexOf(x),1);}
 get firstChild(){return this.children[0];}
 getAttribute(k){return this.attrs[k];}
 setAttribute(k,v){this.attrs[k]=String(v);}
 addEventListener(k,v){this.events[k]=v;}
 set innerHTML(v){this.children=[];this._html=v;}
 get innerHTML(){return this._html||'';}
}
const global={},document={createElement:t=>new Element(t),createElementNS:(_,t)=>new Element(t)};
vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../assets/saint-viz.js'),'utf8'),{window:global,document,Math,Number,String,Object,Array});
const m=global.SaintMath;
for(const x of [-3,-1,0,1,3]){const r=m.context(x);ok(Math.abs(r.weights.reduce((a,b)=>a+b,0)-1)<1e-12);ok(r.value>=2&&r.value<=8);}
ok(Math.abs(m.context(0).value-3.613648528219971)<1e-12);
ok(Math.abs(m.context(3).value-7.284782467867294)<1e-12);
ok(m.weights([1000,1000]).every(v=>v===.5));
ok(m.contrast(.2).loss<m.contrast(2).loss);
function nodes(root){return [root,...root.children.flatMap(nodes)];}
function check(root){
 const svg=nodes(root).find(x=>x.tagName==='svg');const [, , W,H]=svg.attrs.viewBox.split(' ').map(Number);
 const ns=nodes(svg);ok(ns.filter(x=>x.tagName==='text').length>=8,'diagram labels');
 ok(nodes(root).find(x=>x.tagName==='output').textContent.length>50,'caption');
 for(const n of ns.filter(x=>x.tagName==='rect')){
  const a=n.attrs,x=+a.x,y=+a.y,w=+a.width,h=+a.height;
  ok(x>=0&&y>=0&&w>=0&&h>=0&&x+w<=W+.01&&y+h<=H+.01,'rect out of bounds: '+JSON.stringify(a));
 }
 const boxes=ns.filter(x=>x.attrs['data-box']).map(x=>x.children[0].attrs);
 for(let i=0;i<boxes.length;i++)for(let j=i+1;j<boxes.length;j++){
  const a=boxes[i],b=boxes[j];ok(!(Math.max(+a.x,+b.x)<Math.min(+a.x+ +a.width,+b.x+ +b.width)&&Math.max(+a.y,+b.y)<Math.min(+a.y+ +a.height,+b.y+ +b.height)),'overlapping cards');
 }
 for(const n of ns.filter(x=>x.tagName==='text'))ok(+n.attrs.x>=0&&+n.attrs.x<=W&&+n.attrs.y>=0&&+n.attrs.y<=H,'text anchor out of bounds');
 return svg;
}
function escape(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');}
function serialize(n){return '<'+n.tagName+Object.entries(n.attrs).map(([k,v])=>' '+k+'="'+escape(v)+'"').join('')+'>'+escape(n.textContent)+n.children.map(serialize).join('')+'</'+n.tagName+'>';}
const outDir=process.argv[2];if(outDir)fs.mkdirSync(outDir,{recursive:true});let snapshots=[];
function exercise(root){
 for(const button of nodes(root).filter(n=>n.events.click)){button.events.click();check(root);}
 for(const input of nodes(root).filter(n=>n.events.input)){for(const v of [input.min,input.max]){input.value=v;input.events.input();check(root);}}
}
function save(root,name){const svg=check(root);if(outDir){let i=0;nodes(svg).filter(n=>n.tagName==='text').forEach(n=>n.attrs.id='label-'+i++);fs.writeFileSync(path.join(outDir,name+'.svg'),serialize(svg));snapshots.push(name);}}
for(const width of [640,330]){
 function mount(kind){const r=new Element('div');r.clientWidth=width;return [r,global[kind].mount(r)];}
 let [r,a]=mount('SaintAxesViz');for(let i=0;i<3;i++){ok(a.setStage(i)===i);save(r,'axes-'+width+'-'+i);}ok(a.setStage(99)===2);ok(a.setStage(-1)===0);
 exercise(r);[r,a]=mount('SaintArchitectureViz');for(let i=0;i<5;i++){ok(a.setStage(i)===i);save(r,'architecture-'+width+'-'+i);}
 exercise(r);[r,a]=mount('SaintContextViz');for(const v of [-3,0,3]){const state=a.setKey(v);ok(Math.abs(state.value-m.context(v).value)<1e-12);save(r,'context-'+width+'-'+v);}ok(a.setKey(-20).value===m.context(-3).value);ok(a.setKey(20).value===m.context(3).value);
 exercise(r);[r,a]=mount('SaintViewsViz');for(const cut of [false,true])for(const v of [0,.2,1]){a.setCutMix(cut);const state=a.setAlpha(v);ok(state.length===2&&state.every(Number.isFinite));save(r,'views-'+width+'-'+cut+'-'+v);}a.setCutMix(true);ok(JSON.stringify(a.setAlpha(1))==='[1,3]');ok(JSON.stringify(a.setAlpha(0))==='[3,1]');
 exercise(r);[r,a]=mount('SaintContrastViz');for(const v of [.2,.7,2]){const state=a.setTemperature(v);ok(state.loss>0);save(r,'contrast-'+width+'-'+v);}
 // Exercise native event handlers, not just the public setters.
 for(const b of nodes(r).filter(n=>n.events.click))b.events.click();
 for(const input of nodes(r).filter(n=>n.events.input)){input.value=1;input.events.input();}
 check(r);
}
vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../assets/cd-diagram-viz.js'),'utf8'),{window:global,document,Math,Number});
for(const p of ['0.368','0.001']){
 const root=new Element('div');const api=global.CdDiagramViz.mount(root,{models:[{name:'Feature-only',rank:5/3},{name:'Feature + row',rank:5/3},{name:'CatBoost',rank:8/3}],cd:1.9136235155,N:3,friedmanP:p});
 ok(root.children[1].innerHTML.includes(p==='0.368'?'no overall rank difference':'an overall rank difference'));
 api.select('Feature + row');ok(root.children[1].innerHTML.includes('Feature + row'));
}
console.log(count+' SAINT checks PASS; '+snapshots.length+' desktop/mobile SVG states exported. Browser rendering is a separate check.');
