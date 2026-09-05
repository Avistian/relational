/* SVG geometry and real widget handlers, not a browser rendering claim. */
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
let count=0;function ok(x,m){assert(x,m);count++;}
class Element {
 constructor(tag){this.tagName=tag;this.children=[];this.attrs={};this.events={};this.style={};this.textContent='';this.clientWidth=640;}
 appendChild(x){this.children.push(x);return x;} removeChild(x){this.children.splice(this.children.indexOf(x),1);}
 get firstChild(){return this.children[0];} setAttribute(k,v){this.attrs[k]=String(v);} getAttribute(k){return this.attrs[k];}
 addEventListener(k,v){this.events[k]=v;}
}
const global={},document={createElement:t=>new Element(t),createElementNS:(_,t)=>new Element(t)};
vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../assets/dcn-viz.js'),'utf8'),{window:global,document,Math,Number,Object,Array});
const m=global.DcnMath;
ok(JSON.stringify(m.cross(1).out)==='[14,9]','worked example');
for(const v of [-1,0,1,2]){ok(m.cross(v).out[0]===8+6*v);ok(m.cross(v).out[1]===9);}
ok(m.lowrank(4).error===0);ok(m.lowrank(2).cost===16);ok(m.lowrank(1).cost<16);
for(let r=1;r<=4;r++){const q=m.lowrank(r);let error=0;for(let i=0;i<4;i++)for(let j=0;j<4;j++)error+=(q.full[i][j]-q.approx[i][j])**2;ok(Math.abs(Math.sqrt(error)-q.error)<1e-12);}
for(const t of [-4,0,4]){const q=m.mix(t);ok(q.out[0]>=2&&q.out[0]<=2+q.e1[0]);ok(q.out[1]>=3&&q.out[1]<=3+q.e2[1]);}
function nodes(n){return [n,...n.children.flatMap(nodes)];}
function geometry(svg){
 const [,,W,H]=svg.attrs.viewBox.split(' ').map(Number),ns=nodes(svg);
 const rects=ns.filter(n=>n.tagName==='rect');
 for(const n of rects){const a=n.attrs;ok(+a.x>=0&&+a.y>=0&&+a.x+ +a.width<=W+.01&&+a.y+ +a.height<=H+.01,'rectangle bounds');}
 const boxes=ns.filter(n=>n.attrs['data-box']).map(n=>n.children[0].attrs);
 for(let i=0;i<boxes.length;i++)for(let j=i+1;j<boxes.length;j++){const a=boxes[i],b=boxes[j];ok(!(Math.max(+a.x,+b.x)<Math.min(+a.x+ +a.width,+b.x+ +b.width)&&Math.max(+a.y,+b.y)<Math.min(+a.y+ +a.height,+b.y+ +b.height)),'overlapping cards');}
 for(const n of ns.filter(n=>n.tagName==='text'))ok(+n.attrs.x>=0&&+n.attrs.x<=W&&+n.attrs.y>=0&&+n.attrs.y<=H,'text anchor bounds');
}
function escape(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');}
function serialize(n){return '<'+n.tagName+Object.entries(n.attrs).map(([k,v])=>' '+k+'="'+escape(v)+'"').join('')+'>'+escape(n.textContent)+n.children.map(serialize).join('')+'</'+n.tagName+'>';}
const out=process.argv[2];if(out)fs.mkdirSync(out,{recursive:true});let states=0;
for(const width of [640,330])for(const [kind,values] of Object.entries({cross:[-1,0,1,2],degree:[1,2,3,4],rank:[1,2,3,4],architecture:[0,1],mix:[-4,0,4]})){
 const root=new Element('div');root.clientWidth=width;const api=global.DcnViz.mount(root,{kind});
 for(const v of values){ok(api.set(v)===v);geometry(api.svg);ok(api.output.textContent.length>80,'explanatory feedback');if(out)fs.writeFileSync(path.join(out,kind+'-'+width+'-'+v+'.svg'),serialize(api.svg));states++;}
 for(const n of nodes(root)){if(n.events.click){n.events.click();geometry(api.svg);}if(n.events.input){n.value=n.min;n.events.input();geometry(api.svg);n.value=n.max;n.events.input();geometry(api.svg);}}
 ok(api.set(-999)===values[0]);ok(api.set(999)===values[values.length-1]);geometry(api.svg);
}
console.log(count+' checks passed; '+states+' desktop/mobile SVG states. Browser not checked.');
