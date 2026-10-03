/* RT-v1 fixture: four masks × nine readers. Sources always precede attention.
   Mask rows = readers; columns = sources. Means illustrate one scalar value
   coordinate at equal logits, not predictions from the trained RT model. */
(function(){'use strict';
const cells=[['T0.y',100,0,0,10,0],['T0.time',100,0,1,10,10],['C0.age',10,1,0,-2,42],['C0.name',10,1,2,-2,1],['O0.amount',20,2,0,10,6],['O0.time',20,2,1,10,7],['T1.y',101,0,0,10,1],['T1.time',101,0,1,10,2],['C1.age',11,1,0,-2,20]];
function permits(a,b,k){return k==='full'||(k==='col'&&a[2]===b[2]&&a[3]===b[3])||(k==='feat'&&(a[1]===b[1]||a[4]===b[1]))||(k==='nbr'&&a[1]===b[4]);}
for(const root of document.querySelectorAll('[data-b10="masks"]')){
 const kind=root.querySelector('[name=kind]'),reader=root.querySelector('[name=reader]'),grid=root.querySelector('.b10-cells'),out=root.querySelector('output');
 function draw(){const i=+reader.value,k=kind.value;grid.replaceChildren();let ids=[],sum=0;
 cells.forEach((c,j)=>{const allowed=permits(cells[i],c,k);if(allowed){ids.push(j);sum+=c[5];}const d=document.createElement('div');d.className='b10-cell'+(allowed?' allowed':'')+(i===j?' reader':'');const strong=document.createElement('strong');strong.textContent=j+' · '+c[0];d.append(strong);const s=document.createElement('small');s.textContent=(allowed?'Allowed source':'Blocked source')+(i===j?' · reader':'');d.append(s);grid.append(d);});
 const mean=ids.length?sum/ids.length:0;out.dataset.count=ids.length;out.dataset.mean=mean;out.textContent='Reader '+cells[i][0]+' can read '+(ids.length?ids.map(j=>cells[j][0]).join(', '):'no cells')+'. Equal-score coordinate: '+(ids.length?sum+' / '+ids.length+' = '+mean.toFixed(3):'0 (empty neighborhood)')+'. T0.y uses a mask coordinate 0; its hidden target is never a value source.';
 }
 kind.addEventListener('change',draw);reader.addEventListener('change',draw);root.querySelector('button').addEventListener('click',()=>{kind.value='feat';reader.value='0';draw();});draw();
}
for(const root of document.querySelectorAll('[data-b10="time"]')){
 const slider=root.querySelector('input'),out=root.querySelector('output'),value=root.querySelector('[data-value]');
 function draw(){const c=+slider.value;value.textContent=c;const rows=[['order',7,7,0],['old task label',8,11,3],['scheduled race',12,9,0]];const text=rows.map(([n,e,a,h])=>n+': '+(e<=c&&a<=c&&e+h<=c?'eligible':'excluded'));const eligible=rows.filter(([,e,a,h])=>e<=c&&a<=c&&e+h<=c).length;out.dataset.count=eligible;out.textContent='Cutoff day '+c+'. '+text.join('; ')+'. The strict course policy checks event time AND recorded arrival AND label-window completion. A schedule known on day 9 but dated day 12 stays excluded until day 12 under this policy.';}
 slider.addEventListener('input',draw);root.querySelector('button').addEventListener('click',()=>{slider.value='10';draw();});draw();
}
if(window.RetrievalBank)RetrievalBank.mount(document.getElementById('b10-warmup'),{upTo:200.10,count:3});
if(window.Predict)Predict.mount(document.getElementById('b10-predict'),{prompt:'A task cell has no child rows. What should neighbor attention contribute?',options:[{label:'Return a zero vector',value:'zero'},{label:'Read every parent cell',value:'parent'}],correct:'zero',reveal:'Neighbor attention reads child rows. With none available, its update is zero. Feature attention handles the parent direction.'});
if(window.Teachback)Teachback.mount(document.getElementById('b10-teachback'),{prompt:'Explain how a masked task cell can use past labels without seeing its own answer. Include the four attention masks and the temporal failure found in the saved contexts.',points:['Query target is replaced before encoding','Feature and neighbor directions differ','Full attention requires safe context selection first','No target gradients does not mean no contextual labels','Event time and historical availability are different'],model:'The target cell becomes a learned type-specific mask vector. Valid past task labels remain ordinary context. Column attention links matching table/column identities, feature attention reads the same row and linked parents, neighbor attention reads children, and full attention reads the admitted context. Sampling must remove forbidden evidence first because later full attention can mix it everywhere. The saved RT-v1 contexts contain future-dated race fields; this violates the event-time rule, while their historical arrival remains unknown.'});
})();
