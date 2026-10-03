/* Course-designed contract simulator; it does not inspect model inputs. */
(function(g){
'use strict';
function evaluate(values){
 const keys=['features','support','visibility'];
 const mismatches=keys.filter(k=>values[k]==='different'),unknown=keys.filter(k=>values[k]==='unknown');
 const status=mismatches.length?'INCOMPARABLE':unknown.length?'NOT_ESTABLISHED':'MATCHED';
 let claim=values.claim==='learner_mastery'?'PENDING_WRITTEN_DEFENSE':values.claim==='fresh_inference'?'NOT_RUN':values.claim==='architecture_cause'?'NOT_ESTABLISHED':values.evidence==='INCOMPLETE'?'INCOMPLETE':status==='MATCHED'?'SUPPORTED_DESCRIPTIVE_REPLAY':status;
 return {status,claim,mismatches,unknown};
}
function mount(board){
 const selects=Array.from(board.querySelectorAll('select')),out=board.querySelector('output');
 function update(){const v={};selects.forEach(s=>v[s.name]=s.value);const r=evaluate(v);out.textContent=r.status+' → '+r.claim+'. Mismatched: '+(r.mismatches.join(', ')||'none')+'. Unknown: '+(r.unknown.join(', ')||'none')+'. Other fields are held at their declared baseline.';out.dataset.status=r.status;out.dataset.claim=r.claim;}
 selects.forEach(s=>s.addEventListener('change',update));board.querySelector('button').addEventListener('click',()=>{selects.forEach(s=>s.selectedIndex=0);update();});update();
}
function mountFamilies(el){const select=el.querySelector('select');function update(){el.querySelectorAll('[data-family]').forEach(card=>card.hidden=card.dataset.family!==select.value);}select.addEventListener('change',update);update();}
g.ComparisonContract={evaluate,mount,mountFamilies};
document.querySelectorAll('.comparison-board').forEach(mount);document.querySelectorAll('.family-explorer').forEach(mountFamilies);
})(window);
