/* State dependencies: predictions may be unchanged even when identity changes. */
(()=>{'use strict';const states={
none:['Reuse','Reuse','Reuse','Reuse','All dependencies unchanged.'],
support:['Rebuild','Recompute','Recompute','Revalidate or retrain','Support identity changed, even if this query receives the same probability.'],
preprocess:['Rebuild','Recompute','Recompute','Revalidate or retrain','Transformed support or query representation changed.'],
query:['Reuse','Recompute','Recompute','Reuse','Support cache remains valid; memoized query outputs do not.'],
background:['Reuse','Reuse','Recompute','Reuse','Only the explanation reference changed; the predictor is unchanged.']};
for(const panel of document.querySelectorAll('[data-context-state]')){const select=panel.querySelector('select'),out=panel.querySelector('output');function show(){const s=states[select.value];out.textContent=`Support cache: ${s[0]}. Query prediction: ${s[1]}. Explanation result: ${s[2]}. Distilled student: ${s[3]}. ${s[4]}`;out.dataset.state=select.value;}select.addEventListener('change',show);panel.querySelector('button').addEventListener('click',()=>{select.value='none';show();});show();}})();
