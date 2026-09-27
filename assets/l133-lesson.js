(() => {
  const host = document.getElementById('l133-layer');
  if (!host) return;
  host.innerHTML = `<label for="l133-root">Customer vector <input id="l133-root" type="range" min="0" max="8" value="4" step="1"></label>
  <label for="l133-presence">Tickets relation <select id="l133-presence"><option value="present">Present: one neighbor, value 5</option><option value="empty">Present: empty edge tensor</option><option value="absent">Absent: no relation key</option></select></label>
  <p><output aria-live="polite"></output></p><button type="button">Reset</button><p>Scalar pre-normalization arithmetic. Orders always have neighbors 1 and 2. Baseline: customer 4, tickets present, total 12.5.</p>`;
  const root=host.querySelector('input'),select=host.querySelector('select'),out=host.querySelector('output');
  function update(){const h=Number(root.value),orders=7+3*h,tickets=select.value==='absent'?0:(select.value==='empty'?0:2.5)-1-2*h;
    out.textContent=`Customer ${h}: orders ${orders}; tickets ${select.value==='absent'?'skipped':tickets}; total ${orders+tickets}.`;
  }
  root.addEventListener('input',update);select.addEventListener('change',update);
  host.querySelector('button').addEventListener('click',()=>{root.value='4';select.value='present';update();});update();
})();
