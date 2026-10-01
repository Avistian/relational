/* Reusable evidence-origin intervention. All numerical examples are synthetic. */
(function (global) {
  'use strict';
  function compute(metric, origin, complete, units) {
    const higher = metric === 'AUROC';
    const model = higher ? .70 : 3, baseline = higher ? .60 : 4;
    const gap = higher ? model - baseline : baseline - model;
    const scale = higher && units === 'display' ? 100 : 1;
    return { model: model * scale, baseline: baseline * scale, gap: gap * scale,
      unit: higher ? (scale === 100 ? 'percentage points' : 'AUROC fraction') : 'positions',
      claim: !complete ? 'INCOMPLETE — no final comparison' :
        origin === 'published' ? 'PUBLISHED_CONTEXT_ONLY — local win NOT_ESTABLISHED' :
        'LOCAL_MATCHED_DESCRIPTIVE — fixture model has the better score',
      coverage: complete ? '1/1 completed fixture task' : '0/1 completed fixture task' };
  }
  function mount(host) {
    host.innerHTML = '<div class="evidence-kicker">Synthetic intervention · not measured results</div>' +
      '<h3>Keep the score. Change the claim.</h3><div class="evidence-controls">' +
      '<label>Metric<select data-metric><option>MAE</option><option>AUROC</option></select></label>' +
      '<label>Baseline evidence<select data-origin><option value="published">Published context</option><option value="local">Matched local fixture</option></select></label>' +
      '<label>Units<select data-units><option value="display">Display units</option><option value="raw">Stored units</option></select></label>' +
      '<label class="evidence-complete"><input type="checkbox" checked data-complete> Complete required runs</label></div>' +
      '<div class="evidence-numbers"><p>Model<strong data-model></strong></p><p>Baseline<strong data-baseline></strong></p><p>Benefit<strong data-gap></strong></p></div>' +
      '<p data-claim role="status" aria-live="polite"></p><p data-coverage></p>' +
      '<p class="evidence-footnote">Positive benefit favors the model. Matched local fixtures assume the same population and evaluation rule; no significance claim follows.</p>' +
      '<button type="button">Reset example</button>';
    const metric=host.querySelector('[data-metric]'), origin=host.querySelector('[data-origin]'),
      units=host.querySelector('[data-units]'), complete=host.querySelector('[data-complete]');
    function draw() {
      const r=compute(metric.value,origin.value,complete.checked,units.value);
      host.querySelector('[data-model]').textContent=r.model.toFixed(2);
      host.querySelector('[data-baseline]').textContent=r.baseline.toFixed(2);
      host.querySelector('[data-gap]').textContent='+'+r.gap.toFixed(2)+' '+r.unit;
      host.querySelector('[data-claim]').textContent=r.claim;
      host.querySelector('[data-coverage]').textContent=r.coverage;
      host.dataset.gap=String(r.gap);host.dataset.claim=r.claim;
    }
    host.addEventListener('change',draw);
    host.querySelector('button').addEventListener('click',()=>{
      metric.value='MAE';origin.value='published';units.value='display';complete.checked=true;draw();
    });
    draw();
  }
  global.PortfolioEvidence={compute,mount};
})(window);
