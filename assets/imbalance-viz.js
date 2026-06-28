/**
 * Reusable class-imbalance visualizer: the accuracy paradox + threshold tuning.
 *
 * A fixed, deterministic population of scored examples (most negative, few
 * positive). A threshold slider classifies them; the widget shows a live
 * confusion matrix and three metrics. The lesson: accuracy can stay high while
 * recall collapses, and moving the threshold trades precision against recall.
 *
 * Usage: ImbalanceViz.mount(containerElement, { posRate: 0.1 })
 * Companion CSS lives in the lesson <style> block (class prefix .imb-viz).
 */
(function (global) {
  "use strict";

  // Deterministic score distribution. Each example: { score, label }.
  // Negatives cluster low, positives skew high but overlap — a realistic, hard
  // imbalanced problem. 200 examples, ~10% positive.
  function buildData() {
    var data = [];
    var i;
    // a small linear congruential PRNG for stable, seedless determinism
    var seed = 12345;
    function rand() {
      seed = (1103515245 * seed + 12345) % 2147483648;
      return seed / 2147483648;
    }
    function clamp(v) { return Math.max(0.001, Math.min(0.999, v)); }
    for (i = 0; i < 180; i++) {
      // negatives: centered ~0.30
      data.push({ score: clamp(0.30 + (rand() - 0.5) * 0.5), label: 0 });
    }
    for (i = 0; i < 20; i++) {
      // positives: centered ~0.62, overlapping the negatives
      data.push({ score: clamp(0.62 + (rand() - 0.5) * 0.5), label: 1 });
    }
    return data;
  }

  var DATA = buildData();

  function metricsAt(threshold) {
    var tp = 0, fp = 0, tn = 0, fn = 0;
    DATA.forEach(function (d) {
      var pred = d.score >= threshold ? 1 : 0;
      if (pred === 1 && d.label === 1) tp++;
      else if (pred === 1 && d.label === 0) fp++;
      else if (pred === 0 && d.label === 0) tn++;
      else fn++;
    });
    var acc = (tp + tn) / DATA.length;
    var recall = tp + fn === 0 ? 0 : tp / (tp + fn);
    var precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    return { tp: tp, fp: fp, tn: tn, fn: fn, acc: acc, recall: recall, precision: precision };
  }

  function pct(x) { return (x * 100).toFixed(0) + "%"; }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.classList.add("imb-viz");

    var posRate = DATA.filter(function (d) { return d.label === 1; }).length / DATA.length;

    var intro = document.createElement("p");
    intro.className = "imb-viz-intro";
    intro.innerHTML =
      "200 cases, <strong>" + pct(posRate) + " positive</strong>. Drag the threshold: " +
      "everything at or right of the line is predicted positive. Watch accuracy stay high " +
      "while recall swings.";
    container.appendChild(intro);

    // strip of dots
    var strip = document.createElement("div");
    strip.className = "imb-viz-strip";
    container.appendChild(strip);

    var line = document.createElement("div");
    line.className = "imb-viz-line";
    strip.appendChild(line);

    DATA.forEach(function (d) {
      var dot = document.createElement("span");
      dot.className = "imb-viz-dot " + (d.label === 1 ? "pos" : "neg");
      dot.style.left = (d.score * 100) + "%";
      dot.style.top = (d.label === 1 ? 18 : 58) + "%";
      strip.appendChild(dot);
    });

    // slider
    var slider = document.createElement("input");
    slider.type = "range";
    slider.min = "1"; slider.max = "99"; slider.value = "50";
    slider.className = "imb-viz-slider";
    container.appendChild(slider);

    // readouts
    var grid = document.createElement("div");
    grid.className = "imb-viz-readout";
    container.appendChild(grid);

    function render() {
      var t = parseInt(slider.value, 10) / 100;
      line.style.left = (t * 100) + "%";
      var m = metricsAt(t);
      grid.innerHTML =
        '<div class="imb-metric"><span class="imb-k">threshold</span><span class="imb-v">' + t.toFixed(2) + '</span></div>' +
        '<div class="imb-metric"><span class="imb-k">accuracy</span><span class="imb-v">' + pct(m.acc) + '</span></div>' +
        '<div class="imb-metric hot"><span class="imb-k">recall</span><span class="imb-v">' + pct(m.recall) + '</span></div>' +
        '<div class="imb-metric"><span class="imb-k">precision</span><span class="imb-v">' + pct(m.precision) + '</span></div>' +
        '<div class="imb-cm">TP ' + m.tp + ' &nbsp; FP ' + m.fp + ' &nbsp; FN ' + m.fn + ' &nbsp; TN ' + m.tn + '</div>';
    }

    slider.addEventListener("input", render);
    render();
  }

  global.ImbalanceViz = { mount: mount };
})(window);
