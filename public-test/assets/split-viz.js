/**
 * Stratified vs random K-fold class-balance visualizer.
 * Usage: SplitViz.mount(containerElement, config?)
 * config: { nSamples, positiveRate, nSplits, seed }
 */
(function (global) {
  "use strict";

  function mulberry32(seed) {
    return function () {
      seed |= 0;
      seed = (seed + 0x6d2b79f5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function makeLabels(n, positiveRate) {
    var nPos = Math.round(n * positiveRate);
    var labels = [];
    var i;
    for (i = 0; i < nPos; i++) labels.push(1);
    for (i = nPos; i < n; i++) labels.push(0);
    return labels;
  }

  function shuffle(arr, rng) {
    var a = arr.slice();
    var i = a.length - 1;
    var j;
    var tmp;
    while (i > 0) {
      j = Math.floor(rng() * (i + 1));
      tmp = a[i];
      a[i] = a[j];
      a[j] = tmp;
      i -= 1;
    }
    return a;
  }

  function kFoldIndices(n, k) {
    var indices = [];
    var i;
    for (i = 0; i < n; i++) indices.push(i);
    var foldSizes = [];
    var base = Math.floor(n / k);
    var rem = n % k;
    for (i = 0; i < k; i++) foldSizes.push(base + (i < rem ? 1 : 0));
    var folds = [];
    var start = 0;
    for (i = 0; i < k; i++) {
      folds.push(indices.slice(start, start + foldSizes[i]));
      start += foldSizes[i];
    }
    return folds;
  }

  function stratifiedKFold(labels, k, rng) {
    var pos = [];
    var neg = [];
    var i;
    for (i = 0; i < labels.length; i++) {
      if (labels[i] === 1) pos.push(i);
      else neg.push(i);
    }
    pos = shuffle(pos, rng);
    neg = shuffle(neg, rng);
    var folds = [];
    for (i = 0; i < k; i++) folds.push([]);
    for (i = 0; i < pos.length; i++) folds[i % k].push(pos[i]);
    for (i = 0; i < neg.length; i++) folds[i % k].push(neg[i]);
    return folds;
  }

  function foldStats(folds, labels) {
    return folds.map(function (idxs) {
      var pos = 0;
      var j;
      for (j = 0; j < idxs.length; j++) {
        if (labels[idxs[j]] === 1) pos += 1;
      }
      var n = idxs.length;
      return { n: n, pos: pos, neg: n - pos, rate: n ? pos / n : 0 };
    });
  }

  function mount(container, config) {
    config = config || {};
    var nSamples = config.nSamples || 100;
    var positiveRate = config.positiveRate != null ? config.positiveRate : 0.1;
    var nSplits = config.nSplits || 5;
    var seed = config.seed != null ? config.seed : 42;

    var labels = makeLabels(nSamples, positiveRate);
    var rngRandom = mulberry32(seed);
    var rngStrat = mulberry32(seed);

    container.innerHTML = "";
    container.classList.add("split-viz");

    var intro = document.createElement("p");
    intro.className = "split-viz-intro";
    intro.textContent =
      "Toy dataset: " +
      nSamples +
      " rows, " +
      Math.round(positiveRate * 100) +
      "% positive class (imbalanced). Compare how each fold's validation slice preserves the label ratio.";
    container.appendChild(intro);

    var controls = document.createElement("div");
    controls.className = "split-viz-controls";
    container.appendChild(controls);

    var btnRandom = document.createElement("button");
    btnRandom.type = "button";
    btnRandom.className = "split-viz-btn";
    btnRandom.textContent = "Random K-fold";
    controls.appendChild(btnRandom);

    var btnStrat = document.createElement("button");
    btnStrat.type = "button";
    btnStrat.className = "split-viz-btn active";
    btnStrat.textContent = "Stratified K-fold";
    controls.appendChild(btnStrat);

    var chart = document.createElement("div");
    chart.className = "split-viz-chart";
    container.appendChild(chart);

    var caption = document.createElement("div");
    caption.className = "split-viz-caption";
    container.appendChild(caption);

    function render(mode) {
      btnRandom.classList.toggle("active", mode === "random");
      btnStrat.classList.toggle("active", mode === "stratified");

      var shuffled = shuffle(
        labels.map(function (_, i) {
          return i;
        }),
        mulberry32(seed)
      );
      var folds;
      if (mode === "random") {
        folds = kFoldIndices(nSamples, nSplits);
        folds = folds.map(function (fold) {
          return fold.map(function (i) {
            return shuffled[i];
          });
        });
      } else {
        folds = stratifiedKFold(labels, nSplits, rngStrat);
      }

      var stats = foldStats(folds, labels);
      chart.innerHTML = "";

      stats.forEach(function (s, fi) {
        var row = document.createElement("div");
        row.className = "split-viz-row";

        var label = document.createElement("span");
        label.className = "split-viz-fold-label";
        label.textContent = "Fold " + (fi + 1);
        row.appendChild(label);

        var bar = document.createElement("div");
        bar.className = "split-viz-bar";
        var negSeg = document.createElement("div");
        negSeg.className = "split-viz-seg neg";
        negSeg.style.flex = s.neg;
        var posSeg = document.createElement("div");
        posSeg.className = "split-viz-seg pos";
        posSeg.style.flex = s.pos;
        bar.appendChild(negSeg);
        bar.appendChild(posSeg);
        row.appendChild(bar);

        var pct = document.createElement("span");
        pct.className = "split-viz-pct";
        pct.textContent = (s.rate * 100).toFixed(1) + "% pos (n=" + s.n + ")";
        row.appendChild(pct);

        chart.appendChild(row);
      });

      var rates = stats.map(function (s) {
        return s.rate;
      });
      var minR = Math.min.apply(null, rates);
      var maxR = Math.max.apply(null, rates);
      var spread = ((maxR - minR) * 100).toFixed(1);

      if (mode === "stratified") {
        caption.textContent =
          "Stratified: each fold's positive rate stays near " +
          Math.round(positiveRate * 100) +
          "%. Spread across folds: " +
          spread +
          " pp — stable metrics for rare labels.";
      } else {
        caption.textContent =
          "Random: some folds can be nearly all negative (e.g. Fold with 0% pos). Spread: " +
          spread +
          " pp — misleading PR-AUC on small folds.";
      }
    }

    btnRandom.addEventListener("click", function () {
      render("random");
    });
    btnStrat.addEventListener("click", function () {
      render("stratified");
    });

    render("stratified");
  }

  global.SplitViz = { mount: mount };
})(window);
