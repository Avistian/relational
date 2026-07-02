/**
 * Ensemble variance visualizer (bagging).
 * Shows B high-variance step predictors (individual trees on bootstraps) and their
 * average collapsing toward a smooth target as B grows. Illustrative, deterministic.
 * Usage: EnsembleViz.mount(container, { maxTrees, caption })
 * Expected: faint jagged step curves; bold ensemble mean smooths toward the dashed target
 * as the slider increases; a variance readout drops as B rises.
 */
(function (global) {
  "use strict";

  // Small seeded PRNG (mulberry32) so the picture is stable across reloads.
  function rng(seed) {
    return function () {
      seed |= 0;
      seed = (seed + 0x6d2b79f5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function target(x) {
    // Smooth true function on [0,1].
    return 0.5 + 0.42 * Math.sin(2 * Math.PI * x);
  }

  // Build one "tree": a piecewise-constant fit to noisy samples of target().
  function makeTree(seed) {
    var r = rng(seed);
    var nBreaks = 4 + Math.floor(r() * 3); // 4-6 leaves
    var breaks = [];
    for (var i = 0; i < nBreaks - 1; i++) breaks.push(r());
    breaks.sort(function (a, b) { return a - b; });
    breaks = [0].concat(breaks, [1]);
    var levels = [];
    for (var j = 0; j < breaks.length - 1; j++) {
      var mid = (breaks[j] + breaks[j + 1]) / 2;
      var noise = (r() - 0.5) * 0.6; // high variance per tree
      levels.push(Math.max(0.02, Math.min(0.98, target(mid) + noise)));
    }
    return { breaks: breaks, levels: levels };
  }

  function predict(tree, x) {
    for (var j = 0; j < tree.breaks.length - 1; j++) {
      if (x >= tree.breaks[j] && x <= tree.breaks[j + 1]) return tree.levels[j];
    }
    return tree.levels[tree.levels.length - 1];
  }

  function mount(container, config) {
    config = config || {};
    var maxTrees = config.maxTrees || 40;

    container.innerHTML = "";
    container.className = "ensemble-viz tree-viz";

    var trees = [];
    for (var s = 0; s < maxTrees; s++) trees.push(makeTree(s * 2654435761 + 12345));

    var label = document.createElement("p");
    label.className = "tree-viz-intro";
    container.appendChild(label);

    var canvas = document.createElement("canvas");
    canvas.width = 340;
    canvas.height = 220;
    canvas.className = "tree-viz-canvas";
    container.appendChild(canvas);

    var control = document.createElement("div");
    control.style.margin = "0.5rem 0";
    control.style.fontFamily = "var(--font-sans)";
    control.style.fontSize = "0.85rem";
    var slider = document.createElement("input");
    slider.type = "range";
    slider.min = "1";
    slider.max = String(maxTrees);
    slider.value = "1";
    slider.style.width = "100%";
    control.appendChild(slider);
    container.appendChild(control);

    var readout = document.createElement("p");
    readout.className = "tree-viz-caption";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "tree-viz-caption";
    cap.textContent = config.caption ||
      "Averaging decorrelated high-variance trees shrinks variance ~1/B — bias stays put.";
    container.appendChild(cap);

    var ctx = canvas.getContext("2d");
    var pad = 26;
    var w = canvas.width - pad * 2;
    var h = canvas.height - pad * 2;
    var N = 120;

    function px(x) { return pad + x * w; }
    function py(v) { return pad + h - v * h; }

    // Stable single-tree variance: spread across the FULL pool of trees, averaged over x.
    // (Using the whole pool keeps this constant as the slider moves, so the story is
    //  "single-tree variance is fixed; the ensemble mean's variance is that / B".)
    var singleTreeVar = (function () {
      var tot = 0, cnt = 0;
      for (var k = 0; k <= N; k++) {
        var xx = k / N, mean = 0, vals = [];
        for (var t = 0; t < maxTrees; t++) { var pv = predict(trees[t], xx); mean += pv; vals.push(pv); }
        mean /= maxTrees;
        var vv = 0;
        for (var q = 0; q < vals.length; q++) vv += (vals[q] - mean) * (vals[q] - mean);
        tot += vv / maxTrees; cnt++;
      }
      return tot / cnt;
    })();

    function draw(B) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#f8f9fa";
      ctx.fillRect(pad, pad, w, h);
      ctx.strokeStyle = "#d5d8dc";
      ctx.strokeRect(pad, pad, w, h);

      // Individual trees (faint).
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(149,165,166,0.5)";
      for (var b = 0; b < B; b++) {
        ctx.beginPath();
        for (var i = 0; i <= N; i++) {
          var x = i / N;
          var v = predict(trees[b], x);
          if (i === 0) ctx.moveTo(px(x), py(v)); else ctx.lineTo(px(x), py(v));
        }
        ctx.stroke();
      }

      // Ensemble mean (bold).
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = "#1a5276";
      ctx.beginPath();
      for (var k = 0; k <= N; k++) {
        var xx = k / N, mean = 0;
        for (var t = 0; t < B; t++) mean += predict(trees[t], xx);
        mean /= B;
        if (k === 0) ctx.moveTo(px(xx), py(mean)); else ctx.lineTo(px(xx), py(mean));
      }
      ctx.stroke();

      // True function (dashed green).
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "#1e6b3c";
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      for (var m = 0; m <= N; m++) {
        var tx = m / N;
        if (m === 0) ctx.moveTo(px(tx), py(target(tx))); else ctx.lineTo(px(tx), py(target(tx)));
      }
      ctx.stroke();
      ctx.setLineDash([]);

      var ensembleVar = singleTreeVar / B;       // decorrelated-average variance ~ var/B
      label.textContent = "Trees averaged: B = " + B;
      readout.innerHTML =
        "single-tree variance ≈ <strong>" + singleTreeVar.toFixed(3) +
        "</strong> · ensemble-mean variance ≈ <strong>" + ensembleVar.toFixed(3) +
        "</strong> (= single ÷ B; blue → green as B grows)";
    }

    slider.addEventListener("input", function () { draw(Number(slider.value)); });
    draw(1);
  }

  global.EnsembleViz = { mount: mount };
})(window);
