/**
 * Boosting residual visualizer (gradient boosting, stagewise additive).
 * Shows a 1-D regression where the model starts as a flat mean and, each round,
 * fits a depth-1 stump to the CURRENT residuals and adds a shrunken step.
 * Red bars are residuals (prediction -> data); bold blue is the running fit;
 * dashed green is the true function.
 * Usage: ResidualViz.mount(container, { rounds, lr, caption })
 * Expected states:
 *   - M = 0: prediction is a flat line at the mean; residual bars are large; MSE high.
 *   - As M grows: the blue step curve bends toward the green truth; residual bars
 *     shrink; the MSE-vs-truth readout falls toward ~0 (bias reduction, round by round).
 */
(function (global) {
  "use strict";

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
    return 0.5 + 0.4 * Math.sin(2 * Math.PI * x);
  }

  // Fit a depth-1 regression stump to (x, r): pick the threshold minimizing SSE.
  function fitStump(xs, r) {
    var n = xs.length;
    var best = { thr: xs[0], left: 0, right: 0, sse: Infinity };
    // Candidate thresholds = midpoints between consecutive sorted x values.
    for (var s = 1; s < n; s++) {
      var thr = (xs[s - 1] + xs[s]) / 2;
      var ls = 0, lc = 0, rs = 0, rc = 0;
      for (var i = 0; i < n; i++) {
        if (xs[i] <= thr) { ls += r[i]; lc++; } else { rs += r[i]; rc++; }
      }
      if (lc === 0 || rc === 0) continue;
      var lm = ls / lc, rm = rs / rc, sse = 0;
      for (var j = 0; j < n; j++) {
        var m = xs[j] <= thr ? lm : rm;
        sse += (r[j] - m) * (r[j] - m);
      }
      if (sse < best.sse) best = { thr: thr, left: lm, right: rm, sse: sse };
    }
    return best;
  }

  function mount(container, config) {
    config = config || {};
    var maxRounds = config.rounds || 40;
    var lr = config.lr || 0.3;

    container.innerHTML = "";
    container.className = "residual-viz tree-viz";

    // Build a fixed noisy dataset.
    var r = rng(7);
    var n = 40;
    var xs = [];
    for (var i = 0; i < n; i++) xs.push(r());
    xs.sort(function (a, b) { return a - b; });
    var ys = [];
    for (var k = 0; k < n; k++) ys.push(target(xs[k]) + (r() - 0.5) * 0.28);

    // Precompute the ensemble prediction at every round on the data points AND
    // on a dense grid (for drawing the step function).
    var N = 160;
    var grid = [];
    for (var g = 0; g <= N; g++) grid.push(g / N);

    var predData = new Array(n);
    var mean0 = ys.reduce(function (a, b) { return a + b; }, 0) / n;
    for (var d = 0; d < n; d++) predData[d] = mean0;
    var predGrid = new Array(grid.length);
    for (var e = 0; e < grid.length; e++) predGrid[e] = mean0;

    var stages = [{ dataPred: predData.slice(), gridPred: predGrid.slice() }];
    for (var m = 0; m < maxRounds; m++) {
      var resid = [];
      for (var a = 0; a < n; a++) resid.push(ys[a] - predData[a]);
      var stump = fitStump(xs, resid);
      for (var b = 0; b < n; b++) {
        predData[b] += lr * (xs[b] <= stump.thr ? stump.left : stump.right);
      }
      for (var c = 0; c < grid.length; c++) {
        predGrid[c] += lr * (grid[c] <= stump.thr ? stump.left : stump.right);
      }
      stages.push({ dataPred: predData.slice(), gridPred: predGrid.slice() });
    }

    // MSE vs the TRUE function on the grid, per stage.
    function mseVsTruth(gridPred) {
      var tot = 0;
      for (var q = 0; q < grid.length; q++) {
        var diff = gridPred[q] - target(grid[q]);
        tot += diff * diff;
      }
      return tot / grid.length;
    }

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
    var slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = String(maxRounds);
    slider.value = "0";
    slider.style.width = "100%";
    control.appendChild(slider);
    container.appendChild(control);

    var readout = document.createElement("p");
    readout.className = "tree-viz-caption";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "tree-viz-caption";
    cap.textContent = config.caption ||
      "Each round fits a stump to the residuals and adds a shrunken step — bias falls as rounds grow.";
    container.appendChild(cap);

    var ctx = canvas.getContext("2d");
    var pad = 26;
    var w = canvas.width - pad * 2;
    var h = canvas.height - pad * 2;

    // y-range for plotting.
    var lo = -0.15, hi = 1.15;
    function px(x) { return pad + x * w; }
    function py(v) { return pad + h - ((v - lo) / (hi - lo)) * h; }

    function draw(M) {
      var st = stages[M];
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#f8f9fa";
      ctx.fillRect(pad, pad, w, h);
      ctx.strokeStyle = "#d5d8dc";
      ctx.strokeRect(pad, pad, w, h);

      // Residual bars (prediction -> data point).
      ctx.strokeStyle = "rgba(192,57,43,0.55)";
      ctx.lineWidth = 1;
      for (var i = 0; i < n; i++) {
        ctx.beginPath();
        ctx.moveTo(px(xs[i]), py(st.dataPred[i]));
        ctx.lineTo(px(xs[i]), py(ys[i]));
        ctx.stroke();
      }

      // Data points.
      ctx.fillStyle = "rgba(52,73,94,0.6)";
      for (var d = 0; d < n; d++) {
        ctx.beginPath();
        ctx.arc(px(xs[d]), py(ys[d]), 2.2, 0, 2 * Math.PI);
        ctx.fill();
      }

      // True function (dashed green).
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "#1e6b3c";
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      for (var t = 0; t <= N; t++) {
        var tx = t / N;
        if (t === 0) ctx.moveTo(px(tx), py(target(tx))); else ctx.lineTo(px(tx), py(target(tx)));
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Current ensemble prediction (bold blue step).
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = "#1a5276";
      ctx.beginPath();
      for (var k = 0; k < grid.length; k++) {
        if (k === 0) ctx.moveTo(px(grid[k]), py(st.gridPred[k]));
        else ctx.lineTo(px(grid[k]), py(st.gridPred[k]));
      }
      ctx.stroke();

      var mse = mseVsTruth(st.gridPred);
      label.textContent = "Boosting rounds: M = " + M + "  (learning rate " + lr + ")";
      readout.innerHTML =
        "MSE vs truth ≈ <strong>" + mse.toFixed(3) +
        "</strong> · red bars = residuals the next stump will fit · blue → green as M grows";
    }

    slider.addEventListener("input", function () { draw(Number(slider.value)); });
    draw(0);
  }

  global.ResidualViz = { mount: mount };
})(window);
