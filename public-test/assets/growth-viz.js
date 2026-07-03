/**
 * Leaf-wise (best-first) vs level-wise tree growth (LightGBM; Ke et al. 2017).
 *
 * Two ways to spend a fixed budget of splits on the SAME candidate tree:
 *   - level-wise (XGBoost default): split breadth-first, one whole depth at a
 *     time. Produces balanced trees.
 *   - leaf-wise (LightGBM default): always split the frontier leaf with the
 *     largest loss reduction (gain). Produces deep, unbalanced trees and, for a
 *     fixed number of leaves, reaches lower training loss.
 *
 * The widget draws ONE tree at a time (toggle the mode) and a splits slider.
 * A readout always reports the remaining training loss for BOTH strategies at
 * the current split count, so the comparison is explicit: leaf-wise ≤ level-wise.
 *
 * Usage: GrowthViz.mount(container, { caption })
 * Expected states:
 *   - 0 splits: both strategies show just the root; equal loss.
 *   - a few splits, leaf-wise: tree dives down the high-gain path (unbalanced),
 *     and the leaf-wise remaining loss is strictly lower than level-wise.
 *   - level-wise: fills depth 1 fully before depth 2; balanced tree.
 *   - max splits: max_depth note explains why LightGBM caps leaf-wise depth.
 */
(function (global) {
  "use strict";

  // A fixed complete binary tree of candidate nodes (depth 4 => 31 nodes).
  // Each node carries the loss reduction (gain) unlocked by splitting it.
  // Gains are hand-set so one deep path is very rewarding — that is where
  // leaf-wise dives while level-wise wastes splits on low-gain breadth.
  var DEPTH = 4;
  var GAIN = {
    0: 10.0,
    1: 8.5, 2: 2.0,
    3: 7.0, 4: 1.2, 5: 1.0, 6: 0.8,
    7: 6.0, 8: 0.6, 9: 0.5, 10: 0.4, 11: 0.4, 12: 0.3, 13: 0.3, 14: 0.2,
    15: 5.2, 16: 0.3, 17: 0.2, 18: 0.2, 19: 0.2, 20: 0.15, 21: 0.15, 22: 0.15,
    23: 0.15, 24: 0.1, 25: 0.1, 26: 0.1, 27: 0.1, 28: 0.1, 29: 0.1, 30: 0.1,
  };

  function left(i) { return 2 * i + 1; }
  function right(i) { return 2 * i + 2; }
  function depthOf(i) { return Math.floor(Math.log2(i + 1)); }
  function exists(i) { return i in GAIN; }

  var INIT_LOSS = 40.0;

  // Return the set of split nodes chosen after `n` splits under a strategy.
  function grow(n, mode) {
    var splits = [];
    if (mode === "level") {
      // breadth-first: node indices 0,1,2,3,... are already in BFS order.
      for (var i = 0; i < GAIN_KEYS.length && splits.length < n; i++) {
        var idx = GAIN_KEYS[i];
        if (depthOf(idx) < DEPTH) splits.push(idx); // only internal nodes can split
      }
    } else {
      // best-first: maintain a frontier of splittable leaves; repeatedly take
      // the one with max gain, then add its children to the frontier.
      var frontier = [0];
      for (var s = 0; s < n; s++) {
        if (!frontier.length) break;
        var best = -1, bestGain = -Infinity;
        for (var f = 0; f < frontier.length; f++) {
          var node = frontier[f];
          if (GAIN[node] > bestGain) { bestGain = GAIN[node]; best = f; }
        }
        var chosen = frontier.splice(best, 1)[0];
        splits.push(chosen);
        var l = left(chosen), r = right(chosen);
        if (exists(l) && depthOf(l) < DEPTH) frontier.push(l);
        if (exists(r) && depthOf(r) < DEPTH) frontier.push(r);
      }
    }
    return splits;
  }

  var GAIN_KEYS = Object.keys(GAIN).map(Number).sort(function (a, b) { return a - b; });

  function lossAfter(splits) {
    var reduced = 0;
    for (var i = 0; i < splits.length; i++) reduced += GAIN[splits[i]];
    return Math.max(0, INIT_LOSS - reduced);
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "growth-viz tree-viz";

    var intro = document.createElement("p");
    intro.className = "tree-viz-intro";
    container.appendChild(intro);

    var canvas = document.createElement("canvas");
    canvas.width = 360;
    canvas.height = 220;
    canvas.className = "tree-viz-canvas";
    container.appendChild(canvas);
    var ctx = canvas.getContext("2d");

    // mode toggle
    var toggleWrap = document.createElement("div");
    toggleWrap.style.cssText = "margin:0.5rem 0;display:flex;gap:0.5rem;";
    var mode = "leaf";
    var btnLeaf = mkBtn("Leaf-wise (LightGBM)");
    var btnLevel = mkBtn("Level-wise (XGBoost)");
    toggleWrap.appendChild(btnLeaf);
    toggleWrap.appendChild(btnLevel);
    container.appendChild(toggleWrap);

    function mkBtn(txt) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = txt;
      b.style.cssText =
        "flex:1;font-size:0.8rem;padding:0.35rem 0.4rem;border:1px solid var(--border);" +
        "border-radius:5px;background:var(--bg-soft);cursor:pointer;font-family:var(--font-sans);";
      return b;
    }

    // splits slider
    var sWrap = document.createElement("div");
    sWrap.style.margin = "0.4rem 0";
    var sLab = document.createElement("label");
    sLab.style.cssText = "font-size:0.82rem;font-family:var(--font-mono);display:block;";
    var sInp = document.createElement("input");
    sInp.type = "range";
    sInp.min = "0"; sInp.max = "6"; sInp.step = "1"; sInp.value = "3";
    sInp.style.width = "100%";
    sWrap.appendChild(sLab);
    sWrap.appendChild(sInp);
    container.appendChild(sWrap);

    var readout = document.createElement("p");
    readout.className = "tree-viz-caption";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "tree-viz-caption";
    cap.textContent = config.caption ||
      "Same split budget, two strategies. Leaf-wise dives down the high-gain path; " +
      "for the same number of leaves it reaches lower training loss — but can overfit, " +
      "so LightGBM caps it with max_depth.";
    container.appendChild(cap);

    function nodeXY(idx) {
      var d = depthOf(idx);
      var firstInRow = Math.pow(2, d) - 1;
      var posInRow = idx - firstInRow;
      var rowCount = Math.pow(2, d);
      var x = (canvas.width) * (posInRow + 0.5) / rowCount;
      var y = 26 + d * 52;
      return { x: x, y: y };
    }

    function draw() {
      var n = Number(sInp.value);
      var splits = grow(n, mode);
      var splitSet = {};
      splits.forEach(function (s) { splitSet[s] = true; });

      // which nodes are visible: root + any child whose parent was split
      var visible = { 0: true };
      splits.forEach(function (s) {
        visible[left(s)] = exists(left(s));
        visible[right(s)] = exists(right(s));
      });

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // edges first
      Object.keys(visible).forEach(function (k) {
        var idx = Number(k);
        if (idx === 0) return;
        var parent = Math.floor((idx - 1) / 2);
        if (!visible[parent]) return;
        var a = nodeXY(parent), b = nodeXY(idx);
        ctx.strokeStyle = "#b8c2cc";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      });

      // nodes
      Object.keys(visible).forEach(function (k) {
        var idx = Number(k);
        var p = nodeXY(idx);
        var isSplit = splitSet[idx];
        ctx.beginPath();
        ctx.arc(p.x, p.y, 9, 0, 2 * Math.PI);
        // split (internal) nodes filled; current leaves outlined
        ctx.fillStyle = isSplit ? (mode === "leaf" ? "#1e6b3c" : "#2980b9") : "#ffffff";
        ctx.strokeStyle = isSplit ? (mode === "leaf" ? "#1e6b3c" : "#2980b9") : "#7f8c8d";
        ctx.lineWidth = 2;
        ctx.fill();
        ctx.stroke();
      });

      var lvlLoss = lossAfter(grow(n, "level"));
      var leafLoss = lossAfter(grow(n, "leaf"));

      btnLeaf.style.outline = mode === "leaf" ? "2px solid #1e6b3c" : "none";
      btnLevel.style.outline = mode === "level" ? "2px solid #2980b9" : "none";
      sLab.textContent = "splits = " + n + "  (leaves = " + (n + 1) + ")";
      intro.innerHTML = "Showing <strong>" +
        (mode === "leaf" ? "leaf-wise" : "level-wise") + "</strong> growth after <strong>" +
        n + "</strong> split" + (n === 1 ? "" : "s") + ".";
      readout.innerHTML =
        "remaining training loss &mdash; level-wise <strong>" + lvlLoss.toFixed(1) +
        "</strong> · leaf-wise <strong>" + leafLoss.toFixed(1) +
        "</strong>" + (leafLoss < lvlLoss ? " &larr; lower" : "");
    }

    btnLeaf.addEventListener("click", function () { mode = "leaf"; draw(); });
    btnLevel.addEventListener("click", function () { mode = "level"; draw(); });
    sInp.addEventListener("input", draw);
    draw();
  }

  global.GrowthViz = { mount: mount };
})(window);
