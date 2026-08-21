/*
 * node-tree-viz.js — a differentiable Oblivious Decision Tree (NODE, Popov et al. 2019), Lesson 044.
 *
 * Shows the load-bearing NODE mechanism live: a depth-3 oblivious tree routes a row to 2^3 = 8 leaves
 * SOFTLY. Each level i has one shared (feature, threshold); the standardised gap z_i drives an entmoid
 * soft split c_i = entmoid15(z_i / tau) (the two-class 1.5-entmax). The leaf weights are the OUTER
 * PRODUCT over levels of [c_i, 1-c_i], so they always sum to 1, and the tree output is the
 * weight-weighted average of the 8 leaf responses. Drag the temperature tau: at small tau the routing
 * collapses to ONE leaf (an ordinary hard decision tree); at large tau it spreads across many leaves
 * (smooth, differentiable — which is the whole point).
 *
 * Expected states (verify in browser):
 *  - default tau ~ 1.0, z = [+1.2, -0.8, +0.4]: routing concentrated but not one-hot; entropy shown.
 *  - tau -> 0.1: leaf distribution becomes ~one-hot on the argmax leaf; "hard tree" toggle matches it.
 *  - tau -> 3: leaf distribution flattens toward uniform (all 8 near 0.125); output -> mean response.
 *  - every leaf-weight bar row sums to 1.000 (shown in the readout).
 *
 * File://-safe: pure DOM/SVG, no deps. CSS prefix `.nod-` lives in the lesson <style>.
 */
(function () {
  "use strict";

  // two-class 1.5-entmax ("entmoid") closed form (NODE repo lib/nn_utils.py). Returns P(go right).
  function entmoid15(x) {
    var ax = Math.abs(x);
    var tau = (ax + Math.sqrt(Math.max(0, 8 - ax * ax))) / 2;
    if (tau <= ax) tau = 2.0;
    var yNeg = 0.25 * Math.pow(Math.max(0, tau - ax), 2);
    return x >= 0 ? 1 - yNeg : yNeg;
  }

  var FEATS = ["age", "income", "balance"];      // the 3 shared split features (one per level)
  var RESP = [-2.0, -0.7, 0.4, 1.1, -1.3, 0.2, 1.6, 2.4];   // fixed leaf responses R (8 leaves)

  // outer product over levels: leaf bit r_i selects c_i (right) or 1-c_i (left) — pure, testable.
  function leafWeights(c) {
    var w = [];
    for (var leaf = 0; leaf < 8; leaf++) {
      var p = 1;
      for (var i = 0; i < 3; i++) {
        var bit = (leaf >> i) & 1;   // 1 => right
        p *= bit ? c[i] : (1 - c[i]);
      }
      w.push(p);
    }
    return w;
  }

  // Pure, DOM-free forward pass — the load-bearing math, shared by the widget and the headless check.
  function model(z, tau, hard) {
    var c = z.map(function (zi) {
      if (hard) return zi >= 0 ? 1 : 0;
      return entmoid15(zi / tau);
    });
    var w = leafWeights(c);
    var out = 0;
    for (var k = 0; k < 8; k++) out += w[k] * RESP[k];
    return { choices: c, weights: w, output: out };
  }

  function mount(el, opts) {
    opts = opts || {};
    var state = {
      z: [1.2, -0.8, 0.4],   // standardised gaps (feature - threshold) per level
      tau: 1.0,
      hard: false
    };

    el.classList.add("nod-viz");
    el.innerHTML =
      '<div class="nod-ctl">' +
      '  <label class="nod-glab">temperature τ <input type="range" class="nod-slider" id="nod-tau" min="0.1" max="3" step="0.05" value="1.0"><span id="nod-tauval" class="nod-mono"></span></label>' +
      '  <button id="nod-hard" type="button">hard tree (argmax)</button>' +
      '  <button id="nod-reset" type="button">reset row</button>' +
      '</div>' +
      '<div class="nod-levels" id="nod-levels"></div>' +
      '<svg class="nod-svg" id="nod-leaves" viewBox="0 0 560 150" preserveAspectRatio="xMidYMid meet"></svg>' +
      '<div class="nod-readout" id="nod-readout"></div>';

    var levelsBox = el.querySelector("#nod-levels");
    // one slider row per level
    for (var i = 0; i < 3; i++) {
      var row = document.createElement("div");
      row.className = "nod-lrow";
      row.innerHTML =
        '<span class="nod-lname">level ' + (i + 1) + ' · split on <strong>' + FEATS[i] + '</strong></span>' +
        '<input type="range" class="nod-zslider" data-i="' + i + '" min="-3" max="3" step="0.1" value="' + state.z[i] + '">' +
        '<span class="nod-lval nod-mono" id="nod-lval-' + i + '"></span>';
      levelsBox.appendChild(row);
    }

    var tau = el.querySelector("#nod-tau");
    var tauval = el.querySelector("#nod-tauval");
    var hardBtn = el.querySelector("#nod-hard");
    var resetBtn = el.querySelector("#nod-reset");
    var svg = el.querySelector("#nod-leaves");
    var readout = el.querySelector("#nod-readout");

    tau.addEventListener("input", function () { state.tau = parseFloat(tau.value); render(); });
    hardBtn.addEventListener("click", function () { state.hard = !state.hard; render(); });
    resetBtn.addEventListener("click", function () {
      state.z = [1.2, -0.8, 0.4]; state.tau = 1.0; state.hard = false; tau.value = "1.0";
      el.querySelectorAll(".nod-zslider").forEach(function (s) { s.value = state.z[+s.dataset.i]; });
      render();
    });
    el.querySelectorAll(".nod-zslider").forEach(function (s) {
      s.addEventListener("input", function () { state.z[+s.dataset.i] = parseFloat(s.value); render(); });
    });

    function choices() { return model(state.z, state.tau, state.hard).choices; }

    function render() {
      var c = choices();
      tauval.textContent = state.hard ? "(n/a)" : state.tau.toFixed(2);
      hardBtn.classList.toggle("nod-on", state.hard);
      tau.disabled = state.hard;

      for (var i = 0; i < 3; i++) {
        var lval = el.querySelector("#nod-lval-" + i);
        var side = c[i] > 0.5 ? "right" : "left";
        lval.textContent = "gap " + (state.z[i] >= 0 ? "+" : "") + state.z[i].toFixed(1) +
          " → c=" + c[i].toFixed(2) + " (" + (c[i] === 1 || c[i] === 0 ? "hard " : "") + side + ")";
      }

      var w = leafWeights(c);
      var sum = w.reduce(function (a, b) { return a + b; }, 0);
      var out = 0;
      for (var k = 0; k < 8; k++) out += w[k] * RESP[k];

      // entropy + effective #leaves
      var ent = 0;
      w.forEach(function (p) { if (p > 1e-9) ent -= p * Math.log2(p); });
      var effLeaves = Math.pow(2, ent);

      // draw 8 leaf bars
      var W = 560, pad = 30, bw = (W - 2 * pad) / 8, base = 110, maxh = 82;
      var maxw = Math.max.apply(null, w);
      var parts = [];
      for (var j = 0; j < 8; j++) {
        var h = (w[j] / Math.max(maxw, 1e-6)) * maxh;
        var x = pad + j * bw;
        var strong = w[j] === maxw;
        parts.push('<rect x="' + (x + 3) + '" y="' + (base - h) + '" width="' + (bw - 6) +
          '" height="' + h + '" rx="2" fill="' + (strong ? "#2e6fb0" : "#9db8d2") + '"></rect>');
        parts.push('<text class="nod-leaflab" x="' + (x + bw / 2) + '" y="' + (base + 12) +
          '" text-anchor="middle">' + j.toString(2).padStart(3, "0") + '</text>');
        parts.push('<text class="nod-leafw" x="' + (x + bw / 2) + '" y="' + (base - h - 4) +
          '" text-anchor="middle">' + (w[j] >= 0.005 ? w[j].toFixed(2) : "") + '</text>');
        parts.push('<text class="nod-leafr" x="' + (x + bw / 2) + '" y="' + (base + 24) +
          '" text-anchor="middle">R=' + RESP[j].toFixed(1) + '</text>');
      }
      parts.push('<text class="nod-axis" x="' + pad + '" y="16">leaf routing weights  (bits = right/left per level, sum = ' + sum.toFixed(3) + ')</text>');
      svg.innerHTML = parts.join("");

      var mode = state.hard
        ? '<strong>Hard tree.</strong> Each split is a yes/no on the sign of the gap, so exactly <strong>one</strong> leaf gets weight 1 — an ordinary decision tree. It is not differentiable: nudging a threshold either does nothing or flips a whole leaf.'
        : (effLeaves < 1.5
          ? '<strong>Nearly hard.</strong> At τ=' + state.tau.toFixed(2) + ' the routing is almost one-hot (≈' + effLeaves.toFixed(1) + ' effective leaves): the soft tree is imitating a hard one, but a gradient still flows.'
          : '<strong>Soft routing.</strong> The row is split across ≈' + effLeaves.toFixed(1) + ' of the 8 leaves. Every leaf response contributes a fraction, so the output is a smooth function of the thresholds and features — that is what lets NODE train by gradient descent.');

      readout.innerHTML =
        '<p>' + mode + '</p>' +
        '<p class="nod-mono">tree output  ŷ = Σ wₖ·Rₖ = <strong>' + out.toFixed(3) + '</strong>' +
        '  ·  leaf entropy ' + ent.toFixed(2) + ' bits  ·  effective leaves ' + effLeaves.toFixed(1) + '</p>' +
        '<p class="nod-note">Drag τ toward 0.1 to watch the 8 soft leaves collapse onto the single argmax leaf (compare the "hard tree" toggle); push it toward 3 to watch them spread toward uniform. The feature <em>chosen</em> at each level is itself learned by a sparse entmax over all columns — here fixed for legibility.</p>';
    }

    render();

    // headless-testable handle (mirrors the L043 viz API)
    return {
      state: function () {
        var c = choices();
        var w = leafWeights(c);
        var out = 0;
        for (var k = 0; k < 8; k++) out += w[k] * RESP[k];
        return { z: state.z.slice(), tau: state.tau, hard: state.hard, choices: c, weights: w, output: out };
      },
      getTau: function () { return state.tau; },
      getHard: function () { return state.hard; },
      setTau: function (v) { state.tau = v; if (tau) tau.value = String(v); render(); },
      setHard: function (v) { state.hard = !!v; render(); },
      setZ: function (i, v) { state.z[i] = v; render(); }
    };
  }

  window.NodeTreeViz = {
    mount: mount, model: model, entmoid15: entmoid15, leafWeights: leafWeights, responses: RESP
  };
})();
