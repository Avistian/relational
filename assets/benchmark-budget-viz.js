/**
 * The Grinsztajn benchmark-aggregation curve (Grinsztajn et al. 2022, §4).
 *
 * The methodological heart of the paper: do NOT report one tuned number per model.
 * Instead, for each model run a random search over hyperparameters and plot the
 * expected TEST score of the best-VALIDATION config found so far, as a function of
 * the number of random-search iterations (the "budget"). Averaging over random
 * orderings of the draws turns a noisy search into a smooth curve you can compare.
 *
 * The two curves below are a real single-dataset reproduction on OpenML credit-g
 * (labs/_verify_l024.py): a GBT (HistGradientBoosting) and a neural net (MLP).
 * The GBT curve sits ABOVE the neural net at every budget, and is already strong at
 * budget 1 (a good default) — the qualitative shape of Grinsztajn's Figure 1.
 *
 * Two views (toggle):
 *   - "raw"        : raw test accuracy on this dataset.
 *   - "normalized" : affine min-max per dataset (worst model → 0, best → 1). This is
 *                    the transform that makes datasets commensurable so the 45-dataset
 *                    suite can be averaged (echo of Demšar, L023: don't average raw acc).
 * A budget slider marks iteration k and reads out both scores + the gap.
 *
 * Usage: BenchmarkBudgetViz.mount(container, { caption })
 * Expected states:
 *   - budget k=1 : GBT ~0.791 vs MLP ~0.729 raw (normalized 0.96 vs 0.00) — GBT wins by default.
 *   - budget k=30: GBT ~0.785 vs MLP ~0.770 raw — gap shrinks but GBT still on top.
 *   - GBT line is above MLP at every k in both views.
 *   - Toggling normalized rescales the y-axis to [0,1] without changing the ordering.
 */
(function (global) {
  "use strict";

  // Real curves from labs/_verify_l024.py (credit-g, 40 shuffles, 30 configs each).
  var GBT = [0.7906,0.7924,0.7925,0.7915,0.791,0.791,0.7921,0.7927,0.7929,0.7925,
             0.7917,0.7914,0.7915,0.7914,0.7906,0.7899,0.7884,0.7884,0.7882,0.7871,
             0.7866,0.7865,0.7865,0.7865,0.7862,0.7861,0.7859,0.7855,0.7851,0.785];
  var MLP = [0.7286,0.7539,0.7683,0.7705,0.7699,0.7703,0.7685,0.7675,0.7689,0.7676,
             0.768,0.7676,0.768,0.7689,0.7688,0.7678,0.7683,0.7683,0.7683,0.7683,
             0.77,0.77,0.77,0.77,0.77,0.77,0.77,0.77,0.77,0.77];

  var LO = Math.min.apply(null, GBT.concat(MLP));
  var HI = Math.max.apply(null, GBT.concat(MLP));
  function normz(v) { return (v - LO) / (HI - LO); }

  var svgNS = "http://www.w3.org/2000/svg";
  function el(name, attrs) {
    var e = document.createElementNS(svgNS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "benchmark-budget-viz";

    var mode = "raw";      // or "normalized"
    var budget = 1;        // 1..30
    var N = GBT.length;

    // controls
    var ctl = document.createElement("div");
    ctl.className = "bb-ctl";
    var rawBtn = document.createElement("button");
    rawBtn.type = "button"; rawBtn.textContent = "Raw accuracy";
    var normBtn = document.createElement("button");
    normBtn.type = "button"; normBtn.textContent = "Normalized (per dataset)";
    var slabel = document.createElement("label");
    slabel.className = "bb-slabel";
    slabel.textContent = "budget k = ";
    var slider = document.createElement("input");
    slider.type = "range"; slider.min = "1"; slider.max = String(N); slider.value = "1";
    slider.className = "bb-slider";
    var kval = document.createElement("span");
    kval.className = "bb-kval"; kval.textContent = "1";
    slabel.appendChild(slider); slabel.appendChild(kval);
    ctl.appendChild(rawBtn); ctl.appendChild(normBtn); ctl.appendChild(slabel);
    container.appendChild(ctl);

    var W = 460, H = 300;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", class: "bb-svg" });
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "bb-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "bb-caption";
    cap.textContent = config.caption ||
      "Expected test score of the best-validation config found after k random-search " +
      "iterations, on OpenML credit-g (avg of 40 orderings). The GBT sits above the " +
      "neural net at every budget and is already strong by default (k=1).";
    container.appendChild(cap);

    // plot geometry (log-x)
    var PL = 52, PR = W - 14, PT = 22, PB = H - 42;
    function sx(k) { // k in 1..N, log scale
      var t = Math.log(k) / Math.log(N);
      return PL + t * (PR - PL);
    }
    function yrange() {
      if (mode === "normalized") return [0, 1];
      return [0.70, 0.80];
    }
    function sy(v) {
      var r = yrange();
      return PB - (v - r[0]) / (r[1] - r[0]) * (PB - PT);
    }
    function val(arr, i) { return mode === "normalized" ? normz(arr[i]) : arr[i]; }

    function line(arr, color) {
      var d = "M";
      for (var i = 0; i < N; i++) {
        d += (i ? "L" : "") + sx(i + 1).toFixed(1) + "," + sy(val(arr, i)).toFixed(1) + " ";
      }
      return el("path", { d: d, fill: "none", stroke: color, "stroke-width": 2.2 });
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      // axes box
      svg.appendChild(el("rect", { x: PL, y: PT, width: PR - PL, height: PB - PT,
        fill: "var(--bg)", stroke: "var(--border)", "stroke-width": 1 }));

      // y gridlines / ticks
      var r = yrange();
      var steps = mode === "normalized" ? [0, 0.25, 0.5, 0.75, 1] : [0.70, 0.72, 0.74, 0.76, 0.78, 0.80];
      steps.forEach(function (v) {
        var yy = sy(v);
        svg.appendChild(el("line", { x1: PL, y1: yy, x2: PR, y2: yy,
          stroke: "var(--border)", "stroke-width": 0.5, opacity: 0.6 }));
        svg.appendChild(el("text", { x: PL - 6, y: yy + 3, "text-anchor": "end", class: "bb-tick" }))
          .textContent = mode === "normalized" ? v.toFixed(2) : v.toFixed(2);
      });

      // x ticks (log): 1,2,5,10,20,30
      [1, 2, 5, 10, 20, 30].forEach(function (k) {
        if (k > N) return;
        var xx = sx(k);
        svg.appendChild(el("line", { x1: xx, y1: PB, x2: xx, y2: PB + 4, stroke: "var(--muted)", "stroke-width": 1 }));
        svg.appendChild(el("text", { x: xx, y: PB + 15, "text-anchor": "middle", class: "bb-tick" }))
          .textContent = String(k);
      });
      svg.appendChild(el("text", { x: (PL + PR) / 2, y: H - 4, "text-anchor": "middle", class: "bb-lab" }))
        .textContent = "number of random-search iterations (budget, log scale)";
      var yl = el("text", { x: 13, y: (PT + PB) / 2, "text-anchor": "middle", class: "bb-lab",
        transform: "rotate(-90 13 " + ((PT + PB) / 2) + ")" });
      yl.textContent = mode === "normalized" ? "normalized test score" : "test accuracy";
      svg.appendChild(yl);

      // budget marker
      var mx = sx(budget);
      svg.appendChild(el("line", { x1: mx, y1: PT, x2: mx, y2: PB,
        stroke: "var(--muted)", "stroke-width": 1, "stroke-dasharray": "3,3", opacity: 0.8 }));

      // curves
      svg.appendChild(line(MLP, "#2e6fb0"));
      svg.appendChild(line(GBT, "#1e6b3c"));

      // dots at current budget
      var i = budget - 1;
      svg.appendChild(el("circle", { cx: mx, cy: sy(val(GBT, i)), r: 4.5, fill: "#1e6b3c", stroke: "var(--bg)", "stroke-width": 1.5 }));
      svg.appendChild(el("circle", { cx: mx, cy: sy(val(MLP, i)), r: 4.5, fill: "#2e6fb0", stroke: "var(--bg)", "stroke-width": 1.5 }));

      // inline legend
      svg.appendChild(el("rect", { x: PR - 96, y: PT + 6, width: 10, height: 10, fill: "#1e6b3c" }));
      svg.appendChild(el("text", { x: PR - 82, y: PT + 15, class: "bb-leg" })).textContent = "GBT";
      svg.appendChild(el("rect", { x: PR - 96, y: PT + 22, width: 10, height: 10, fill: "#2e6fb0" }));
      svg.appendChild(el("text", { x: PR - 82, y: PT + 31, class: "bb-leg" })).textContent = "Neural net";

      var g = val(GBT, i), m = val(MLP, i), gap = g - m;
      var fmt = mode === "normalized" ? function (x) { return x.toFixed(3); } : function (x) { return x.toFixed(4); };
      readout.innerHTML =
        "<span class='bb-pill'>" + (mode === "normalized" ? "normalized" : "raw") + "</span> " +
        "at budget <strong>k=" + budget + "</strong>: " +
        "GBT <strong class='bb-win'>" + fmt(g) + "</strong> vs " +
        "Neural net <strong>" + fmt(m) + "</strong> · gap <strong class='bb-win'>+" + fmt(gap) + "</strong>" +
        (budget === 1
          ? " — the default (untuned) GBT already leads."
          : (budget === N ? " — fully tuned, and the GBT is still on top." : ""));
    }

    function setMode(m) {
      mode = m;
      rawBtn.classList.toggle("bb-active", m === "raw");
      normBtn.classList.toggle("bb-active", m === "normalized");
      draw();
    }
    rawBtn.addEventListener("click", function () { setMode("raw"); });
    normBtn.addEventListener("click", function () { setMode("normalized"); });
    slider.addEventListener("input", function () {
      budget = Number(slider.value); kval.textContent = slider.value; draw();
    });

    setMode("raw");

    return { setBudget: function (k) { budget = k; slider.value = String(k); kval.textContent = String(k); draw(); },
             setMode: setMode };
  }

  global.BenchmarkBudgetViz = { mount: mount };
})(window);
