/**
 * CASH — Combined Algorithm Selection and Hyperparameter optimization (Auto-sklearn, Feurer 2015).
 *
 * The single idea L017's grid/random-search viz did NOT show: AutoML searches over WHICH ALGORITHM
 * *and* that algorithm's hyperparameters jointly — one big combined space — and keeps the config with
 * the best VALIDATION score. This scatter is the REAL trace of a 40-iteration random search over a
 * 4-algorithm CASH space on credit_g (labs/_dump_l029_trace.py, seed 0):
 *   - each dot is one evaluated pipeline, x = iteration, y = validation ROC-AUC, colour = algorithm;
 *   - the black step line is the best-validation-AUC-so-far (echoes the L024 budget curve);
 *   - the star marks the winning config (a HistGB, val 0.817, found at iteration 15).
 *
 * Mechanism message: the search visits every algorithm lane, and "best-so-far" climbs with budget.
 * That joint search (plus meta-learning warm-start and ensembling) is what AutoML automates.
 *
 * Usage: CashSearchViz.mount(container, { caption })
 * Handle: { setAlgo(name|null) } filters the scatter to one algorithm (null = all).
 */
(function (global) {
  "use strict";

  // REAL per-iteration trace (labs/_dump_l029_trace.py, credit_g, seed 0).
  var TRACE = [
    {i:1,algo:"logreg",val:0.796,best:0.796},{i:2,algo:"histgb",val:0.790,best:0.796},
    {i:3,algo:"logreg",val:0.800,best:0.800},{i:4,algo:"logreg",val:0.796,best:0.800},
    {i:5,algo:"extratrees",val:0.763,best:0.800},{i:6,algo:"rf",val:0.759,best:0.800},
    {i:7,algo:"histgb",val:0.790,best:0.800},{i:8,algo:"rf",val:0.780,best:0.800},
    {i:9,algo:"logreg",val:0.796,best:0.800},{i:10,algo:"histgb",val:0.782,best:0.800},
    {i:11,algo:"histgb",val:0.802,best:0.802},{i:12,algo:"logreg",val:0.790,best:0.802},
    {i:13,algo:"logreg",val:0.787,best:0.802},{i:14,algo:"logreg",val:0.780,best:0.802},
    {i:15,algo:"histgb",val:0.817,best:0.817},{i:16,algo:"histgb",val:0.811,best:0.817},
    {i:17,algo:"extratrees",val:0.737,best:0.817},{i:18,algo:"histgb",val:0.781,best:0.817},
    {i:19,algo:"histgb",val:0.795,best:0.817},{i:20,algo:"histgb",val:0.817,best:0.817},
    {i:21,algo:"rf",val:0.771,best:0.817},{i:22,algo:"rf",val:0.757,best:0.817},
    {i:23,algo:"extratrees",val:0.733,best:0.817},{i:24,algo:"rf",val:0.781,best:0.817},
    {i:25,algo:"logreg",val:0.807,best:0.817},{i:26,algo:"extratrees",val:0.775,best:0.817},
    {i:27,algo:"extratrees",val:0.737,best:0.817},{i:28,algo:"extratrees",val:0.733,best:0.817},
    {i:29,algo:"extratrees",val:0.795,best:0.817},{i:30,algo:"histgb",val:0.795,best:0.817},
    {i:31,algo:"extratrees",val:0.706,best:0.817},{i:32,algo:"logreg",val:0.778,best:0.817},
    {i:33,algo:"extratrees",val:0.699,best:0.817},{i:34,algo:"logreg",val:0.799,best:0.817},
    {i:35,algo:"extratrees",val:0.780,best:0.817},{i:36,algo:"rf",val:0.753,best:0.817},
    {i:37,algo:"histgb",val:0.815,best:0.817},{i:38,algo:"logreg",val:0.796,best:0.817},
    {i:39,algo:"extratrees",val:0.710,best:0.817},{i:40,algo:"rf",val:0.772,best:0.817}
  ];
  var COLORS = { logreg: "#2e6fb0", rf: "#1e6b3c", extratrees: "#b07a2e", histgb: "#8e44ad" };
  var NAMES = { logreg: "LogReg", rf: "RandForest", extratrees: "ExtraTrees", histgb: "HistGB" };
  var YMIN = 0.68, YMAX = 0.83;

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "cash-search-viz";

    var filter = null;

    var ctl = document.createElement("div");
    ctl.className = "csv-ctl";
    var allBtn = document.createElement("button");
    allBtn.type = "button"; allBtn.textContent = "All algorithms";
    ctl.appendChild(allBtn);
    var algoBtns = {};
    ["logreg", "rf", "extratrees", "histgb"].forEach(function (a) {
      var b = document.createElement("button");
      b.type = "button"; b.textContent = NAMES[a];
      b.style.borderColor = COLORS[a];
      algoBtns[a] = b; ctl.appendChild(b);
    });
    container.appendChild(ctl);

    var svgNS = "http://www.w3.org/2000/svg";
    var W = 520, H = 300;
    var svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("width", "100%");
    svg.setAttribute("class", "csv-svg");
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "csv-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "csv-caption";
    cap.textContent = config.caption ||
      "Real 40-iteration random search over a combined 4-algorithm × hyperparameter space (credit_g). " +
      "Each dot is one evaluated pipeline; colour is the algorithm; the black step line is the best " +
      "validation AUC found so far. Filter to one algorithm with the buttons.";
    container.appendChild(cap);

    var PL = 46, PR = W - 14, PT = 18, PB = H - 34;
    function sx(i) { return PL + ((i - 1) / 39) * (PR - PL); }
    function sy(v) { return PB - ((v - YMIN) / (YMAX - YMIN)) * (PB - PT); }
    function el(name, attrs) {
      var e = document.createElementNS(svgNS, name);
      for (var k in attrs) e.setAttribute(k, attrs[k]);
      return e;
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.appendChild(el("rect", { x: PL, y: PT, width: PR - PL, height: PB - PT,
        fill: "var(--bg)", stroke: "var(--border)", "stroke-width": 1 }));

      // y grid + labels
      [0.70, 0.75, 0.80].forEach(function (v) {
        svg.appendChild(el("line", { x1: PL, y1: sy(v), x2: PR, y2: sy(v),
          stroke: "var(--border)", "stroke-width": 0.5, "stroke-dasharray": "2,3" }));
        svg.appendChild(el("text", { x: PL - 6, y: sy(v) + 3, "text-anchor": "end", class: "csv-tick" }))
          .textContent = v.toFixed(2);
      });
      svg.appendChild(el("text", { x: (PL + PR) / 2, y: H - 6, "text-anchor": "middle", class: "csv-lab" }))
        .textContent = "search iteration (budget) →";
      var yl = el("text", { x: 12, y: (PT + PB) / 2, "text-anchor": "middle", class: "csv-lab",
        transform: "rotate(-90 12 " + ((PT + PB) / 2) + ")" });
      yl.textContent = "validation ROC-AUC"; svg.appendChild(yl);

      // best-so-far step line
      var d = "M";
      TRACE.forEach(function (t, k) {
        var x = sx(t.i), yv = sy(t.best);
        d += (k ? "L" : "") + x.toFixed(1) + "," + sy(TRACE[k].best).toFixed(1) + " ";
        d += "L" + sx(t.i).toFixed(1) + "," + yv.toFixed(1) + " ";
      });
      svg.appendChild(el("path", { d: d, fill: "none", stroke: "var(--fg, #222)", "stroke-width": 1.6, class: "csv-best" }));

      // dots
      TRACE.forEach(function (t) {
        var dim = filter && t.algo !== filter;
        svg.appendChild(el("circle", { cx: sx(t.i), cy: sy(t.val), r: 4.2,
          fill: COLORS[t.algo], opacity: dim ? 0.12 : 0.92, stroke: "var(--bg)", "stroke-width": 1,
          class: "csv-dot csv-dot-" + t.algo }));
      });

      // winner star (iteration 15, first time best hits 0.817)
      var win = TRACE[14];
      svg.appendChild(el("text", { x: sx(win.i), y: sy(win.val) - 7, "text-anchor": "middle", class: "csv-star" }))
        .textContent = "★";
    }

    function setAlgo(a) {
      filter = a;
      allBtn.classList.toggle("csv-on", !a);
      for (var k in algoBtns) algoBtns[k].classList.toggle("csv-on", k === a);
      var counts = { logreg: 11, rf: 7, extratrees: 11, histgb: 11 };
      if (!a) {
        readout.innerHTML = "Searched a <strong>combined</strong> space of <strong>4 algorithms × their " +
          "hyperparameters</strong> (CASH). Best validation AUC climbed <strong>0.796 → 0.817</strong> " +
          "(★, iteration 15); the winning config was a <strong>HistGB</strong>. Selecting the winner by " +
          "<em>validation</em> score is the whole game — that is what AutoML automates.";
      } else {
        readout.innerHTML = "<span style='color:" + COLORS[a] + "'>■</span> <strong>" + NAMES[a] +
          "</strong>: tried <strong>" + counts[a] + "</strong> of 40 configs. The search spreads budget " +
          "across every algorithm lane, not just the eventual winner — that is why it is <em>combined</em> " +
          "algorithm + hyperparameter search.";
      }
      draw();
    }

    allBtn.addEventListener("click", function () { setAlgo(null); });
    Object.keys(algoBtns).forEach(function (a) {
      algoBtns[a].addEventListener("click", function () { setAlgo(a); });
    });

    setAlgo(null);
    return { setAlgo: setAlgo };
  }

  global.CashSearchViz = { mount: mount };
})(window);
