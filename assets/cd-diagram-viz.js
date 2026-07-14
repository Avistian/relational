/**
 * Critical-difference (CD) diagram — Demšar 2006, the headline tool. (Lesson 023)
 *
 * When you compare k classifiers over N datasets, you cannot average accuracies (they are
 * not commensurable across datasets) or run a paired t-test (normality is not safe). Demšar's
 * recipe, made visible here:
 *   1. Rank the k models on each dataset (1 = best); average the ranks down the datasets.
 *   2. The FRIEDMAN test asks whether the average ranks differ more than chance would allow.
 *   3. If so, the NEMENYI post-hoc says two models differ significantly only if their average
 *      ranks differ by at least the CRITICAL DIFFERENCE
 *          CD = q_alpha * sqrt( k(k+1) / (6N) ).
 * The diagram: an axis of average ranks (best on the left); each model hangs at its average
 * rank; a horizontal bar joins any group of models whose ranks are all within CD of each other
 * ("not significantly different"). A CD ruler shows the length of the critical difference.
 *
 * Click a model to highlight every model that is NOT significantly different from it.
 *
 * Defaults reproduce the verified L023 case (labs/_verify_l023.py, 4 models over N=12 datasets):
 *   LogReg 1.083, GaussianNB 2.083, RandomForest 3.417, HistGBDT 3.417; Friedman p=4e-6; CD=1.354.
 *
 * Usage: CdDiagramViz.mount(container, { });
 * Expected states:
 *   - LogReg is leftmost (best avg rank); RF & HistGBDT tie on the right.
 *   - LogReg–GaussianNB joined (gap 1.00 < CD); LogReg–RandomForest NOT joined (gap 2.33 > CD).
 */
(function (global) {
  "use strict";

  var DEFAULT_MODELS = [
    { name: "LogReg", rank: 1.083 },
    { name: "GaussianNB", rank: 2.083 },
    { name: "RandomForest", rank: 3.417 },
    { name: "HistGBDT", rank: 3.417 },
  ];

  function mount(container, config) {
    config = config || {};
    var models = config.models || DEFAULT_MODELS;
    var cd = config.cd != null ? config.cd : 1.354;
    var N = config.N != null ? config.N : 12;
    var friedmanP = config.friedmanP || "4e-6";
    var k = models.length;

    container.innerHTML = "";
    container.className = "cd-diagram-viz";

    var svgNS = "http://www.w3.org/2000/svg";
    var W = 560, H = 230;
    var svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("width", "100%");
    svg.setAttribute("class", "cdv-svg");
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "cdv-readout";
    container.appendChild(readout);

    // rank axis: 1 (best) on the LEFT ... k (worst) on the RIGHT
    var PL = 60, PR = W - 60, axisY = 60;
    var lo = 1, hi = k;
    function rx(r) { return PL + (r - lo) / (hi - lo) * (PR - PL); }

    function el(name, attrs) {
      var e = document.createElementNS(svgNS, name);
      for (var a in attrs) e.setAttribute(a, attrs[a]);
      return e;
    }

    var selected = null;

    // maximal cliques of consecutive (by rank) models within CD — the joining bars
    var sorted = models.slice().sort(function (a, b) { return a.rank - b.rank; });
    var bars = [];
    for (var i = 0; i < sorted.length; i++) {
      var j = i;
      while (j + 1 < sorted.length && sorted[j + 1].rank - sorted[i].rank <= cd + 1e-9) j++;
      if (j > i) {
        // drop if contained in the previous bar
        var prev = bars[bars.length - 1];
        if (!(prev && prev.startRank <= sorted[i].rank && sorted[j].rank <= prev.endRank)) {
          bars.push({ startRank: sorted[i].rank, endRank: sorted[j].rank });
        }
      }
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      // main axis
      svg.appendChild(el("line", { x1: rx(lo), y1: axisY, x2: rx(hi), y2: axisY, stroke: "var(--fg,#222)", "stroke-width": 1.5 }));
      for (var r = lo; r <= hi; r++) {
        svg.appendChild(el("line", { x1: rx(r), y1: axisY - 5, x2: rx(r), y2: axisY, stroke: "var(--fg,#222)", "stroke-width": 1.5 }));
        svg.appendChild(el("text", { x: rx(r), y: axisY - 10, "text-anchor": "middle", class: "cdv-tick" }))
          .textContent = String(r);
      }
      svg.appendChild(el("text", { x: rx(lo), y: axisY - 26, "text-anchor": "middle", class: "cdv-lab" }))
        .textContent = "better ← average rank → worse";

      // CD ruler (top-left)
      var rulerY = 24, x0 = rx(lo), x1 = rx(lo + cd);
      svg.appendChild(el("line", { x1: x0, y1: rulerY, x2: x1, y2: rulerY, stroke: "#8e44ad", "stroke-width": 2 }));
      svg.appendChild(el("line", { x1: x0, y1: rulerY - 4, x2: x0, y2: rulerY + 4, stroke: "#8e44ad", "stroke-width": 2 }));
      svg.appendChild(el("line", { x1: x1, y1: rulerY - 4, x2: x1, y2: rulerY + 4, stroke: "#8e44ad", "stroke-width": 2 }));
      svg.appendChild(el("text", { x: (x0 + x1) / 2, y: rulerY - 7, "text-anchor": "middle", class: "cdv-cd" }))
        .textContent = "CD = " + cd.toFixed(3);

      // model nodes: alternate labels above/below to avoid overlap
      sorted.forEach(function (m, idx) {
        var x = rx(m.rank);
        var above = idx % 2 === 0;
        var elbowY = above ? axisY + 22 + (idx % 2) * 0 : axisY + 22;
        var labelY = axisY + 40 + (idx) * 20;
        // connector from axis down to a stacked label row (classic CD layout stacks labels)
        var ly = axisY + 30 + idx * 22;
        var isHi = selected == null ? false :
          Math.abs(m.rank - selected.rank) <= cd + 1e-9;
        var isSel = selected && m.name === selected.name;
        var color = isSel ? "#8e44ad" : (isHi ? "#1e6b3c" : "var(--fg,#222)");
        svg.appendChild(el("line", { x1: x, y1: axisY, x2: x, y2: ly, stroke: color, "stroke-width": isSel ? 2.5 : 1.2 }));
        svg.appendChild(el("line", { x1: x, y1: ly, x2: PL - 8, y2: ly, stroke: color, "stroke-width": isSel ? 2.5 : 1.2 }));
        var dot = el("circle", { cx: x, cy: axisY, r: isSel ? 5 : 4, fill: color, stroke: "var(--bg)", "stroke-width": 1 });
        svg.appendChild(dot);
        var label = el("text", { x: PL - 12, y: ly + 3.5, "text-anchor": "end", class: "cdv-name" });
        label.textContent = m.name + "  (" + m.rank.toFixed(2) + ")";
        if (isSel) label.setAttribute("font-weight", "700");
        svg.appendChild(label);
        // clickable hit target
        var hit = el("rect", { x: 4, y: ly - 9, width: PL - 8, height: 18, fill: "transparent", style: "cursor:pointer" });
        hit.addEventListener("click", function () { selected = (selected && selected.name === m.name) ? null : m; draw(); });
        dot.addEventListener("click", function () { selected = (selected && selected.name === m.name) ? null : m; draw(); });
        svg.appendChild(hit);
      });

      // joining bars (not-significantly-different cliques), stacked just under the axis
      bars.forEach(function (b, bi) {
        var y = axisY + 8 + bi * 7;
        svg.appendChild(el("line", {
          x1: rx(b.startRank) - 3, y1: y, x2: rx(b.endRank) + 3, y2: y,
          stroke: "#555", "stroke-width": 4, "stroke-linecap": "round",
        }));
      });

      var msg = selected
        ? "<span class='cdv-pill'>" + selected.name + "</span> is <strong>not</strong> significantly different from the models in green " +
          "(average rank within CD = " + cd.toFixed(3) + "); the rest differ significantly. Click again to clear."
        : "Friedman p ≈ " + friedmanP + " over N = " + N + " datasets → the ranks differ. " +
          "Two models are joined by a bar (and so <em>not</em> significantly different) only if their average ranks are within " +
          "CD = " + cd.toFixed(3) + ". <strong>Click a model</strong> to see its non-different group.";
      readout.innerHTML = msg;
    }

    draw();
    return { select: function (name) { selected = models.find(function (m) { return m.name === name; }) || null; draw(); } };
  }

  global.CdDiagramViz = { mount: mount };
})(window);
