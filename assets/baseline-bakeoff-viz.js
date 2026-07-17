/**
 * Honest baseline bake-off — MLP vs ResNet vs tuned GBDT on a real dataset.
 *
 * REAL numbers from labs/_verify_l028.py (credit_g, Tier A OpenML, 5 seeds, 70/30 splits):
 *   labels   = ['MLP', 'ResNet', 'GBDT']
 *   auc_mean = [0.752, 0.743, 0.793]   auc_sd = [0.011, 0.010, 0.025]
 *   acc_mean = [0.734, 0.740, 0.754]
 *
 * The message (Gorishniy 2021): the two neural baselines are competitive WITH EACH OTHER, and a
 * tuned GBDT is strong and cheap on this small categorical dataset (consistent with the Grinsztajn
 * arc, L024-L027). There is NO universal winner. The value of the ResNet is as an honest baseline
 * many "SOTA" DL papers failed to beat — not as a headline winner.
 *
 * Usage: BaselineBakeoffViz.mount(container, { caption })
 * Expected states:
 *   - default: 3 bars (ROC-AUC) with ±sd error whiskers; GBDT tallest, MLP≈ResNet overlapping.
 *   - "Accuracy" / "ROC-AUC" toggle switches the metric.
 */
(function (global) {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var COLORS = { MLP: "#2e6fb0", ResNet: "#1e6b3c", GBDT: "#8a6d3b" };

  var LABELS = ["MLP", "ResNet", "GBDT"];
  var AUC = [0.752, 0.743, 0.793];
  var AUC_SD = [0.011, 0.010, 0.025];
  var ACC = [0.734, 0.740, 0.754];
  var ACC_SD = [0.002, 0.013, 0.019];

  function el(name, attrs, text) {
    var e = document.createElementNS(SVGNS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "baseline-bakeoff-viz";

    var metric = "auc";  // "auc" | "acc"

    var ctl = document.createElement("div");
    ctl.className = "bko-ctl";
    var bAuc = document.createElement("button");
    bAuc.textContent = "ROC-AUC";
    var bAcc = document.createElement("button");
    bAcc.textContent = "Accuracy";
    ctl.appendChild(bAuc); ctl.appendChild(bAcc);
    container.appendChild(ctl);

    var W = 420, H = 300;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", class: "bko-svg" });
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "bko-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "bko-caption";
    cap.textContent = config.caption ||
      "credit_g (real OpenML, 5 seeds). Whiskers are ±1 standard deviation. The point is not who " +
      "wins — it is that the two neural baselines are tied and a tuned GBDT is strong on this small table.";
    container.appendChild(cap);

    var PL = 44, PR = W - 14, PT = 20, PB = H - 40;
    var YLO = 0.5, YHI = 0.85;
    function sy(v) { return PB - (v - YLO) / (YHI - YLO) * (PB - PT); }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var vals = metric === "auc" ? AUC : ACC;
      var sds = metric === "auc" ? AUC_SD : ACC_SD;

      svg.appendChild(el("rect", { x: PL, y: PT, width: PR - PL, height: PB - PT,
        fill: "var(--bg)", stroke: "var(--border)", "stroke-width": 1 }));
      [0.5, 0.6, 0.7, 0.8].forEach(function (v) {
        var yy = sy(v);
        svg.appendChild(el("line", { x1: PL, y1: yy, x2: PR, y2: yy, stroke: "var(--border)", "stroke-width": 0.5, opacity: 0.6 }));
        svg.appendChild(el("text", { x: PL - 6, y: yy + 3, "text-anchor": "end", class: "bko-tick" }, v.toFixed(1)));
      });

      var groupW = (PR - PL) / LABELS.length;
      var barW = groupW * 0.42;
      LABELS.forEach(function (name, g) {
        var cxg = PL + g * groupW + groupW / 2;
        var x = cxg - barW / 2;
        var v = vals[g], sd = sds[g];
        svg.appendChild(el("rect", { x: x, y: sy(v), width: barW, height: PB - sy(v), fill: COLORS[name], opacity: 0.88 }));
        svg.appendChild(el("text", { x: cxg, y: sy(v) - 12, "text-anchor": "middle", class: "bko-val" }, v.toFixed(3)));
        // error whisker
        svg.appendChild(el("line", { x1: cxg, y1: sy(v - sd), x2: cxg, y2: sy(v + sd), stroke: "var(--fg,#222)", "stroke-width": 1.3 }));
        svg.appendChild(el("line", { x1: cxg - 5, y1: sy(v + sd), x2: cxg + 5, y2: sy(v + sd), stroke: "var(--fg,#222)", "stroke-width": 1.3 }));
        svg.appendChild(el("line", { x1: cxg - 5, y1: sy(v - sd), x2: cxg + 5, y2: sy(v - sd), stroke: "var(--fg,#222)", "stroke-width": 1.3 }));
        svg.appendChild(el("text", { x: cxg, y: PB + 15, "text-anchor": "middle", class: "bko-lab" }, name));
      });

      svg.appendChild(el("text", { x: 12, y: (PT + PB) / 2, "text-anchor": "middle", class: "bko-lab",
        transform: "rotate(-90 12 " + ((PT + PB) / 2) + ")" }, metric === "auc" ? "ROC-AUC" : "accuracy"));

      readout.innerHTML = metric === "auc"
        ? "ROC-AUC: <strong style='color:" + COLORS.MLP + "'>MLP 0.752</strong> ≈ " +
          "<strong style='color:" + COLORS.ResNet + "'>ResNet 0.743</strong> (their ±sd bands overlap — a tie), " +
          "with <strong style='color:" + COLORS.GBDT + "'>GBDT 0.793</strong> ahead. On a small, " +
          "categorical, irregular table the tuned tree wins — exactly what L024–L027 predict. The ResNet " +
          "is a <em>competitive, honest baseline</em>, not a headline winner."
        : "Accuracy: MLP 0.734 · ResNet 0.740 · GBDT 0.754 — same story, tighter spread. No universal " +
          "winner: the neural baselines are competitive, the GBDT is strong and far cheaper to train here.";
    }

    bAuc.addEventListener("click", function () { metric = "auc"; bAuc.classList.add("bko-on"); bAcc.classList.remove("bko-on"); draw(); });
    bAcc.addEventListener("click", function () { metric = "acc"; bAcc.classList.add("bko-on"); bAuc.classList.remove("bko-on"); draw(); });

    bAuc.classList.add("bko-on");
    draw();
    return { setMetric: function (m) { metric = m; draw(); } };
  }

  global.BaselineBakeoffViz = { mount: mount };
})(window);
