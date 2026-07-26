/**
 * The estimator of record: one number, two ways to compute it. (Lesson 037)
 *
 * The submission reports top-label ECE twice, for the SAME shipped model on the
 * SAME out-of-fold predictions:
 *   report.md §3.1 model-selection table : 0.0332  (mean of five per-fold ECEs)
 *   README.md + §4 ship-gate table       : 0.018   (pool all 5,587 rows, bin once)
 * Neither is wrong. They are different estimators, and the word "ECE" does not
 * distinguish them.
 *
 * The widget draws the 15 equal-width confidence bins that ECE is built from, so
 * the mechanism is visible: pooling gives bins with 5x the population, and the
 * |accuracy - confidence| gap in a sparse bin is mostly sampling noise. In the
 * pooled view the top populated bin holds 17 rows and shows a 0.19 gap; in one
 * fold it holds 2 rows and shows a 0.40 gap.
 *
 * The readout also quotes the NOISE FLOOR: the same estimator applied to a
 * perfectly calibrated predictor (labels resampled from its own probabilities,
 * so true ECE = 0 by construction), measured at the same sample size.
 *
 * All values measured by labs/_ece_estimator_l037.py on the submission's own
 * artifacts/oof_M2a_lgbm_km.npz (5,587 rows, 5 classes, 15 bins).
 *
 * Plain <script> (file://-safe). Usage: EceEstimatorViz.mount(el, config?)
 *
 * Expected states (headless verification):
 *   - mounts an svg with one bar per populated bin + three mode buttons
 *   - default mode "pooled": readout reports 0.0178 and noise floor 0.0149
 *   - mode "fold": readout reports 0.0332 (mean of 5) and noise floor 0.0335
 *   - mode "slice": readout reports the 107-row ship-gate cell at 0.094
 */
(function (global) {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var C_BAR = "#2e6fb0";
  var C_SPARSE = "#b03a2e";

  // [binIndex, n, conf, acc] for the populated bins. Measured, 15 equal-width bins.
  var POOLED = [
    [3, 582, 0.2479, 0.2371], [4, 1264, 0.3000, 0.2975], [5, 1028, 0.3659, 0.3589],
    [6, 827, 0.4315, 0.4595], [7, 687, 0.4975, 0.5226], [8, 487, 0.5658, 0.6099],
    [9, 330, 0.6319, 0.6394], [10, 216, 0.6964, 0.7315], [11, 104, 0.7602, 0.7308],
    [12, 45, 0.8298, 0.7333], [13, 17, 0.8983, 0.7059]
  ];
  var FOLD0 = [
    [3, 121, 0.2500, 0.2479], [4, 262, 0.2995, 0.2863], [5, 220, 0.3650, 0.3818],
    [6, 151, 0.4325, 0.4768], [7, 135, 0.4982, 0.5333], [8, 94, 0.5670, 0.5638],
    [9, 62, 0.6343, 0.6129], [10, 39, 0.6996, 0.6923], [11, 23, 0.7625, 0.7826],
    [12, 9, 0.8317, 0.7778], [13, 2, 0.9015, 0.5000]
  ];

  var MODES = {
    pooled: {
      label: "Pool once (n = 5,587)",
      bins: POOLED,
      n: 5587,
      reported: "0.0178",
      floor: "0.0149",
      where: "README.md headline and the report.md &sect;4 ship-gate table (as <strong>0.018</strong>)",
      note:
        "One pass over every out-of-fold row. The bins are well populated — the smallest still holds " +
        "<strong>17</strong> rows — so each bin's accuracy is estimated from enough cases to mean " +
        "something. Reported <strong>0.0178</strong> against a noise floor of <strong>0.0149</strong>: " +
        "about <strong>0.003 of actual signal</strong>."
    },
    fold: {
      label: "Bin per fold (5 &times; n = 1,117), then average",
      bins: FOLD0,
      n: 1117,
      reported: "0.0332",
      floor: "0.0335",
      where: "report.md &sect;3.1, the table the shipped model was <em>selected</em> from",
      note:
        "The same rows, cut five ways first. Every bin is a fifth as populated — the top one now holds " +
        "<strong>2 rows</strong>, one right and one wrong, producing a <strong>0.40</strong> gap out of " +
        "pure arithmetic. Reported <strong>0.0332</strong> against a noise floor of " +
        "<strong>0.0335</strong>: a <em>perfectly calibrated</em> model scores the same. This number " +
        "carries essentially no information about calibration."
    },
    slice: {
      label: "One slice (n = 107)",
      bins: null,
      n: 107,
      reported: "0.094",
      floor: "0.1071",
      where: "report.md &sect;4.7 — the only ship-gate the submission <strong>failed</strong>",
      note:
        "The <code>age = missing</code> cell: 107 rows, ECE <strong>0.094</strong>, gate threshold " +
        "0.05, verdict CONDITIONAL. But at n = 107 this estimator reports <strong>0.1071 &plusmn; 0.0297</strong> " +
        "for a model that is calibrated <em>exactly</em> right. The slice gate, as written, cannot be " +
        "passed by any model — the failure is a property of the ruler."
    }
  };

  function el(tag, attrs) {
    var n = document.createElementNS(SVGNS, tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    return n;
  }
  function txt(n, s) { n.textContent = s; return n; }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.classList.add("ee-viz");

    var ctl = document.createElement("div");
    ctl.className = "ee-ctl";
    container.appendChild(ctl);
    var holder = document.createElement("div");
    container.appendChild(holder);
    var readout = document.createElement("div");
    readout.className = "ee-readout";
    container.appendChild(readout);

    var order = ["pooled", "fold", "slice"];
    var current = "pooled";
    var buttons = [];

    function draw() {
      var m = MODES[current];
      buttons.forEach(function (b, i) { b.classList.toggle("ee-on", order[i] === current); });

      holder.innerHTML = "";
      var W = 640, H = 200;
      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%" });
      svg.setAttribute("class", "ee-svg");

      if (!m.bins) {
        var msg = el("text", { x: W / 2, y: 96, "text-anchor": "middle", class: "ee-empty" });
        svg.appendChild(txt(msg,
          "107 rows spread over 15 bins \u2014 about 7 per bin, most of them empty"));
        var msg2 = el("text", { x: W / 2, y: 118, "text-anchor": "middle", class: "ee-empty" });
        svg.appendChild(txt(msg2, "there is nothing left to plot, which is the finding"));
        holder.appendChild(svg);
      } else {
        var x0 = 46, top = 22, barMax = 96, wBar = 44, gap = 8;
        // gap bars, scaled to a fixed 0.45 domain so the two modes are comparable
        var DOM = 0.45;
        svg.appendChild(el("line", {
          x1: x0 - 6, y1: top + barMax, x2: W - 12, y2: top + barMax,
          stroke: "#cbd5e1", "stroke-width": 1
        }));
        var yl = el("text", { x: 8, y: top + 8, class: "ee-axis" });
        svg.appendChild(txt(yl, "|acc\u2212conf|"));

        m.bins.forEach(function (b, i) {
          var n = b[1], gapv = Math.abs(b[3] - b[2]);
          var x = x0 + i * (wBar + gap) * 0.52;
          var h = Math.max(1.5, (gapv / DOM) * barMax);
          var sparse = n < 25;
          svg.appendChild(el("rect", {
            x: x, y: top + barMax - h, width: wBar * 0.52, height: h, rx: 2,
            fill: sparse ? C_SPARSE : C_BAR,
            opacity: sparse ? 0.95 : 0.75
          }));
          var nt = el("text", {
            x: x + wBar * 0.26, y: top + barMax + 14, "text-anchor": "middle",
            class: "ee-n", fill: sparse ? C_SPARSE : "#64748b"
          });
          svg.appendChild(txt(nt, "n=" + n));
          if (sparse) {
            var gt = el("text", {
              x: x + wBar * 0.26, y: top + barMax - h - 5, "text-anchor": "middle",
              class: "ee-gapv", fill: C_SPARSE
            });
            svg.appendChild(txt(gt, gapv.toFixed(2)));
          }
        });

        var cap = el("text", { x: x0, y: top + barMax + 40, class: "ee-cap" });
        svg.appendChild(txt(cap,
          "one bar per populated confidence bin, left = least confident"));
        var cap2 = el("text", { x: x0, y: top + barMax + 56, class: "ee-cap" });
        svg.appendChild(txt(cap2,
          "red = fewer than 25 rows in the bin: the gap there is mostly sampling noise"));
        holder.appendChild(svg);
      }

      var signal = (parseFloat(m.reported) - parseFloat(m.floor));
      var verdict = signal > 0.002
        ? "<span class='ee-good'>above the floor</span>"
        : "<span class='ee-bad'>at or below the floor</span>";
      readout.innerHTML =
        "<p><strong>Reported ECE " + m.reported + "</strong> &nbsp;&middot;&nbsp; " +
        "noise floor at n&nbsp;=&nbsp;" + m.n.toLocaleString() + ": <strong>" + m.floor +
        "</strong> &nbsp;&middot;&nbsp; " + verdict + "</p>" +
        "<p class='ee-where'><strong>Where this number appears:</strong> " + m.where + "</p>" +
        "<p>" + m.note + "</p>";
    }

    order.forEach(function (k) {
      var b = document.createElement("button");
      b.textContent = MODES[k].label;
      b.addEventListener("click", function () { current = k; draw(); });
      ctl.appendChild(b);
      buttons.push(b);
    });

    draw();
    return { setMode: function (k) { current = k; draw(); }, getMode: function () { return current; } };
  }

  global.EceEstimatorViz = { mount: mount };
})(typeof window !== "undefined" ? window : this);
