/**
 * Label efficiency — the payoff of TabTransformer's RTD pre-training (Huang 2020, §3.3; L045 verified).
 *
 * ONE mechanism/claim: pre-training the encoder on UNLABELED rows (RTD) and then fine-tuning on a small
 * labeled set beats training the same model from scratch on that set — and the gain is LARGEST when
 * labels are scarcest, shrinking as labels grow. The slider moves the labeled fraction; two points
 * (from-scratch vs pre-train+fine-tune) and the lift between them update.
 *
 * Numbers are REAL — labs/_verify_l045.py on `adult` (16 000 rows, unlabeled pool ~11 200), test ROC-AUC,
 * mean over 3 seeds. Between the two measured fractions the curve is drawn as a guide; the measured
 * anchors are marked. The honest caption states the lift is small at this down-scaled scale (the paper's
 * +2.1% comes from far more unlabeled data), but positive on every seed at the smallest fraction.
 *
 * Expected states:
 *   - default frac 0.03: scratch 0.825, pretrain 0.833, lift +0.008 (biggest gap).
 *   - frac 0.10        : scratch 0.861, pretrain 0.862, lift +0.001 (gap nearly closed).
 *
 * Usage: LabelEfficiencyViz.mount(container, {})
 */
(function (global) {
  "use strict";

  // Verified anchors (labs/_verify_l045_results.json -> semisupervised.summary). fraction -> {scratch, pretrain}
  var DATA = [
    { frac: 0.03, scratch: 0.825, pretrain: 0.833, allpos: true },
    { frac: 0.10, scratch: 0.861, pretrain: 0.862, allpos: false }
  ];

  // Pure: nearest measured anchor's lift for a given fraction (used by the readout + headless check).
  function anchorAt(frac) {
    var best = DATA[0], bd = Infinity;
    DATA.forEach(function (d) {
      var dd = Math.abs(d.frac - frac);
      if (dd < bd) { bd = dd; best = d; }
    });
    return best;
  }
  function liftAt(frac) { var a = anchorAt(frac); return a.pretrain - a.scratch; }

  var svgNS = "http://www.w3.org/2000/svg";
  function el(name, attrs) {
    var e = document.createElementNS(svgNS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "leff-viz";
    var frac = 0.03;

    var ctl = document.createElement("div");
    ctl.className = "leff-ctl";
    var lab = document.createElement("span");
    lab.className = "leff-glab";
    lab.textContent = "labeled fraction ";
    var fval = document.createElement("b");
    fval.className = "leff-mono";
    fval.textContent = "3%";
    lab.appendChild(fval);
    var slider = document.createElement("input");
    slider.type = "range"; slider.min = "3"; slider.max = "10"; slider.step = "1"; slider.value = "3";
    slider.className = "leff-slider";
    slider.addEventListener("input", function () { frac = (+slider.value) / 100; draw(); });
    ctl.appendChild(lab); ctl.appendChild(slider);
    container.appendChild(ctl);

    var W = 460, H = 260;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", class: "leff-svg" });
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "leff-readout";
    container.appendChild(readout);

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      fval.textContent = Math.round(frac * 100) + "%";
      var PL = 44, PR = W - 90, PT = 18, PB = H - 34;
      var xlo = 0.02, xhi = 0.11, ylo = 0.80, yhi = 0.88;
      function sx(x) { return PL + (x - xlo) / (xhi - xlo) * (PR - PL); }
      function sy(y) { return PB - (y - ylo) / (yhi - ylo) * (PB - PT); }

      svg.appendChild(el("rect", { x: PL, y: PT, width: PR - PL, height: PB - PT,
        fill: "var(--bg)", stroke: "var(--border)", "stroke-width": 1 }));
      // y gridlines
      [0.80, 0.82, 0.84, 0.86, 0.88].forEach(function (yv) {
        svg.appendChild(el("line", { x1: PL, y1: sy(yv), x2: PR, y2: sy(yv),
          stroke: "var(--border)", "stroke-width": 0.5, "stroke-dasharray": "2 3" }));
        var t = el("text", { x: PL - 6, y: sy(yv) + 3, "text-anchor": "end", class: "leff-axis" });
        t.textContent = yv.toFixed(2); svg.appendChild(t);
      });
      // axis labels
      var xl = el("text", { x: (PL + PR) / 2, y: H - 8, "text-anchor": "middle", class: "leff-axis" });
      xl.textContent = "labeled fraction of the training rows"; svg.appendChild(xl);
      var yl = el("text", { x: 12, y: (PT + PB) / 2, "text-anchor": "middle", class: "leff-axis",
        transform: "rotate(-90 12 " + ((PT + PB) / 2) + ")" });
      yl.textContent = "test ROC-AUC"; svg.appendChild(yl);

      // two lines
      function line(getter, color, dash) {
        var pts = DATA.map(function (d) { return sx(d.frac) + "," + sy(getter(d)); }).join(" ");
        svg.appendChild(el("polyline", { points: pts, fill: "none", stroke: color, "stroke-width": 2,
          "stroke-dasharray": dash || "" }));
      }
      line(function (d) { return d.pretrain; }, "#1e6b3c");
      line(function (d) { return d.scratch; }, "#b03a2e", "5 3");
      DATA.forEach(function (d) {
        svg.appendChild(el("circle", { cx: sx(d.frac), cy: sy(d.pretrain), r: 4, fill: "#1e6b3c" }));
        svg.appendChild(el("circle", { cx: sx(d.frac), cy: sy(d.scratch), r: 4, fill: "#b03a2e" }));
      });
      // legend
      svg.appendChild(el("rect", { x: PR + 8, y: PT + 6, width: 10, height: 10, fill: "#1e6b3c" }));
      var lg1 = el("text", { x: PR + 22, y: PT + 15, class: "leff-leg" }); lg1.textContent = "pre-train+finetune"; svg.appendChild(lg1);
      svg.appendChild(el("rect", { x: PR + 8, y: PT + 24, width: 10, height: 10, fill: "#b03a2e" }));
      var lg2 = el("text", { x: PR + 22, y: PT + 33, class: "leff-leg" }); lg2.textContent = "from scratch"; svg.appendChild(lg2);

      // marker at the selected fraction
      var a = anchorAt(frac);
      svg.appendChild(el("line", { x1: sx(a.frac), y1: PT, x2: sx(a.frac), y2: PB,
        stroke: "#2e6fb0", "stroke-width": 1 }));
      // the lift bracket
      svg.appendChild(el("line", { x1: sx(a.frac), y1: sy(a.scratch), x2: sx(a.frac), y2: sy(a.pretrain),
        stroke: "#2e6fb0", "stroke-width": 3 }));

      var lift = (a.pretrain - a.scratch);
      readout.innerHTML = "<strong>At " + Math.round(a.frac * 100) + "% labels:</strong> from scratch " +
        "<b>" + a.scratch.toFixed(3) + "</b> → pre-train + fine-tune <b>" + a.pretrain.toFixed(3) + "</b> " +
        "(<b class='leff-pos'>lift " + (lift >= 0 ? "+" : "") + lift.toFixed(3) + "</b>" +
        (a.allpos ? ", positive on all 3 seeds" : "") + "). " +
        (a.frac <= 0.05
          ? "Scarce labels + a big unlabeled pool is exactly where the free self-supervised signal pays."
          : "With more labels the model already learns the structure supervised, so the pre-training gap nearly closes.");
    }

    draw();
    return { set: function (f) { frac = f; draw(); }, liftAt: liftAt, anchorAt: anchorAt, data: DATA };
  }

  global.LabelEfficiencyViz = { mount: mount, liftAt: liftAt, anchorAt: anchorAt, data: DATA };
})(window);
