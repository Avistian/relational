/**
 * The honest bake-off: default XGBoost vs a tuned XGBoost vs a tiny AutoML (Feurer 2015 mechanisms).
 *
 * REAL numbers (labs/_verify_l029.py, credit_g, 5 seeds, ROC-AUC mean ± sd):
 *   default XGB   0.775 ± 0.025
 *   tuned  XGB    0.806 ± 0.016   (+0.031 over default — the big jump is TUNING)
 *   tiny AutoML   0.803 ± 0.020   (−0.002 vs the hand-tuned XGB — a TIE, at far higher compute)
 *
 * The lesson's point: most of the win comes from *searching at all* (default → tuned), not from which
 * meta-strategy runs the search; AutoML ties a competently tuned GBDT here and buys automation/robustness,
 * not a guaranteed win — and both are still single-table learners. Bars carry ±sd whiskers; toggle the
 * metric label between the three models' means. Bands overlap ⇒ tuned XGB and AutoML are indistinguishable.
 *
 * Usage: AutomlBakeoffViz.mount(container, { caption })
 * Handle: { highlight("default"|"tuned"|"automl"|null) }
 */
(function (global) {
  "use strict";

  var BARS = [
    { key: "default", label: "default XGB", mean: 0.775, sd: 0.025, color: "#b03a2e" },
    { key: "tuned", label: "tuned XGB", mean: 0.806, sd: 0.016, color: "#1e6b3c" },
    { key: "automl", label: "tiny AutoML", mean: 0.803, sd: 0.020, color: "#8e44ad" }
  ];
  var YMIN = 0.70, YMAX = 0.84;

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "automl-bakeoff-viz";

    var hl = null;

    var svgNS = "http://www.w3.org/2000/svg";
    var W = 420, H = 280;
    var svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("width", "100%");
    svg.setAttribute("class", "abo-svg");
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "abo-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "abo-caption";
    cap.textContent = config.caption ||
      "credit_g, 5 seeds. Bars = mean test ROC-AUC; whiskers = ±1 sd. The tuned XGB and the tiny AutoML " +
      "bands overlap (a tie); both clear the untuned default. Hover/click a bar to read its number.";
    container.appendChild(cap);

    var PL = 46, PR = W - 14, PT = 20, PB = H - 40;
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
      [0.72, 0.76, 0.80].forEach(function (v) {
        svg.appendChild(el("line", { x1: PL, y1: sy(v), x2: PR, y2: sy(v),
          stroke: "var(--border)", "stroke-width": 0.5, "stroke-dasharray": "2,3" }));
        svg.appendChild(el("text", { x: PL - 6, y: sy(v) + 3, "text-anchor": "end", class: "abo-tick" }))
          .textContent = v.toFixed(2);
      });
      var yl = el("text", { x: 12, y: (PT + PB) / 2, "text-anchor": "middle", class: "abo-lab",
        transform: "rotate(-90 12 " + ((PT + PB) / 2) + ")" });
      yl.textContent = "test ROC-AUC"; svg.appendChild(yl);

      var n = BARS.length, slot = (PR - PL) / n, bw = 68;
      BARS.forEach(function (b, k) {
        var cx = PL + slot * (k + 0.5), x = cx - bw / 2;
        var dim = hl && hl !== b.key;
        svg.appendChild(el("rect", { x: x, y: sy(b.mean), width: bw, height: PB - sy(b.mean),
          fill: b.color, opacity: dim ? 0.25 : 0.9, class: "abo-bar abo-bar-" + b.key }));
        // ±sd whisker
        svg.appendChild(el("line", { x1: cx, y1: sy(b.mean - b.sd), x2: cx, y2: sy(b.mean + b.sd),
          stroke: "var(--fg, #222)", "stroke-width": 1.4 }));
        svg.appendChild(el("line", { x1: cx - 7, y1: sy(b.mean + b.sd), x2: cx + 7, y2: sy(b.mean + b.sd),
          stroke: "var(--fg, #222)", "stroke-width": 1.4 }));
        svg.appendChild(el("line", { x1: cx - 7, y1: sy(b.mean - b.sd), x2: cx + 7, y2: sy(b.mean - b.sd),
          stroke: "var(--fg, #222)", "stroke-width": 1.4 }));
        svg.appendChild(el("text", { x: cx, y: sy(b.mean + b.sd) - 6, "text-anchor": "middle", class: "abo-val" }))
          .textContent = b.mean.toFixed(3);
        svg.appendChild(el("text", { x: cx, y: PB + 16, "text-anchor": "middle", class: "abo-lab2" }))
          .textContent = b.label;
        var hit = el("rect", { x: x, y: PT, width: bw, height: PB - PT, fill: "transparent", style: "cursor:pointer" });
        hit.addEventListener("click", function () { setHighlight(hl === b.key ? null : b.key); });
        svg.appendChild(hit);
      });
    }

    function setHighlight(k) {
      hl = k;
      if (!k) {
        readout.innerHTML = "The big jump is <strong>tuning</strong>: default XGB <strong>0.775</strong> → " +
          "tuned XGB <strong>0.806</strong> (<strong>+0.031</strong>). The tiny AutoML (<strong>0.803</strong>) " +
          "<strong>ties</strong> the hand-tuned XGB (−0.002; bands overlap) at far higher compute — it buys " +
          "automation and robustness, <em>not</em> a win. Both are single-table learners.";
      } else {
        var b = BARS.filter(function (x) { return x.key === k; })[0];
        readout.innerHTML = "<strong>" + b.label + "</strong>: ROC-AUC <strong>" + b.mean.toFixed(3) +
          " ± " + b.sd.toFixed(3) + "</strong> over 5 seeds.";
      }
      draw();
    }

    setHighlight(null);
    return { highlight: setHighlight };
  }

  global.AutomlBakeoffViz = { mount: mount };
})(window);
