/**
 * Feature-engineering diminishing-returns curve (Lesson 033, Domingos 2012).
 *
 * The central mechanism of the lesson: hand-crafted features buy less and less as
 * you add more, and past a point they *hurt* (Domingos' "overfitting has many
 * faces"). We hold the MODEL fixed and add engineered features one at a time in a
 * fixed, sensible order (not greedily selected on the label), measuring 5-fold CV
 * ROC-AUC after each addition. The x-axis is thus a proxy for "modeling effort
 * spent on features"; the y-axis is honest held-out performance.
 *
 * Two real curves from labs/_verify_l033.py on OpenML credit_g (seed 0, 5-fold):
 *   - GBDT (HistGradientBoosting): climbs slightly to a PEAK at k=3 (0.7911), then
 *     DECLINES as more features overfit (down to 0.766 at k=8) — negative returns.
 *   - Linear (scaled LogisticRegression): near-flat, drifting up only ~+0.006 total.
 * Each curve carries a shaded +/-1 CV-std band. The bands are wide (~0.02-0.03) and
 * overlap the whole time: every gain here sits INSIDE the noise envelope (the L023
 * lesson) — you cannot buy a real edge with single-table FE on this dataset.
 *
 * A "where I'd stop" marker sits at the GBDT peak (k=3). The slider reads out both
 * models' AUC at budget k, the delta from the k=0 baseline, and whether that delta
 * is within one CV std (i.e. indistinguishable from noise).
 *
 * Usage: FeReturnsViz.mount(container, { caption })
 * Returns: { setK(k), setModel("gbdt"|"linear"|"both"), toggleBand() } for tests.
 * Expected states:
 *   - default: both curves + bands shown, stop marker at k=3, slider at k=3.
 *   - GBDT k=0 -> 0.7865, k=3 -> 0.7911 (peak, +0.0046, WITHIN noise), k=8 -> 0.7659 (below baseline).
 *   - Linear stays ~0.79 throughout; total drift +0.0061, also within its ~0.02 band.
 *   - readout flags "within noise" whenever |delta| < std at that k.
 */
(function (global) {
  "use strict";

  var GBDT = { name: "GBDT (HistGradientBoosting)", color: "#1e6b3c",
    mean: [0.7865, 0.7828, 0.7863, 0.7911, 0.7856, 0.7740, 0.7748, 0.7748, 0.7659, 0.7756, 0.7756],
    std:  [0.0282, 0.0254, 0.0301, 0.0324, 0.0329, 0.0263, 0.0294, 0.0294, 0.0371, 0.0339, 0.0339] };
  var LIN = { name: "Linear (logistic regression)", color: "#2e6fb0",
    mean: [0.7913, 0.7916, 0.7921, 0.7902, 0.7870, 0.7868, 0.7865, 0.7946, 0.7955, 0.7950, 0.7974],
    std:  [0.0238, 0.0216, 0.0217, 0.0207, 0.0271, 0.0263, 0.0264, 0.0193, 0.0189, 0.0183, 0.0185] };
  var N = GBDT.mean.length;      // 0..10
  var STOP = 3;                  // GBDT peak — "where I'd stop"

  var svgNS = "http://www.w3.org/2000/svg";
  function el(name, attrs) {
    var e = document.createElementNS(svgNS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "fer-viz";

    var show = "both";   // "gbdt" | "linear" | "both"
    var band = true;
    var kSel = STOP;

    // ---- controls ----
    var ctl = document.createElement("div");
    ctl.className = "fer-ctl";
    var bothBtn = document.createElement("button"); bothBtn.type = "button"; bothBtn.textContent = "Both models";
    var gbdtBtn = document.createElement("button"); gbdtBtn.type = "button"; gbdtBtn.textContent = "GBDT only";
    var linBtn = document.createElement("button"); linBtn.type = "button"; linBtn.textContent = "Linear only";
    var bandBtn = document.createElement("button"); bandBtn.type = "button"; bandBtn.textContent = "Noise band: on";
    var slab = document.createElement("label"); slab.className = "fer-slab"; slab.textContent = "features k = ";
    var slider = document.createElement("input");
    slider.type = "range"; slider.min = "0"; slider.max = String(N - 1); slider.value = String(kSel);
    slider.className = "fer-slider";
    var kval = document.createElement("span"); kval.className = "fer-kval"; kval.textContent = String(kSel);
    slab.appendChild(slider); slab.appendChild(kval);
    ctl.appendChild(bothBtn); ctl.appendChild(gbdtBtn); ctl.appendChild(linBtn);
    ctl.appendChild(bandBtn); ctl.appendChild(slab);
    container.appendChild(ctl);

    var W = 480, H = 300;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", class: "fer-svg" });
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "fer-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "fer-caption";
    cap.textContent = config.caption ||
      "5-fold CV ROC-AUC on credit_g as hand-crafted features are added one at a time " +
      "(model fixed). The GBDT peaks at 3 features then declines as extra features overfit; " +
      "shaded ±1 CV-std bands show every change stays inside the noise envelope.";
    container.appendChild(cap);

    // plot geometry
    var PL = 52, PR = W - 14, PT = 18, PB = H - 40;
    function sx(k) { return PL + (k / (N - 1)) * (PR - PL); }
    var YLO = 0.74, YHI = 0.81;
    function sy(v) { return PB - (v - YLO) / (YHI - YLO) * (PB - PT); }

    function linePath(s) {
      var d = "M";
      for (var i = 0; i < N; i++) d += (i ? "L" : "") + sx(i).toFixed(1) + "," + sy(s.mean[i]).toFixed(1) + " ";
      return el("path", { d: d, fill: "none", stroke: s.color, "stroke-width": 2.4 });
    }
    function bandPath(s) {
      var d = "M", i;
      for (i = 0; i < N; i++) d += (i ? "L" : "") + sx(i).toFixed(1) + "," + sy(s.mean[i] + s.std[i]).toFixed(1) + " ";
      for (i = N - 1; i >= 0; i--) d += "L" + sx(i).toFixed(1) + "," + sy(s.mean[i] - s.std[i]).toFixed(1) + " ";
      d += "Z";
      return el("path", { d: d, fill: s.color, opacity: 0.12, stroke: "none" });
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      svg.appendChild(el("rect", { x: PL, y: PT, width: PR - PL, height: PB - PT,
        fill: "var(--bg)", stroke: "var(--border)", "stroke-width": 1 }));

      // y gridlines
      [0.74, 0.76, 0.78, 0.80].forEach(function (v) {
        var yy = sy(v);
        svg.appendChild(el("line", { x1: PL, y1: yy, x2: PR, y2: yy,
          stroke: "var(--border)", "stroke-width": 0.5, opacity: 0.6 }));
        svg.appendChild(el("text", { x: PL - 6, y: yy + 3, "text-anchor": "end", class: "fer-tick" }))
          .textContent = v.toFixed(2);
      });
      // x ticks 0..10
      for (var k = 0; k < N; k++) {
        var xx = sx(k);
        svg.appendChild(el("line", { x1: xx, y1: PB, x2: xx, y2: PB + 4, stroke: "var(--muted)", "stroke-width": 1 }));
        svg.appendChild(el("text", { x: xx, y: PB + 15, "text-anchor": "middle", class: "fer-tick" }))
          .textContent = String(k);
      }
      svg.appendChild(el("text", { x: (PL + PR) / 2, y: H - 3, "text-anchor": "middle", class: "fer-lab" }))
        .textContent = "hand-crafted features added (modeling effort →)";
      var yl = el("text", { x: 13, y: (PT + PB) / 2, "text-anchor": "middle", class: "fer-lab",
        transform: "rotate(-90 13 " + ((PT + PB) / 2) + ")" });
      yl.textContent = "5-fold CV ROC-AUC";
      svg.appendChild(yl);

      // "where I'd stop" marker
      var mx = sx(STOP);
      svg.appendChild(el("line", { x1: mx, y1: PT, x2: mx, y2: PB,
        stroke: "#b0632e", "stroke-width": 1.3, "stroke-dasharray": "4,3", opacity: 0.9 }));
      svg.appendChild(el("text", { x: mx + 4, y: PT + 11, class: "fer-stop" })).textContent = "stop here";

      // selected-k marker
      var kx = sx(kSel);
      svg.appendChild(el("line", { x1: kx, y1: PT, x2: kx, y2: PB,
        stroke: "var(--muted)", "stroke-width": 1, "stroke-dasharray": "2,2", opacity: 0.7 }));

      var series = show === "gbdt" ? [GBDT] : show === "linear" ? [LIN] : [LIN, GBDT];
      if (band) series.forEach(function (s) { svg.appendChild(bandPath(s)); });
      series.forEach(function (s) { svg.appendChild(linePath(s)); });
      // dots at selected k
      series.forEach(function (s) {
        svg.appendChild(el("circle", { cx: kx, cy: sy(s.mean[kSel]), r: 4.5,
          fill: s.color, stroke: "var(--bg)", "stroke-width": 1.5 }));
      });

      // legend
      var ly = PT + 8;
      series.forEach(function (s) {
        svg.appendChild(el("rect", { x: PR - 150, y: ly - 8, width: 10, height: 10, fill: s.color }));
        svg.appendChild(el("text", { x: PR - 136, y: ly + 1, class: "fer-leg" }))
          .textContent = s.name.indexOf("GBDT") === 0 ? "GBDT" : "Linear";
        ly += 15;
      });

      // readout
      function row(s) {
        var d = s.mean[kSel] - s.mean[0];
        var within = Math.abs(d) < s.std[kSel];
        return "<div class='fer-rrow'><span class='fer-dot' style='background:" + s.color + "'></span>" +
          "<strong>" + (s.name.indexOf("GBDT") === 0 ? "GBDT" : "Linear") + "</strong> " +
          s.mean[kSel].toFixed(4) + " ±" + s.std[kSel].toFixed(4) +
          " · Δ vs k=0 <strong class='" + (d >= 0 ? "fer-up" : "fer-down") + "'>" +
          (d >= 0 ? "+" : "") + d.toFixed(4) + "</strong> " +
          (within ? "<span class='fer-noise'>within noise</span>" : "<span class='fer-real'>exceeds 1σ</span>") +
          "</div>";
      }
      var head = "<div class='fer-rhead'>at <strong>k=" + kSel + "</strong> features" +
        (kSel === STOP ? " — the GBDT peak (where I'd stop)" :
         kSel > STOP ? " — past the peak: extra features are not paying off" : "") + "</div>";
      var body = (show === "gbdt" ? [GBDT] : show === "linear" ? [LIN] : [GBDT, LIN]).map(row).join("");
      readout.innerHTML = head + body;
    }

    function setShow(m) {
      show = m;
      bothBtn.classList.toggle("fer-on", m === "both");
      gbdtBtn.classList.toggle("fer-on", m === "gbdt");
      linBtn.classList.toggle("fer-on", m === "linear");
      draw();
    }
    bothBtn.addEventListener("click", function () { setShow("both"); });
    gbdtBtn.addEventListener("click", function () { setShow("gbdt"); });
    linBtn.addEventListener("click", function () { setShow("linear"); });
    bandBtn.addEventListener("click", function () {
      band = !band; bandBtn.textContent = "Noise band: " + (band ? "on" : "off"); draw();
    });
    slider.addEventListener("input", function () {
      kSel = Number(slider.value); kval.textContent = slider.value; draw();
    });

    setShow("both");

    return {
      setK: function (k) { kSel = k; slider.value = String(k); kval.textContent = String(k); draw(); },
      setModel: setShow,
      toggleBand: function () { band = !band; bandBtn.textContent = "Noise band: " + (band ? "on" : "off"); draw(); }
    };
  }

  global.FeReturnsViz = { mount: mount };
})(window);
