/**
 * Grinsztajn's §5.2 target-smoothing experiment (the "Finding 1" summary plot).
 *
 * On an irregular multi-D regression, Gaussian-smooth the TARGET at a growing length-scale h and
 * retrain both models. Plotting each model's test R^2 vs h shows the mechanism as one curve:
 * the tree's advantage lives entirely in the high-frequency structure, so smoothing the target
 * (which destroys that structure) collapses the GBT-vs-MLP gap toward zero.
 *
 * REAL numbers from labs/_verify_l025.py (mean over 5 seeds, irregular 8-D regression, GBT =
 * HistGradientBoostingRegressor, MLP = 256x256). The "variance kept" annotation is the fraction of
 * the target's variance surviving the smoothing — the gap closes exactly as the variance is erased.
 *
 * Usage: SmoothnessGapViz.mount(container, { caption })
 * Expected states:
 *   - h=0 (raw):    GBT 0.938 vs MLP 0.717  → gap +0.22 (tree exploits the jags).
 *   - h=1.0:        GBT 0.885 vs MLP 0.905  → gap ~0 once ~80% of the variance is smoothed away.
 *   - slider marks h and reads out both R^2 + the gap.
 */
(function (global) {
  "use strict";

  var SCALES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
  var GBT = [0.9381, 0.9264, 0.8853, 0.9142, 0.8935, 0.8191];
  var MLP = [0.7166, 0.6968, 0.9053, 0.9283, 0.8937, 0.8069];
  var VARKEPT = [1.00, 0.91, 0.19, 0.05, 0.02, 0.01];

  var svgNS = "http://www.w3.org/2000/svg";
  function el(name, attrs) {
    var e = document.createElementNS(svgNS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "smoothness-gap-viz";

    var idx = 0;
    var N = SCALES.length;

    var ctl = document.createElement("div");
    ctl.className = "sg-ctl";
    var slabel = document.createElement("label");
    slabel.className = "sg-slabel";
    slabel.textContent = "target smoothing length-scale h = ";
    var slider = document.createElement("input");
    slider.type = "range"; slider.min = "0"; slider.max = String(N - 1); slider.step = "1"; slider.value = "0";
    slider.className = "sg-slider";
    var hval = document.createElement("span");
    hval.className = "sg-hval"; hval.textContent = "0.0";
    slabel.appendChild(slider); slabel.appendChild(hval);
    ctl.appendChild(slabel);
    container.appendChild(ctl);

    var W = 460, H = 300;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", class: "sg-svg" });
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "sg-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "sg-caption";
    cap.textContent = config.caption ||
      "Test R² of a GBT and an MLP as the target is Gaussian-smoothed (mean of 5 seeds). The gap " +
      "closes as the target's high-frequency variance is destroyed — the tree's edge was that variance.";
    container.appendChild(cap);

    var PL = 44, PR = W - 14, PT = 18, PB = H - 40;
    var YLO = 0.65, YHI = 0.97;
    function sx(i) { return PL + (i / (N - 1)) * (PR - PL); }
    function sy(v) { return PB - (v - YLO) / (YHI - YLO) * (PB - PT); }

    function line(arr, color) {
      var d = "M";
      for (var i = 0; i < N; i++) d += (i ? "L" : "") + sx(i).toFixed(1) + "," + sy(arr[i]).toFixed(1) + " ";
      return el("path", { d: d, fill: "none", stroke: color, "stroke-width": 2.3 });
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      svg.appendChild(el("rect", { x: PL, y: PT, width: PR - PL, height: PB - PT,
        fill: "var(--bg)", stroke: "var(--border)", "stroke-width": 1 }));

      [0.7, 0.75, 0.8, 0.85, 0.9, 0.95].forEach(function (v) {
        var yy = sy(v);
        svg.appendChild(el("line", { x1: PL, y1: yy, x2: PR, y2: yy,
          stroke: "var(--border)", "stroke-width": 0.5, opacity: 0.6 }));
        svg.appendChild(el("text", { x: PL - 5, y: yy + 3, "text-anchor": "end", class: "sg-tick" }))
          .textContent = v.toFixed(2);
      });

      SCALES.forEach(function (h, i) {
        var xx = sx(i);
        svg.appendChild(el("line", { x1: xx, y1: PB, x2: xx, y2: PB + 4, stroke: "var(--muted)", "stroke-width": 1 }));
        svg.appendChild(el("text", { x: xx, y: PB + 15, "text-anchor": "middle", class: "sg-tick" }))
          .textContent = h.toFixed(1);
      });
      svg.appendChild(el("text", { x: (PL + PR) / 2, y: H - 4, "text-anchor": "middle", class: "sg-lab" }))
        .textContent = "target smoothing length-scale h (0 = raw, right = smoother)";
      var yl = el("text", { x: 12, y: (PT + PB) / 2, "text-anchor": "middle", class: "sg-lab",
        transform: "rotate(-90 12 " + ((PT + PB) / 2) + ")" });
      yl.textContent = "test R²";
      svg.appendChild(yl);

      var mx = sx(idx);
      svg.appendChild(el("line", { x1: mx, y1: PT, x2: mx, y2: PB,
        stroke: "var(--muted)", "stroke-width": 1, "stroke-dasharray": "3,3", opacity: 0.8 }));

      svg.appendChild(line(MLP, "#2e6fb0"));
      svg.appendChild(line(GBT, "#1e6b3c"));

      svg.appendChild(el("circle", { cx: mx, cy: sy(GBT[idx]), r: 4.5, fill: "#1e6b3c", stroke: "var(--bg)", "stroke-width": 1.5 }));
      svg.appendChild(el("circle", { cx: mx, cy: sy(MLP[idx]), r: 4.5, fill: "#2e6fb0", stroke: "var(--bg)", "stroke-width": 1.5 }));

      svg.appendChild(el("rect", { x: PR - 92, y: PT + 6, width: 10, height: 10, fill: "#1e6b3c" }));
      svg.appendChild(el("text", { x: PR - 78, y: PT + 15, class: "sg-leg" })).textContent = "GBT";
      svg.appendChild(el("rect", { x: PR - 92, y: PT + 22, width: 10, height: 10, fill: "#2e6fb0" }));
      svg.appendChild(el("text", { x: PR - 78, y: PT + 31, class: "sg-leg" })).textContent = "MLP";

      var gap = GBT[idx] - MLP[idx];
      var sign = gap >= 0 ? "+" : "";
      readout.innerHTML =
        "<span class='sg-pill'>h = " + SCALES[idx].toFixed(1) + "</span> " +
        "GBT R² <strong class='sg-gbt'>" + GBT[idx].toFixed(3) + "</strong> vs " +
        "MLP R² <strong class='sg-mlp'>" + MLP[idx].toFixed(3) + "</strong> · " +
        "gap <strong>" + sign + gap.toFixed(3) + "</strong> · " +
        "variance kept <strong>" + Math.round(VARKEPT[idx] * 100) + "%</strong>" +
        (idx === 0
          ? " — the raw irregular target: the tree wins big."
          : (gap < 0.05 ? " — the jags are gone; the gap has closed." : " — smoothing away the jags."));
    }

    slider.addEventListener("input", function () {
      idx = Number(slider.value);
      hval.textContent = SCALES[idx].toFixed(1);
      draw();
    });

    draw();
    return { setIdx: function (i) { idx = i; slider.value = String(i); hval.textContent = SCALES[i].toFixed(1); draw(); } };
  }

  global.SmoothnessGapViz = { mount: mount };
})(window);
