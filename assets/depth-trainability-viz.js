/**
 * Depth trainability — plain MLP vs ResNet as depth grows (the skip-connection payoff).
 *
 * REAL numbers from labs/_verify_l028.py (synthetic n=8000, 32 features, dropout=0, BatchNorm on
 * both, AdamW 60 epochs; identical architecture with the residual skip ON vs OFF):
 *   depths       = [1, 2, 4, 8, 16, 32]
 *   plain_test   = [0.917, 0.905, 0.900, 0.885, 0.884, 0.866]
 *   resnet_test  = [0.912, 0.897, 0.909, 0.902, 0.897, 0.899]
 *   plain_train  = [1.000, 0.998, 0.994, 0.985, 0.981, 0.927]
 *   resnet_train = [1.000, 0.998, 0.999, 1.000, 0.997, 0.997]
 *
 * The plain net degrades with depth — and its TRAIN accuracy falls too (1.000 -> 0.927), so the
 * failure is optimization (the "degradation problem", He et al. 2015), NOT overfitting. Both nets
 * use BatchNorm, so it is not vanishing gradients. The ResNet holds flat because the skip makes
 * the identity free.
 *
 * Usage: DepthTrainabilityViz.mount(container, { caption })
 * Expected states:
 *   - default: TEST-accuracy lines for plain (blue, sloping down) and resnet (green, flat).
 *   - "Show training accuracy" toggle: adds dashed train-accuracy lines; plain's train line
 *     also sags at depth 32 (the degradation signature); resnet's train line stays ~1.0.
 */
(function (global) {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var PLAIN = "#2e6fb0", RESNET = "#1e6b3c";

  var DEPTHS = [1, 2, 4, 8, 16, 32];
  var PLAIN_TEST = [0.917, 0.905, 0.900, 0.885, 0.884, 0.866];
  var RESNET_TEST = [0.912, 0.897, 0.909, 0.902, 0.897, 0.899];
  var PLAIN_TRAIN = [1.000, 0.998, 0.994, 0.985, 0.981, 0.927];
  var RESNET_TRAIN = [1.000, 0.998, 0.999, 1.000, 0.997, 0.997];

  function el(name, attrs, text) {
    var e = document.createElementNS(SVGNS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "depth-trainability-viz";

    var showTrain = false;

    var ctl = document.createElement("div");
    ctl.className = "dtr-ctl";
    var btn = document.createElement("button");
    btn.textContent = "Show training accuracy";
    ctl.appendChild(btn);
    container.appendChild(ctl);

    var W = 480, H = 320;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", class: "dtr-svg" });
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "dtr-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "dtr-caption";
    cap.textContent = config.caption ||
      "Test accuracy vs depth (number of blocks) for the same architecture with the residual " +
      "skip ON (ResNet) vs OFF (plain MLP). Synthetic data, mean over the run; real numbers from the lab.";
    container.appendChild(cap);

    var PL = 46, PR = W - 16, PT = 20, PB = H - 44;
    var YLO = 0.84, YHI = 1.01;
    function sy(v) { return PB - (v - YLO) / (YHI - YLO) * (PB - PT); }
    function sx(i) { return PL + i / (DEPTHS.length - 1) * (PR - PL); }

    function polyline(vals, color, dash) {
      var pts = vals.map(function (v, i) { return sx(i) + "," + sy(v); }).join(" ");
      var attrs = { points: pts, fill: "none", stroke: color, "stroke-width": 2.2 };
      if (dash) attrs["stroke-dasharray"] = "5,4";
      return el("polyline", attrs);
    }

    function dots(vals, color) {
      vals.forEach(function (v, i) {
        svg.appendChild(el("circle", { cx: sx(i), cy: sy(v), r: 3, fill: color }));
      });
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      svg.appendChild(el("rect", { x: PL, y: PT, width: PR - PL, height: PB - PT,
        fill: "var(--bg)", stroke: "var(--border)", "stroke-width": 1 }));

      [0.85, 0.90, 0.95, 1.00].forEach(function (v) {
        var yy = sy(v);
        svg.appendChild(el("line", { x1: PL, y1: yy, x2: PR, y2: yy, stroke: "var(--border)", "stroke-width": 0.5, opacity: 0.6 }));
        svg.appendChild(el("text", { x: PL - 6, y: yy + 3, "text-anchor": "end", class: "dtr-tick" }, v.toFixed(2)));
      });
      DEPTHS.forEach(function (d, i) {
        svg.appendChild(el("text", { x: sx(i), y: PB + 15, "text-anchor": "middle", class: "dtr-tick" }, d));
      });
      svg.appendChild(el("text", { x: (PL + PR) / 2, y: H - 6, "text-anchor": "middle", class: "dtr-lab" }, "depth (number of blocks)"));
      svg.appendChild(el("text", { x: 12, y: (PT + PB) / 2, "text-anchor": "middle", class: "dtr-lab",
        transform: "rotate(-90 12 " + ((PT + PB) / 2) + ")" }, "accuracy"));

      if (showTrain) {
        svg.appendChild(polyline(PLAIN_TRAIN, PLAIN, true));
        svg.appendChild(polyline(RESNET_TRAIN, RESNET, true));
      }
      svg.appendChild(polyline(PLAIN_TEST, PLAIN, false));
      svg.appendChild(polyline(RESNET_TEST, RESNET, false));
      dots(PLAIN_TEST, PLAIN);
      dots(RESNET_TEST, RESNET);

      // legend
      svg.appendChild(el("rect", { x: PR - 128, y: PT + 4, width: 12, height: 3, fill: RESNET }));
      svg.appendChild(el("text", { x: PR - 112, y: PT + 10, class: "dtr-leg" }, "ResNet (skip)"));
      svg.appendChild(el("rect", { x: PR - 128, y: PT + 18, width: 12, height: 3, fill: PLAIN }));
      svg.appendChild(el("text", { x: PR - 112, y: PT + 24, class: "dtr-leg" }, "plain MLP"));

      readout.innerHTML = showTrain
        ? "Dashed = <strong>training</strong> accuracy. The plain net's train accuracy falls " +
          "<strong>1.000 → 0.927</strong> as depth goes 1 → 32: it cannot even fit the training set, " +
          "so this is an <strong>optimization (degradation) problem, not overfitting</strong>. The " +
          "ResNet's train accuracy stays <strong>~1.00</strong> at every depth — the skip keeps it trainable."
        : "Solid = <strong>test</strong> accuracy. Adding depth <em>hurts</em> the plain MLP " +
          "(<strong>0.917 → 0.866</strong>) but leaves the ResNet essentially flat " +
          "(<strong>0.912 → 0.899</strong>). Same architecture, same BatchNorm — the only difference " +
          "is the skip connection. Toggle the training curves to see it is a degradation problem.";
    }

    btn.addEventListener("click", function () {
      showTrain = !showTrain;
      btn.textContent = showTrain ? "Hide training accuracy" : "Show training accuracy";
      btn.classList.toggle("dtr-on", showTrain);
      draw();
    });

    draw();
    return { setTrain: function (v) { showTrain = v; btn.classList.toggle("dtr-on", v); draw(); } };
  }

  global.DepthTrainabilityViz = { mount: mount };
})(window);
