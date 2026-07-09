/**
 * Concept-drift visualizer: the rule the model must learn ROTATES over time.
 *
 * Left panel: grouped bars of each feature's correlation with the label across
 * 5 time buckets (x0 fades, x1 rises, x2 stays flat) — the verified numbers from
 * labs/_verify_l021.py. Right panel: the decision-boundary NORMAL vector rotating
 * from the x0 axis toward the x1 axis as the selected bucket advances. A slider
 * selects the "current" time bucket (same mechanism under one knob).
 *
 * Plain <script> (file://-safe). Usage: DriftViz.mount(el, config?).
 */
(function (global) {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var C0 = "#2e6fb0"; // x0
  var C1 = "#d68910"; // x1
  var C2 = "#9aa5b1"; // x2 (stable)

  // verified corr(feature, y) per bucket (labs/_verify_l021.py)
  var X0 = [0.72, 0.59, 0.48, 0.34, 0.12];
  var X1 = [0.10, 0.34, 0.49, 0.62, 0.71];
  var X2 = [0.29, 0.30, 0.30, 0.32, 0.31];
  var LABELS = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"];

  function el(tag, attrs) {
    var n = document.createElementNS(SVGNS, tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    return n;
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.classList.add("drift-viz");

    var W = 560, H = 220;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, class: "dv-svg", width: "100%" });
    container.appendChild(svg);

    // ---- left: grouped bars ----
    var bx = 40, by = 24, bw = 300, bh = 150; // plot box
    var maxV = 0.8;
    var nb = 5;
    var groupW = bw / nb;
    var barW = groupW / 4;

    // axes
    svg.appendChild(el("line", { x1: bx, y1: by, x2: bx, y2: by + bh, stroke: "#c9c9c9", "stroke-width": 1 }));
    svg.appendChild(el("line", { x1: bx, y1: by + bh, x2: bx + bw, y2: by + bh, stroke: "#c9c9c9", "stroke-width": 1 }));
    var yl = el("text", { x: bx - 6, y: by - 10, class: "dv-axis" }); yl.textContent = "corr(feature, y)"; svg.appendChild(yl);

    // bucket highlight (drawn BEFORE bars so it sits behind them)
    var groupW0 = bw / 5;
    var bucketHi = el("rect", { x: bx, y: by, width: groupW0, height: bh, fill: "#000", "fill-opacity": 0.06 });
    svg.appendChild(bucketHi);

    var bars = []; // {rect, feat, bucket}
    function barH(v) { return (v / maxV) * bh; }
    var b, feats = [X0, X1, X2], cols = [C0, C1, C2];
    for (b = 0; b < nb; b++) {
      var gx = bx + b * groupW + groupW * 0.12;
      for (var f = 0; f < 3; f++) {
        var v = feats[f][b];
        var h = barH(v);
        var rect = el("rect", {
          x: gx + f * barW, y: by + bh - h, width: barW - 1.5, height: h,
          fill: cols[f], "fill-opacity": 0.85,
        });
        svg.appendChild(rect);
        bars.push({ rect: rect, f: f });
      }
      var xl = el("text", { x: bx + b * groupW + groupW / 2, y: by + bh + 14, class: "dv-tick", "text-anchor": "middle" });
      xl.textContent = LABELS[b];
      svg.appendChild(xl);
    }

    // ---- right: rotating normal vector ----
    var cxp = 460, cyp = 100, R = 66;
    svg.appendChild(el("circle", { cx: cxp, cy: cyp, r: R, fill: "none", stroke: "#e0e0e0", "stroke-width": 1 }));
    // axes for x0 (horizontal) and x1 (vertical)
    svg.appendChild(el("line", { x1: cxp - R - 6, y1: cyp, x2: cxp + R + 6, y2: cyp, stroke: C0, "stroke-width": 1, "stroke-opacity": 0.5 }));
    svg.appendChild(el("line", { x1: cxp, y1: cyp + R + 6, x2: cxp, y2: cyp - R - 6, stroke: C1, "stroke-width": 1, "stroke-opacity": 0.5 }));
    var lx0 = el("text", { x: cxp + R + 8, y: cyp + 4, class: "dv-mini", fill: C0 }); lx0.textContent = "x0"; svg.appendChild(lx0);
    var lx1 = el("text", { x: cxp - 6, y: cyp - R - 10, class: "dv-mini", fill: C1, "text-anchor": "middle" }); lx1.textContent = "x1"; svg.appendChild(lx1);
    var arrow = el("line", { x1: cxp, y1: cyp, x2: cxp + R, y2: cyp, stroke: "#111", "stroke-width": 3 });
    svg.appendChild(arrow);
    var head = el("circle", { cx: cxp + R, cy: cyp, r: 4, fill: "#111" });
    svg.appendChild(head);
    var rt = el("text", { x: cxp, y: cyp + R + 26, class: "dv-mini", "text-anchor": "middle" }); rt.textContent = "rule direction"; svg.appendChild(rt);

    // ---- controls + readout ----
    var controls = document.createElement("div");
    controls.className = "dv-controls";
    container.appendChild(controls);
    var lab = document.createElement("label");
    lab.className = "dv-label";
    lab.textContent = "Time bucket: ";
    var slider = document.createElement("input");
    slider.type = "range"; slider.min = "0"; slider.max = "4"; slider.step = "1"; slider.value = "0";
    slider.className = "dv-slider";
    lab.appendChild(slider);
    controls.appendChild(lab);

    var readout = document.createElement("div");
    readout.className = "dv-readout";
    container.appendChild(readout);

    function render(bi) {
      bucketHi.setAttribute("x", bx + bi * groupW);
      // rotate arrow: angle from x0 axis (0) to x1 axis (90deg) proportional to corr mix
      var theta = (bi / (nb - 1)) * (Math.PI / 2); // 0 -> 90deg
      var ex = cxp + R * Math.cos(theta);
      var ey = cyp - R * Math.sin(theta);
      arrow.setAttribute("x2", ex); head.setAttribute("cx", ex); head.setAttribute("cy", ey);
      // emphasize the two moving bars in the selected bucket
      bars.forEach(function (obj, idx) {
        var grp = Math.floor(idx / 3);
        obj.rect.setAttribute("fill-opacity", grp === bi ? 0.95 : 0.45);
      });
      readout.innerHTML =
        "Bucket <strong>" + LABELS[bi] + "</strong>: corr(x0,y)=<strong style='color:" + C0 + "'>" + X0[bi].toFixed(2) +
        "</strong>, corr(x1,y)=<strong style='color:" + C1 + "'>" + X1[bi].toFixed(2) +
        "</strong>, corr(x2,y)=" + X2[bi].toFixed(2) +
        ". The rule rotates from x0 toward x1; a model trained only on early buckets learns to trust x0 and is stale by the late buckets.";
    }
    slider.addEventListener("input", function () { render(parseInt(slider.value, 10)); });
    render(0);

    return { render: render, x0: X0, x1: X1 };
  }

  global.DriftViz = { mount: mount };
})(window);
