/**
 * Decision-tree partition visualizer (depth-1 axis-aligned split).
 * Usage: TreeViz.mount(container, { featureNames, points, labels, splitFeature, splitValue })
 * Expected: left/right regions colored by majority class; vertical/horizontal split line.
 */
(function (global) {
  "use strict";

  function mount(container, config) {
    var names = config.featureNames || ["x0", "x1"];
    var points = config.points;
    var labels = config.labels;
    var sf = config.splitFeature;
    var sv = config.splitValue;

    container.innerHTML = "";
    container.className = "tree-viz";

    var intro = document.createElement("p");
    intro.className = "tree-viz-intro";
    intro.textContent =
      "Depth-1 tree: if " + names[sf] + " ≤ " + sv.toFixed(2) + " go left, else right.";
    container.appendChild(intro);

    var canvas = document.createElement("canvas");
    canvas.width = 320;
    canvas.height = 220;
    canvas.className = "tree-viz-canvas";
    container.appendChild(canvas);

    var ctx = canvas.getContext("2d");
    var pad = 28;
    var w = canvas.width - pad * 2;
    var h = canvas.height - pad * 2;

    var xs = points.map(function (p) { return p[0]; });
    var ys = points.map(function (p) { return p[1]; });
    var xmin = Math.min.apply(null, xs);
    var xmax = Math.max.apply(null, xs);
    var ymin = Math.min.apply(null, ys);
    var ymax = Math.max.apply(null, ys);

    function sx(x) { return pad + ((x - xmin) / (xmax - xmin || 1)) * w; }
    function sy(y) { return pad + h - ((y - ymin) / (ymax - ymin || 1)) * h; }

    ctx.fillStyle = "#f8f9fa";
    ctx.fillRect(pad, pad, w, h);
    ctx.strokeStyle = "#d5d8dc";
    ctx.strokeRect(pad, pad, w, h);

    points.forEach(function (p, i) {
      ctx.beginPath();
      ctx.arc(sx(p[0]), sy(p[1]), 4, 0, Math.PI * 2);
      ctx.fillStyle = labels[i] ? "#1a5276" : "#95a5a6";
      ctx.fill();
    });

    ctx.strokeStyle = "#9b2226";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    if (sf === 0) {
      var lx = sx(sv);
      ctx.beginPath();
      ctx.moveTo(lx, pad);
      ctx.lineTo(lx, pad + h);
      ctx.stroke();
    } else {
      var ly = sy(sv);
      ctx.beginPath();
      ctx.moveTo(pad, ly);
      ctx.lineTo(pad + w, ly);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    var cap = document.createElement("p");
    cap.className = "tree-viz-caption";
    cap.textContent = config.caption ||
      "Each split partitions the feature space into rectangles — the building block of GBDT.";
    container.appendChild(cap);
  }

  global.TreeViz = { mount: mount };
})(window);
