/**
 * Reusable reliability-diagram (calibration) visualizer.
 *
 * Plots binned (predicted probability -> observed frequency) points against the
 * y = x diagonal. A perfectly calibrated model sits on the diagonal. Toggle
 * "raw" (a sigmoid-distorted model, like an out-of-the-box RandomForest) vs
 * "calibrated" (points snapped onto the diagonal) to see what calibration does.
 *
 * Data are the verified reliability readings from Lesson 008 (sklearn 1.9,
 * RandomForest, 17.7% positive): predicted bin center -> observed fraction.
 *
 * Usage: ReliabilityViz.mount(containerElement)
 * Companion CSS lives in the lesson <style> block (class prefix .rel-viz).
 */
(function (global) {
  "use strict";

  // Verified raw RandomForest reliability curve (uniform 10 bins).
  var RAW = [
    [0.051, 0.042], [0.143, 0.088], [0.247, 0.161], [0.349, 0.382],
    [0.442, 0.577], [0.548, 0.724], [0.648, 0.884], [0.750, 0.929],
    [0.843, 0.982], [0.921, 1.000]
  ];

  function mount(container) {
    container.innerHTML = "";
    container.classList.add("rel-viz");

    var controls = document.createElement("div");
    controls.className = "rel-viz-controls";
    container.appendChild(controls);

    var W = 320, H = 320, pad = 36;
    var svgNS = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("class", "rel-viz-svg");
    container.appendChild(svg);

    var caption = document.createElement("p");
    caption.className = "rel-viz-caption";
    container.appendChild(caption);

    function x(p) { return pad + p * (W - 2 * pad); }
    function y(p) { return (H - pad) - p * (H - 2 * pad); }

    var current = "raw";

    ["raw", "calibrated"].forEach(function (mode) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "rel-viz-btn";
      btn.textContent = mode === "raw" ? "Raw model" : "Calibrated";
      btn.addEventListener("click", function () { current = mode; render(); });
      controls.appendChild(btn);
    });

    function line(x1, y1, x2, y2, cls) {
      var l = document.createElementNS(svgNS, "line");
      l.setAttribute("x1", x1); l.setAttribute("y1", y1);
      l.setAttribute("x2", x2); l.setAttribute("y2", y2);
      l.setAttribute("class", cls);
      return l;
    }
    function text(tx, ty, str, cls) {
      var t = document.createElementNS(svgNS, "text");
      t.setAttribute("x", tx); t.setAttribute("y", ty);
      t.setAttribute("class", cls);
      t.textContent = str;
      return t;
    }

    function render() {
      svg.innerHTML = "";

      Array.prototype.forEach.call(controls.children, function (b) {
        var isActive = (b.textContent === "Raw model" && current === "raw") ||
                       (b.textContent === "Calibrated" && current === "calibrated");
        b.classList.toggle("active", isActive);
      });

      // axes
      svg.appendChild(line(pad, H - pad, W - pad, H - pad, "rel-axis"));
      svg.appendChild(line(pad, pad, pad, H - pad, "rel-axis"));
      // perfect-calibration diagonal
      svg.appendChild(line(x(0), y(0), x(1), y(1), "rel-diag"));
      svg.appendChild(text(W / 2 - 30, H - 8, "predicted probability", "rel-lbl"));
      var yl = text(12, H / 2 + 40, "observed frequency", "rel-lbl");
      yl.setAttribute("transform", "rotate(-90 12 " + (H / 2) + ")");
      svg.appendChild(yl);

      // points + connecting path
      var pts = RAW.map(function (d) {
        var pred = d[0];
        var obs = current === "calibrated" ? pred : d[1];
        return [pred, obs];
      });

      var d = "M";
      pts.forEach(function (p, i) { d += (i ? " L" : "") + x(p[0]) + " " + y(p[1]); });
      var path = document.createElementNS(svgNS, "path");
      path.setAttribute("d", d);
      path.setAttribute("class", "rel-path " + current);
      svg.appendChild(path);

      pts.forEach(function (p) {
        var c = document.createElementNS(svgNS, "circle");
        c.setAttribute("cx", x(p[0])); c.setAttribute("cy", y(p[1]));
        c.setAttribute("r", 4);
        c.setAttribute("class", "rel-dot " + current);
        svg.appendChild(c);
      });

      caption.innerHTML = current === "raw"
        ? "<strong>Raw model.</strong> Points bow below the diagonal in the mid-range \u2014 " +
          "where it predicts 0.55, the event actually happens 72% of the time. The scores " +
          "rank well but are not trustworthy as probabilities."
        : "<strong>Calibrated.</strong> Platt (sigmoid) or isotonic mapping pulls each point " +
          "onto the diagonal: now a predicted 0.7 means roughly a 70% chance. Ranking is " +
          "unchanged \u2014 only the probability values move.";
    }

    render();
  }

  global.ReliabilityViz = { mount: mount };
})(window);
