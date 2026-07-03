/**
 * Reusable feature-engineering visualizer.
 *
 * Shows the central idea of Lesson 009: structure a model cannot see in the raw
 * coordinates becomes obvious after the right engineered feature. The target is
 * a "ratio of differences" y = (a - b) / (c - d) (Heaton 2016 — the one feature
 * none of NN/RF/GBDT/SVM could synthesize on their own).
 *
 *  - "Raw feature":  y plotted against a single raw column (a) -> a shapeless
 *                    cloud; no axis-aligned model can carve the relationship out.
 *  - "Engineered":   y plotted against the engineered ratio (a-b)/(c-d) -> a tight
 *                    monotonic line a linear model fits almost perfectly.
 *
 * Points are generated deterministically (seeded LCG) so the picture is stable
 * and matches the verified Lesson 009 story (Ridge raw R2 0.64 -> 1.00 with the
 * ratio engineered in).
 *
 * Usage: FeatureViz.mount(containerElement)
 * Companion CSS lives in the lesson <style> block (class prefix .feat-viz).
 */
(function (global) {
  "use strict";

  // Deterministic linear-congruential generator so the scatter never changes.
  function makePoints() {
    var seed = 1234567;
    function rnd() { seed = (1103515245 * seed + 12345) % 2147483648; return seed / 2147483648; }
    var pts = [];
    for (var i = 0; i < 90; i++) {
      var a = 1 + rnd() * 8;
      var b = 1 + rnd() * 8;
      var c = 1 + rnd() * 8;
      var d = 9.5 + rnd() * 1.5;            // c - d in ~[-10, -0.5]
      var ratio = (a - b) / (c - d);
      var y = ratio + (rnd() - 0.5) * 0.1;  // small noise
      pts.push({ a: a, ratio: ratio, y: y });
    }
    return pts;
  }

  function mount(container) {
    container.innerHTML = "";
    container.classList.add("feat-viz");

    var controls = document.createElement("div");
    controls.className = "feat-viz-controls";
    container.appendChild(controls);

    var W = 340, H = 300, pad = 40;
    var svgNS = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("class", "feat-viz-svg");
    container.appendChild(svg);

    var caption = document.createElement("p");
    caption.className = "feat-viz-caption";
    container.appendChild(caption);

    var PTS = makePoints();
    var current = "raw";

    ["raw", "engineered"].forEach(function (mode) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "feat-viz-btn";
      btn.textContent = mode === "raw" ? "Raw feature (a)" : "Engineered ratio";
      btn.addEventListener("click", function () { current = mode; render(); });
      controls.appendChild(btn);
    });

    function text(tx, ty, str, cls) {
      var t = document.createElementNS(svgNS, "text");
      t.setAttribute("x", tx); t.setAttribute("y", ty);
      t.setAttribute("class", cls);
      t.textContent = str;
      return t;
    }
    function line(x1, y1, x2, y2, cls) {
      var l = document.createElementNS(svgNS, "line");
      l.setAttribute("x1", x1); l.setAttribute("y1", y1);
      l.setAttribute("x2", x2); l.setAttribute("y2", y2);
      l.setAttribute("class", cls);
      return l;
    }

    function render() {
      svg.innerHTML = "";

      Array.prototype.forEach.call(controls.children, function (b) {
        var isActive = (b.textContent.indexOf("Raw") === 0 && current === "raw") ||
                       (b.textContent.indexOf("Engineered") === 0 && current === "engineered");
        b.classList.toggle("active", isActive);
      });

      var xField = current === "raw" ? "a" : "ratio";
      var xs = PTS.map(function (p) { return p[xField]; });
      var ys = PTS.map(function (p) { return p.y; });
      var xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs);
      var ymin = Math.min.apply(null, ys), ymax = Math.max.apply(null, ys);
      function X(v) { return pad + (v - xmin) / (xmax - xmin) * (W - 2 * pad); }
      function Y(v) { return (H - pad) - (v - ymin) / (ymax - ymin) * (H - 2 * pad); }

      svg.appendChild(line(pad, H - pad, W - pad, H - pad, "feat-axis"));
      svg.appendChild(line(pad, pad, pad, H - pad, "feat-axis"));
      svg.appendChild(text(W / 2 - 40, H - 8,
        current === "raw" ? "raw feature  a" : "engineered  (a\u2212b)/(c\u2212d)", "feat-lbl"));
      var yl = text(12, H / 2 + 12, "target  y", "feat-lbl");
      yl.setAttribute("transform", "rotate(-90 12 " + (H / 2) + ")");
      svg.appendChild(yl);

      PTS.forEach(function (p) {
        var c = document.createElementNS(svgNS, "circle");
        c.setAttribute("cx", X(p[xField])); c.setAttribute("cy", Y(p.y));
        c.setAttribute("r", 3.2);
        c.setAttribute("class", "feat-dot " + current);
        svg.appendChild(c);
      });

      caption.innerHTML = current === "raw"
        ? "<strong>Raw feature.</strong> Plotted against any single raw column the target is a " +
          "shapeless cloud \u2014 the signal is split across four columns as a ratio of " +
          "differences. A linear model scores R\u00b2 \u2248 0.64; no axis-aligned split sees it cleanly."
        : "<strong>Engineered ratio.</strong> Compute (a\u2212b)/(c\u2212d) once and the relationship " +
          "is a near-perfect line \u2014 the same linear model jumps to R\u00b2 \u2248 1.00. You " +
          "encoded the structure the model could not synthesize itself.";
    }

    render();
  }

  global.FeatureViz = { mount: mount };
})(window);
