/**
 * Grid vs random hyperparameter search (Bergstra & Bengio 2012, JMLR).
 *
 * The iconic figure: a 2-D search space where ONE axis (horizontal, "important")
 * strongly changes the objective (green peaked marginal drawn on top) and the
 * other axis (vertical, "unimportant") barely matters (near-flat marginal on the
 * left). Under an EQUAL budget of n trials:
 *   - GRID lays points on a k x k lattice, so it only tries k = sqrt(n) DISTINCT
 *     values on the important axis (many points share the same x -> wasted).
 *   - RANDOM draws n independent points, so it tries ~n distinct values on the
 *     important axis and samples the peak far more densely.
 * Dashed lines project every trial onto the important axis; ticks show the
 * distinct values actually tried there. The readout reports the best value of
 * the important marginal that each strategy found.
 *
 * This is why random >= grid once a few knobs are irrelevant ("low effective
 * dimensionality"): grid's per-axis resolution collapses as dimensions grow,
 * random's does not.
 *
 * Usage: SearchViz.mount(container, { caption })
 * Expected states:
 *   - Grid, budget 9: 3 distinct important-axis values; may or may not land near
 *     the peak (a lucky 3-point axis can do fine in 2-D).
 *   - Random, budget 9: ~9 distinct important-axis values; denser peak coverage.
 *   - Raise the budget: both improve; distinct-value gap (k vs n) stays.
 *   - Resample (random only): the point cloud + best-found change with the draw.
 */
(function (global) {
  "use strict";

  var SIGMA = 0.09; // width of the peak on the important axis (peak at 0.5)

  function importance(x) {
    return Math.exp(-((x - 0.5) * (x - 0.5)) / (2 * SIGMA * SIGMA));
  }

  function lcg(seed) {
    var s = seed || 1;
    return function () {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      return s / 0x7fffffff;
    };
  }

  function gridPoints(budget) {
    var k = Math.max(1, Math.round(Math.sqrt(budget)));
    var pts = [];
    for (var i = 0; i < k; i++) {
      for (var j = 0; j < k; j++) {
        pts.push({ x: (i + 0.5) / k, y: (j + 0.5) / k });
      }
    }
    return { pts: pts, distinct: k };
  }

  function randomPoints(budget, seed) {
    var r = lcg(seed);
    var pts = [];
    var xs = {};
    for (var i = 0; i < budget; i++) {
      var x = r(), y = r();
      pts.push({ x: x, y: y });
      xs[x.toFixed(4)] = true;
    }
    return { pts: pts, distinct: Object.keys(xs).length };
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "search-viz";

    var mode = "grid";
    var budget = 9;
    var seed = 3;

    // controls
    var ctl = document.createElement("div");
    ctl.className = "sv-ctl";
    var gridBtn = document.createElement("button");
    gridBtn.type = "button";
    gridBtn.textContent = "Grid";
    var randBtn = document.createElement("button");
    randBtn.type = "button";
    randBtn.textContent = "Random";
    var sel = document.createElement("select");
    [9, 16, 25].forEach(function (b) {
      var o = document.createElement("option");
      o.value = String(b);
      o.textContent = "budget " + b;
      sel.appendChild(o);
    });
    var resample = document.createElement("button");
    resample.type = "button";
    resample.textContent = "Resample";
    ctl.appendChild(gridBtn);
    ctl.appendChild(randBtn);
    ctl.appendChild(sel);
    ctl.appendChild(resample);
    container.appendChild(ctl);

    var svgNS = "http://www.w3.org/2000/svg";
    var W = 320, H = 300;
    var svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("width", "100%");
    svg.setAttribute("class", "sv-svg");
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "sv-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "sv-caption";
    cap.textContent = config.caption ||
      "Green marginal on top = the important knob (sharp peak at the middle); the " +
      "left marginal is the near-useless knob. Dashed lines project each trial onto " +
      "the important axis. Grid reuses only sqrt(n) values there; random tries ~n.";
    container.appendChild(cap);

    // plot geometry
    var PL = 52, PR = W - 16, PT = 56, PB = H - 40; // plot rectangle
    function sx(x) { return PL + x * (PR - PL); }
    function sy(y) { return PB - y * (PB - PT); } // y up

    function el(name, attrs) {
      var e = document.createElementNS(svgNS, name);
      for (var k in attrs) e.setAttribute(k, attrs[k]);
      return e;
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      // plot border
      svg.appendChild(el("rect", {
        x: PL, y: PT, width: PR - PL, height: PB - PT,
        fill: "var(--bg)", stroke: "var(--border)", "stroke-width": 1,
      }));

      // top marginal: importance(x) peaked curve
      var topH = 38, topBase = PT - 6, topTop = PT - 6 - topH;
      var path = "M";
      for (var i = 0; i <= 60; i++) {
        var x = i / 60;
        var px = sx(x);
        var py = topBase - importance(x) * topH;
        path += (i ? "L" : "") + px.toFixed(1) + "," + py.toFixed(1) + " ";
      }
      svg.appendChild(el("path", { d: path, fill: "none", stroke: "#1e6b3c", "stroke-width": 2 }));
      svg.appendChild(el("text", {
        x: (PL + PR) / 2, y: topTop - 2, "text-anchor": "middle", class: "sv-lab",
      })).textContent = "important knob  →  large effect";

      // left marginal: near-flat curve
      var leftW = 34, leftBase = PL - 8;
      var lpath = "M";
      for (var j = 0; j <= 60; j++) {
        var yv = j / 60;
        var val = 0.5 + 0.12 * Math.cos(6 * yv); // near-flat
        var lx = leftBase - val * leftW;
        var ly = sy(yv);
        lpath += (j ? "L" : "") + lx.toFixed(1) + "," + ly.toFixed(1) + " ";
      }
      svg.appendChild(el("path", { d: lpath, fill: "none", stroke: "#b07a2e", "stroke-width": 2 }));
      var ylab = el("text", { x: 14, y: (PT + PB) / 2, "text-anchor": "middle", class: "sv-lab",
        transform: "rotate(-90 14 " + ((PT + PB) / 2) + ")" });
      ylab.textContent = "useless knob";
      svg.appendChild(ylab);

      // projection axis (bottom)
      svg.appendChild(el("line", {
        x1: PL, y1: PB + 10, x2: PR, y2: PB + 10, stroke: "var(--muted)", "stroke-width": 1,
      }));

      var data = mode === "grid" ? gridPoints(budget) : randomPoints(budget, seed);
      var pts = data.pts;

      // dashed projections + points
      var best = 0;
      for (var p = 0; p < pts.length; p++) {
        var cx = sx(pts[p].x), cy = sy(pts[p].y);
        svg.appendChild(el("line", {
          x1: cx, y1: cy, x2: cx, y2: PB + 10,
          stroke: "var(--muted)", "stroke-width": 0.75, "stroke-dasharray": "2,2", opacity: 0.55,
        }));
        svg.appendChild(el("circle", {
          cx: cx, cy: cy, r: 4,
          fill: mode === "grid" ? "#2e6fb0" : "#8e44ad", stroke: "var(--bg)", "stroke-width": 1,
        }));
        // tick on projection axis
        svg.appendChild(el("circle", {
          cx: cx, cy: PB + 10, r: 2.4,
          fill: mode === "grid" ? "#2e6fb0" : "#8e44ad",
        }));
        var v = importance(pts[p].x);
        if (v > best) best = v;
      }

      readout.innerHTML =
        "<span class='sv-pill'>" + (mode === "grid" ? "Grid" : "Random") + "</span> " +
        "budget <strong>" + budget + "</strong> · distinct values on the important axis: " +
        "<strong>" + data.distinct + "</strong> · best peak value found: <strong>" +
        best.toFixed(3) + "</strong> / 1.000";
    }

    function setMode(m) {
      mode = m;
      gridBtn.classList.toggle("sv-active", m === "grid");
      randBtn.classList.toggle("sv-active", m === "random");
      resample.style.visibility = m === "random" ? "visible" : "hidden";
      draw();
    }

    gridBtn.addEventListener("click", function () { setMode("grid"); });
    randBtn.addEventListener("click", function () { setMode("random"); });
    sel.addEventListener("change", function () { budget = Number(sel.value); draw(); });
    resample.addEventListener("click", function () {
      seed = (seed * 6364136223 + 1) & 0x7fffffff;
      draw();
    });

    setMode("grid");
  }

  global.SearchViz = { mount: mount };
})(window);
