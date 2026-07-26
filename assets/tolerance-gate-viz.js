/**
 * Where to set the reproduction tolerance. (Lesson 037)
 *
 * A package that regenerates its headline number needs a gate: |new - expected|
 * <= tol, or the run fails loudly. Choosing tol is the whole skill, because the
 * candidate values span four orders of magnitude and each one buys a different
 * set of false alarms and blind spots.
 *
 * The scale is |delta mean log-loss| on log10 axes, with the measured landmarks
 * from this pipeline placed on it:
 *   0            eight of nine perturbations were bit-identical (drawn at the far left)
 *   1.33e-3      the float32 -> float64 cast (labs/_repro2_l037.py)
 *   3.2e-3       the M2a - M1 margin that chose the shipped model (L036)
 *   1.66e-2      the spread across five CV-splitter seeds (labs/_repro3_l037.py)
 *   3.50e-2      one fold's standard deviation (labs/_repro2_l037.py, ref config)
 *
 * A slider moves tol; the readout names what a gate there catches, what it waves
 * through, and what it will fire on for no reason.
 *
 * Plain <script> (file://-safe). Usage: ToleranceGateViz.mount(el, config?)
 *
 * Expected states (headless verification):
 *   - mounts an svg with one marker per landmark + a range input
 *   - default tol = 1e-3: readout says the gate CATCHES the dtype cast
 *   - tol = 1e-2: readout says it waves the shipping margin through
 *   - tol = 5e-2: readout says the gate can no longer fail
 */
(function (global) {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";

  var LANDMARKS = [
    {
      v: 1.33e-3, label: "float32 \u2192 float64",
      note: "one undocumented cast; 258 predicted classes change"
    },
    {
      v: 3.2e-3, label: "M2a \u2212 M1 margin",
      note: "the difference that decided which model shipped (L036)"
    },
    {
      v: 1.66e-2, label: "CV-splitter seed spread",
      note: "same code, five fold draws: 1.6191 \u2013 1.6357"
    },
    {
      v: 3.50e-2, label: "one fold's \u03c3",
      note: "fold-to-fold spread within a single run"
    }
  ];

  // log10 domain
  var LO = -5, HI = -1;

  function el(tag, attrs) {
    var n = document.createElementNS(SVGNS, tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    return n;
  }
  function txt(n, s) { n.textContent = s; return n; }
  function fmt(v) {
    if (v === 0) return "0";
    var e = Math.floor(Math.log10(v));
    var m = v / Math.pow(10, e);
    return (Math.round(m * 100) / 100) + "e" + e;
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.classList.add("tg-viz");

    var ctl = document.createElement("div");
    ctl.className = "tg-ctl";
    var slider = document.createElement("input");
    slider.type = "range";
    slider.min = "-5";
    slider.max = "-1";
    slider.step = "0.05";
    slider.value = "-3";
    var sLabel = document.createElement("span");
    sLabel.className = "tg-slabel";
    ctl.appendChild(document.createTextNode("tolerance "));
    ctl.appendChild(slider);
    ctl.appendChild(sLabel);
    container.appendChild(ctl);

    var holder = document.createElement("div");
    container.appendChild(holder);
    var readout = document.createElement("div");
    readout.className = "tg-readout";
    container.appendChild(readout);

    function xOf(v, W) {
      var lg = Math.log10(v);
      var t = (lg - LO) / (HI - LO);
      return 70 + Math.max(0, Math.min(1, t)) * (W - 100);
    }

    function draw() {
      var tol = Math.pow(10, parseFloat(slider.value));
      sLabel.textContent = "|\u0394 log-loss| \u2264 " + fmt(tol);

      holder.innerHTML = "";
      var W = 640, H = 170, axisY = 108;
      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%" });
      svg.setAttribute("class", "tg-svg");

      // shaded PASS region left of tol
      svg.appendChild(el("rect", {
        x: 40, y: axisY - 46, width: xOf(tol, W) - 40, height: 46,
        fill: "#1e6b3c", opacity: 0.08
      }));
      svg.appendChild(el("line", {
        x1: 40, y1: axisY, x2: W - 20, y2: axisY, stroke: "#334155", "stroke-width": 1.2
      }));

      for (var e = LO; e <= HI; e++) {
        var x = xOf(Math.pow(10, e), W);
        svg.appendChild(el("line", { x1: x, y1: axisY, x2: x, y2: axisY + 5, stroke: "#94a3b8" }));
        var t = el("text", { x: x, y: axisY + 18, "text-anchor": "middle", class: "tg-tick" });
        svg.appendChild(txt(t, "1e" + e));
      }

      // bit-identical marker, pinned at the left edge
      svg.appendChild(el("circle", { cx: 46, cy: axisY, r: 5, fill: "#1e6b3c" }));
      var zt = el("text", { x: 46, y: axisY + 34, "text-anchor": "middle", class: "tg-zero" });
      svg.appendChild(txt(zt, "0 \u2014 8 of 9 runs"));

      LANDMARKS.forEach(function (m, i) {
        var x = xOf(m.v, W);
        var passes = m.v <= tol;
        var yTop = axisY - 22 - (i % 2) * 26;
        svg.appendChild(el("line", {
          x1: x, y1: axisY, x2: x, y2: yTop, stroke: passes ? "#1e6b3c" : "#b03a2e",
          "stroke-width": 1.2, "stroke-dasharray": passes ? "3 2" : ""
        }));
        svg.appendChild(el("circle", {
          cx: x, cy: axisY, r: 4.5, fill: passes ? "#1e6b3c" : "#b03a2e"
        }));
        var lt = el("text", {
          x: x, y: yTop - 4, "text-anchor": "middle", class: "tg-mark",
          fill: passes ? "#1e6b3c" : "#b03a2e"
        });
        svg.appendChild(txt(lt, m.label));
      });

      // the tolerance line itself
      var tx = xOf(tol, W);
      svg.appendChild(el("line", {
        x1: tx, y1: axisY - 52, x2: tx, y2: axisY + 8, stroke: "#7a1f16", "stroke-width": 2.5
      }));
      var tt = el("text", { x: tx, y: axisY - 58, "text-anchor": "middle", class: "tg-tolmark" });
      svg.appendChild(txt(tt, "tol"));
      var pl = el("text", { x: 44, y: axisY - 52, class: "tg-region" });
      svg.appendChild(txt(pl, "\u2190 gate passes"));

      holder.appendChild(svg);

      var caught = LANDMARKS.filter(function (m) { return m.v > tol; });
      var waved = LANDMARKS.filter(function (m) { return m.v <= tol; });

      var verdict;
      if (tol < 1.33e-3) {
        verdict = "<span class='tg-pill tg-tight'>TIGHT</span> Only a bit-identical run passes. On " +
          "this pipeline that is achievable and therefore the right default \u2014 eight of nine " +
          "perturbations were byte-for-byte identical, so a gate this tight will not cry wolf. " +
          "It would, however, be unusable the moment anything genuinely stochastic enters " +
          "(a GPU reduction, an early-stopping split, torch), which is Year&nbsp;2 onward.";
      } else if (tol < 1.66e-2) {
        verdict = "<span class='tg-pill tg-mid'>PERMISSIVE</span> The gate now waves through changes " +
          "as large as the margin that chose which model shipped. It will still catch a genuinely " +
          "broken environment, but it can no longer protect a <em>decision</em> \u2014 the class of " +
          "defect L036 found to be the most expensive.";
      } else if (tol < 3.5e-2) {
        verdict = "<span class='tg-pill tg-loose'>TOO LOOSE</span> Everything smaller than a " +
          "re-draw of the folds passes. A gate here answers \"did the code roughly still work\", " +
          "not \"did it reproduce\", and it cannot support any comparison you would publish.";
      } else {
        verdict = "<span class='tg-pill tg-loose'>CANNOT FAIL</span> The tolerance now exceeds one " +
          "fold's own standard deviation, so a gate here passes runs that share nothing but a " +
          "filename. A test that cannot fail is not a test \u2014 it is a comment.";
      }

      readout.innerHTML =
        "<p>" + verdict + "</p>" +
        "<p><strong>Fires on:</strong> " +
        (caught.length
          ? caught.map(function (m) {
              return "<span class='tg-c'>" + m.label + "</span> <em>(" + m.note + ")</em>";
            }).join(" &middot; ")
          : "<em>nothing \u2014 no measured perturbation exceeds this tolerance</em>") +
        "</p><p><strong>Waves through:</strong> " +
        (waved.length
          ? waved.map(function (m) {
              return "<span class='tg-w'>" + m.label + "</span>";
            }).join(" &middot; ")
          : "<em>nothing but an exactly identical run</em>") +
        "</p>";
    }

    slider.addEventListener("input", draw);
    slider.addEventListener("change", draw);
    draw();
    return {
      setTol: function (v) { slider.value = String(Math.log10(v)); draw(); },
      getTol: function () { return Math.pow(10, parseFloat(slider.value)); }
    };
  }

  global.ToleranceGateViz = { mount: mount };
})(typeof window !== "undefined" ? window : this);
