/**
 * Grinsztajn-2022 inductive-bias visualizer — why tree-based models win on tabular data.
 *
 * Two spatial biases on a shared checkerboard motif (verified numbers baked in from
 * labs/_verify_l019.py, synthetic Tier-C, test accuracy):
 *
 *  - Mode "Irregular target": two boards show an N×N checkerboard (tile buttons 2 / 4 / 8).
 *    TREE board is drawn sharp (axis-aligned cells it can carve exactly). NEURAL NET board is
 *    the SAME checkerboard through a fixed Gaussian blur (feGaussianBlur) — the NN's smoothness
 *    bias. As tiles get finer the blur washes the cells to gray, mirroring the collapse in
 *    accuracy (8×8: tree 0.969 vs NN 0.837).
 *  - Mode "Rotate the board": boundary-line diagram of a 2-feature XOR. Axis-aligned: the tree's
 *    two axis-aligned splits (green) match the boundary exactly (0.998). Rotated: the boundary
 *    turns diagonal; the tree can only approximate it with a red STAIRCASE of axis-aligned steps
 *    (0.831), while the NN's boundary rotates with the data (invariant, 0.976).
 *
 * Expected states:
 *  - Irregular, tiles=2: each board has 4 checker rects; readout "tree 1.000 · NN 0.994".
 *  - Irregular, tiles=8: each board has 64 rects; NN board group carries filter=url(#bias-blur);
 *    readout "tree 0.969 · NN 0.837".
 *  - Rotation, axis-aligned: tree overlay has straight split lines, no staircase; readout 0.998/0.979.
 *  - Rotation, rotated: tree overlay contains a polyline (staircase); readout 0.831/0.976.
 *
 * Usage: BiasesViz.mount(container, { caption })
 */
(function (global) {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";

  // verified test-accuracy readouts (labs/_verify_l019.py)
  var IRREG = { 2: [1.000, 0.994], 4: [0.999, 0.975], 8: [0.969, 0.837] };
  var ROT = { aligned: [0.998, 0.979], rotated: [0.831, 0.976] };

  var DARK = "#2e6fb0", LIGHT = "#dce9f5";

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "biases-viz";

    var mode = "irregular"; // "irregular" | "rotation"
    var tiles = 2;
    var rotated = false;

    // ---- mode controls
    var ctl = document.createElement("div");
    ctl.className = "bv-ctl";
    var bIrreg = document.createElement("button");
    bIrreg.textContent = "Irregular target";
    var bRot = document.createElement("button");
    bRot.textContent = "Rotate the board";
    ctl.appendChild(bIrreg);
    ctl.appendChild(bRot);
    container.appendChild(ctl);

    // ---- contextual controls (rebuilt per mode)
    var sub = document.createElement("div");
    sub.className = "bv-sub";
    container.appendChild(sub);

    var svg = document.createElementNS(SVGNS, "svg");
    svg.setAttribute("viewBox", "0 0 360 210");
    svg.setAttribute("class", "bv-svg");
    container.appendChild(svg);

    var readout = document.createElement("p");
    readout.className = "bv-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "bv-caption";
    cap.textContent = config.caption || "";
    container.appendChild(cap);

    function el(tag, attrs, text) {
      var n = document.createElementNS(SVGNS, tag);
      for (var k in attrs) n.setAttribute(k, attrs[k]);
      if (text != null) n.textContent = text;
      return n;
    }

    // board geometry
    var TX = 25, NX = 205, BY = 34, S = 130;

    function checker(x, y, size, t, group) {
      var cell = size / t;
      for (var i = 0; i < t; i++) {
        for (var j = 0; j < t; j++) {
          group.appendChild(el("rect", {
            x: x + j * cell, y: y + i * cell, width: cell, height: cell,
            fill: (i + j) % 2 ? DARK : LIGHT,
          }));
        }
      }
    }

    function boardLabel(x, sub2) {
      svg.appendChild(el("text", { x: x + S / 2, y: 24, class: "bv-title", "text-anchor": "middle" },
        sub2));
    }

    function drawIrregular() {
      boardLabel(TX, "TREE (axis-aligned)");
      boardLabel(NX, "NEURAL NET (smooth)");

      // tree board — sharp
      var gTree = el("g", {});
      checker(TX, BY, S, tiles, gTree);
      svg.appendChild(gTree);
      svg.appendChild(el("rect", { x: TX, y: BY, width: S, height: S, fill: "none", stroke: "#34495e", "stroke-width": 1.5 }));

      // nn board — same checkerboard, blurred (smoothness bias)
      var gNN = el("g", { filter: "url(#bias-blur)" });
      checker(NX, BY, S, tiles, gNN);
      svg.appendChild(gNN);
      svg.appendChild(el("rect", { x: NX, y: BY, width: S, height: S, fill: "none", stroke: "#34495e", "stroke-width": 1.5 }));

      var acc = IRREG[tiles];
      readout.innerHTML =
        "Checkerboard target with <strong>" + tiles + "×" + tiles + "</strong> tiles. Verified test accuracy: " +
        "tree <strong style='color:#1e6b3c'>" + acc[0].toFixed(3) + "</strong> · " +
        "NN <strong style='color:#b03a2e'>" + acc[1].toFixed(3) + "</strong>. " +
        "The finer (more irregular) the target, the more the smooth-biased NN blurs the cells away.";
    }

    function drawRotation() {
      boardLabel(TX, "TREE (axis-aligned splits)");
      boardLabel(NX, "NEURAL NET (rotates)");
      var deg = rotated ? 30 : 0;

      // faint checker context (2×2) for each board, rotated with the data
      [TX, NX].forEach(function (bx) {
        var cx = bx + S / 2, cy = BY + S / 2;
        var g = el("g", { transform: "rotate(" + deg + " " + cx + " " + cy + ")", opacity: 0.28 });
        checker(bx, BY, S, 2, g);
        svg.appendChild(g);
        svg.appendChild(el("rect", { x: bx, y: BY, width: S, height: S, fill: "none", stroke: "#34495e", "stroke-width": 1.5 }));
      });

      var cxT = TX + S / 2, cyT = BY + S / 2;
      // TREE overlay — axis-aligned splits
      if (!rotated) {
        svg.appendChild(el("line", { x1: cxT, y1: BY, x2: cxT, y2: BY + S, stroke: "#1e6b3c", "stroke-width": 2.5 }));
        svg.appendChild(el("line", { x1: TX, y1: cyT, x2: TX + S, y2: cyT, stroke: "#1e6b3c", "stroke-width": 2.5 }));
      } else {
        // red staircase approximating a 30° diagonal boundary through the centre
        var pts = [], steps = 6, step = S / steps;
        var x0 = TX, y0 = BY + S * 0.78;
        for (var k = 0; k <= steps; k++) {
          var px = x0 + k * step;
          var py = y0 - k * step * 0.6;
          pts.push(px + "," + py);
          if (k < steps) pts.push((px + step) + "," + py);
        }
        svg.appendChild(el("polyline", { points: pts.join(" "), fill: "none", stroke: "#b03a2e", "stroke-width": 2.5, class: "bv-stair" }));
      }

      // NN overlay — a boundary line that rotates WITH the data (invariant)
      var cxN = NX + S / 2, cyN = BY + S / 2;
      var gN = el("g", { transform: "rotate(" + deg + " " + cxN + " " + cyN + ")" });
      gN.appendChild(el("line", { x1: cxN, y1: BY + 6, x2: cxN, y2: BY + S - 6, stroke: "#2e6fb0", "stroke-width": 2.5 }));
      gN.appendChild(el("line", { x1: NX + 6, y1: cyN, x2: NX + S - 6, y2: cyN, stroke: "#2e6fb0", "stroke-width": 2.5 }));
      svg.appendChild(gN);

      var acc = rotated ? ROT.rotated : ROT.aligned;
      readout.innerHTML = rotated
        ? "Board rotated: the XOR boundary is now diagonal. The tree can only <strong>staircase</strong> it with " +
          "axis-aligned steps → tree <strong style='color:#b03a2e'>" + acc[0].toFixed(3) + "</strong>, while the " +
          "rotation-invariant NN is unmoved <strong style='color:#1e6b3c'>" + acc[1].toFixed(3) + "</strong>."
        : "Axis-aligned XOR: the tree's two splits match the boundary exactly → tree " +
          "<strong style='color:#1e6b3c'>" + acc[0].toFixed(3) + "</strong>, NN " + acc[1].toFixed(3) +
          ". Orientation carries the tree's edge — press <em>Rotated</em> to destroy it.";
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      // blur filter def (used by irregular NN board)
      var defs = el("defs", {});
      var filt = el("filter", { id: "bias-blur", x: "-20%", y: "-20%", width: "140%", height: "140%" });
      filt.appendChild(el("feGaussianBlur", { in: "SourceGraphic", stdDeviation: 5 }));
      defs.appendChild(filt);
      svg.appendChild(defs);

      if (mode === "irregular") drawIrregular();
      else drawRotation();
    }

    function buildSub() {
      sub.innerHTML = "";
      if (mode === "irregular") {
        [2, 4, 8].forEach(function (t) {
          var b = document.createElement("button");
          b.textContent = t + "×" + t;
          b.classList.toggle("bv-on", t === tiles);
          b.addEventListener("click", function () { tiles = t; buildSub(); draw(); });
          sub.appendChild(b);
        });
      } else {
        var bA = document.createElement("button");
        bA.textContent = "Axis-aligned";
        bA.classList.toggle("bv-on", !rotated);
        bA.addEventListener("click", function () { rotated = false; buildSub(); draw(); });
        var bR = document.createElement("button");
        bR.textContent = "Rotated";
        bR.classList.toggle("bv-on", rotated);
        bR.addEventListener("click", function () { rotated = true; buildSub(); draw(); });
        sub.appendChild(bA);
        sub.appendChild(bR);
      }
    }

    function setMode(m) {
      mode = m;
      bIrreg.classList.toggle("bv-active", m === "irregular");
      bRot.classList.toggle("bv-active", m === "rotation");
      buildSub();
      draw();
    }

    bIrreg.addEventListener("click", function () { setMode("irregular"); });
    bRot.addEventListener("click", function () { setMode("rotation"); });

    setMode("irregular");
  }

  global.BiasesViz = { mount: mount };
})(window);
