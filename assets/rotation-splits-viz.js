/**
 * Rotation bias — geometry beat (Grinsztajn 2022 §5.4, Finding 3).
 *
 * WHY a decision tree is NOT rotationally invariant: its splits are axis-aligned, so they are
 * tied to the original column basis. An axis-aligned class region (here the top-right quadrant,
 * x>0 AND y>0) is carved by two straight splits (x=0, y=0) exactly. Apply a random rotation and
 * the SAME region becomes a diagonal wedge: axis-aligned splits can now only approximate it with a
 * red STAIRCASE, so the tree loses accuracy. An MLP's boundary rotates with the data, so it is
 * unaffected (rotationally invariant, in the sense of Ng 2004).
 *
 * Test-accuracy readouts baked in from labs/_verify_l026.py (synthetic axis-aligned task, Tier C,
 * mean of 5 seeds): tree 0.987 (original) -> 0.747 (rotated); MLP 0.862 -> 0.869 (≈unchanged).
 *
 * Expected states:
 *   - "original": green straight splits (2 <line>), no staircase; readout has "Original basis",
 *     "axis-aligned", tree "0.987".
 *   - "rotated": a red staircase <polyline> appears; readout has "Rotated basis", "staircase",
 *     tree "0.747".
 *
 * Usage: RotationSplitsViz.mount(container, { caption })
 */
(function (global) {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var GREEN = "#1e6b3c", RED = "#b03a2e", BLUE = "#2e6fb0";
  var CLASS1 = "#cfe3d4", GRID = "#dce9f5";

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "rotation-splits-viz";

    var rotated = false;

    var ctl = document.createElement("div");
    ctl.className = "rs-ctl";
    var bA = document.createElement("button");
    bA.textContent = "Original basis";
    var bR = document.createElement("button");
    bR.textContent = "Rotated basis";
    ctl.appendChild(bA);
    ctl.appendChild(bR);
    container.appendChild(ctl);

    var svg = document.createElementNS(SVGNS, "svg");
    svg.setAttribute("viewBox", "0 0 360 210");
    svg.setAttribute("class", "rs-svg");
    container.appendChild(svg);

    var readout = document.createElement("p");
    readout.className = "rs-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "rs-caption";
    cap.textContent = config.caption ||
      "A tree carves the class region with axis-aligned splits. Rotating the feature space turns the " +
      "boundary diagonal, so the tree can only staircase it — while the MLP's boundary just rotates too.";
    container.appendChild(cap);

    function el(tag, attrs, text) {
      var n = document.createElementNS(SVGNS, tag);
      for (var k in attrs) n.setAttribute(k, attrs[k]);
      if (text != null) n.textContent = text;
      return n;
    }

    // board geometry (two boards: TREE frame, NEURAL-NET frame)
    var TX = 25, NX = 205, BY = 30, S = 140;

    function boardLabel(x, txt) {
      svg.appendChild(el("text", { x: x + S / 2, y: 20, class: "rs-title", "text-anchor": "middle" }, txt));
    }

    function rot(cx, cy, px, py, deg) {
      var r = deg * Math.PI / 180, c = Math.cos(r), s = Math.sin(r);
      var dx = px - cx, dy = py - cy;
      return [cx + dx * c - dy * s, cy + dx * s + dy * c];
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var deg = rotated ? 30 : 0;

      boardLabel(TX, "TREE (axis-aligned)");
      boardLabel(NX, "NEURAL NET (invariant)");

      [TX, NX].forEach(function (bx) {
        var cx = bx + S / 2, cy = BY + S / 2;

        // class-1 region = top-right quadrant of the board, rotated WITH the data
        var q = [[cx, cy], [cx + S / 2, cy], [cx + S / 2, cy - S / 2], [cx, cy - S / 2]];
        var pts = q.map(function (p) { return rot(cx, cy, p[0], p[1], deg); });
        svg.appendChild(el("polygon", {
          points: pts.map(function (p) { return p[0].toFixed(1) + "," + p[1].toFixed(1); }).join(" "),
          fill: CLASS1, stroke: "#9ec4a9", "stroke-width": 1,
        }));

        // board frame
        svg.appendChild(el("rect", { x: bx, y: BY, width: S, height: S, fill: "none", stroke: "#34495e", "stroke-width": 1.4 }));
      });

      // ---- TREE overlay ----
      var cxT = TX + S / 2, cyT = BY + S / 2;
      if (!rotated) {
        // two straight splits at x=0, y=0 fit the quadrant exactly
        svg.appendChild(el("line", { x1: cxT, y1: BY, x2: cxT, y2: BY + S, stroke: GREEN, "stroke-width": 2.6 }));
        svg.appendChild(el("line", { x1: TX, y1: cyT, x2: TX + S, y2: cyT, stroke: GREEN, "stroke-width": 2.6 }));
      } else {
        // red staircase approximating the rotated diagonal edge of the wedge
        var stair = [];
        var steps = 6, step = (S / 2) / steps;
        // approximate the rotated lower edge (originally the +x axis, now at 30 deg)
        var x0 = cxT, y0 = cyT;
        for (var k = 0; k < steps; k++) {
          var hx = x0 + (k + 1) * step;
          var hy = y0 - k * step * Math.tan(30 * Math.PI / 180);
          stair.push(hx.toFixed(1) + "," + (y0 - k * step * Math.tan(30 * Math.PI / 180)).toFixed(1));
          stair.push(hx.toFixed(1) + "," + (y0 - (k + 1) * step * Math.tan(30 * Math.PI / 180)).toFixed(1));
        }
        svg.appendChild(el("polyline", {
          points: stair.join(" "), fill: "none", stroke: RED, "stroke-width": 2.6, class: "rs-stair",
        }));
      }

      // ---- NEURAL NET overlay: a boundary that rotates WITH the data (invariant) ----
      var cxN = NX + S / 2, cyN = BY + S / 2;
      var v = [rot(cxN, cyN, cxN, BY, deg), rot(cxN, cyN, cxN, BY + S, deg)];
      var h = [rot(cxN, cyN, NX, cyN, deg), rot(cxN, cyN, NX + S, cyN, deg)];
      svg.appendChild(el("line", { x1: v[0][0], y1: v[0][1], x2: v[1][0], y2: v[1][1], stroke: BLUE, "stroke-width": 2.4 }));
      svg.appendChild(el("line", { x1: h[0][0], y1: h[0][1], x2: h[1][0], y2: h[1][1], stroke: BLUE, "stroke-width": 2.4 }));

      readout.innerHTML = rotated
        ? "<strong>Rotated basis.</strong> The class region is now a diagonal wedge. The tree's " +
          "axis-aligned splits can only <strong>staircase</strong> it, so its accuracy falls " +
          "<strong style='color:" + RED + "'>0.987 → 0.747</strong>. The MLP's boundary rotated with " +
          "the data — it is <strong>rotationally invariant</strong>, so it is unmoved (0.862 → 0.869)."
        : "<strong>Original basis.</strong> The class region is the top-right quadrant. The tree fits it " +
          "with <strong>two axis-aligned splits</strong> (x=0, y=0) exactly → accuracy " +
          "<strong style='color:" + GREEN + "'>0.987</strong> (MLP 0.862). Press <em>Rotated basis</em> " +
          "to spin the axes and watch the tree's edge collapse.";
    }

    function setRotated(v) {
      rotated = v;
      bA.classList.toggle("rs-on", !rotated);
      bR.classList.toggle("rs-on", rotated);
      draw();
    }

    bA.addEventListener("click", function () { setRotated(false); });
    bR.addEventListener("click", function () { setRotated(true); });

    setRotated(false);
    return { setRotated: setRotated };
  }

  global.RotationSplitsViz = { mount: mount };
})(window);
