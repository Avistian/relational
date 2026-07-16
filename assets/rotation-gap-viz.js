/**
 * Rotation bias — experiment-result beat (Grinsztajn 2022 §5.4, Fig 5a).
 *
 * Randomly rotate the feature space (same orthogonal Q on train + test) and refit. Tree-based
 * models are NOT rotationally invariant, so their accuracy collapses; an MLP IS invariant, so it
 * is essentially unchanged — and the model ranking REVERSES (the tree beats the MLP originally, the
 * MLP beats the tree after rotation).
 *
 * REAL numbers from labs/_verify_l026.py (synthetic axis-aligned task, Tier C, mean of 5 seeds):
 *   original: Tree 0.987 · GBT 0.997 · RF 0.994 · MLP 0.862
 *   rotated : Tree 0.747 · GBT 0.824 · RF 0.812 · MLP 0.869
 * MLP change +0.008 (invariant); tree change −0.240 (not invariant).
 *
 * Usage: RotationGapViz.mount(container, { caption })
 * Expected states:
 *   - default: 8 bars (4 models × {original, rotated}); readout notes the reversal + MLP invariance.
 *   - "Show ranking reversal" toggle draws two change connectors (tree collapses, MLP flat).
 */
(function (global) {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var ORIG = "#1e6b3c", ROT = "#d98a2b", MUTE = "#8a94a6";

  var LABELS = ["Tree", "GBT", "RF", "MLP"];
  var ORIG_ACC = [0.987, 0.997, 0.994, 0.862];
  var ROT_ACC = [0.747, 0.824, 0.812, 0.869];

  function el(name, attrs, text) {
    var e = document.createElementNS(SVGNS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "rotation-gap-viz";

    var showReversal = false;

    var ctl = document.createElement("div");
    ctl.className = "rg-ctl";
    var btn = document.createElement("button");
    btn.textContent = "Show ranking reversal";
    ctl.appendChild(btn);
    container.appendChild(ctl);

    var W = 460, H = 300;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", class: "rg-svg" });
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "rg-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "rg-caption";
    cap.textContent = config.caption ||
      "Test accuracy on an axis-aligned task, original vs randomly-rotated features (mean of 5 seeds). " +
      "Tree-based models collapse; the MLP is unmoved — and the ranking flips.";
    container.appendChild(cap);

    var PL = 40, PR = W - 14, PT = 20, PB = H - 42;
    var YLO = 0.5, YHI = 1.0;
    function sy(v) { return PB - (v - YLO) / (YHI - YLO) * (PB - PT); }

    var nGroups = LABELS.length;
    var groupW = (PR - PL) / nGroups;
    var barW = groupW * 0.30;

    function barX(g, which) {
      var gx = PL + g * groupW + groupW / 2;
      return which === 0 ? gx - barW - 2 : gx + 2;
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      svg.appendChild(el("rect", { x: PL, y: PT, width: PR - PL, height: PB - PT,
        fill: "var(--bg)", stroke: "var(--border)", "stroke-width": 1 }));

      [0.5, 0.6, 0.7, 0.8, 0.9, 1.0].forEach(function (v) {
        var yy = sy(v);
        svg.appendChild(el("line", { x1: PL, y1: yy, x2: PR, y2: yy,
          stroke: "var(--border)", "stroke-width": 0.5, opacity: 0.6 }));
        svg.appendChild(el("text", { x: PL - 5, y: yy + 3, "text-anchor": "end", class: "rg-tick" }, v.toFixed(1)));
      });

      LABELS.forEach(function (name, g) {
        var xo = barX(g, 0), xr = barX(g, 1);
        svg.appendChild(el("rect", { x: xo, y: sy(ORIG_ACC[g]), width: barW, height: PB - sy(ORIG_ACC[g]), fill: ORIG }));
        svg.appendChild(el("rect", { x: xr, y: sy(ROT_ACC[g]), width: barW, height: PB - sy(ROT_ACC[g]), fill: ROT }));
        svg.appendChild(el("text", { x: PL + g * groupW + groupW / 2, y: PB + 15, "text-anchor": "middle", class: "rg-lab" }, name));
        svg.appendChild(el("text", { x: xo + barW / 2, y: sy(ORIG_ACC[g]) - 3, "text-anchor": "middle", class: "rg-val" }, ORIG_ACC[g].toFixed(2)));
        svg.appendChild(el("text", { x: xr + barW / 2, y: sy(ROT_ACC[g]) - 3, "text-anchor": "middle", class: "rg-val" }, ROT_ACC[g].toFixed(2)));
      });

      // legend
      svg.appendChild(el("rect", { x: PR - 150, y: PT + 4, width: 10, height: 10, fill: ORIG }));
      svg.appendChild(el("text", { x: PR - 136, y: PT + 13, class: "rg-leg" }, "original"));
      svg.appendChild(el("rect", { x: PR - 78, y: PT + 4, width: 10, height: 10, fill: ROT }));
      svg.appendChild(el("text", { x: PR - 64, y: PT + 13, class: "rg-leg" }, "rotated"));

      if (showReversal) {
        // connectors: tree collapses, MLP flat
        var treeMid = PL + 0 * groupW + groupW / 2;
        var mlpMid = PL + 3 * groupW + groupW / 2;
        svg.appendChild(el("line", { x1: treeMid, y1: sy(ORIG_ACC[0]), x2: mlpMid, y2: sy(ORIG_ACC[3]),
          stroke: ORIG, "stroke-width": 1.4, "stroke-dasharray": "4,3", opacity: 0.8 }));
        svg.appendChild(el("line", { x1: treeMid, y1: sy(ROT_ACC[0]), x2: mlpMid, y2: sy(ROT_ACC[3]),
          stroke: ROT, "stroke-width": 1.4, "stroke-dasharray": "4,3", opacity: 0.9 }));
      }

      var yl = el("text", { x: 11, y: (PT + PB) / 2, "text-anchor": "middle", class: "rg-lab",
        transform: "rotate(-90 11 " + ((PT + PB) / 2) + ")" }, "test accuracy");
      svg.appendChild(yl);

      readout.innerHTML =
        "Original basis: <strong style='color:" + ORIG + "'>Tree 0.987</strong> beats " +
        "<strong>MLP 0.862</strong> (gap +0.126). After a random rotation: " +
        "<strong style='color:" + ROT + "'>Tree 0.747</strong> vs <strong>MLP 0.869</strong> — the " +
        "<strong>ranking reverses</strong>. The MLP moved by just <strong>+0.008</strong> " +
        "(rotationally <strong>invariant</strong>); the tree lost <strong>0.240</strong> (not invariant)." +
        (showReversal ? " <em>Dashed lines: the tree's edge collapses while the MLP holds flat.</em>" : "");
    }

    btn.addEventListener("click", function () {
      showReversal = !showReversal;
      btn.classList.toggle("rg-on", showReversal);
      draw();
    });

    draw();
    return { setReversal: function (v) { showReversal = v; btn.classList.toggle("rg-on", v); draw(); } };
  }

  global.RotationGapViz = { mount: mount };
})(window);
