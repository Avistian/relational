/**
 * Uninformative-features bias — the GATING mechanism (why a tree is robust and an MLP is not).
 *
 * A decision tree chooses each split by GAIN (impurity decrease). On the full data a pure-noise
 * column has ~zero gain, so the greedy tree never picks it near the top — its structure is
 * "anchored" by the informative features. An MLP has NO such gate: every input feeds the first
 * linear layer, and (being rotationally invariant) it cannot cheaply zero-out a junk direction, so
 * it spends nearly as much first-layer weight on junk as on signal.
 *
 * Two panels, both bars normalized to the informative mean (= 1.0) so the JUNK bar shows the
 * fraction of the informative magnitude the model still spends on noise:
 *   - Tree ROOT-SPLIT GAIN: informative 0.0442 vs junk 0.000376 → junk ≈ 1% (a ~118× gate).
 *   - MLP 1st-layer |weight|: informative 0.1354 vs junk 0.0757  → junk ≈ 56% (only 1.8×, no gate).
 *
 * REAL numbers from labs/_verify_l027.py (5 informative + 20 junk features, seed 0).
 *
 * Usage: UninformativeMechanismViz.mount(container, { caption })
 * Expected states:
 *   - two labelled panels ("Tree: root-split gain", "MLP: 1st-layer |weight|"), each with an
 *     informative bar (full height) and a junk bar (fraction); readout gives raw values + ratios.
 */
(function (global) {
  "use strict";

  var svgNS = "http://www.w3.org/2000/svg";
  var C_INFORM = "#1e6b3c", C_JUNK = "#b03a2e";

  // raw values (labs/_verify_l027.py, Part C); junk gain kept at full precision so the ratio is 118×
  var TREE_INFORM = 0.0442, TREE_JUNK = 0.000376;
  var MLP_INFORM = 0.1354, MLP_JUNK = 0.0757;

  function el(name, attrs, text) {
    var e = document.createElementNS(svgNS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "uninformative-mechanism-viz";

    var W = 460, H = 300;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", class: "um-svg" });
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "um-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "um-caption";
    cap.textContent = config.caption ||
      "Each panel is normalized so the informative bar = 1.0; the junk bar is the fraction of that " +
      "magnitude the model still spends on pure noise. The tree gates junk out; the MLP cannot.";
    container.appendChild(cap);

    var PT = 26, PB = H - 52;

    function panel(x0, x1, title, inform, junk, gateColor) {
      var mid = (x0 + x1) / 2;
      var barW = (x1 - x0) * 0.26;
      var informFrac = 1.0;
      var junkFrac = junk / inform;
      function bh(frac) { return (PB - PT) * frac; }

      // baseline
      svg.appendChild(el("line", { x1: x0 + 8, y1: PB, x2: x1 - 8, y2: PB,
        stroke: "var(--border)", "stroke-width": 1 }));
      // panel title
      svg.appendChild(el("text", { x: mid, y: PT - 10, "text-anchor": "middle", class: "um-title" }, title));

      var xi = mid - barW - 6, xj = mid + 6;
      // informative bar (full)
      svg.appendChild(el("rect", { x: xi, y: PB - bh(informFrac), width: barW, height: bh(informFrac), fill: C_INFORM }));
      svg.appendChild(el("text", { x: xi + barW / 2, y: PB - bh(informFrac) - 4, "text-anchor": "middle", class: "um-val" }, inform.toFixed(4)));
      svg.appendChild(el("text", { x: xi + barW / 2, y: PB + 14, "text-anchor": "middle", class: "um-lab" }, "informative"));
      // junk bar (fraction)
      var jy = PB - bh(junkFrac);
      svg.appendChild(el("rect", { x: xj, y: jy, width: barW, height: Math.max(bh(junkFrac), 1), fill: C_JUNK }));
      svg.appendChild(el("text", { x: xj + barW / 2, y: jy - 4, "text-anchor": "middle", class: "um-val" }, junk.toFixed(4)));
      svg.appendChild(el("text", { x: xj + barW / 2, y: PB + 14, "text-anchor": "middle", class: "um-lab" }, "junk"));
      // fraction / ratio annotation
      svg.appendChild(el("text", { x: mid, y: PB + 32, "text-anchor": "middle", class: "um-ratio", fill: gateColor },
        "junk = " + Math.round(junkFrac * 100) + "% of informative (" + (inform / junk).toFixed(0) + "× gate)"));
    }

    panel(14, W / 2, "Tree: root-split GAIN", TREE_INFORM, TREE_JUNK, C_INFORM);
    panel(W / 2, W - 14, "MLP: 1st-layer |weight|", MLP_INFORM, MLP_JUNK, C_JUNK);

    // divider
    svg.appendChild(el("line", { x1: W / 2, y1: PT - 4, x2: W / 2, y2: PB + 36,
      stroke: "var(--border)", "stroke-width": 0.5, opacity: 0.6, "stroke-dasharray": "3,3" }));

    readout.innerHTML =
      "The tree picks splits by <strong>gain</strong>: a noise column's best root split has gain " +
      "<strong>" + TREE_JUNK.toFixed(4) + "</strong> vs <strong>" + TREE_INFORM.toFixed(4) + "</strong> " +
      "for informative — a <strong>" + Math.round(TREE_INFORM / TREE_JUNK) + "× gate</strong>, so junk is " +
      "never chosen at the top. The MLP has <strong>no gate</strong>: it still spends " +
      "<strong>" + Math.round(MLP_JUNK / MLP_INFORM * 100) + "%</strong> as much first-layer weight on " +
      "junk as on signal (only a <strong>" + (MLP_INFORM / MLP_JUNK).toFixed(1) + "× </strong>difference).";

    return {};
  }

  global.UninformativeMechanismViz = { mount: mount };
})(window);
