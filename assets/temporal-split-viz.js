/**
 * Temporal-vs-random split geometry visualizer.
 *
 * Renders the SAME time-ordered stream of rows twice, stacked, so the contrast is
 * immediate (not a toggle — two different concepts side by side):
 *   - Random split: ~test% of rows are drawn at random across the WHOLE timeline,
 *     so test rows sit among train rows from the same period (the future bleeds
 *     into training).
 *   - Temporal split: a single cut; everything before it is train (past), everything
 *     after is test (future). The model must extrapolate forward.
 *
 * Plain <script> (file://-safe). Usage: TemporalSplitViz.mount(el, { n, testFrac, seed }).
 */
(function (global) {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var TRAIN = "#2e6fb0";
  var TEST = "#b03a2e";

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function el(tag, attrs) {
    var n = document.createElementNS(SVGNS, tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    return n;
  }

  function randomTestMask(n, testFrac, rng) {
    // choose round(n*testFrac) distinct indices uniformly across the whole timeline
    var k = Math.round(n * testFrac);
    var order = [];
    var i;
    for (i = 0; i < n; i++) order.push(i);
    for (i = n - 1; i > 0; i--) {
      var j = Math.floor(rng() * (i + 1));
      var tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }
    var mask = new Array(n).fill(false);
    for (i = 0; i < k; i++) mask[order[i]] = true;
    return mask;
  }

  function strip(svg, x0, y0, w, rowH, n, isTest, cutIndex, title) {
    var r = Math.max(2.5, Math.min(6, (w / n) * 0.42));
    var g = el("g", {});
    svg.appendChild(g);
    // title
    var t = el("text", { x: x0, y: y0 - 8, class: "tsv-title" });
    t.textContent = title;
    g.appendChild(t);
    // baseline
    g.appendChild(el("line", { x1: x0, y1: y0 + rowH / 2, x2: x0 + w, y2: y0 + rowH / 2, stroke: "#d9d9d9", "stroke-width": 1 }));
    var i;
    for (i = 0; i < n; i++) {
      var cx = x0 + ((i + 0.5) / n) * w;
      var test = isTest(i);
      g.appendChild(el("circle", {
        cx: cx, cy: y0 + rowH / 2, r: r,
        fill: test ? TEST : TRAIN,
        "fill-opacity": test ? 0.95 : 0.65,
      }));
    }
    if (cutIndex != null) {
      var cxCut = x0 + (cutIndex / n) * w;
      g.appendChild(el("line", { x1: cxCut, y1: y0 - 2, x2: cxCut, y2: y0 + rowH + 2, stroke: "#111", "stroke-width": 1.5, "stroke-dasharray": "3 2" }));
    }
    return g;
  }

  function mount(container, config) {
    config = config || {};
    var n = config.n || 60;
    var testFrac = config.testFrac != null ? config.testFrac : 0.2;
    var seed = config.seed != null ? config.seed : 7;

    container.innerHTML = "";
    container.classList.add("temporal-split-viz");

    var W = 560, H = 210;
    var padL = 14, padR = 14, plotW = W - padL - padR;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, class: "tsv-svg", width: "100%" });
    container.appendChild(svg);

    var rng = mulberry32(seed);
    var mask = randomTestMask(n, testFrac, rng);
    var cut = Math.round(n * (1 - testFrac));

    strip(svg, padL, 34, plotW, 26, n, function (i) { return mask[i]; }, null, "RANDOM SPLIT  (test rows scattered across all time)");
    strip(svg, padL, 118, plotW, 26, n, function (i) { return i >= cut; }, cut, "TEMPORAL SPLIT  (cut once; test = the future)");

    // time axis label
    var ax = el("text", { x: padL, y: 168, class: "tsv-axis" });
    ax.textContent = "\u25C0 older";
    svg.appendChild(ax);
    var ax2 = el("text", { x: padL + plotW, y: 168, class: "tsv-axis", "text-anchor": "end" });
    ax2.textContent = "newer \u25B6";
    svg.appendChild(ax2);

    // legend
    var lg = el("g", {});
    svg.appendChild(lg);
    lg.appendChild(el("circle", { cx: padL + 6, cy: 190, r: 5, fill: TRAIN, "fill-opacity": 0.65 }));
    var l1 = el("text", { x: padL + 16, y: 194, class: "tsv-legend" }); l1.textContent = "train"; lg.appendChild(l1);
    lg.appendChild(el("circle", { cx: padL + 70, cy: 190, r: 5, fill: TEST })); 
    var l2 = el("text", { x: padL + 80, y: 194, class: "tsv-legend" }); l2.textContent = "test"; lg.appendChild(l2);

    var controls = document.createElement("div");
    controls.className = "tsv-controls";
    container.appendChild(controls);
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "tsv-btn";
    btn.textContent = "Reshuffle the random split";
    controls.appendChild(btn);

    var readout = document.createElement("div");
    readout.className = "tsv-readout";
    container.appendChild(readout);

    function setReadout() {
      // does any TRAIN row sit LATER (in time) than the earliest TEST row, under random?
      var firstTest = mask.indexOf(true);
      var lateTrain = 0, i;
      for (i = firstTest; i < n; i++) if (!mask[i]) lateTrain++;
      readout.innerHTML =
        "<strong>Random:</strong> " + lateTrain + " training rows fall <em>after</em> the earliest test row in time — " +
        "the model trains on the very period it is scored on. " +
        "<strong>Temporal:</strong> every test row is strictly after the cut, so the model must predict a period it never saw.";
    }
    setReadout();

    btn.addEventListener("click", function () {
      mask = randomTestMask(n, testFrac, mulberry32((seed = (seed * 1103515245 + 12345) & 0x7fffffff)));
      // redraw only by re-mounting the top strip: simplest is a full re-render
      svg.innerHTML = "";
      strip(svg, padL, 34, plotW, 26, n, function (i) { return mask[i]; }, null, "RANDOM SPLIT  (test rows scattered across all time)");
      strip(svg, padL, 118, plotW, 26, n, function (i) { return i >= cut; }, cut, "TEMPORAL SPLIT  (cut once; test = the future)");
      svg.appendChild(ax); svg.appendChild(ax2); svg.appendChild(lg);
      setReadout();
    });

    return { getMask: function () { return mask.slice(); }, getCut: function () { return cut; } };
  }

  global.TemporalSplitViz = { mount: mount };
})(window);
