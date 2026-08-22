/**
 * sparsemax as a solved-for threshold (Martins & Astudillo 2016, Alg. 1) — the picture behind the
 * equation TabNet's attentive transformer uses (Arik & Pfister 2019, Fig. 4d).
 *
 * The equation is four lines of sorting and subtraction; the geometry is one sentence: draw the logits
 * as bars, drop a water line at height tau, and keep only what pokes above it. tau is chosen so the
 * poking-out parts sum to exactly 1 — so the TOTAL SHADED AREA IS INVARIANT no matter how the slider is
 * dragged. That is the whole algorithm, seen.
 *
 *     sort z descending -> k(z) = max{ k : k*z_(k) > (sum_{j<=k} z_(j)) - 1 }
 *     tau(z) = ((sum_{j<=k(z)} z_(j)) - 1) / k(z)
 *     sparsemax(z)_i = max(z_i - tau(z), 0)
 *
 * The third row runs softmax on the SAME logits, so the contrast the lesson turns on is visible rather
 * than asserted: exp(z) > 0 always, so softmax's smallest weight is small but never zero.
 *
 * Note on the drawing: both softmax and sparsemax are shift-invariant (adding a constant to every logit
 * changes nothing), so the logits are drawn shifted to sit on a zero baseline. tau is computed on the
 * same shifted vector, so the water line and the bars are on one consistent scale. The arithmetic on
 * screen is exact, computed live.
 *
 * Slider = "attention spread", a multiplier on the logits (an inverse temperature). It is the honest
 * knob: sparsity is a property of how SEPARATED the logits are, not a threshold anyone sets.
 *
 * Expected states (D = 8 features):
 *   - default (spread 1.0x): k = 4 survive, 4 exact zeros, tau ~ 0.54
 *   - "flat" (0.2x):   k = 8 — nothing is switched off, and tau goes NEGATIVE (line below the baseline)
 *   - "peaked" (2.0x): k = 2 — six exact zeros
 *   - every spread: the sparsemax row sums to 1, the softmax row sums to 1, and softmax has NO zeros
 *   - sparsemax entropy < softmax entropy at every spread (this is what L_sparse penalises)
 *
 * Usage: SparsemaxViz.mount(container, { caption })
 * api: { getSpread(), setSpread(s), state() }
 */
(function (global) {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var W = 520, H = 234;
  var PL = 98, PR = 508;

  // One illustrative row of attention logits, in feature order (names shared with tabnet-mask-viz so
  // the two widgets read as the same running example).
  var FEATURES = ["age", "income", "job", "savings", "debt", "hours", "region", "sex"];
  var LOGITS = [0.9, 1.0, 0.35, 0.7, 0.15, 0.0, 0.55, 0.25];
  var D = FEATURES.length;

  var BASE = 104;          // the zero line for the logit bars
  var PXU = 40;            // pixels per logit unit — FIXED, so dragging the slider really moves the bars
  var SPREAD_MAX = 2;      // max bar = LOGITS max (1.0) * SPREAD_MAX * PXU = 80px, the headroom above BASE
  var EPS = 1e-15;

  /** Euclidean projection onto the simplex, with the threshold and support size exposed. */
  function sparsemax(z) {
    var sorted = z.slice().sort(function (a, b) { return b - a; });
    var cum = 0, k = 1, tau = sorted[0] - 1;
    for (var i = 0; i < sorted.length; i++) {
      cum += sorted[i];
      if ((i + 1) * sorted[i] > cum - 1) { k = i + 1; tau = (cum - 1) / k; }
    }
    return { p: z.map(function (v) { return Math.max(v - tau, 0); }), tau: tau, k: k };
  }

  function softmax(z) {
    var m = Math.max.apply(null, z);
    var e = z.map(function (v) { return Math.exp(v - m); });
    var s = e.reduce(function (a, b) { return a + b; }, 0);
    return e.map(function (v) { return v / s; });
  }

  /** Shannon entropy in nats — the quantity L_sparse adds to the loss. */
  function entropy(p) {
    return p.reduce(function (acc, v) { return acc - v * Math.log(v + EPS); }, 0);
  }

  function run(spread) {
    // Shift to a zero baseline for drawing; sparsemax and softmax are both shift-invariant.
    var scaled = LOGITS.map(function (v) { return v * spread; });
    var lo = Math.min.apply(null, scaled);
    var z = scaled.map(function (v) { return v - lo; });
    var sm = sparsemax(z);
    var sx = softmax(z);
    return {
      z: z, tau: sm.tau, k: sm.k, mask: sm.p, soft: sx,
      zeros: sm.p.filter(function (v) { return v === 0; }).length,
      maskEntropy: entropy(sm.p), softEntropy: entropy(sx),
      softMin: Math.min.apply(null, sx)
    };
  }

  function el(name, attrs, text) {
    var e = document.createElementNS(SVGNS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  function fmt(v) { return v.toFixed(2).replace(/^0\./, ".").replace(/^-0\./, "−."); }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "spx-viz";

    var spread = 1.0;

    var ctl = document.createElement("div");
    ctl.className = "spx-ctl";
    var presets = [["flat", 0.2], ["default", 1.0], ["peaked", SPREAD_MAX]];
    var pbtn = {};
    presets.forEach(function (p) {
      var b = document.createElement("button");
      b.textContent = p[0];
      b.addEventListener("click", function () { setSpread(p[1]); });
      pbtn[p[0]] = b;
      ctl.appendChild(b);
    });
    var lab = document.createElement("span");
    lab.className = "spx-lab";
    var slider = document.createElement("input");
    slider.type = "range"; slider.min = "0.2"; slider.max = String(SPREAD_MAX); slider.step = "0.05";
    slider.value = String(spread); slider.className = "spx-slider";
    slider.addEventListener("input", function () { spread = parseFloat(slider.value); draw(); });
    ctl.appendChild(lab);
    ctl.appendChild(slider);
    container.appendChild(ctl);

    var scroll = document.createElement("div");
    scroll.className = "spx-scroll";
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, class: "spx-svg",
      role: "img", "aria-label": "sparsemax as a threshold on attention logits" });
    scroll.appendChild(svg);
    container.appendChild(scroll);

    var readout = document.createElement("div");
    readout.className = "spx-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "spx-caption";
    cap.textContent = config.caption ||
      "sparsemax is one subtraction: drop a threshold τ across the logits and keep what pokes above it. " +
      "τ is not set by hand — it is solved for, so the surviving weights sum to exactly 1, which is why " +
      "the total shaded area never changes as you drag the slider. Softmax, on the same logits, gives " +
      "every feature a non-zero weight at every setting.";
    container.appendChild(cap);

    var cw = (PR - PL) / D;

    function strip(y, vals, colour, label, max) {
      svg.appendChild(el("text", { x: PL - 8, y: y + 14, "text-anchor": "end", class: "spx-rowlab" },
        label));
      for (var j = 0; j < D; j++) {
        var v = vals[j];
        svg.appendChild(el("rect", {
          x: PL + j * cw + 1, y: y, width: cw - 2, height: 20, rx: 2,
          fill: colour, opacity: 0.1 + 0.9 * Math.min(v / max, 1),
          stroke: v === 0 ? "#b03a2e" : "var(--border)",
          "stroke-width": v === 0 ? 1.2 : 0.5,
          "stroke-dasharray": v === 0 ? "2,1.5" : "none"
        }));
        svg.appendChild(el("text", {
          x: PL + j * cw + cw / 2, y: y + 14, "text-anchor": "middle",
          class: "spx-cell" + (v / max > 0.55 ? " spx-cell-on" : "")
        }, v === 0 ? "0" : fmt(v)));
      }
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var st = run(spread);
      lab.textContent = "attention spread = " + spread.toFixed(2) + "×";
      presets.forEach(function (p) {
        pbtn[p[0]].className = Math.abs(spread - p[1]) < 1e-9 ? "spx-on" : "";
      });

      var tauY = BASE - st.tau * PXU;

      // feature names
      for (var j = 0; j < D; j++) {
        svg.appendChild(el("text", {
          x: PL + j * cw + cw / 2, y: 12, "text-anchor": "middle",
          class: "spx-feat" + (st.mask[j] > 0 ? " spx-feat-on" : "")
        }, FEATURES[j]));
      }

      // baseline
      svg.appendChild(el("line", { x1: PL - 4, y1: BASE, x2: PR, y2: BASE,
        stroke: "var(--border)", "stroke-width": 1 }));
      svg.appendChild(el("text", { x: PL - 8, y: BASE - 6, "text-anchor": "end",
        class: "spx-rowlab" }, "logits z"));

      // the bars: grey below tau, green above it (the part sparsemax keeps)
      for (j = 0; j < D; j++) {
        var top = BASE - st.z[j] * PXU;
        var x = PL + j * cw + 4, w = cw - 8;
        var cut = Math.max(top, Math.min(tauY, BASE));   // where the water line crosses this bar
        svg.appendChild(el("rect", { x: x, y: cut, width: w, height: Math.max(BASE - cut, 0),
          fill: "#9aa4ae", opacity: 0.45, rx: 1.5 }));
        if (top < cut) {
          svg.appendChild(el("rect", { x: x, y: top, width: w, height: cut - top,
            fill: "#1e6b3c", opacity: 0.85, rx: 1.5 }));
        }
        // values on a fixed row below the baseline: above the bar they collide with the tau line
        svg.appendChild(el("text", { x: x + w / 2, y: BASE + 12, "text-anchor": "middle",
          class: "spx-val" }, fmt(st.z[j])));
      }

      // the water line
      svg.appendChild(el("line", { x1: PL - 4, y1: tauY, x2: PR, y2: tauY,
        stroke: "#b03a2e", "stroke-width": 1.4, "stroke-dasharray": "5,3" }));
      svg.appendChild(el("text", { x: PL - 8, y: tauY + 3.5, "text-anchor": "end",
        class: "spx-tau" }, "τ = " + st.tau.toFixed(3)));

      svg.appendChild(el("text", { x: (PL + PR) / 2, y: 131, "text-anchor": "middle",
        class: "spx-axis" },
        "green = the part above τ · the green areas always total exactly 1"));

      strip(142, st.mask, "#1e6b3c", "sparsemax(z)", 0.5);
      strip(190, st.soft, "#2e6fb0", "softmax(z)", 0.5);
      svg.appendChild(el("text", { x: PR, y: 177, "text-anchor": "end", class: "spx-axis" },
        st.zeros + " exact zeros — those features are switched OFF"));
      svg.appendChild(el("text", { x: PR, y: 225, "text-anchor": "end", class: "spx-axis" },
        "no zeros, ever — smallest weight " + st.softMin.toFixed(3)));

      var html = "<p><strong>k(z) = " + st.k + "</strong> of " + D + " features clear the threshold, so " +
        "<span class='spx-m'>τ = " + st.tau.toFixed(3) + "</span> and <strong>" + st.zeros +
        "</strong> features come out at exactly zero. τ is not a hyper-parameter and not a percentile: " +
        "it is <em>solved for</em>, by taking the largest <span class='spx-m'>k</span> whose top-k logits " +
        "can still support a total of 1 after the same amount is shaved off each of them.</p>";
      if (st.tau < 0) {
        html += "<p>Notice τ has gone <strong>negative</strong> — the line sits below the baseline, every " +
          "feature survives, and sparsemax has degenerated into a soft, dense mask. Sparsity is not " +
          "something sparsemax imposes; it is what happens when the logits are <em>separated</em>. Flat " +
          "logits, no selection.</p>";
      } else if (st.k <= 2) {
        html += "<p>With the logits this separated, one feature's lead exceeds the whole budget of 1 that " +
          "the mask has to spend, so all but " + st.k + " are crushed to zero. This is the regime the " +
          "sparsity penalty <span class='spx-m'>λ_sparse</span> pushes the model toward.</p>";
      }
      html += "<p>Softmax on the <em>same</em> logits keeps every feature alive: its smallest weight is " +
        "<span class='spx-m'>" + st.softMin.toFixed(3) + "</span> — small, but not zero, so every column " +
        "still leaks into the step and still receives gradient. That is the whole reason TabNet cannot use " +
        "it: <span class='spx-m'>exp(z) > 0</span> for every finite z, so softmax has no way to say " +
        "<em>never mind this column</em>.</p>";
      html += "<p class='spx-note'>Mask entropy: <strong>" + st.maskEntropy.toFixed(2) +
        " nats</strong> for sparsemax vs <strong>" + st.softEntropy.toFixed(2) +
        " nats</strong> for softmax (the most a mask over " + D + " features can have is ln " + D + " = " +
        Math.log(D).toFixed(2) + "). Entropy is exactly the quantity <span class='spx-m'>L_sparse</span> " +
        "adds to the loss, so minimising it means pushing this number down — toward a mask that has " +
        "committed. Both operators are <strong>shift-invariant</strong> — adding the same constant to " +
        "every logit changes neither output — which is why the bars can be drawn from a zero baseline " +
        "without changing a single number on screen.</p>";
      readout.innerHTML = html;
    }

    // clamped: the bar heights are on a fixed pixel scale, so a spread past the slider's range would
    // draw off the top of the canvas
    function setSpread(s) {
      spread = Math.min(Math.max(s, 0.2), SPREAD_MAX);
      slider.value = String(spread);
      draw();
    }

    draw();
    return {
      getSpread: function () { return spread; },
      setSpread: setSpread,
      state: function () { return run(spread); }
    };
  }

  global.SparsemaxViz = { mount: mount };
})(window);
