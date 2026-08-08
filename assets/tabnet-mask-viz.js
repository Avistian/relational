/**
 * Sequential attention — TabNet's masks and the prior scale (Arik & Pfister 2019, Fig. 4a/4d).
 *
 * The one mechanism that makes TabNet "TabNet": at each decision step i it builds a SPARSE mask over
 * the features and reasons only from the ones it selects,
 *
 *     M[i] = sparsemax( P[i-1] . h_i(a[i-1]) )        P[i] = prod_{j<=i} (gamma - M[j]),  P[0] = 1
 *
 * so a feature already spent at an earlier step is pushed down in the next step's logits. `gamma` is the
 * relaxation knob: at gamma = 1 a fully-used feature is banned outright (its prior hits 0), and as gamma
 * grows the model may revisit it.
 *
 * What this viz makes seeable, per step:
 *   - the mask M[i] as a strip over the features (exact zeros — sparsemax, not softmax);
 *   - the prior P[i] that the previous steps' masks produced, i.e. what is still "affordable";
 *   - the running aggregate M_agg, the paper's global feature-importance attribution.
 * Drag gamma to watch step 2/3 either avoid (low gamma) or reuse (high gamma) step 1's features.
 *
 * The masks here are ILLUSTRATIVE (a small hand-built pattern for a 10-feature row, so the mechanism is
 * legible), not fitted scores; the lesson's *verified* mask numbers are the Syn2/Syn4 results from
 * labs/_verify_l043.py. The arithmetic on screen (sparsemax renormalisation, the prior product, the
 * aggregate) is computed live and is exact.
 *
 * Usage: TabnetMaskViz.mount(container, { caption })
 * Expected states:
 *   - default: step 1 selected, gamma 1.5; step 1's prior is all ones ("nothing spent yet").
 *   - stepping to 2 or 3: prior drops below 1 exactly on the features the earlier steps used.
 *   - gamma = 1.0: a feature that received mask 1.0 has prior exactly 0 at the next step (banned).
 *   - the aggregate strip always sums to 1.
 * api: { getStep(), setStep(i), getGamma(), setGamma(g), state() }
 */
(function (global) {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var STEP_C = ["#2e6fb0", "#1e6b3c", "#b9770e"];

  // A 10-feature row. Raw attention logits per step (pre-prior, pre-sparsemax) chosen so the steps
  // naturally want DIFFERENT feature groups — which is what sequential attention is for.
  var FEATURES = ["age", "income", "job", "savings", "debt", "hours", "region", "sex", "car", "noise"];
  var LOGITS = [
    [2.6, 2.2, 0.3, 0.1, 0.0, 0.2, -0.4, -0.6, -0.8, -1.0],   // step 1 wants age/income
    [0.9, 0.8, 2.4, 2.1, 0.4, 0.1, -0.3, -0.5, -0.7, -1.0],   // step 2 wants job/savings
    [0.5, 0.4, 0.7, 0.6, 2.3, 2.0, -0.2, -0.4, -0.6, -1.0]    // step 3 wants debt/hours
  ];
  var N_STEPS = 3, D = FEATURES.length;

  /** Euclidean projection onto the simplex (Martins & Astudillo 2016, Alg. 1) — exact, with zeros. */
  function sparsemax(z) {
    var sorted = z.slice().sort(function (a, b) { return b - a; });
    var cum = 0, k = 0, tau = 0;
    for (var i = 0; i < sorted.length; i++) {
      cum += sorted[i];
      if ((i + 1) * sorted[i] > cum - 1) { k = i + 1; tau = (cum - 1) / k; }
    }
    return z.map(function (v) { return Math.max(v - tau, 0); });
  }

  /** Run all steps for a given gamma: returns masks, priors (the prior USED at each step), aggregate. */
  function run(gamma) {
    var prior = [], masks = [], priors = [], i, j;
    for (j = 0; j < D; j++) prior.push(1);                       // P[0] = 1
    for (i = 0; i < N_STEPS; i++) {
      priors.push(prior.slice());
      var scaled = LOGITS[i].map(function (v, jj) { return prior[jj] * v; });
      var M = sparsemax(scaled);
      masks.push(M);
      prior = prior.map(function (p, jj) { return Math.max(p * (gamma - M[jj]), 0); });
    }
    // M_agg with eta = 1 per step (the illustrative case): normalise so it sums to 1.
    var agg = [], total = 0;
    for (j = 0; j < D; j++) {
      var s = 0;
      for (i = 0; i < N_STEPS; i++) s += masks[i][j];
      agg.push(s); total += s;
    }
    agg = agg.map(function (v) { return total > 0 ? v / total : 0; });
    return { masks: masks, priors: priors, agg: agg };
  }

  function el(name, attrs, text) {
    var e = document.createElementNS(SVGNS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "tnm-viz";

    var step = 0, gamma = 1.5;

    // ---- controls
    var ctl = document.createElement("div");
    ctl.className = "tnm-ctl";
    var stepBtns = [];
    for (var i = 0; i < N_STEPS; i++) {
      (function (idx) {
        var b = document.createElement("button");
        b.textContent = "step " + (idx + 1);
        b.addEventListener("click", function () { setStep(idx); });
        ctl.appendChild(b);
        stepBtns.push(b);
      })(i);
    }
    var gLab = document.createElement("span");
    gLab.className = "tnm-glab";
    var gSlider = document.createElement("input");
    gSlider.type = "range"; gSlider.min = "1"; gSlider.max = "2.5"; gSlider.step = "0.1";
    gSlider.value = String(gamma); gSlider.className = "tnm-slider";
    gSlider.addEventListener("input", function () { gamma = parseFloat(gSlider.value); draw(); });
    ctl.appendChild(gLab);
    ctl.appendChild(gSlider);
    container.appendChild(ctl);

    var W = 520, H = 250;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", class: "tnm-svg" });
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "tnm-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "tnm-caption";
    cap.textContent = config.caption ||
      "Sequential attention on one row. Each step's mask is a sparsemax over the features, scaled by the " +
      "prior P[i-1] that records what earlier steps already spent — so the steps divide the features " +
      "between them instead of all looking at the same ones. Drag gamma: at 1.0 a fully-used feature is " +
      "banned from later steps; higher gamma lets the model revisit it.";
    container.appendChild(cap);

    var PL = 66, PR = W - 16;
    var cw = (PR - PL) / D;

    function cellRow(y, vals, colour, label, opts) {
      opts = opts || {};
      svg.appendChild(el("text", { x: PL - 8, y: y + 13, "text-anchor": "end", class: "tnm-rowlab" },
        label));
      for (var j = 0; j < D; j++) {
        var v = vals[j];
        var frac = opts.max ? Math.min(v / opts.max, 1) : Math.min(v, 1);
        svg.appendChild(el("rect", {
          x: PL + j * cw + 1, y: y, width: cw - 2, height: 18, rx: 2,
          fill: colour, opacity: 0.12 + 0.88 * frac,
          stroke: v === 0 ? "#b03a2e" : "var(--border)",
          "stroke-width": v === 0 ? 1.2 : 0.5,
          "stroke-dasharray": v === 0 ? "2,1.5" : "none"
        }));
        svg.appendChild(el("text", {
          x: PL + j * cw + cw / 2, y: y + 13, "text-anchor": "middle",
          class: "tnm-cell" + (frac > 0.55 ? " tnm-cell-on" : "")
        }, v === 0 ? "0" : v.toFixed(2).replace(/^0/, "")));
      }
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var st = run(gamma);
      gLab.textContent = "γ = " + gamma.toFixed(1);
      stepBtns.forEach(function (b, idx) {
        b.className = idx === step ? "tnm-on" : "";
      });

      // feature names (column headers)
      for (var j = 0; j < D; j++) {
        svg.appendChild(el("text", {
          x: PL + j * cw + cw / 2, y: 12, "text-anchor": "middle",
          class: "tnm-feat" + (st.masks[step][j] > 0 ? " tnm-feat-on" : "")
        }, FEATURES[j]));
      }

      // prior used at this step, this step's mask, then the aggregate
      cellRow(22, st.priors[step], "#7f8c8d", "P[" + step + "]", { max: Math.max(1, gamma) });
      cellRow(58, st.masks[step], STEP_C[step], "M[" + (step + 1) + "]", { max: 0.6 });

      svg.appendChild(el("line", { x1: PL, y1: 88, x2: PR, y2: 88,
        stroke: "var(--border)", "stroke-width": 1 }));

      // all steps stacked, so the division of labour is visible at a glance
      for (var i = 0; i < N_STEPS; i++) {
        cellRow(96 + i * 26, st.masks[i], STEP_C[i], "step " + (i + 1), { max: 0.6 });
      }
      cellRow(96 + N_STEPS * 26 + 8, st.agg, "#6c3483", "M_agg", { max: 0.35 });
      svg.appendChild(el("text", { x: PL - 8, y: 96 + N_STEPS * 26 + 40, "text-anchor": "end",
        class: "tnm-rowlab" }, ""));
      svg.appendChild(el("text", { x: (PL + PR) / 2, y: H - 4, "text-anchor": "middle",
        class: "tnm-axis" },
        "each row sums to 1 · red dashed = exact zero (feature switched off at that step)"));

      // readout
      var M = st.masks[step], P = st.priors[step];
      var picked = [], zeros = 0, jj;
      for (jj = 0; jj < D; jj++) {
        if (M[jj] > 0) picked.push(FEATURES[jj] + " " + M[jj].toFixed(2));
        if (M[jj] === 0) zeros++;
      }
      // An untouched feature keeps prior exactly gamma at every step after the first; anything BELOW
      // gamma has been spent earlier, and exactly 0 means banned outright.
      var banned = [];
      for (jj = 0; jj < D; jj++) if (P[jj] === 0) banned.push(FEATURES[jj]);
      var spentBefore = [];
      for (jj = 0; jj < D; jj++) if (P[jj] > 0 && P[jj] < gamma - 1e-9) spentBefore.push(FEATURES[jj]);

      var html = "<p><strong>Step " + (step + 1) + "</strong> selects <strong>" + picked.length +
        "</strong> of " + D + " features (" + zeros + " are exactly zero — sparsemax, not softmax): " +
        "<span class='tnm-mono'>" + picked.join(" · ") + "</span></p>";
      if (step === 0) {
        html += "<p>Its prior <span class='tnm-mono'>P[0]</span> is all ones: nothing has been spent yet, " +
          "so the mask is decided purely by the attention logits.</p>";
      } else {
        html += "<p>An untouched feature carries prior <span class='tnm-mono'>γ = " + gamma.toFixed(1) +
          "</span>. This step's prior <span class='tnm-mono'>P[" + step + "] = ∏(γ − M[j])</span> sits " +
          "<em>below</em> that on " + (spentBefore.length + banned.length) + " feature(s) earlier steps " +
          "spent" + (spentBefore.length ? " (" + spentBefore.join(", ") + ")" : "") +
          (banned.length ? ", and at exactly <strong>0</strong> on " + banned.join(", ") +
            " — banned outright" : "") +
          ". Multiplying the logits by that prior is what pushes this step toward <em>different</em> " +
          "features. At γ = 1 the leftover budget is exactly <span class='tnm-mono'>1 − M</span>, so a " +
          "feature used in full (M = 1) is banned and partial use is suppressed in proportion.</p>";
      }
      html += "<p class='tnm-note'><strong>M_agg</strong> is the paper's global attribution: the masks " +
        "summed across steps (weighted by each step's decision contribution η) and normalised — the " +
        "number you read as \"how much did this model use this feature\".</p>";
      readout.innerHTML = html;
    }

    function setStep(i) { step = i; draw(); }
    function setGamma(g) { gamma = g; gSlider.value = String(g); draw(); }

    draw();
    return {
      getStep: function () { return step; },
      setStep: setStep,
      getGamma: function () { return gamma; },
      setGamma: setGamma,
      state: function () { return run(gamma); }
    };
  }

  global.TabnetMaskViz = { mount: mount };
})(window);
