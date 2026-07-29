/**
 * Review-triage map — the load-bearing visual of L038 ("Peer review your evaluation").
 *
 * A reviewer's job is not to LIST defects but to TRIAGE them on two independent axes:
 *   x — impact on the CONCLUSION: does fixing it change the paper's claim / the shipped decision?
 *   y — severity in the ARTIFACT: how broken is the thing itself, regardless of the headline?
 * The whole point is that these come apart — a defect can be severe in the artifact yet leave
 * the reported number untouched (a group-straddle inside the training block), or trivially small
 * in the artifact yet decide the entire conclusion (a 0.0032-nat selection margin).
 *
 * Every point is a defect already MEASURED in L036 (the audit) or L037 (the package); no new
 * numbers are introduced here — this lesson re-reads that evidence with a reviewer's eye. Clicking
 * a point fills a readout with the axis it belongs to, the verified evidence, the severity/impact
 * verdict, and the required change. Colour = the checklist axis (leakage / tuning / metrics /
 * reporting-repro).
 *
 * Plain <script> (file://-safe). Usage: ReviewTriageViz.mount(el, config?).
 *
 * Expected states (for headless verification):
 *   - mounts a .rt-viz container with a legend, an <svg> holder, and a .rt-readout
 *   - draws exactly 7 defect points (one <circle> per defect) plus 2 quadrant guide lines
 *   - a default point is selected (the selection-on-noise blocker, code T1) and its readout shown
 *   - clicking any point / calling select(code) updates the readout to that defect
 *   - getSel() returns the current code; points() returns all 7 codes
 */
(function (global) {
  "use strict";

  // axis palette (matches the lesson's checklist sections)
  var AXES = {
    LEAK:   { label: "Leakage",           colour: "#b9770e" },
    TUNE:   { label: "Tuning / selection", colour: "#2e6fb0" },
    METRIC: { label: "Metrics",           colour: "#1e6b3c" },
    REPORT: { label: "Reporting / repro", colour: "#64748b" }
  };

  // x = conclusion-impact (0 = artifact-only, 1 = changes the claim/decision)
  // y = artifact severity (0 = nit, 1 = blocker-severity in the thing itself)
  var DEFECTS = [
    {
      code: "L1", axis: "LEAK", x: 0.20, y: 0.72,
      title: "Ungrouped inner calibration split",
      origin: "L036",
      evidence: "Outer CV is correctly person-grouped, but CalibratedClassifierCV(\u2026, cv=5) takes a plain " +
        "StratifiedKFold of the training block with no groups, so a person trains the base model while " +
        "their other rows fit the isotonic map. Re-measured with a grouped inner split: log-loss " +
        "1.4232 vs the reported 1.4248, ECE 0.0360 vs 0.0363 \u2014 the change sits far inside the 0.039 fold \u03c3.",
      impact: "LOW on the conclusion \u2014 the leak is confined to the training block, so the reported test-fold " +
        "numbers are honest measurements of the model actually built.",
      severity: "HIGH in the artifact \u2014 the shipped isotonic map was tuned against optimistically sharp scores " +
        "and is mis-shaped for unseen persons.",
      required: "Required fix (artifact), not a blocker on the reported metric. Pass " +
        "cv=StratifiedGroupKFold(\u2026).split(X, y, groups)."
    },
    {
      code: "T1", axis: "TUNE", x: 0.90, y: 0.60,
      title: "Model selected on a 0.0032-nat margin",
      origin: "L036",
      evidence: "M2a shipped over M1 on \u0394 = 0.0032 nats \u2014 8 % of one fold's \u03c3, losing 2 of 5 folds, naive paired " +
        "p = 0.64, Nadeau\u2013Bengio corrected p = 0.75 \u2014 and the winner flips to M1 the moment fold 2 (which " +
        "supplies M2a's entire margin) is dropped.",
      impact: "HIGH on the conclusion \u2014 this margin IS the headline claim \"M2a is the best model\", and it is " +
        "inside the noise.",
      severity: "MID in the artifact \u2014 every number is correctly measured; the defect is the inference drawn " +
        "from them (the winner's curse).",
      required: "Blocker on the ranking. Give the tie to the simpler model (M1) or report \"no significant winner\"; " +
        "an argmin over correlated folds is a decision, not a measurement."
    },
    {
      code: "T2", axis: "TUNE", x: 0.46, y: 0.50,
      title: "Undocumented float32 cast",
      origin: "L037",
      evidence: "A .astype(np.float32) in a notebook cell \u2014 in neither the README nor src/ \u2014 flips the predicted " +
        "class for 258 of 5,587 people (max |\u0394p| = 0.326) while moving mean log-loss by only +0.00133. That " +
        "+0.00133 is 42 % of the 0.0032 margin that chose which model shipped.",
      impact: "AMBIGUOUS \u2014 negligible on the aggregate, but 42 % of the selection margin, so it silently corrupts " +
        "the very comparison in T1.",
      severity: "MID\u2013HIGH in the artifact \u2014 258 real people receive a different predicted intervention.",
      required: "Pull it into the diffable config with its measured consequence; a representation choice that decides " +
        "4.6 % of predictions is part of the model, not a memory tweak."
    },
    {
      code: "M1", axis: "METRIC", x: 0.75, y: 0.55,
      title: "\"The ECE\" is two different numbers",
      origin: "L037",
      evidence: "The same model's same out-of-fold predictions read top-label ECE 0.0332 (binned per fold, averaged) " +
        "in the selection table and 0.018 (pooled, binned once) in the README and ship-gate \u2014 a 1.87\u00d7 spread " +
        "that says nothing about the model, because ECE sums absolute gaps and thin bins bias it upward.",
      impact: "HIGH on the conclusion \u2014 the model was selected on the larger estimator and ship-gated on the " +
        "smaller one; the two are not comparable.",
      severity: "MID in the artifact \u2014 both numbers are correctly computed; the defect is that neither names its " +
        "estimator.",
      required: "Emit the estimator of record with every value: {metric: ece_top, estimator: pooled_oof, bins: 15, n: 5587}."
    },
    {
      code: "M2", axis: "METRIC", x: 0.70, y: 0.52,
      title: "A ship-gate below its own noise floor",
      origin: "L037",
      evidence: "The one failed gate \u2014 the 107-row age=missing slice, ECE 0.094 against a 0.05 threshold \u2014 sits " +
        "below the 0.1071 \u00b1 0.0297 a perfectly-calibrated model scores at n = 107. The gate cannot be passed " +
        "by ANY model, so its CONDITIONAL verdict is uninterpretable in either direction.",
      impact: "HIGH on the conclusion \u2014 the report's one caveat rests on a measurement that cannot see what it " +
        "claims to.",
      severity: "MID in the artifact \u2014 there may be a real problem in that slice; this estimator at this n simply " +
        "cannot detect it.",
      required: "Compare every threshold to its estimator's noise floor; drop or re-scope a gate that a perfect model " +
        "would fail."
    },
    {
      code: "R1", axis: "REPORT", x: 0.16, y: 0.76,
      title: "No lockfile \u2014 only >= constraints",
      origin: "L037",
      evidence: "requirements ship as lightgbm>=4.0 / scikit-learn>=1.5 with no lockfile; lightgbm 4.5.0 + " +
        "scikit-learn 1.9.0 \u2014 both satisfying those constraints \u2014 raise TypeError (force_all_finite). The " +
        "specific resolution that produced every reported number is written down nowhere.",
      impact: "LOW on this number \u2014 it does not change the value, but it blocks a reviewer from ever regenerating it.",
      severity: "HIGH for reproducibility \u2014 a review's implicit \"I could reproduce this\" is simply false.",
      required: "Ship a lockfile (exact versions + hashes for every transitive dependency); a constraint describes a " +
        "space of environments, not the one you ran."
    },
    {
      code: "R2", axis: "REPORT", x: 0.80, y: 0.45,
      title: "A single fold draw, reported as the result",
      origin: "L037",
      evidence: "Re-drawing the person-grouped folds (splitter seed 0\u21924) moves the headline mean log-loss over a " +
        "0.0166-nat range \u2014 5\u00d7 the 0.0032 margin that chose the shipped model. The report quotes one draw with " +
        "no across-split spread.",
      impact: "HIGH on the conclusion \u2014 the reported ranking is a sample of one from a distribution wide enough to " +
        "reorder the models.",
      severity: "MID in the artifact \u2014 the number is real; it is the uncertainty around it that is missing.",
      required: "Report repeated cross-validation (e.g. 5\u00d75) with a spread, so the split lottery is visible (L023)."
    }
  ];

  var VB_W = 660, VB_H = 440;
  var M = { l: 62, r: 22, t: 26, b: 58 };
  var PW = VB_W - M.l - M.r, PH = VB_H - M.t - M.b;

  function X(v) { return M.l + v * PW; }
  function Y(v) { return M.t + (1 - v) * PH; }

  function svgEl(tag, attrs) {
    var e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (var k in attrs) { if (attrs.hasOwnProperty(k)) e.setAttribute(k, attrs[k]); }
    return e;
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.classList.add("rt-viz");

    // legend
    var legend = document.createElement("div");
    legend.className = "rt-legend";
    Object.keys(AXES).forEach(function (k) {
      var span = document.createElement("span");
      span.className = "rt-key";
      span.innerHTML = '<i class="rt-dot" style="background:' + AXES[k].colour + '"></i>' + AXES[k].label;
      legend.appendChild(span);
    });
    container.appendChild(legend);

    // svg holder
    var holder = document.createElement("div");
    holder.className = "rt-svg-holder";
    container.appendChild(holder);

    var svg = svgEl("svg", { viewBox: "0 0 " + VB_W + " " + VB_H, class: "rt-svg",
      role: "img", "aria-label": "Defects triaged by conclusion-impact and artifact severity" });
    holder.appendChild(svg);

    // quadrant guides
    svg.appendChild(svgEl("line", { x1: X(0.5), y1: Y(0), x2: X(0.5), y2: Y(1),
      class: "rt-guide" }));
    svg.appendChild(svgEl("line", { x1: X(0), y1: Y(0.5), x2: X(1), y2: Y(0.5),
      class: "rt-guide" }));

    // quadrant labels
    var quads = [
      { x: 0.02, y: 0.97, t: "fix the artifact", anchor: "start" },
      { x: 0.98, y: 0.97, t: "blockers \u2014 fix before you believe it", anchor: "end" },
      { x: 0.02, y: 0.03, t: "nits", anchor: "start" },
      { x: 0.98, y: 0.03, t: "overstated conclusion", anchor: "end" }
    ];
    quads.forEach(function (q) {
      var tx = svgEl("text", { x: X(q.x), y: Y(q.y), class: "rt-quad", "text-anchor": q.anchor });
      tx.appendChild(document.createTextNode(q.t));
      svg.appendChild(tx);
    });

    // axes
    var xlab = svgEl("text", { x: X(0.5), y: VB_H - 16, class: "rt-axis", "text-anchor": "middle" });
    xlab.appendChild(document.createTextNode("impact on the conclusion \u2192"));
    svg.appendChild(xlab);
    var ylab = svgEl("text", { x: 16, y: Y(0.5), class: "rt-axis",
      "text-anchor": "middle", transform: "rotate(-90 16 " + Y(0.5) + ")" });
    ylab.appendChild(document.createTextNode("severity in the artifact \u2192"));
    svg.appendChild(ylab);

    // points
    var readout = document.createElement("div");
    readout.className = "rt-readout";

    var byCode = {};
    var nodes = [];
    DEFECTS.forEach(function (d) {
      byCode[d.code] = d;
      var g = svgEl("g", { class: "rt-pt", "data-code": d.code });
      var c = svgEl("circle", { cx: X(d.x), cy: Y(d.y), r: 15,
        fill: AXES[d.axis].colour, class: "rt-circ" });
      var t = svgEl("text", { x: X(d.x), y: Y(d.y) + 4, class: "rt-code", "text-anchor": "middle" });
      t.appendChild(document.createTextNode(d.code));
      g.appendChild(c);
      g.appendChild(t);
      g.addEventListener("click", function () { select(d.code); });
      svg.appendChild(g);
      nodes.push(g);
    });

    container.appendChild(readout);

    function select(code) {
      var d = byCode[code];
      nodes.forEach(function (g) {
        if (g.getAttribute("data-code") === code) g.classList.add("rt-active");
        else g.classList.remove("rt-active");
      });
      readout.innerHTML =
        '<div class="rt-r-head"><span class="rt-r-badge" style="background:' + AXES[d.axis].colour + '">' +
          d.code + " \u00b7 " + AXES[d.axis].label + '</span> <span class="rt-r-title">' + d.title +
          '</span> <span class="rt-r-origin">measured in ' + d.origin + '</span></div>' +
        '<p class="rt-r-ev">' + d.evidence + "</p>" +
        '<p class="rt-r-line"><strong>Conclusion-impact:</strong> ' + d.impact + "</p>" +
        '<p class="rt-r-line"><strong>Artifact severity:</strong> ' + d.severity + "</p>" +
        '<p class="rt-r-req">' + d.required + "</p>";
    }

    select(config.start || "T1");

    return {
      select: select,
      getSel: function () {
        var a = nodes.filter(function (g) { return g.classList.contains("rt-active"); });
        return a.length ? a[0].getAttribute("data-code") : null;
      },
      points: function () { return DEFECTS.map(function (d) { return d.code; }); }
    };
  }

  global.ReviewTriageViz = { mount: mount };
})(window);
