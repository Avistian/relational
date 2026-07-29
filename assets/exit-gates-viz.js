/**
 * Year-1 exit protocol gates — second mechanistic visual of L040 ("Year 1 exit exam").
 *
 * Before any of BEAT / TIE / EXPLAIN is a legitimate exit verdict, six gates must close.
 * Click a gate chip to see (a) what it demands, (b) the Year-1 lesson that installed it,
 * and (c) the failure mode if you skip it. Default = regenerable-baseline (the exit's
 * first concrete deliverable).
 *
 * Plain <script> (file://-safe). Usage: ExitGatesViz.mount(el, config?).
 *
 * Expected states:
 *   - mounts .eg-viz with legend, gate chips, and .eg-readout
 *   - exactly 6 gates
 *   - default selection "g-regen"
 *   - select(id) / clicking updates readout; getSel() / gates() for tests
 */
(function (global) {
  "use strict";

  var GATES = [
    {
      id: "g-regen", label: "1 · Regenerable baseline", colour: "#64748b",
      title: "A tuned (or fixed-default) XGBoost number anyone can regenerate",
      demand: "Emit the reference ROC-AUC (and challenger) from a disclosed config: same data, same seed, preprocessing inside the pipeline. Prefer a number that lands in a published band (adult fixed-default XGB ≈ 0.92–0.93; L020 verified 0.9282).",
      lesson: "L010 / L020 / L037 — the leakage spine, the fair-comparison contract, and the package that regenerates its own headline.",
      failure: "A peak number without a regenerable recipe is not an exit pass — it is configuration debt (Sculley 2015 / Pineau 2021)."
    },
    {
      id: "g-protocol", label: "2 · Fair protocol", colour: "#2e6fb0",
      title: "Five things fixed before the sixth (the verdict)",
      demand: "Same dataset, same split (test touched once), same metric with prevalence stated, disclosed tuning budget on train only, same preprocessing scope (per-fold pipeline). Then report the verdict honestly.",
      lesson: "L020 fair-comparison contract; L021 deployment-matched split; L038 two-pipelines-one-standard.",
      failure: "Beating an undertuned, leaky, or differently-split XGB is how 'SOTA' papers die under review."
    },
    {
      id: "g-challenger", label: "3 · Honest challenger", colour: "#1e6b3c",
      title: "One real attempt under the same contract",
      demand: "Fit at least one Year-1 challenger (tuned LightGBM, leak-free stack, or another disclosed single-table upgrade) under the identical protocol. A beat without a challenger is undefined; an explain-why-not without an attempt is unearned.",
      lesson: "L015–L018 (boosters, tuning, stacking); L028–L033 (what else Year 1 already tried).",
      failure: "Declaring 'trees cannot be beaten' without running a challenger is folklore, not an exit."
    },
    {
      id: "g-verdict", label: "4 · Honest fork", colour: "#b9770e",
      title: "Classify BEAT / TIE / EXPLAIN — never soft-sell a match as a win",
      demand: "Compare challenger − reference to a disclosed noise band (here ±0.002 ROC-AUC on adult, matching L020 CHECK tolerance). Above → BEAT candidate; inside → TIE; below → FAIL then EXPLAIN. Ties are full passes.",
      lesson: "L023 corrected tests; L030 'no significant winner' as a legitimate checkpoint verdict.",
      failure: "Calling Δ=+0.001 a 'beat' is the same defect as the homework's 0.0032-nat winner's curse (L036)."
    },
    {
      id: "g-biases", label: "5 · Three biases written", colour: "#7c3aed",
      title: "Grinsztajn's mechanisms in your own words",
      demand: "Whether you beat or not, write the three inductive biases (irregular targets, privileged orientation, junk-feature robustness) with one evidence-of-record number each and one flip condition each. This is the curriculum's written exit criterion.",
      lesson: "L025–L027 mechanisms; L039 synthesis Section II; M47 repair ('not more powerful').",
      failure: "A number without a mechanism does not prove Year-1 understanding — fluency ≠ storage strength."
    },
    {
      id: "g-essay", label: "6 · Essay stance", colour: "#b03a2e",
      title: "Stand on or revise the L039 synthesis claim",
      demand: "One explicit sentence: STAND (exit result agrees with the working claim) or REVISE (name what changed — a real beat, a broken boundary, a protocol defect). The open burden (no fair-bar RDL win yet) must stay open unless you actually have that win.",
      lesson: "L039 working claim + credibility coda; thesis dossier C1/C3/C4.",
      failure: "Finishing Year 1 without linking the experiment to the written claim wastes the synthesis."
    }
  ];

  function find(id) {
    for (var i = 0; i < GATES.length; i++) if (GATES[i].id === id) return GATES[i];
    return null;
  }

  function renderReadout(el, id) {
    var g = find(id);
    if (!g) return;
    el.innerHTML =
      '<p class="eg-r-head"><span class="eg-r-badge" style="background:' + g.colour + '">' + g.label +
      '</span> <span class="eg-r-title">' + g.title + "</span></p>" +
      '<p class="eg-r-ev"><strong>Demand.</strong> ' + g.demand + "</p>" +
      '<p class="eg-r-ev"><strong>Installed by.</strong> ' + g.lesson + "</p>" +
      '<p class="eg-r-fail"><strong>If skipped.</strong> ' + g.failure + "</p>";
  }

  function mount(root, config) {
    config = config || {};
    var sel = config.defaultSel || "g-regen";
    root.innerHTML = "";
    var wrap = document.createElement("div");
    wrap.className = "eg-viz";

    var legend = document.createElement("div");
    legend.className = "eg-legend";
    legend.innerHTML =
      '<span class="eg-key"><span class="eg-dot" style="background:#64748b"></span> regenerable</span>' +
      '<span class="eg-key"><span class="eg-dot" style="background:#2e6fb0"></span> protocol</span>' +
      '<span class="eg-key"><span class="eg-dot" style="background:#1e6b3c"></span> challenger</span>' +
      '<span class="eg-key"><span class="eg-dot" style="background:#b9770e"></span> fork</span>' +
      '<span class="eg-key"><span class="eg-dot" style="background:#7c3aed"></span> biases</span>' +
      '<span class="eg-key"><span class="eg-dot" style="background:#b03a2e"></span> essay</span>';
    wrap.appendChild(legend);

    var chips = document.createElement("div");
    chips.className = "eg-chips";
    GATES.forEach(function (g) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "eg-chip";
      b.setAttribute("data-id", g.id);
      b.style.setProperty("--eg-c", g.colour);
      b.innerHTML = '<span class="eg-chip-lab">' + g.label + "</span>";
      b.addEventListener("click", function () { api.select(g.id); });
      chips.appendChild(b);
    });
    wrap.appendChild(chips);

    var readout = document.createElement("div");
    readout.className = "eg-readout";
    wrap.appendChild(readout);
    root.appendChild(wrap);

    var api = {
      select: function (id) {
        sel = id;
        renderReadout(readout, id);
        var buttons = wrap.querySelectorAll(".eg-chip");
        for (var i = 0; i < buttons.length; i++) {
          var on = buttons[i].getAttribute("data-id") === id;
          if (on) buttons[i].classList.add("eg-on");
          else buttons[i].classList.remove("eg-on");
        }
      },
      getSel: function () { return sel; },
      gates: function () { return GATES.map(function (g) { return g.id; }); }
    };
    api.select(sel);
    return api;
  }

  global.ExitGatesViz = { mount: mount };
})(typeof window !== "undefined" ? window : global);
