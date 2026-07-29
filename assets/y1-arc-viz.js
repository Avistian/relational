/**
 * Year-1 argument map — the load-bearing visual of L039 ("Year 1 synthesis essay").
 *
 * A synthesis essay is not a chronological rehash. It is an ARGUMENT with four load-bearing
 * moves, and each quarter of Year 1 supplied one of them:
 *   Q1 — evaluation discipline (the leakage spine; you cannot claim anything without it)
 *   Q2 — the incumbent (a tuned GBDT / stack you can actually beat or fail to)
 *   Q3 — why the incumbent wins on flat tables (Grinsztajn's three inductive biases + honest bar)
 *   Q4 — where the claim stops (single-table cleverness exhausted; the join destroys structure)
 *
 * Click a quarter band or a milestone chip to fill the readout with (a) what that piece is,
 * (b) the verified number / finding it contributes, and (c) the sentence it earns in the essay.
 * Colour = quarter. Default selection = Q3 / three biases (the essay's "why" section).
 *
 * Plain <script> (file://-safe). Usage: Y1ArcViz.mount(el, config?).
 *
 * Expected states (for headless verification):
 *   - mounts a .ya-viz with legend, four quarter bands, and a .ya-readout
 *   - exactly 12 milestone chips (3 per quarter)
 *   - default selection is "q3-biases"
 *   - select(id) / clicking a chip updates the readout; getSel() returns the current id
 *   - quarters() returns ["Q1","Q2","Q3","Q4"]; chips() returns all 12 ids
 */
(function (global) {
  "use strict";

  var QUARTERS = [
    {
      id: "Q1", label: "Q1 · Evaluation discipline", colour: "#64748b",
      blurb: "You cannot claim a win you cannot measure honestly.",
      chips: [
        {
          id: "q1-leakage", label: "Leakage spine",
          title: "Design matrix, splits, pipelines",
          evidence: "L001–L005 / L010: the held-out test is touched once; every fit-bearing transform lives inside a per-fold pipeline; grouped / nested CV when rows are dependent. The Q1 checkpoint ships a leak-free HistGBDT baseline (CV-PR 0.470 vs Dummy 0.154).",
          essay: "Essay move: open by naming the measurement contract. A claim about what trees beat is worthless without the spine that keeps the number honest."
        },
        {
          id: "q1-metrics", label: "Metrics & imbalance",
          title: "ROC vs PR, calibration, missingness",
          evidence: "L006–L009: missingness has a taxonomy (MCAR/MAR/MNAR); class imbalance makes ROC lie and PR + prevalence the honest pair; calibration is a separate claim from discrimination; feature engineering is a budget, not an infinite treadmill.",
          essay: "Essay move: pick the metric before the model. Under imbalance, lead with PR-AUC and state prevalence; if probabilities are used, name the calibration estimator."
        },
        {
          id: "q1-checkpoint", label: "Q1 checkpoint",
          title: "A reproducible single-table baseline",
          evidence: "L010: Dummy PR-AUC = prevalence; Logistic CV-PR 0.381; HistGBDT 0.470 / ROC 0.789 / Brier 0.108 — the tree wins the Q1 bake-off under the leakage spine. This is the floor every later model must clear.",
          essay: "Essay move: the floor is high and cheap. Any later 'DL beats trees' claim that loses to this HistGBDT is not a contribution."
        }
      ]
    },
    {
      id: "Q2", label: "Q2 · The incumbent", colour: "#2e6fb0",
      blurb: "A tuned gradient-boosted tree is the thing you must beat.",
      chips: [
        {
          id: "q2-boosters", label: "Tree ensembles",
          title: "Partitions → bagging → boosting → XGB/LGBM/CatBoost",
          evidence: "L011–L016: a tree is recursive axis-aligned partitions; bagging averages variance; boosting fits residuals; XGBoost / LightGBM / CatBoost are the production incarnations. Tuning a strong default barely moves it — the incumbent is stubborn.",
          essay: "Essay move: name the incumbent precisely. 'Trees' means a leak-free, tuned GBDT (and, later, a stacked ensemble) — not an untuned DecisionTreeClassifier."
        },
        {
          id: "q2-tuning", label: "Tuning & stacking",
          title: "Fixed budget, OOF meta-learner",
          evidence: "L017–L018: RandomizedSearchCV with a disclosed trial budget on train only; stacking builds out-of-fold meta-features so the meta-learner never sees its own training rows. The real single-table bar is a leak-free stacked ensemble, not one default.",
          essay: "Essay move: disclose the budget. A peak number without a budget curve (L024) is an invitation for a skeptic to retune the baseline harder than you did."
        },
        {
          id: "q2-preview", label: "When trees win (preview)",
          title: "Three inductive biases, sketched",
          evidence: "L019–L020: Grinsztajn preview — irregular targets, uninformative features, orientation. Q2 checkpoint on adult: fixed-default XGBoost ROC-AUC 0.928; a big 'win' over it is a leak hypothesis first.",
          essay: "Essay move: preview the 'why' so the later full treatment (Q3) has somewhere to land. The checkpoint proves you can reproduce the incumbent under a fair contract."
        }
      ]
    },
    {
      id: "Q3", label: "Q3 · Why trees win", colour: "#1e6b3c",
      blurb: "Inductive bias match — not 'more power'.",
      chips: [
        {
          id: "q3-rigor", label: "Evaluation rigor",
          title: "Temporal splits, leakage audit, significance",
          evidence: "L021–L023: random-CV 0.846 vs temporal 0.758 on drifting data; Kapoor & Narayanan leakage across 17 fields / 329 papers; a naive paired t on overlapping folds over-rejects (demo +0.0098, naive p=1.2e−5 vs corrected p=0.19). A win needs a corrected test and an effect size.",
          essay: "Essay move: prove the gap is not noise before explaining it. Significance without magnitude, or a mean without a test, is half an answer."
        },
        {
          id: "q3-biases", label: "Three biases",
          title: "Smoothness · rotation · uninformative features",
          evidence: "L024–L027 (Grinsztajn 2022): (1) trees fit irregular targets; smoothing the target collapses the GBT–MLP gap. (2) trees are NOT rotationally invariant — a lossless random rotation reverses the ranking (tree 0.987→0.747, MLP 0.862→0.869). (3) trees gate junk via split gain; adding 100 noise columns costs MLP 0.084 vs GBT 0.032 and reverses a smooth-target MLP win.",
          essay: "Essay move: this IS the 'why'. Trees win when tabular targets are jagged, columns carry individual meaning, and junk features abound — the default regime of flat business tables."
        },
        {
          id: "q3-bar", label: "Honest bar",
          title: "ResNet + AutoML + checkpoint tie",
          evidence: "L028–L030: a tuned MLP/ResNet is the honest neural baseline (Gorishniy); AutoML only TIES a tuned XGB on credit_g; the Q3 checkpoint finds GBDT vs MLP +0.0081 ROC-AUC over 25 folds with corrected p=0.64 → no significant winner. Returns to single-table search/architecture are nearly exhausted.",
          essay: "Essay move: state what trees beat (weak nets, untuned baselines) and what they merely match (honest nets, AutoML). An honest essay reports the tie."
        }
      ]
    },
    {
      id: "Q4", label: "Q4 · Where the claim stops", colour: "#b9770e",
      blurb: "Single-table cleverness dies; the join is the open frontier.",
      chips: [
        {
          id: "q4-deflate", label: "Exhaustion cascade",
          title: "Embeddings, attention, FE all tie or fail",
          evidence: "L031–L033: entity embeddings TIE fair one-hot on credit_g; TabTransformer MATCHES trees on supervised tabular (+1.0% is over other DL); hand-crafted FE peaks at 3 features (+0.005 inside ±0.03 noise) then goes negative. Six consecutive lessons: more single-table cleverness does not buy a win.",
          essay: "Essay move: the claim has a ceiling inside one table. Remaining upside is representational — and the representations that still look unpaid live across the join."
        },
        {
          id: "q4-join", label: "What joins destroy",
          title: "Flatten is lossy; aggregation collisions",
          evidence: "L034–L035: the design matrix is manufactured (grain → join → aggregate → PIT). Ada (rising spend, 3 products) and Bo (falling, 1 product) flatten to the identical row n=3/total=90/avg=30/max=50, so a model returns P(churn)=0.502 for both though labels differ. Cardinality, identity, order, multi-hop paths are discarded — Fey 2024 issue (4).",
          essay: "Essay move: this is where 'trees win on tabular' STOPS being the whole story. Trees win on the flattened table; the thesis is that the flatten already threw the signal away."
        },
        {
          id: "q4-cred", label: "Credibility arc",
          title: "Audit → package → peer review",
          evidence: "L036–L038: the learner's own pipeline — ungrouped inner calibration (artifact, not number); 0.0032-nat winner's curse (corrected p=0.75); float32 = 42% of that margin; ECE 0.0332 vs 0.018; splitter seed spans 0.0166 nats. Peer-review verdict: major revision. Two pipelines, one standard.",
          essay: "Essay move: close with the immune system. You may only claim 'trees beat X' (or 'RDL beats trees') after a severity-graded review that holds the baseline to the model's standard."
        }
      ]
    }
  ];

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "ya-viz";

    var sel = config.defaultId || "q3-biases";
    var chipIndex = {};

    var legend = document.createElement("div");
    legend.className = "ya-legend";
    QUARTERS.forEach(function (q) {
      var key = document.createElement("span");
      key.className = "ya-key";
      key.innerHTML = '<span class="ya-dot" style="background:' + q.colour + '"></span>' + q.label;
      legend.appendChild(key);
    });
    container.appendChild(legend);

    var bands = document.createElement("div");
    bands.className = "ya-bands";
    container.appendChild(bands);

    QUARTERS.forEach(function (q) {
      var band = document.createElement("div");
      band.className = "ya-band";
      band.setAttribute("data-q", q.id);
      band.style.borderLeftColor = q.colour;

      var head = document.createElement("div");
      head.className = "ya-band-head";
      head.innerHTML = '<span class="ya-band-title" style="color:' + q.colour + '">' + q.label +
        '</span><span class="ya-band-blurb">' + q.blurb + "</span>";
      band.appendChild(head);

      var row = document.createElement("div");
      row.className = "ya-chips";
      q.chips.forEach(function (c) {
        chipIndex[c.id] = { quarter: q, chip: c };
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "ya-chip";
        btn.setAttribute("data-id", c.id);
        btn.style.setProperty("--ya-c", q.colour);
        btn.innerHTML = '<span class="ya-chip-lab">' + c.label + "</span>";
        btn.addEventListener("click", function () { select(c.id); });
        row.appendChild(btn);
      });
      band.appendChild(row);
      bands.appendChild(band);
    });

    var readout = document.createElement("div");
    readout.className = "ya-readout";
    container.appendChild(readout);

    function select(id) {
      if (!chipIndex[id]) return;
      sel = id;
      var buttons = container.querySelectorAll(".ya-chip");
      for (var i = 0; i < buttons.length; i++) {
        if (buttons[i].getAttribute("data-id") === id) buttons[i].classList.add("ya-on");
        else buttons[i].classList.remove("ya-on");
      }
      var entry = chipIndex[id];
      var q = entry.quarter;
      var c = entry.chip;
      readout.innerHTML =
        '<p class="ya-r-head"><span class="ya-r-badge" style="background:' + q.colour + '">' + q.id +
        '</span> <span class="ya-r-title">' + c.title + "</span></p>" +
        '<p class="ya-r-ev"><strong>What it contributed.</strong> ' + c.evidence + "</p>" +
        '<p class="ya-r-essay"><strong>Essay sentence it earns.</strong> ' + c.essay + "</p>";
    }

    function chips() {
      var ids = [];
      QUARTERS.forEach(function (q) { q.chips.forEach(function (c) { ids.push(c.id); }); });
      return ids;
    }

    select(sel);
    return {
      select: select,
      getSel: function () { return sel; },
      chips: chips,
      quarters: function () { return QUARTERS.map(function (q) { return q.id; }); }
    };
  }

  global.Y1ArcViz = { mount: mount };
})(typeof window !== "undefined" ? window : global);
