/**
 * Year-1 exit verdict fork — load-bearing visual of L040 ("Year 1 exit exam").
 *
 * The curriculum exit is not "you must beat XGBoost." It is a three-way fork:
 *   BEAT     — a regenerable challenger clears the XGB bar by more than disclosed noise,
 *              under a fair protocol, with a significance / effect-size story.
 *   TIE      — the challenger matches the bar within noise (the usual Year-1 outcome on
 *              typical flat tables; L020 adult Δ≈0.001; L030 credit_g corrected p=0.64).
 *   EXPLAIN  — no beat, and you write *why* using Grinsztajn's three inductive biases +
 *              the exhaustion cascade — standing on or revising the L039 synthesis.
 *
 * Click a fork panel or a point inside it. Default = TIE / adult-ceiling (the honest modal
 * outcome of the exit). Every number is evidence of record from earlier lessons.
 *
 * Plain <script> (file://-safe). Usage: ExitVerdictViz.mount(el, config?).
 *
 * Expected states:
 *   - mounts .ev-viz with 3 fork panels + readout
 *   - exactly 9 points (3 per fork)
 *   - default selection "tie-adult"
 *   - select(id) updates readout; getSel() / points() / forks() for tests
 */
(function (global) {
  "use strict";

  var FORKS = [
    {
      id: "BEAT", label: "Beat", colour: "#1e6b3c",
      tagline: "A real win under the fair contract.",
      points: [
        {
          id: "beat-gap", label: "Gap > noise",
          title: "The challenger clears the bar by more than disclosed noise",
          evidence: "A beat is not 'any positive delta.' On adult (L020) tuned XGB 0.9294 vs fixed-default ref 0.9282 is Δ=+0.0012 — inside the ±0.002 noise band the protocol treats as a match, not a beat. On credit_g (L030) GBDT−MLP +0.008 had corrected p=0.64. A Year-1 exit beat needs a gap that survives the same discipline.",
          essay: "Exit move: quote the regenerable numbers, the disclosed noise / test, and only then say 'beat'."
        },
        {
          id: "beat-protocol", label: "Fair protocol",
          title: "Same data, split, metric, budget, preprocessing — test touched once",
          evidence: "L020 fair-comparison contract + L038 two-pipelines-one-standard. A beat measured under a leaky split, an undertuned XGB, or a different metric than the bar is not an exit pass — it is the failure mode Kapoor & Narayanan (L022) and Lones (L038) exist to catch.",
          essay: "Exit move: restate the five fixed things before the sixth (the honest verdict)."
        },
        {
          id: "beat-rare", label: "Rare on flat tables",
          title: "Year 1 evidence says large beats of a tuned GBDT are uncommon",
          evidence: "L028–L033 exhaustion cascade: honest MLP/ResNet, AutoML, embeddings, TabTransformer, and hand FE repeatedly tie or fail against a tuned GBDT on flat tables. A genuine exit beat is allowed — but it is the surprising fork, and a skeptic's first hypothesis is still 'leak or undertuned bar.'",
          essay: "Exit move: if you claim BEAT, lead with the leak audit, then the number."
        }
      ]
    },
    {
      id: "TIE", label: "Tie / match", colour: "#2e6fb0",
      tagline: "Within noise of the XGB bar — a legitimate exit.",
      points: [
        {
          id: "tie-adult", label: "Adult ceiling",
          title: "L020: tuning and stacking barely move a strong default",
          evidence: "Fixed-default XGB ROC-AUC 0.9282; tuned XGB 0.9294; tuned LGBM 0.9296; stack 0.9297; OOF correlation XGB↔LGBM 0.997. Two GBDTs are near-redundant, so stacking adds nothing. Matching the bar is the modal honest result on this flat task.",
          essay: "Exit move: 'matched the regenerable XGB bar within ±0.002; no significant beat' is a full pass."
        },
        {
          id: "tie-credit", label: "Credit_g tie",
          title: "L030: GBDT vs honest MLP is not significant",
          evidence: "Mean gap +0.008 ROC-AUC across paired folds; naive p=0.22, corrected resampled t p=0.64. The checkpoint's load-bearing skill was reporting a tie. The Year-1 exit inherits that honesty.",
          essay: "Exit move: a mean lead that a correct test cannot distinguish from noise is a TIE, not a soft beat."
        },
        {
          id: "tie-automl", label: "AutoML ceiling",
          title: "L029: CASH search only ties a tuned GBDT",
          evidence: "Default XGB 0.775 → tuned 0.806 ≈ AutoML 0.803 on credit_g. Automation is not unpaid accuracy. The exit challenger (another booster, a stack, a small FE pass) is in the same regime.",
          essay: "Exit move: if your challenger is still single-table cleverness, expect a tie and prepare the explain path."
        }
      ]
    },
    {
      id: "EXPLAIN", label: "Explain why not", colour: "#b9770e",
      tagline: "No beat — write the inductive-bias reason.",
      points: [
        {
          id: "ex-biases", label: "Three biases",
          title: "Grinsztajn: irregular targets, orientation, junk robustness",
          evidence: "On typical flat tables the target is jagged, columns carry individual meaning, and junk abounds — so a tree's axis-aligned, gain-gated splits match the regime and an MLP's smoothness + rotation invariance do not (L025–L027). Explaining a non-beat means naming these biases with one verified number each, not saying 'trees are more powerful' (M47).",
          essay: "Exit move: three short sentences — one bias, one number, one flip condition."
        },
        {
          id: "ex-exhaust", label: "Exhaustion cascade",
          title: "Further single-table effort has repeatedly tied or failed",
          evidence: "L028–L033: honest nets, AutoML, embeddings, TabTransformer, hand FE. The unpaid upside is not 'one more booster hyperparameter.' Standing on the L039 synthesis means your explain-why-not paragraph ends by pointing at that cascade.",
          essay: "Exit move: 'I did not beat XGB because Year-1 single-table upgrades plateau' — cite two cascade rungs."
        },
        {
          id: "ex-frontier", label: "Join silence",
          title: "The flat claim cannot speak to signal a lossy join destroyed",
          evidence: "L035 aggregation collision: Ada and Bo flatten to the same row → any flat model predicts P=0.502 for both. An exit that fails to beat XGB on the manufactured table may still be consistent with the thesis — the open burden (recover structure, beat the fair bar) is Years 3–6, not a Year-1 homework.",
          essay: "Exit move: end the explanation by naming the open burden without claiming an RDL win."
        }
      ]
    }
  ];

  function findPoint(id) {
    for (var i = 0; i < FORKS.length; i++) {
      for (var j = 0; j < FORKS[i].points.length; j++) {
        if (FORKS[i].points[j].id === id) return { fork: FORKS[i], point: FORKS[i].points[j] };
      }
    }
    return null;
  }

  function renderReadout(el, id) {
    var hit = findPoint(id);
    if (!hit) return;
    var f = hit.fork, p = hit.point;
    el.innerHTML =
      '<p class="ev-r-head"><span class="ev-r-badge" style="background:' + f.colour + '">' + f.id +
      '</span> <span class="ev-r-title">' + p.title + "</span></p>" +
      '<p class="ev-r-ev"><strong>Evidence of record.</strong> ' + p.evidence + "</p>" +
      '<p class="ev-r-essay">' + p.essay + "</p>";
  }

  function mount(root, config) {
    config = config || {};
    var sel = config.defaultSel || "tie-adult";
    root.innerHTML = "";
    var wrap = document.createElement("div");
    wrap.className = "ev-viz";

    var grid = document.createElement("div");
    grid.className = "ev-grid";

    var apis = [];
    FORKS.forEach(function (fork) {
      var zone = document.createElement("div");
      zone.className = "ev-zone";
      zone.style.setProperty("--ev-c", fork.colour);
      zone.style.borderTopColor = fork.colour;

      var head = document.createElement("button");
      head.type = "button";
      head.className = "ev-zone-head";
      head.innerHTML =
        '<span class="ev-zone-lab">' + fork.label + "</span>" +
        '<span class="ev-zone-tag">' + fork.tagline + "</span>";
      head.addEventListener("click", function () {
        api.select(fork.points[0].id);
      });
      zone.appendChild(head);

      var pts = document.createElement("div");
      pts.className = "ev-points";
      fork.points.forEach(function (pt) {
        var b = document.createElement("button");
        b.type = "button";
        b.className = "ev-pt";
        b.setAttribute("data-id", pt.id);
        b.textContent = pt.label;
        b.addEventListener("click", function () {
          api.select(pt.id);
        });
        pts.appendChild(b);
      });
      zone.appendChild(pts);
      grid.appendChild(zone);
    });
    wrap.appendChild(grid);

    var readout = document.createElement("div");
    readout.className = "ev-readout";
    wrap.appendChild(readout);
    root.appendChild(wrap);

    var api = {
      select: function (id) {
        sel = id;
        renderReadout(readout, id);
        var buttons = wrap.querySelectorAll(".ev-pt");
        for (var i = 0; i < buttons.length; i++) {
          var on = buttons[i].getAttribute("data-id") === id;
          if (on) buttons[i].classList.add("ev-on");
          else buttons[i].classList.remove("ev-on");
        }
      },
      getSel: function () { return sel; },
      points: function () {
        var out = [];
        FORKS.forEach(function (f) {
          f.points.forEach(function (p) { out.push(p.id); });
        });
        return out;
      },
      forks: function () {
        return FORKS.map(function (f) { return f.id; });
      }
    };
    api.select(sel);
    apis.push(api);
    return api;
  }

  global.ExitVerdictViz = { mount: mount };
})(typeof window !== "undefined" ? window : global);
