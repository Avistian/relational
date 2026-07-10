/**
 * Leakage taxonomy map (Kapoor & Narayanan 2022, Table/§ taxonomy).
 *
 * An interactive map of the paper's 8 leakage types grouped into 3 families. Clicking
 * a type chip fills a detail panel with: the definition, WHERE the learner has already
 * met it in this course (spacing / interleaving), and the fix. This is the load-bearing
 * structural visual: it turns eight scattered "leaks" into one navigable ontology.
 *
 * Plain <script> (file://-safe). Usage: LeakageTaxonomyViz.mount(el, config?).
 *
 * Expected states (for headless verification):
 *   - mounts a .ltv container with 3 family blocks and 8 chips total
 *   - a detail panel starts on the first chip (L1.2)
 *   - clicking any chip updates the detail panel title to that chip's code
 */
(function (global) {
  "use strict";

  var FAMILIES = [
    {
      key: "L1",
      title: "1 · No clean train/test separation",
      blurb: "Information from the test rows sneaks into training.",
      types: [
        {
          code: "L1.1", name: "No held-out test set",
          def: "Performance is reported on the same data the model was trained on — there is no independent test set at all.",
          seen: "Lesson 003 (train/valid/test): why you always hold out a test set the model never touches.",
          fix: "Always keep a final test set (or nested CV) untouched until the very end."
        },
        {
          code: "L1.2", name: "Preprocessing on all data",
          def: "A data-learning step (scaler, imputer, SMOTE, PCA) is fit on the full dataset before splitting, so test statistics leak into training.",
          seen: "Lesson 005 (pipelines): fit-on-all vs per-fold — 0.78 mirage vs 0.44 honest on pure noise. Lesson 007: SMOTE before the split.",
          fix: "Fit every data-learning step inside the CV fold (a Pipeline), on training data only."
        },
        {
          code: "L1.3", name: "Feature selection on all data",
          def: "Features are chosen (or target-encoded) using the whole dataset — including the labels of rows that later become test rows.",
          seen: "Lesson 009 (feature engineering): whole-column target encoding scored 0.76 on a pure-noise category vs 0.50 out-of-fold.",
          fix: "Select / encode inside the fold (out-of-fold), never on the full labelled data."
        },
        {
          code: "L1.4", name: "Duplicates across the split",
          def: "The same (or near-identical) record appears in both train and test, so the model half-memorises the answer.",
          seen: "Lesson 021 (splits): near-duplicate rows (TabReD's electricity). Today's lab: 0.948 random vs 0.876 grouped.",
          fix: "De-duplicate, or use a grouped split so no underlying record straddles the fold."
        }
      ]
    },
    {
      key: "L2",
      title: "2 · Illegitimate features",
      blurb: "A feature would not exist — or would not be knowable — at prediction time.",
      types: [
        {
          code: "L2", name: "Feature is a proxy for the label",
          def: "An input column is a stand-in for the outcome (computed after it, or otherwise unavailable when you actually predict). It hands the model the answer.",
          seen: "Lesson 002 (design matrix): future_orders / avg_order_after_t. Today's lab: a leaked column inflates RF's apparent win over LR from ~0 to +0.22.",
          fix: "Audit provenance — could this value be known strictly before the label? A model info sheet forces the question."
        }
      ]
    },
    {
      key: "L3",
      title: "3 · Test set ≠ distribution of interest",
      blurb: "The test set does not match what the model will actually face.",
      types: [
        {
          code: "L3.1", name: "Temporal leakage",
          def: "The model trains on data from the future relative to its test rows — impossible in deployment, where you always predict forward.",
          seen: "Lesson 021 (splits in the wild): random-CV 0.846 vs temporal 0.758 under drift. Lesson 002: point-in-time cutoff.",
          fix: "Temporal split (train past → test future) / TimeSeriesSplit; a temporal validation slice too."
        },
        {
          code: "L3.2", name: "Non-independence (spatial/group)",
          def: "Train and test rows are dependent — the same patient, device, or household in both — so the test set is not a fresh sample.",
          seen: "Lesson 004 (grouped & nested CV): GroupKFold so no entity straddles the split.",
          fix: "Group the split by the dependent entity (GroupKFold / grouped+temporal)."
        },
        {
          code: "L3.3", name: "Sampling bias in the test set",
          def: "The test set is drawn from a narrower or different population than the one the claim is about (e.g. only easy cases, one hospital).",
          seen: "New here — the reason a benchmark number can be honest yet still not generalise to the population of interest.",
          fix: "Define the target population up front; test on a representative sample of it."
        }
      ]
    }
  ];

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.classList.add("ltv");

    var grid = document.createElement("div");
    grid.className = "ltv-grid";
    container.appendChild(grid);

    var detail = document.createElement("div");
    detail.className = "ltv-detail";
    container.appendChild(detail);

    var chips = [];
    var byCode = {};

    FAMILIES.forEach(function (fam) {
      var block = document.createElement("div");
      block.className = "ltv-fam ltv-fam-" + fam.key;

      var h = document.createElement("div");
      h.className = "ltv-fam-title";
      h.textContent = fam.title;
      block.appendChild(h);

      var bl = document.createElement("div");
      bl.className = "ltv-fam-blurb";
      bl.textContent = fam.blurb;
      block.appendChild(bl);

      var row = document.createElement("div");
      row.className = "ltv-chips";
      fam.types.forEach(function (t) {
        var chip = document.createElement("button");
        chip.type = "button";
        chip.className = "ltv-chip ltv-chip-" + fam.key;
        chip.innerHTML = '<span class="ltv-code">' + t.code + "</span> " + t.name;
        chip.addEventListener("click", function () { select(t.code); });
        row.appendChild(chip);
        chips.push({ el: chip, code: t.code });
        byCode[t.code] = t;
      });
      block.appendChild(row);
      grid.appendChild(block);
    });

    function select(code) {
      var t = byCode[code];
      chips.forEach(function (c) {
        if (c.code === code) c.el.classList.add("active");
        else c.el.classList.remove("active");
      });
      detail.innerHTML =
        '<div class="ltv-d-code">' + t.code + " — " + t.name + "</div>" +
        '<p class="ltv-d-def">' + t.def + "</p>" +
        '<p class="ltv-d-seen"><strong>You have met it:</strong> ' + t.seen + "</p>" +
        '<p class="ltv-d-fix"><strong>Fix:</strong> ' + t.fix + "</p>";
    }

    select("L1.2");
    return { select: select, chips: chips };
  }

  global.LeakageTaxonomyViz = { mount: mount };
})(window);
