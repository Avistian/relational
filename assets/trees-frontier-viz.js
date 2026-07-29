/**
 * Trees-win frontier map — second mechanistic visual of L039 ("Year 1 synthesis essay").
 *
 * The synthesis claim has THREE zones a reader must keep apart:
 *   WIN        — where trees beat honest neural nets (Grinsztajn's three biases; verified numbers)
 *   EXHAUSTED  — where more single-table cleverness stops buying wins (L028–L033 cascade of ties)
 *   FRONTIER   — where the flat-table claim stops being the whole story (join loss; thesis lives here)
 *
 * Click a zone header or a point inside it. Default = WIN / smoothness (the essay's first "why").
 * Every number is one already measured in an earlier lesson — L039 introduces no new bake-offs.
 *
 * Plain <script> (file://-safe). Usage: TreesFrontierViz.mount(el, config?).
 *
 * Expected states:
 *   - mounts .tf-viz with 3 zone panels + readout
 *   - exactly 9 points (3 per zone)
 *   - default selection "win-smooth"
 *   - select(id) updates readout; getSel() / points() / zones() for tests
 */
(function (global) {
  "use strict";

  var ZONES = [
    {
      id: "WIN", label: "Where trees win", colour: "#1e6b3c",
      tagline: "Inductive bias matches flat tabular defaults.",
      points: [
        {
          id: "win-smooth", label: "Irregular targets",
          title: "Smoothness (spectral) bias — Grinsztajn §5.2",
          evidence: "Gradient descent fits smooth functions easily and irregular ones poorly; tabular targets are often jagged (thresholds, fees, hard cuts). Ablation: Gaussian-smoothing the target collapses the GBT–MLP R² gap from +0.33 toward 0; on an 8×8 checkerboard the tree holds 0.969 while a 2×256 MLP falls to 0.837.",
          essay: "Write: trees win when the target is irregular, because a tree places hard steps and a net rounds them off."
        },
        {
          id: "win-rotate", label: "Orientation",
          title: "Rotation bias — Grinsztajn §5.4 / Ng 2004",
          evidence: "A tree's axis-aligned splits attend to each meaningful column; an MLP is rotationally invariant (W·(Qx)=(WQ)·x). A lossless random rotation collapses the tree (0.987→0.747) and leaves the MLP unmoved (0.862→0.869), reversing the ranking. Invariance is a liability when columns carry individual meaning.",
          essay: "Write: trees win when the original feature basis is privileged — the default for business tables, and the opposite of image pixels."
        },
        {
          id: "win-junk", label: "Junk features",
          title: "Uninformative-features bias — Grinsztajn §5.3",
          evidence: "Trees gate junk via greedy, gain-gated splits (implicit selection); MLPs feed every column into layer 1 and, being rotation-invariant, need ≥ linearly more samples per junk feature (Ng 2004). On a smooth target where the MLP wins clean (0.986 vs GBT 0.945), adding 100 pure-noise columns costs the MLP 0.084 vs the GBT 0.032 and reverses the ranking.",
          essay: "Write: trees win when junk columns abound — surrogate keys, audit fields, denormalized dupes — because they can ignore them."
        }
      ]
    },
    {
      id: "EXHAUSTED", label: "Where cleverness dies", colour: "#2e6fb0",
      tagline: "More single-table machinery stops buying wins.",
      points: [
        {
          id: "ex-neural", label: "Honest neural bar",
          title: "Tuned MLP / ResNet — Gorishniy 2021",
          evidence: "Once you compare against a properly-tuned MLP and a pre-activation ResNet, much prior tabular-DL 'progress' evaporates. On credit_g the GBDT still leads (0.793) with MLP (0.752) ≈ ResNet (0.743) tied. Residual skips fix depth degradation (plain 0.917→0.866 over depth 1→32; ResNet holds ~0.90) — but they do not repeal the three biases.",
          essay: "Write: trees beat weak nets; against an honest neural baseline they often merely lead, and the gap needs a corrected test."
        },
        {
          id: "ex-automl", label: "AutoML ceiling",
          title: "CASH search only ties a tuned GBDT — Feurer 2015",
          evidence: "On credit_g the jump is tuning at all (default XGB 0.775 → tuned 0.806, +0.031); a 4-algorithm AutoML with ensembling then only ties the tuned XGB (0.803, bands overlap). AutoML searches algorithms and knobs on an already-flattened table — it never recovers what the join discarded.",
          essay: "Write: automating single-table search does not overturn the incumbent; it confirms the returns to search are nearly exhausted."
        },
        {
          id: "ex-repro", label: "Embeddings & FE",
          title: "Entity embeddings, TabTransformer, Domingos FE",
          evidence: "Entity-embedding MLP ties fair one-hot on credit_g (L031). TabTransformer matches trees on supervised tabular — its +1.0% is over other DL, not GBDTs (L032). Hand-crafted FE with the model fixed peaks at 3 features (+0.0046 inside ±0.03 CV band) then declines below baseline (L033). Six lessons, one pattern: single-table cleverness plateaus.",
          essay: "Write: representation tricks inside one table also plateau — the unpaid representations sit across foreign keys."
        }
      ]
    },
    {
      id: "FRONTIER", label: "Where the claim stops", colour: "#b9770e",
      tagline: "The flatten already threw the signal away.",
      points: [
        {
          id: "fr-smooth-win", label: "When nets win",
          title: "The honest flip conditions",
          evidence: "Grinsztajn / L019: smooth the target and the tree's edge shrinks; rotate the features and the ranking reverses; remove junk and the MLP recovers. Clean, low-noise, orientation-free data is where an MLP can win — and that is not the default regime of relational business tables.",
          essay: "Write: 'trees win' is conditional. Name the flip conditions so a skeptic cannot accuse you of universalising a medium-sized-table result."
        },
        {
          id: "fr-collision", label: "Aggregation collision",
          title: "Join + aggregate is a lossy map — Fey 2024 §2",
          evidence: "Ada (rising $10→$30→$50, 3 products) and Bo (falling $50→$30→$10, 1 product) flatten to the byte-identical row n=3/total=90/avg=30/max=50. A fitted classifier returns P(churn)=0.502 for both though labels are 0 vs 1. The loss is information, upstream of every model — cardinality, identity, order, multi-hop paths.",
          essay: "Write: trees can only win on what survives the flatten. An aggregation collision proves the single-table claim can be vacuously true while the database still holds recoverable signal."
        },
        {
          id: "fr-thesis", label: "Thesis lives here",
          title: "C1/C2 still undemonstrated as a fair win",
          evidence: "Year 1 has demonstrated the cost of flattening (L034–L035) and built the honest bar (L010–L030) plus the credibility apparatus (L036–L038). It has NOT yet shown a relational model recovering discarded structure to beat that bar — that is the Y1-exit argument's open burden and the Y3–Y4 empirical programme.",
          essay: "Write: the synthesis ends in an honest gap. 'Trees beat DL on flat tables' is established; 'RDL beats a fair flat bar by keeping structure' is the claim Years 3–6 must earn."
        }
      ]
    }
  ];

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "tf-viz";

    var sel = config.defaultId || "win-smooth";
    var index = {};

    var grid = document.createElement("div");
    grid.className = "tf-grid";
    container.appendChild(grid);

    ZONES.forEach(function (z) {
      var panel = document.createElement("div");
      panel.className = "tf-zone";
      panel.style.borderTopColor = z.colour;
      panel.setAttribute("data-zone", z.id);

      var head = document.createElement("button");
      head.type = "button";
      head.className = "tf-zone-head";
      head.innerHTML = '<span class="tf-zone-lab" style="color:' + z.colour + '">' + z.label +
        '</span><span class="tf-zone-tag">' + z.tagline + "</span>";
      head.addEventListener("click", function () { select(z.points[0].id); });
      panel.appendChild(head);

      var list = document.createElement("div");
      list.className = "tf-points";
      z.points.forEach(function (p) {
        index[p.id] = { zone: z, point: p };
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "tf-pt";
        btn.setAttribute("data-id", p.id);
        btn.style.setProperty("--tf-c", z.colour);
        btn.textContent = p.label;
        btn.addEventListener("click", function () { select(p.id); });
        list.appendChild(btn);
      });
      panel.appendChild(list);
      grid.appendChild(panel);
    });

    var readout = document.createElement("div");
    readout.className = "tf-readout";
    container.appendChild(readout);

    function select(id) {
      if (!index[id]) return;
      sel = id;
      var buttons = container.querySelectorAll(".tf-pt");
      for (var i = 0; i < buttons.length; i++) {
        if (buttons[i].getAttribute("data-id") === id) buttons[i].classList.add("tf-on");
        else buttons[i].classList.remove("tf-on");
      }
      var entry = index[id];
      var z = entry.zone;
      var p = entry.point;
      readout.innerHTML =
        '<p class="tf-r-head"><span class="tf-r-badge" style="background:' + z.colour + '">' + z.id +
        '</span> <span class="tf-r-title">' + p.title + "</span></p>" +
        '<p class="tf-r-ev"><strong>Evidence already in hand.</strong> ' + p.evidence + "</p>" +
        '<p class="tf-r-essay"><strong>Sentence for the essay.</strong> ' + p.essay + "</p>";
    }

    function points() {
      var ids = [];
      ZONES.forEach(function (z) { z.points.forEach(function (p) { ids.push(p.id); }); });
      return ids;
    }

    select(sel);
    return {
      select: select,
      getSel: function () { return sel; },
      points: points,
      zones: function () { return ZONES.map(function (z) { return z.id; }); }
    };
  }

  global.TreesFrontierViz = { mount: mount };
})(typeof window !== "undefined" ? window : global);
