/**
 * Neural-tabular landscape map — primary visual of L041 ("The deep-tabular landscape & rtdl").
 *
 * The Year-2 Q1 syllabus laid out as a map, not a leaderboard. Click a model chip to see
 * (a) what it does mechanistically, (b) its inductive-bias stance versus the Year-1 GBDT bar,
 * and (c) the Y2-Q1 lesson that teaches it in full. Colour = family. The incumbent GBDT bar
 * sits in its own family because every neural model on the map is measured against it.
 *
 * Default = ResNet: Gorishniy 2021's "do these first" strong simple baseline (L042).
 *
 * Plain <script> (file://-safe). Usage: TabularDLMapViz.mount(el, config?).
 *
 * Expected states:
 *   - mounts .tdm-viz with a family legend, model chips, and a .tdm-readout
 *   - 10 models across 5 families
 *   - default selection "resnet"
 *   - select(id) / clicking a chip updates the readout; getSel() / models() for tests
 */
(function (global) {
  "use strict";

  var FAMILIES = {
    baseline: { label: "baseline (do first)", colour: "#2e6fb0" },
    attention: { label: "attention / transformer", colour: "#7c3aed" },
    treelike: { label: "tree-inspired", colour: "#1e6b3c" },
    crosses: { label: "explicit feature crosses", colour: "#b9770e" },
    incumbent: { label: "incumbent bar", colour: "#64748b" }
  };

  var MODELS = [
    {
      id: "mlp", label: "MLP", family: "baseline", lesson: "L042",
      title: "Multilayer perceptron — the minimal neural baseline",
      what: "A stack of fully-connected blocks (Linear → ReLU → Dropout). Every feature feeds every unit of layer 1, so the model must learn any feature selection or interaction from scratch.",
      bias: "Rotationally invariant (L026): mixes columns freely, so it needs ~linearly more samples per uninformative column (L027) and rounds off jagged targets (L025). The honest floor every fancier model must clear."
    },
    {
      id: "resnet", label: "ResNet", family: "baseline", lesson: "L042",
      title: "MLP + residual blocks — Gorishniy's strong simple baseline",
      what: "The MLP with skip (residual) connections and normalisation, so gradients flow through deep stacks. Trained with a shared tuning protocol, it is far stronger than the weak nets earlier DL-tabular papers compared against — which is the whole point of 'Revisiting'.",
      bias: "Same rotational-invariance family as the MLP, but the residual path makes depth trainable. Gorishniy 2021: this baseline alone matches many published 'novel' architectures — so run it first, before believing any new model's win."
    },
    {
      id: "ft", label: "FT-Transformer", family: "attention", lesson: "L046",
      title: "Feature Tokenizer + Transformer — the strong universal DL model",
      what: "A Feature Tokenizer turns EVERY feature into an embedding token — numeric x_j → x_j·W_j + b_j, categorical → embedding lookup — prepends a learnable [CLS] token, runs L Transformer self-attention layers, and reads the final [CLS] vector through a linear head.",
      bias: "Attention lets any feature attend to any other, so it can build interactions and is less orientation-privileged than a tree. Gorishniy 2021's headline architecture: the most reliably strong classic neural single-table model — but still not a universal GBDT-beater."
    },
    {
      id: "tabtransformer", label: "TabTransformer", family: "attention", lesson: "L045",
      title: "Contextual embeddings for categoricals only",
      what: "Categorical entity embeddings pass through Transformer layers to become contextual (the 'bank' = river vs savings analogy, L032); continuous features bypass attention and are concatenated before an MLP head.",
      bias: "Only categoricals are contextualised — numerics are second-class, which FT-Transformer fixes by tokenising them too. Honest verdict (L032): ties tree ensembles on supervised tabular; real wins are noise/missingness robustness and a semi-supervised lift."
    },
    {
      id: "saint", label: "SAINT", family: "attention", lesson: "L047",
      title: "Attention over columns AND rows (inter-sample)",
      what: "Adds inter-sample attention: a row can attend to OTHER ROWS in the batch, not just its own columns — a learned, soft analogue of nearest-neighbour lookup on top of the usual feature attention.",
      bias: "Inter-sample attention is the first hint of the retrieval idea that TabR (Y2 Q2) makes central: use the training rows themselves, not only learned weights. Costs quadratic attention in the batch."
    },
    {
      id: "excelformer", label: "ExcelFormer / Trompt", family: "attention", lesson: "L049",
      title: "2023 transformers claiming to surpass GBDT",
      what: "Later attention architectures (ExcelFormer, Trompt) that report beating tuned GBDT on broad suites. Read them the L038 way: is the GBDT baseline equally tuned, freshly implemented, and tested on the same splits?",
      bias: "The critical-reading station of the map. A 'surpasses GBDT' claim is only as strong as the baseline it beats and the protocol it beats it under — Year 1's whole discipline, applied to a hot claim."
    },
    {
      id: "node", label: "NODE", family: "treelike", lesson: "L044",
      title: "Differentiable oblivious decision trees",
      what: "Ensembles of SOFT (differentiable) oblivious trees: instead of a hard split, each 'decision' is a smooth gate (entmax) so the whole forest is trainable end-to-end by gradient descent.",
      bias: "An explicit attempt to give a neural model the tree inductive bias for irregular targets (L025). Where it helps: genuinely piecewise/irregular functions a plain MLP smooths over."
    },
    {
      id: "tabnet", label: "TabNet", family: "treelike", lesson: "L043",
      title: "Sequential attention with instance-wise feature masks",
      what: "At each decision step a learnable mask selects a sparse subset of features to attend to, sequentially — an instance-wise feature selection you can read out as an interpretability map.",
      bias: "The sparse mask is a soft imitation of a tree's gated feature use (L027 junk robustness), and the masks are its selling point (interpretability). Fair-protocol results are mixed — a strong ResNet often matches it."
    },
    {
      id: "dcnv2", label: "DCNv2", family: "crosses", lesson: "L048",
      title: "Explicit bounded-degree feature crosses",
      what: "A cross network builds explicit feature interactions of bounded degree (x, then x⊗x-style crosses) in parallel with a deep network, so products of features are modelled directly instead of being hoped for from an MLP.",
      bias: "Bakes in the interaction structure an MLP must otherwise discover — useful when known low-order crosses carry the signal (recsys/CTR heritage). Explicit crosses vs learned attention is the design axis here."
    },
    {
      id: "gbdt", label: "GBDT (XGB / LGBM / CatBoost)", family: "incumbent", lesson: "Y1",
      title: "The Year-1 bar every model on this map is measured against",
      what: "Gradient-boosted decision trees — the tuned, leak-free, regenerable baseline built and closed in Year 1 (L014–L020, exit L040). Not a neural model; it is the incumbent the whole year is honest about.",
      bias: "Wins on typical flat tables via three inductive biases — irregular targets, privileged column orientation, junk-feature robustness (Grinsztajn 2022, L024–L027). Gorishniy 2021 confirms it from the architecture side: no neural model universally beats it."
    }
  ];

  function find(id) {
    for (var i = 0; i < MODELS.length; i++) if (MODELS[i].id === id) return MODELS[i];
    return null;
  }

  function renderReadout(el, id) {
    var m = find(id);
    if (!m) return;
    var fam = FAMILIES[m.family];
    el.innerHTML =
      '<p class="tdm-r-head"><span class="tdm-r-badge" style="background:' + fam.colour + '">' +
      m.label + "</span> <span class=\"tdm-r-lesson\">" + m.lesson + "</span> <span class=\"tdm-r-title\">" +
      m.title + "</span></p>" +
      '<p class="tdm-r-ev"><strong>What it is.</strong> ' + m.what + "</p>" +
      '<p class="tdm-r-ev"><strong>Inductive-bias stance.</strong> ' + m.bias + "</p>";
  }

  function mount(root, config) {
    config = config || {};
    var sel = config.defaultSel || "resnet";
    root.innerHTML = "";
    var wrap = document.createElement("div");
    wrap.className = "tdm-viz";

    var legend = document.createElement("div");
    legend.className = "tdm-legend";
    var legendHtml = "";
    Object.keys(FAMILIES).forEach(function (k) {
      legendHtml += '<span class="tdm-key"><span class="tdm-dot" style="background:' +
        FAMILIES[k].colour + '"></span> ' + FAMILIES[k].label + "</span>";
    });
    legend.innerHTML = legendHtml;
    wrap.appendChild(legend);

    var chips = document.createElement("div");
    chips.className = "tdm-chips";
    MODELS.forEach(function (m) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "tdm-chip";
      b.setAttribute("data-id", m.id);
      b.style.setProperty("--tdm-c", FAMILIES[m.family].colour);
      b.innerHTML = '<span class="tdm-chip-lab">' + m.label + "</span>" +
        '<span class="tdm-chip-lsn">' + m.lesson + "</span>";
      b.addEventListener("click", function () { api.select(m.id); });
      chips.appendChild(b);
    });
    wrap.appendChild(chips);

    var readout = document.createElement("div");
    readout.className = "tdm-readout";
    wrap.appendChild(readout);
    root.appendChild(wrap);

    var api = {
      select: function (id) {
        sel = id;
        renderReadout(readout, id);
        var buttons = wrap.querySelectorAll(".tdm-chip");
        for (var i = 0; i < buttons.length; i++) {
          var on = buttons[i].getAttribute("data-id") === id;
          if (on) buttons[i].classList.add("tdm-on");
          else buttons[i].classList.remove("tdm-on");
        }
      },
      getSel: function () { return sel; },
      models: function () { return MODELS.map(function (m) { return m.id; }); },
      families: function () { return Object.keys(FAMILIES); }
    };
    api.select(sel);
    return api;
  }

  global.TabularDLMapViz = { mount: mount };
})(typeof window !== "undefined" ? window : global);
