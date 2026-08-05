/**
 * Protocol bake-off — three baselines under ONE shared tuning protocol (Gorishniy 2021 §3.2).
 *
 * The lesson's headline result: run the strong simple baselines (MLP, ResNet) FIRST, tuned under the
 * SAME protocol as everything else, and read them honestly against a tuned GBDT. The bars are the
 * L042 evidence of record — VERIFIED in this workspace (labs/_verify_l042.py, our from-scratch
 * relkit.nets models validated against rtdl, OpenML credit_g, test ROC-AUC, mean ± std over 3 seeds,
 * 6-trial shared search budget):
 *   MLP         0.802 ± 0.020   (simplest net floor — here nominally on top)
 *   ResNet      0.790 ± 0.033   (the "do these first" strong simple baseline)
 *   tuned GBDT  0.780 ± 0.016   (incumbent single-table bar — here it does NOT lead)
 * These are the credit_g row of the multi-dataset run (labs/_verify_l042.py, 4 datasets); the lesson's
 * cross-dataset rank summary + the numeric-skew caveat live in the "across datasets" section.
 * The three overlap within one standard deviation: a statistical TIE — "no universal winner"
 * (L041/L024) — and, crucially, the properly-tuned neural baselines are right there with the tree
 * (Gorishniy's actual message, not "GBDT always wins"). The shared FRAME is what makes it fair: same
 * split (L020), same metric, same search budget, validation-selected — only the per-model search SPACE
 * differs. Whiskers show ± std across seeds so you can SEE the gaps are inside the noise.
 *
 * Clicking a model reveals its role + its search space (what the shared budget is spent on).
 *
 * Usage: ProtocolBakeoffViz.mount(container, { caption })
 * Expected states:
 *   - default: ResNet selected (the do-these-first baseline); readout names "strong simple baseline".
 *   - clicking GBDT: readout says "incumbent" and "no universal winner — the tuned nets match it".
 *   - clicking MLP: readout names it the simplest net floor.
 *   - every bar shows its ROC-AUC value + a ± std whisker; the shared-protocol banner is always visible.
 * api: { getSel(), select(id), models() }
 */
(function (global) {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";

  // colour by role
  var C = { gbdt: "#b9770e", resnet: "#1e6b3c", mlp: "#2e6fb0" };

  var MODELS = [
    {
      id: "resnet", label: "ResNet", score: 0.790, sd: 0.033, color: C.resnet,
      role: "Strong simple baseline — run it FIRST",
      body: "An MLP whose blocks add a residual skip, output = x + f(x), so depth stays trainable " +
        "(L028 — the from-scratch block, here validated against rtdl, |Δ|=0.000). Gorishniy 2021's " +
        "headline: tuned under a shared protocol it alone matches many published 'novel' architectures " +
        "— so a new model that does not beat it has shown nothing. Here (verified, credit_g) it lands " +
        "0.790 — inside a whisker of the GBDT.",
      space: "n_blocks, d_block (main width), d_hidden_multiplier (block width), dropout1 + dropout2, " +
        "learning rate, weight decay — AdamW, early stopping on the validation metric."
    },
    {
      id: "mlp", label: "MLP", score: 0.802, sd: 0.020, color: C.mlp,
      role: "Simplest net floor",
      body: "A plain stack of Dropout(ReLU(Linear)) blocks — no skips. The most basic honest neural " +
        "baseline; if a fancy model cannot beat a tuned MLP, the architecture is not the reason it won. " +
        "Verified here at 0.802 — nominally the top score, but within one std of the other two (a tie).",
      space: "n_blocks, d_block (width), dropout, learning rate, weight decay — AdamW, early stopping."
    },
    {
      id: "gbdt", label: "tuned GBDT", score: 0.780, sd: 0.016, color: C.gbdt,
      role: "Incumbent single-table bar",
      body: "The Year-1 bar (XGBoost / LightGBM / CatBoost; HistGBT here). On this small categorical " +
        "table it does NOT lead — the two properly-tuned nets match it within noise. That is the real " +
        "'no universal winner' (L041/L024): the best model is dataset-dependent, and a fair protocol " +
        "keeps the neural baselines honestly in the race.",
      space: "n_estimators (early-stopped), max_leaf_nodes, learning rate, L2 (lambda) — " +
        "same trial budget as each net."
    }
  ];

  function el(name, attrs, text) {
    var e = document.createElementNS(SVGNS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "pbo-viz";

    var sel = "resnet";

    // shared-protocol banner
    var banner = document.createElement("div");
    banner.className = "pbo-banner";
    banner.innerHTML = "<strong>One shared protocol</strong> · same split (L020) · same metric " +
      "(ROC-AUC) · same search budget · validation-selected — only the per-model search space differs";
    container.appendChild(banner);

    var W = 480, H = 200;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", class: "pbo-svg" });
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "pbo-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "pbo-caption";
    cap.textContent = config.caption ||
      "The three baselines on OpenML credit_g (test ROC-AUC, mean ± std over 3 seeds), tuned under one " +
      "shared protocol — VERIFIED here with our from-scratch models validated against rtdl " +
      "(labs/_verify_l042.py). Click a bar for its role and search space. The whiskers overlap: the tuned " +
      "neural baselines match the GBDT — a tie on this ONE table (the cross-dataset verdict is below).";
    container.appendChild(cap);

    var PL = 96, PR = W - 60, PT = 18, PB = H - 22;
    var XLO = 0.72, XHI = 0.86;
    function sx(v) { return PL + (v - XLO) / (XHI - XLO) * (PR - PL); }

    var bars = {};

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      // x-axis gridlines
      [0.75, 0.78, 0.81, 0.84].forEach(function (v) {
        var xx = sx(v);
        svg.appendChild(el("line", { x1: xx, y1: PT, x2: xx, y2: PB,
          stroke: "var(--border)", "stroke-width": 0.5, opacity: 0.6 }));
        svg.appendChild(el("text", { x: xx, y: PB + 14, "text-anchor": "middle", class: "pbo-tick" },
          v.toFixed(2)));
      });
      svg.appendChild(el("text", { x: (PL + PR) / 2, y: H - 4, "text-anchor": "middle", class: "pbo-axis" },
        "test ROC-AUC (credit_g, mean ± std)"));

      var bh = 30, gap = ((PB - PT) - MODELS.length * bh) / (MODELS.length + 1);
      MODELS.forEach(function (m, i) {
        var y = PT + gap + i * (bh + gap);
        var on = m.id === sel;

        // model name (clickable label)
        var lab = el("text", { x: PL - 10, y: y + bh / 2 + 4, "text-anchor": "end",
          class: "pbo-name" + (on ? " pbo-name-on" : ""), "data-id": m.id }, m.label);
        lab.addEventListener("click", function () { select(m.id); });
        svg.appendChild(lab);

        // track + bar
        svg.appendChild(el("rect", { x: PL, y: y, width: PR - PL, height: bh, rx: 4,
          fill: "var(--bg)", stroke: "var(--border)", "stroke-width": 1 }));
        var bar = el("rect", { x: PL, y: y, width: sx(m.score) - PL, height: bh, rx: 4,
          fill: m.color, opacity: on ? 1 : 0.5, cursor: "pointer", "data-id": m.id });
        bar.addEventListener("click", function () { select(m.id); });
        svg.appendChild(bar);
        bars[m.id] = bar;

        // ± std whisker (shows the gaps are inside the noise)
        var cy = y + bh / 2;
        var xlo = sx(Math.max(XLO, m.score - m.sd)), xhi = sx(Math.min(XHI, m.score + m.sd));
        svg.appendChild(el("line", { x1: xlo, y1: cy, x2: xhi, y2: cy,
          stroke: "var(--fg)", "stroke-width": 1.5, opacity: on ? 0.95 : 0.55 }));
        [xlo, xhi].forEach(function (xx) {
          svg.appendChild(el("line", { x1: xx, y1: cy - 5, x2: xx, y2: cy + 5,
            stroke: "var(--fg)", "stroke-width": 1.5, opacity: on ? 0.95 : 0.55 }));
        });

        // value label
        svg.appendChild(el("text", { x: sx(m.score + m.sd) + 6, y: y + bh / 2 + 4, "text-anchor": "start",
          class: "pbo-val" }, m.score.toFixed(3)));

        if (on) {
          svg.appendChild(el("rect", { x: PL, y: y, width: PR - PL, height: bh, rx: 4,
            fill: "none", stroke: m.color, "stroke-width": 2 }));
        }
      });

      var m = MODELS.filter(function (x) { return x.id === sel; })[0];
      readout.innerHTML =
        "<p class='pbo-r-head'><span class='pbo-r-badge' style='background:" + m.color + "'>" +
        m.label + "</span> <span class='pbo-r-role'>" + m.role + "</span> " +
        "<span class='pbo-r-score'>" + m.score.toFixed(3) + " ± " + m.sd.toFixed(3) + " ROC-AUC</span></p>" +
        "<p>" + m.body + "</p>" +
        "<p class='pbo-r-space'><strong>Search space (one shared budget):</strong> " + m.space + "</p>";
    }

    function select(id) { sel = id; draw(); }

    draw();
    return {
      getSel: function () { return sel; },
      select: select,
      models: function () { return MODELS.map(function (m) { return m.id; }); }
    };
  }

  global.ProtocolBakeoffViz = { mount: mount };
})(window);
