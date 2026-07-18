/**
 * The four ways to turn ONE categorical column into numbers a model can eat.
 *
 * Uses the real credit_g `checking_status` column (4 levels) from labs/_verify_l031.py. A toggle
 * switches the SAME column between one-hot / ordinal / target(mean) / entity-embedding encodings so the
 * learner sees, side by side, what each representation costs and captures. One mechanism (encoding a
 * categorical), one knob (which scheme) — so a single toggled widget is the right call here.
 *
 * REAL numbers (credit_g, verify script):
 *   levels & smoothed target means: <0 0.52 (n274), 0<=X<200 0.616 (n269), >=200 0.759 (n63),
 *   no checking 0.874 (n394); global P(good)=0.70; one-hot adds 54 columns across the 13 categoricals.
 * Embedding vectors shown are the learned 2-D projection stand-in (illustrative dense vectors).
 *
 * Usage: EncodingTaxonomyViz.mount(container, { caption })
 * Expected states:
 *   - one-hot: 4 levels -> 4 sparse 0/1 columns (wide; no order, no similarity)
 *   - ordinal: 4 levels -> one integer 0..3 (compact but invents a FALSE order)
 *   - target : 4 levels -> one number = P(good|level) (compact, ordered by risk; LEAKAGE-prone)
 *   - embedding: 4 levels -> a short learned dense vector (compact + learned similarity)
 */
(function (global) {
  "use strict";

  var LEVELS = [
    { level: "<0", n: 274, target: 0.520, ord: 0, emb: [-0.9, -0.5] },
    { level: "0<=X<200", n: 269, target: 0.616, ord: 1, emb: [-0.2, 0.6] },
    { level: ">=200", n: 63, target: 0.759, ord: 2, emb: [0.7, 0.4] },
    { level: "no checking", n: 394, target: 0.874, ord: 3, emb: [1.0, -0.7] }
  ];
  var GLOBAL = 0.70;

  var SCHEMES = [
    { id: "onehot", label: "one-hot" },
    { id: "ordinal", label: "ordinal" },
    { id: "target", label: "target (mean)" },
    { id: "embedding", label: "embedding" }
  ];

  function ramp(v) {
    // 0.5 -> red-ish, 0.9 -> green-ish (risk of "good")
    var t = Math.max(0, Math.min(1, (v - 0.5) / 0.4));
    var r = Math.round(176 + (30 - 176) * t);
    var g = Math.round(58 + (107 - 58) * t);
    var b = Math.round(46 + (60 - 46) * t);
    return "rgb(" + r + "," + g + "," + b + ")";
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "etx-viz";
    var scheme = "onehot";

    var ctl = document.createElement("div");
    ctl.className = "etx-ctl";
    SCHEMES.forEach(function (s) {
      var b = document.createElement("button");
      b.textContent = s.label;
      if (s.id === scheme) b.className = "etx-on";
      b.addEventListener("click", function () {
        scheme = s.id;
        Array.prototype.forEach.call(ctl.children, function (c) { c.className = ""; });
        b.className = "etx-on";
        draw();
      });
      ctl.appendChild(b);
    });
    container.appendChild(ctl);

    var grid = document.createElement("div");
    grid.className = "etx-grid";
    container.appendChild(grid);

    var readout = document.createElement("div");
    readout.className = "etx-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "etx-caption";
    cap.textContent = config.caption ||
      "The credit_g `checking_status` column (4 levels) under each encoding. Toggle to compare what each " +
      "representation costs (width) and captures (order, similarity).";
    container.appendChild(cap);

    function cell(txt, bg, mono) {
      var d = document.createElement("div");
      d.className = "etx-cell" + (mono ? " etx-mono" : "");
      d.textContent = txt;
      if (bg) { d.style.background = bg; d.style.color = "#fff"; }
      return d;
    }

    function draw() {
      grid.innerHTML = "";
      var tbl = document.createElement("div");
      tbl.className = "etx-tbl";

      if (scheme === "onehot") {
        tbl.style.gridTemplateColumns = "150px repeat(4, 1fr)";
        tbl.appendChild(cell("level", "", false));
        LEVELS.forEach(function (l) { tbl.appendChild(cell("is_" + l.level, "", true)); });
        LEVELS.forEach(function (l, i) {
          tbl.appendChild(cell(l.level, "", false));
          LEVELS.forEach(function (_, j) {
            tbl.appendChild(cell(i === j ? "1" : "0", i === j ? "#2e6fb0" : "", true));
          });
        });
        grid.appendChild(tbl);
        readout.innerHTML = "<strong>one-hot:</strong> 4 levels → <strong>4 columns</strong> (one per level; " +
          "across all 13 categoricals that is <strong>+54 columns</strong>). Sparse, and every level is " +
          "equidistant — the model sees <em>no order and no similarity</em>. Safe from leakage; costly in width.";
      } else if (scheme === "ordinal") {
        tbl.style.gridTemplateColumns = "150px 1fr";
        tbl.appendChild(cell("level", "", false));
        tbl.appendChild(cell("code", "", true));
        LEVELS.forEach(function (l) {
          tbl.appendChild(cell(l.level, "", false));
          tbl.appendChild(cell(String(l.ord), "#7a4fb0", true));
        });
        grid.appendChild(tbl);
        readout.innerHTML = "<strong>ordinal:</strong> 4 levels → <strong>1 integer</strong> (0,1,2,3). " +
          "Compact — but it <em>invents a false order</em>: it asserts &lt;0 &lt; 0–200 &lt; ≥200 &lt; none and " +
          "that the gaps are equal. A <em>linear</em> model must obey that fiction (AUC 0.782→0.739 on credit_g); " +
          "a tree can split around it.";
      } else if (scheme === "target") {
        tbl.style.gridTemplateColumns = "150px 90px 1fr";
        tbl.appendChild(cell("level", "", false));
        tbl.appendChild(cell("n", "", true));
        tbl.appendChild(cell("P(good | level)", "", true));
        LEVELS.forEach(function (l) {
          tbl.appendChild(cell(l.level, "", false));
          tbl.appendChild(cell(String(l.n), "", true));
          tbl.appendChild(cell(l.target.toFixed(3), ramp(l.target), true));
        });
        grid.appendChild(tbl);
        readout.innerHTML = "<strong>target (mean) encoding:</strong> each level → <strong>one number</strong>, " +
          "the (smoothed) mean of the label for that level — ordered by risk, and compact even at huge " +
          "cardinality. But it is computed <em>from the label</em>, so it <strong>leaks</strong> unless done " +
          "out-of-fold (next viz). Note it is literally a <em>1-D learned embedding</em>.";
      } else {
        tbl.style.gridTemplateColumns = "150px 1fr 1fr";
        tbl.appendChild(cell("level", "", false));
        tbl.appendChild(cell("emb[0]", "", true));
        tbl.appendChild(cell("emb[1]", "", true));
        LEVELS.forEach(function (l) {
          tbl.appendChild(cell(l.level, "", false));
          tbl.appendChild(cell(l.emb[0].toFixed(2), "#1e6b3c", true));
          tbl.appendChild(cell(l.emb[1].toFixed(2), "#1e6b3c", true));
        });
        grid.appendChild(tbl);
        readout.innerHTML = "<strong>entity embedding:</strong> each level → a short <strong>learned dense " +
          "vector</strong> (here 2-D; Guo &amp; Berkhahn use up to 50). Trained end-to-end with the model, so " +
          "similar levels move close together — a <em>d-dimensional</em> generalisation of target encoding that " +
          "captures similarity, not just a single risk number.";
      }
    }

    draw();
    return { set: function (s) { scheme = s; draw(); } };
  }

  global.EncodingTaxonomyViz = { mount: mount };
})(window);
