/**
 * Target-encoding leakage — the pitfall Lesson 031 is named for.
 *
 * NAIVE target encoding computes each level's mean target INCLUDING the row being encoded (and, worse,
 * often on the whole dataset before the split). On a high-cardinality / near-unique column the encoding
 * becomes a copy of the row's own label — the model reads the answer. OUT-OF-FOLD encoding computes each
 * row's value from OTHER folds only, so a unique level falls back to the global prior and the fake signal
 * vanishes.
 *
 * Illustrative rows show the mechanism on a unique `customer_id`; the payoff AUCs are REAL, from
 * labs/_verify_l031.py on a synthetic near-unique id column (2 rows/level) fed to a GBDT:
 *   naive  id-only feature: train AUC 0.891, test AUC 0.891  (pure leakage — the column has NO real signal)
 *   oof    id-only feature: train AUC 0.514, test AUC 0.504  (honest: chance, the truth)
 *
 * Usage: TargetLeakViz.mount(container, { caption })
 * Expected states:
 *   - naive : each row's encoding == its own label; red "= the answer" flags; AUC 0.891 on noise.
 *   - oof   : each row's encoding == global prior 0.70; no signal; AUC 0.504 (chance).
 */
(function (global) {
  "use strict";

  var GLOBAL = 0.70;
  var ROWS = [
    { id: "cust_8841", label: 1 },
    { id: "cust_2093", label: 0 },
    { id: "cust_5507", label: 1 },
    { id: "cust_1174", label: 1 },
    { id: "cust_6620", label: 0 }
  ];
  var AUC = {
    naive: { train: 0.891, test: 0.891 },
    oof: { train: 0.514, test: 0.504 }
  };

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "tlk-viz";
    var mode = "naive";

    var ctl = document.createElement("div");
    ctl.className = "tlk-ctl";
    [["naive", "naive (leaky)"], ["oof", "out-of-fold (honest)"]].forEach(function (m) {
      var b = document.createElement("button");
      b.textContent = m[1];
      if (m[0] === mode) b.className = "tlk-on";
      b.addEventListener("click", function () {
        mode = m[0];
        Array.prototype.forEach.call(ctl.children, function (c) { c.className = ""; });
        b.className = "tlk-on";
        draw();
      });
      ctl.appendChild(b);
    });
    container.appendChild(ctl);

    var grid = document.createElement("div");
    grid.className = "tlk-grid";
    container.appendChild(grid);

    var readout = document.createElement("div");
    readout.className = "tlk-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "tlk-caption";
    cap.textContent = config.caption ||
      "A unique customer_id target-encoded. Naive encoding copies each row's own label; out-of-fold " +
      "encoding falls back to the prior. The AUCs below are real (synthetic near-unique id, credit_g run).";
    container.appendChild(cap);

    function cell(txt, cls) {
      var d = document.createElement("div");
      d.className = "tlk-cell" + (cls ? " " + cls : "");
      d.innerHTML = txt;
      return d;
    }

    function draw() {
      grid.innerHTML = "";
      var tbl = document.createElement("div");
      tbl.className = "tlk-tbl";
      tbl.style.gridTemplateColumns = "120px 70px 150px 1fr";
      tbl.appendChild(cell("customer_id", "tlk-head"));
      tbl.appendChild(cell("label", "tlk-head"));
      tbl.appendChild(cell(mode === "naive" ? "naive encode" : "out-of-fold encode", "tlk-head"));
      tbl.appendChild(cell("what the model sees", "tlk-head"));

      ROWS.forEach(function (r) {
        tbl.appendChild(cell(r.id, "tlk-mono"));
        tbl.appendChild(cell(String(r.label), "tlk-mono"));
        if (mode === "naive") {
          tbl.appendChild(cell(r.label.toFixed(2), "tlk-mono tlk-leak"));
          tbl.appendChild(cell("= its own label ⚠", "tlk-flag"));
        } else {
          tbl.appendChild(cell(GLOBAL.toFixed(2), "tlk-mono tlk-safe"));
          tbl.appendChild(cell("= prior (no other rows share this id)", "tlk-ok"));
        }
      });
      grid.appendChild(tbl);

      var a = AUC[mode];
      if (mode === "naive") {
        readout.innerHTML =
          "<span class='tlk-pill tlk-bad'>naive</span> The encoded feature <em>is</em> the label. Fed to a " +
          "GBDT, an id column with <strong>no real signal</strong> scores train AUC <strong>" +
          a.train.toFixed(3) + "</strong> and — because the encoding was built before the split — even " +
          "test AUC <strong>" + a.test.toFixed(3) + "</strong>. A pure mirage.";
      } else {
        readout.innerHTML =
          "<span class='tlk-pill tlk-good'>out-of-fold</span> Each row is encoded from <em>other</em> folds " +
          "only, so a unique id falls back to the prior 0.70 and carries no information. The same column now " +
          "scores train AUC <strong>" + a.train.toFixed(3) + "</strong>, test AUC <strong>" +
          a.test.toFixed(3) + "</strong> — chance, the honest truth.";
      }
    }

    draw();
    return { set: function (m) { mode = m; draw(); } };
  }

  global.TargetLeakViz = { mount: mount };
})(window);
