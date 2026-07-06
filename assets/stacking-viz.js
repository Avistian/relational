/**
 * Stacking / out-of-fold meta-feature visualizer (Wolpert 1992, stacked generalization).
 *
 * Shows how a level-1 meta-learner's training features are built from a level-0 base
 * model, and why they MUST be out-of-fold.
 *
 *  - Mode "Out-of-fold (correct)": step through the K folds. For the active fold, its
 *    rows are PREDICT (held out) and the rest are TRAIN; the meta-feature cell for the
 *    held-out rows is filled with a HONEST prediction (green). After stepping through
 *    all folds the whole meta column is filled cleanly.
 *  - Mode "In-sample (leak)": every row is TRAIN and PREDICT at once, so every meta cell
 *    fills instantly with a LEAKED prediction (red) — the base model saw each row's own
 *    label, so the value is a memorized copy, not a forecast.
 *
 * Expected states:
 *  - OOF, fold 0 active: rows 0..2 outlined green (predict), rows 3..11 blue (train),
 *    only rows 0..2 meta cells filled; readout says "3/12 rows filled honestly".
 *  - OOF after Next through all folds: 12/12 filled, all green, "honest — ready for the meta-learner".
 *  - In-sample: all 12 cells red immediately, readout warns "leak: base model saw its own label".
 *
 * Usage: StackingViz.mount(container, { caption })
 */
(function (global) {
  "use strict";

  var NROWS = 12;
  var K = 4;
  var PER = NROWS / K; // 3 rows per fold

  function foldOf(i) { return Math.floor(i / PER); }

  // Deterministic pseudo-predictions so the picture is stable.
  function honestPred(i) { return 0.15 + 0.7 * (((i * 37) % 100) / 100); }
  function leakPred(i, label) { return label ? 0.97 : 0.03; } // memorized copy of the label

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "stacking-viz";

    var mode = "oof";        // "oof" | "insample"
    var activeFold = 0;      // for oof stepping
    var filled = [];         // per-row: null | "honest" | "leak"
    for (var z = 0; z < NROWS; z++) filled.push(null);
    // toy labels for the leak illustration
    var labels = [];
    for (var l = 0; l < NROWS; l++) labels.push(l % 3 === 0 ? 1 : 0);

    // ---- controls
    var ctl = document.createElement("div");
    ctl.className = "stk-ctl";
    var bOof = document.createElement("button");
    bOof.textContent = "Out-of-fold (correct)";
    var bLeak = document.createElement("button");
    bLeak.textContent = "In-sample (leak)";
    var bNext = document.createElement("button");
    bNext.textContent = "next fold ▶";
    var bReset = document.createElement("button");
    bReset.textContent = "Reset";
    ctl.appendChild(bOof);
    ctl.appendChild(bLeak);
    ctl.appendChild(bNext);
    ctl.appendChild(bReset);
    container.appendChild(ctl);

    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 360 300");
    svg.setAttribute("class", "stk-svg");
    container.appendChild(svg);

    var readout = document.createElement("p");
    readout.className = "stk-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "stk-caption";
    cap.textContent = config.caption || "";
    container.appendChild(cap);

    var SVGNS = "http://www.w3.org/2000/svg";
    function el(tag, attrs, text) {
      var n = document.createElementNS(SVGNS, tag);
      for (var k in attrs) n.setAttribute(k, attrs[k]);
      if (text != null) n.textContent = text;
      return n;
    }

    var FOLD_FILL = ["#eaf2fb", "#eef7f0", "#fdf3e7", "#f6eef8"];

    function fillFold(f) {
      for (var i = 0; i < NROWS; i++) if (foldOf(i) === f) filled[i] = "honest";
    }

    function resetFill() {
      for (var i = 0; i < NROWS; i++) filled[i] = null;
      activeFold = 0;
    }

    function computeInSample() {
      for (var i = 0; i < NROWS; i++) filled[i] = "leak";
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      // headers
      svg.appendChild(el("text", { x: 12, y: 16, class: "stk-lab" }, "TRAINING ROWS"));
      svg.appendChild(el("text", { x: 150, y: 16, class: "stk-lab" }, "LEVEL-0 BASE MODEL"));
      svg.appendChild(el("text", { x: 262, y: 16, class: "stk-lab" }, "META-FEATURE"));

      var rowH = 18, top = 26, boxX = 12, boxW = 96, metaX = 262, metaW = 70;

      for (var i = 0; i < NROWS; i++) {
        var y = top + i * rowH;
        var f = foldOf(i);
        var isPredict, isTrain;
        if (mode === "oof") {
          isPredict = f === activeFold;
          isTrain = !isPredict;
        } else {
          isPredict = true; isTrain = true;
        }
        // training-row cell
        var rect = el("rect", {
          x: boxX, y: y, width: boxW, height: rowH - 3, rx: 3,
          fill: FOLD_FILL[f],
          stroke: isPredict ? "#1e6b3c" : "#9fb6cf",
          "stroke-width": isPredict ? 2 : 1,
        });
        svg.appendChild(rect);
        svg.appendChild(el("text", { x: boxX + 6, y: y + 12, class: "stk-cell" },
          "row " + i + " · fold " + f));
        // role tag
        var role = mode === "insample" ? "train+predict" : (isPredict ? "predict" : "train");
        svg.appendChild(el("text", {
          x: boxX + boxW - 4, y: y + 12, class: "stk-role",
          "text-anchor": "end",
          fill: mode === "insample" ? "#b03a2e" : (isPredict ? "#1e6b3c" : "#5a7ba6"),
        }, role));

        // meta-feature cell
        var mf = el("rect", {
          x: metaX, y: y, width: metaW, height: rowH - 3, rx: 3,
          fill: filled[i] === "honest" ? "#d7ecdd" : (filled[i] === "leak" ? "#f7dcd7" : "#f4f4f4"),
          stroke: filled[i] === "honest" ? "#1e6b3c" : (filled[i] === "leak" ? "#b03a2e" : "#d5d8dc"),
          "stroke-width": 1,
        });
        svg.appendChild(mf);
        if (filled[i]) {
          var val = filled[i] === "leak" ? leakPred(i, labels[i]) : honestPred(i);
          svg.appendChild(el("text", {
            x: metaX + metaW / 2, y: y + 12, class: "stk-val", "text-anchor": "middle",
            fill: filled[i] === "leak" ? "#b03a2e" : "#1e6b3c",
          }, val.toFixed(2)));
        }
      }

      // arrow base -> meta
      svg.appendChild(el("line", {
        x1: boxX + boxW + 4, y1: 150, x2: metaX - 4, y2: 150,
        stroke: "#9aa0a6", "stroke-width": 1, "marker-end": "url(#stk-arrow)",
      }));
      var defs = el("defs", {});
      var marker = el("marker", { id: "stk-arrow", markerWidth: 7, markerHeight: 7, refX: 6, refY: 3, orient: "auto" });
      marker.appendChild(el("path", { d: "M0,0 L6,3 L0,6 Z", fill: "#9aa0a6" }));
      defs.appendChild(marker);
      svg.appendChild(defs);

      var nFilled = filled.filter(function (v) { return v; }).length;
      var allHonest = nFilled === NROWS && filled.every(function (v) { return v === "honest"; });
      if (mode === "insample") {
        readout.innerHTML =
          "<strong style='color:#b03a2e'>Leak:</strong> each base prediction was made on the row it " +
          "trained on — the meta-feature is a memorized copy of the label, not a forecast. The meta-learner " +
          "will over-trust whichever base memorizes best.";
      } else if (allHonest) {
        readout.innerHTML =
          "<strong style='color:#1e6b3c'>Honest:</strong> every meta-feature came from a fold the base model " +
          "never saw (" + NROWS + "/" + NROWS + " filled). Ready to train the level-1 meta-learner.";
      } else {
        readout.innerHTML =
          "Out-of-fold: fold <strong>" + activeFold + "</strong> is held out and predicted by a model trained " +
          "on the other folds (" + nFilled + "/" + NROWS + " rows filled). Press <em>fold ▶</em> to fill the rest.";
      }
    }

    function setMode(m) {
      mode = m;
      resetFill();
      if (m === "insample") computeInSample();
      else fillFold(0); // land on fold 0 and predict it honestly right away
      bOof.classList.toggle("stk-active", m === "oof");
      bLeak.classList.toggle("stk-active", m === "insample");
      draw();
    }

    function nextFold() {
      if (mode !== "oof") return;
      if (activeFold < K - 1) {
        activeFold += 1;
        fillFold(activeFold);
      }
      draw();
    }

    bOof.addEventListener("click", function () { setMode("oof"); });
    bLeak.addEventListener("click", function () { setMode("insample"); });
    bNext.addEventListener("click", nextFold);
    bReset.addEventListener("click", function () { setMode("oof"); });

    setMode("oof");
  }

  global.StackingViz = { mount: mount };
})(window);
