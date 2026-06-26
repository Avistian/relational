/**
 * Preprocessing-leakage visualizer: "fit on all data" vs "Pipeline (fit per fold)".
 *
 * Shows nSplits CV rows. In each row one fold is the held-out test (red) and the
 * rest are train (blue). A ring on a cell means "this row was seen by the
 * preprocessing step when it computed its statistics" (e.g. the mean/scale of a
 * StandardScaler, the median of a SimpleImputer, the columns SelectKBest keeps).
 *
 *   - Fit on all data (LEAK): the transformer was fit once on the whole table, so
 *     EVERY cell is ringed — including the test cells. Test statistics leak into
 *     training, and the CV score is optimistic.
 *   - Pipeline (per fold): the transformer is refit inside each fold on TRAIN only,
 *     so test cells are never ringed. The score is honest.
 *
 * Usage: PipelineViz.mount(containerElement, config?)
 * config: { nSamples, nSplits }
 */
(function (global) {
  "use strict";

  function mount(container, config) {
    config = config || {};
    var nSamples = config.nSamples || 10;
    var nSplits = config.nSplits || 5;

    container.innerHTML = "";
    container.classList.add("pipe-viz");

    var intro = document.createElement("p");
    intro.className = "pipe-viz-intro";
    intro.textContent =
      "Each row is one CV fold: the red square is the held-out test fold, blue squares are training rows. " +
      "A glowing ring means the preprocessing step (scaler, imputer, feature selector) used that row to compute its statistics.";
    container.appendChild(intro);

    var controls = document.createElement("div");
    controls.className = "pipe-viz-controls";
    container.appendChild(controls);

    var btnLeak = document.createElement("button");
    btnLeak.type = "button";
    btnLeak.className = "pipe-viz-btn";
    btnLeak.textContent = "Fit on all data (leak)";
    controls.appendChild(btnLeak);

    var btnPipe = document.createElement("button");
    btnPipe.type = "button";
    btnPipe.className = "pipe-viz-btn active";
    btnPipe.textContent = "Pipeline (fit per fold)";
    controls.appendChild(btnPipe);

    var grid = document.createElement("div");
    grid.className = "pipe-viz-grid";
    container.appendChild(grid);

    var caption = document.createElement("div");
    caption.className = "pipe-viz-caption";
    container.appendChild(caption);

    function testFoldForRow(rowIdx, sampleIdx) {
      // Contiguous blocks: fold k owns samples [k*size, (k+1)*size).
      var size = Math.ceil(nSamples / nSplits);
      return Math.floor(sampleIdx / size) === rowIdx;
    }

    function render(mode) {
      btnLeak.classList.toggle("active", mode === "leak");
      btnPipe.classList.toggle("active", mode === "pipe");

      grid.innerHTML = "";
      var leakedCells = 0;

      for (var r = 0; r < nSplits; r++) {
        var rowEl = document.createElement("div");
        rowEl.className = "pipe-viz-row";

        var label = document.createElement("span");
        label.className = "pipe-viz-rlabel";
        label.textContent = "Fold " + (r + 1);
        rowEl.appendChild(label);

        var cells = document.createElement("div");
        cells.className = "pipe-viz-cells";

        for (var s = 0; s < nSamples; s++) {
          var isTest = testFoldForRow(r, s);
          var cell = document.createElement("span");
          cell.className = "pipe-viz-cell " + (isTest ? "test" : "train");

          // Was this cell seen by preprocessing?
          // leak: every cell seen. pipe: only train cells seen.
          var seen = mode === "leak" ? true : !isTest;
          if (seen) cell.classList.add("seen");
          if (seen && isTest) {
            cell.classList.add("leaked");
            leakedCells += 1;
          }
          cells.appendChild(cell);
        }
        rowEl.appendChild(cells);
        grid.appendChild(rowEl);
      }

      if (mode === "leak") {
        caption.textContent =
          "Fit on all data: the transformer was fit once before splitting, so it read every test cell (" +
          leakedCells +
          " test cells peeked at). Test statistics bleed into training — the CV score is optimistically biased.";
      } else {
        caption.textContent =
          "Pipeline: the transformer is refit inside each fold on training rows only. No test cell is ever read (" +
          leakedCells +
          " leaked). The CV score is an honest estimate of generalization.";
      }
    }

    btnLeak.addEventListener("click", function () {
      render("leak");
    });
    btnPipe.addEventListener("click", function () {
      render("pipe");
    });

    render("pipe");
  }

  global.PipelineViz = { mount: mount };
})(window);
