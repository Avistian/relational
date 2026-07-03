/**
 * Reusable missingness-mechanism visualizer (MCAR / MAR / MNAR).
 *
 * Shows two columns: an observed driver column (X2) and a target column (X1)
 * whose cells go missing. Toggling the mechanism reveals WHAT the probability
 * of being missing depends on:
 *   MCAR  - nothing (a fixed random subset)
 *   MAR   - the OBSERVED column X2 (rings the driving X2 cells)
 *   MNAR  - the target's own hidden value X1 (ghosts the would-be-high values)
 *
 * Usage: MissingnessViz.mount(containerElement, { nRows: 12 })
 * Companion CSS lives in the lesson <style> block (class prefix .miss-viz).
 */
(function (global) {
  "use strict";

  // Fixed dataset so the picture is stable across reloads and print.
  // Each row: x1 = target value, x2 = observed driver value (both 0..1).
  var DATA = [
    { x1: 0.12, x2: 0.40 },
    { x1: 0.88, x2: 0.20 },
    { x1: 0.35, x2: 0.92 },
    { x1: 0.67, x2: 0.55 },
    { x1: 0.05, x2: 0.78 },
    { x1: 0.95, x2: 0.33 },
    { x1: 0.50, x2: 0.10 },
    { x1: 0.28, x2: 0.85 },
    { x1: 0.73, x2: 0.62 },
    { x1: 0.18, x2: 0.48 },
    { x1: 0.82, x2: 0.07 },
    { x1: 0.42, x2: 0.70 }
  ];
  // A fixed "coin flip" per row for MCAR (no dependence on x1 or x2).
  var MCAR_FLAGS = [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0];

  var MODES = {
    MCAR: {
      label: "MCAR",
      caption:
        "MCAR \u2014 missingness depends on nothing. The gaps fall by pure chance; " +
        "neither X2 nor the hidden X1 value predicts them. Complete-case analysis stays unbiased.",
      missing: function (row, i) { return MCAR_FLAGS[i] === 1; },
      driver: "none"
    },
    MAR: {
      label: "MAR",
      caption:
        "MAR \u2014 missingness depends on the OBSERVED column. X1 goes missing exactly where " +
        "X2 is high (ringed). Because the driver is observed, conditioning on X2 (imputation) can fix it.",
      missing: function (row) { return row.x2 >= 0.70; },
      driver: "x2"
    },
    MNAR: {
      label: "MNAR",
      caption:
        "MNAR \u2014 missingness depends on the value that went missing. X1 hides exactly its own " +
        "high values (ghosted). Nothing observed explains it, so imputation cannot recover the truth.",
      missing: function (row) { return row.x1 >= 0.70; },
      driver: "x1"
    }
  };

  function shade(v) {
    // light -> dark blue ramp for a 0..1 value
    var l = Math.round(90 - v * 55);
    return "hsl(206, 45%, " + l + "%)";
  }

  function mount(container, config) {
    config = config || {};
    var rows = DATA.slice(0, config.nRows || DATA.length);

    container.innerHTML = "";
    container.classList.add("miss-viz");

    var controls = document.createElement("div");
    controls.className = "miss-viz-controls";
    container.appendChild(controls);

    var grid = document.createElement("div");
    grid.className = "miss-viz-grid";
    container.appendChild(grid);

    var caption = document.createElement("p");
    caption.className = "miss-viz-caption";
    container.appendChild(caption);

    var current = "MCAR";

    Object.keys(MODES).forEach(function (key) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "miss-viz-btn";
      btn.textContent = MODES[key].label;
      btn.addEventListener("click", function () {
        current = key;
        render();
      });
      controls.appendChild(btn);
    });

    function render() {
      var mode = MODES[current];

      Array.prototype.forEach.call(controls.children, function (b) {
        b.classList.toggle("active", b.textContent === mode.label);
      });

      grid.innerHTML = "";

      var head = document.createElement("div");
      head.className = "miss-viz-row miss-viz-head";
      head.innerHTML =
        '<span class="miss-viz-rlabel"></span>' +
        '<span class="miss-viz-colh">X2 (observed)</span>' +
        '<span class="miss-viz-colh">X1 (target)</span>';
      grid.appendChild(head);

      rows.forEach(function (row, i) {
        var isMissing = mode.missing(row, i);

        var r = document.createElement("div");
        r.className = "miss-viz-row";

        var label = document.createElement("span");
        label.className = "miss-viz-rlabel";
        label.textContent = "row " + (i + 1);
        r.appendChild(label);

        // X2 driver cell
        var c2 = document.createElement("span");
        c2.className = "miss-viz-cell";
        c2.style.background = shade(row.x2);
        if (mode.driver === "x2" && row.x2 >= 0.70) {
          c2.classList.add("driver");
          c2.title = "high X2 \u2192 drives X1 missing";
        }
        r.appendChild(c2);

        // X1 target cell
        var c1 = document.createElement("span");
        c1.className = "miss-viz-cell";
        if (isMissing) {
          c1.classList.add("missing");
          if (mode.driver === "x1") {
            c1.classList.add("ghost");
            c1.style.background = shade(row.x1);
            c1.title = "its own high value caused the gap";
          }
        } else {
          c1.style.background = shade(row.x1);
        }
        r.appendChild(c1);

        grid.appendChild(r);
      });

      caption.textContent = mode.caption;
    }

    render();
  }

  global.MissingnessViz = { mount: mount };
})(window);
