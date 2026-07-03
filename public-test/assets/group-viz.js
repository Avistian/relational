/**
 * Group-leakage visualizer: random K-fold vs GroupKFold.
 *
 * Shows a grid where each row is a group (e.g. a customer) and each cell is a
 * sample (e.g. one order). Cells are colored by the fold they are assigned to.
 * With random K-fold a single group's samples scatter across folds — the same
 * group lands in both train and test (leakage). With GroupKFold every sample in
 * a group shares one color: the group is atomic and never leaks.
 *
 * Usage: GroupViz.mount(containerElement, config?)
 * config: { nGroups, samplesPerGroup, nSplits, seed }
 */
(function (global) {
  "use strict";

  var PALETTE = ["#1a5276", "#9a7b0a", "#922b21", "#1e6b3c", "#5b2c6f"];

  function mulberry32(seed) {
    return function () {
      seed |= 0;
      seed = (seed + 0x6d2b79f5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function randomFolds(nGroups, samplesPerGroup, nSplits, rng) {
    var assign = [];
    var g;
    var s;
    for (g = 0; g < nGroups; g++) {
      assign.push([]);
      for (s = 0; s < samplesPerGroup; s++) {
        assign[g].push(Math.floor(rng() * nSplits));
      }
    }
    return assign;
  }

  function groupFolds(nGroups, samplesPerGroup, nSplits) {
    var assign = [];
    var g;
    var s;
    for (g = 0; g < nGroups; g++) {
      assign.push([]);
      var fold = g % nSplits;
      for (s = 0; s < samplesPerGroup; s++) assign[g].push(fold);
    }
    return assign;
  }

  function countLeakedGroups(assign) {
    var leaked = 0;
    assign.forEach(function (row) {
      var first = row[0];
      var spans = row.some(function (f) {
        return f !== first;
      });
      if (spans) leaked += 1;
    });
    return leaked;
  }

  function mount(container, config) {
    config = config || {};
    var nGroups = config.nGroups || 8;
    var samplesPerGroup = config.samplesPerGroup || 4;
    var nSplits = config.nSplits || 4;
    var seed = config.seed != null ? config.seed : 7;

    container.innerHTML = "";
    container.classList.add("group-viz");

    var intro = document.createElement("p");
    intro.className = "group-viz-intro";
    intro.textContent =
      "Toy data: " +
      nGroups +
      " groups (customers), " +
      samplesPerGroup +
      " samples each (orders). Color = assigned fold. A group that shows >1 color is split across folds — it will sit in both train and test.";
    container.appendChild(intro);

    var controls = document.createElement("div");
    controls.className = "group-viz-controls";
    container.appendChild(controls);

    var btnRandom = document.createElement("button");
    btnRandom.type = "button";
    btnRandom.className = "group-viz-btn";
    btnRandom.textContent = "Random K-fold";
    controls.appendChild(btnRandom);

    var btnGroup = document.createElement("button");
    btnGroup.type = "button";
    btnGroup.className = "group-viz-btn active";
    btnGroup.textContent = "GroupKFold";
    controls.appendChild(btnGroup);

    var grid = document.createElement("div");
    grid.className = "group-viz-grid";
    container.appendChild(grid);

    var caption = document.createElement("div");
    caption.className = "group-viz-caption";
    container.appendChild(caption);

    function render(mode) {
      btnRandom.classList.toggle("active", mode === "random");
      btnGroup.classList.toggle("active", mode === "group");

      var assign;
      if (mode === "random") {
        assign = randomFolds(nGroups, samplesPerGroup, nSplits, mulberry32(seed));
      } else {
        assign = groupFolds(nGroups, samplesPerGroup, nSplits);
      }

      grid.innerHTML = "";
      assign.forEach(function (row, gi) {
        var rowEl = document.createElement("div");
        rowEl.className = "group-viz-row";

        var label = document.createElement("span");
        label.className = "group-viz-glabel";
        label.textContent = "Group " + (gi + 1);
        rowEl.appendChild(label);

        var cells = document.createElement("div");
        cells.className = "group-viz-cells";
        row.forEach(function (fold) {
          var cell = document.createElement("span");
          cell.className = "group-viz-cell";
          cell.style.background = PALETTE[fold % PALETTE.length];
          cell.title = "Fold " + (fold + 1);
          cells.appendChild(cell);
        });
        rowEl.appendChild(cells);

        var spans = row.some(function (f) {
          return f !== row[0];
        });
        var flag = document.createElement("span");
        flag.className = "group-viz-flag " + (spans ? "leak" : "safe");
        flag.textContent = spans ? "leaks" : "intact";
        rowEl.appendChild(flag);

        grid.appendChild(rowEl);
      });

      var leaked = countLeakedGroups(assign);
      if (mode === "group") {
        caption.textContent =
          "GroupKFold: every group is one color — each group sits entirely in one fold. Leaked groups: " +
          leaked +
          " of " +
          nGroups +
          ". The test score estimates performance on brand-new groups.";
      } else {
        caption.textContent =
          "Random K-fold: " +
          leaked +
          " of " +
          nGroups +
          " groups span multiple folds, so the model trains and tests on the same customer. The score is optimistically biased — it never measures generalization to new groups.";
      }
    }

    btnRandom.addEventListener("click", function () {
      render("random");
    });
    btnGroup.addEventListener("click", function () {
      render("group");
    });

    render("group");
  }

  global.GroupViz = { mount: mount };
})(window);
