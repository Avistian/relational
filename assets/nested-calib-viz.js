/**
 * Nested-fold calibration leak: a library default that ignores your groups. (Lesson 036)
 *
 * The outer CV is person-grouped and correct — StratifiedGroupKFold(groups=person_id)
 * keeps every situation of a person on ONE side of the outer split. But
 * CalibratedClassifierCV(estimator, method="isotonic", cv=5) then splits that outer
 * TRAINING block again, internally, with a plain StratifiedKFold that has never heard
 * of person_id. So inside the training block a person's two situations can land on
 * opposite sides of the base-model / calibrator boundary.
 *
 * Each person is a capsule holding 1-2 situation squares:
 *   - blue   = row used to FIT THE BASE MODEL (LightGBM)
 *   - amber  = row used to FIT THE CALIBRATOR (isotonic map)
 *   - red    = row in the OUTER TEST fold (never touched; this is why the reported
 *              ECE stays honest)
 * A person with rows in BOTH blue and amber is outlined red and flagged: the
 * calibrator is being fit on a person the base model already trained on, so the
 * scores it calibrates against are optimistically sharp, and the isotonic map it
 * learns is the wrong shape for the genuinely-unseen persons of the outer test fold.
 *
 * Two modes (a true toggle: the SAME mechanism under one knob — how the inner split
 * is drawn):
 *   - "cv=5 (library default)"  -> ungrouped inner split, 3 straddling persons
 *   - "grouped inner split"     -> persons kept intact, 0 straddling persons
 *
 * Plain <script> (file://-safe). Usage: NestedCalibViz.mount(el, config?)
 * config: { realPersons, realRows, eceBefore, eceAfter } — the measured numbers
 *         quoted in the readout (defaults are the verified L036 homework values).
 *
 * Expected states:
 *   - default (ungrouped): 3 capsules outlined red, readout "3 of 4" + "leak present"
 *   - grouped: 0 capsules outlined red, readout "0 of 4" + "no person straddles"
 *   - the outer-test capsules (red squares) are never ringed in either mode
 */
(function (global) {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var BASE = "#2e6fb0";   // base-model training rows
  var CALIB = "#c8871b";  // calibrator-fitting rows
  var TEST = "#b03a2e";   // outer test fold
  var WARN = "#b03a2e";

  // Deterministic layout. Each entry: [personLabel, nRows, isOuterTest].
  var PERSONS = [
    ["P1", 2, false], ["P2", 1, false], ["P3", 2, false], ["P4", 1, false],
    ["P5", 1, false], ["P6", 2, false], ["P7", 1, false], ["P8", 1, false],
    ["P9", 2, false],
    ["P10", 2, true], ["P11", 1, true], ["P12", 1, true]
  ];

  // Which rows go to the calibrator, per mode. Keyed "person:rowIndex".
  // Ungrouped: P1, P3, P9 straddle (one row each to the calibrator); P6 stays whole.
  var CALIB_UNGROUPED = { "P1:1": true, "P3:0": true, "P5:0": true, "P9:1": true };
  // Grouped: whole persons go to the calibrator, so nobody straddles.
  var CALIB_GROUPED = { "P5:0": true, "P6:0": true, "P6:1": true, "P8:0": true };

  function el(tag, attrs) {
    var n = document.createElementNS(SVGNS, tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    return n;
  }

  function mount(container, config) {
    config = config || {};
    var realPersons = config.realPersons != null ? config.realPersons : 675;
    var realRows = config.realRows != null ? config.realRows : 1411;
    var eceBefore = config.eceBefore || "0.0363";
    var eceAfter = config.eceAfter || null;

    container.innerHTML = "";
    container.classList.add("nc-viz");

    var controls = document.createElement("div");
    controls.className = "nc-ctl";
    container.appendChild(controls);

    var btnUn = document.createElement("button");
    btnUn.type = "button";
    btnUn.textContent = "cv=5 (library default)";
    controls.appendChild(btnUn);

    var btnGr = document.createElement("button");
    btnGr.type = "button";
    btnGr.textContent = "grouped inner split";
    controls.appendChild(btnGr);

    var W = 580, H = 250;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, class: "nc-svg", width: "100%" });
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "nc-readout";
    container.appendChild(readout);

    var mode = "ungrouped";

    function render() {
      svg.innerHTML = "";
      var calibMap = mode === "ungrouped" ? CALIB_UNGROUPED : CALIB_GROUPED;

      // ---- lane labels -------------------------------------------------
      var l1 = el("text", { x: 10, y: 20, class: "nc-lane" });
      l1.textContent = "OUTER FOLD: TRAINING BLOCK  (person-grouped \u2014 correct)";
      svg.appendChild(l1);

      var l2 = el("text", { x: 10, y: 176, class: "nc-lane" });
      l2.textContent = "OUTER FOLD: TEST BLOCK  (whole persons held out \u2014 never touched)";
      svg.appendChild(l2);

      var straddlers = [];
      var nMulti = 0;
      var x = 12, y = 34;
      var capsuleH = 46, sq = 16, gap = 4;

      PERSONS.forEach(function (p) {
        var label = p[0], nRows = p[1], isTest = p[2];
        if (isTest) return;
        if (nRows > 1) nMulti++;

        var wCap = 10 + nRows * (sq + gap);
        // wrap
        if (x + wCap > W - 12) { x = 12; y += capsuleH + 12; }

        var kinds = [];
        var i;
        for (i = 0; i < nRows; i++) {
          kinds.push(calibMap[label + ":" + i] ? "calib" : "base");
        }
        var hasBase = kinds.indexOf("base") >= 0;
        var hasCalib = kinds.indexOf("calib") >= 0;
        var straddles = hasBase && hasCalib;
        if (straddles) straddlers.push(label);

        var cap = el("rect", {
          x: x, y: y, width: wCap, height: capsuleH, rx: 7,
          fill: "#fff",
          stroke: straddles ? WARN : "#c9c9c9",
          "stroke-width": straddles ? 2.2 : 1,
          "stroke-dasharray": straddles ? "" : ""
        });
        svg.appendChild(cap);

        for (i = 0; i < nRows; i++) {
          svg.appendChild(el("rect", {
            x: x + 5 + i * (sq + gap), y: y + 8, width: sq, height: sq, rx: 3,
            fill: kinds[i] === "calib" ? CALIB : BASE
          }));
        }

        var pl = el("text", { x: x + wCap / 2, y: y + capsuleH - 8, class: "nc-plabel", "text-anchor": "middle" });
        pl.textContent = label + (straddles ? " \u26a0" : "");
        if (straddles) pl.setAttribute("fill", WARN);
        svg.appendChild(pl);

        x += wCap + 10;
      });

      // ---- outer test block --------------------------------------------
      var tx = 12, ty = 190;
      PERSONS.forEach(function (p) {
        var label = p[0], nRows = p[1], isTest = p[2];
        if (!isTest) return;
        var wCap = 10 + nRows * (sq + gap);
        var cap = el("rect", {
          x: tx, y: ty, width: wCap, height: capsuleH, rx: 7,
          fill: "#fff", stroke: "#c9c9c9", "stroke-width": 1
        });
        svg.appendChild(cap);
        var i;
        for (i = 0; i < nRows; i++) {
          svg.appendChild(el("rect", {
            x: tx + 5 + i * (sq + gap), y: ty + 8, width: sq, height: sq, rx: 3, fill: TEST
          }));
        }
        var pl = el("text", { x: tx + wCap / 2, y: ty + capsuleH - 8, class: "nc-plabel", "text-anchor": "middle" });
        pl.textContent = label;
        svg.appendChild(pl);
        tx += wCap + 10;
      });

      // ---- legend -------------------------------------------------------
      var lg = el("g", {});
      svg.appendChild(lg);
      var items = [
        [BASE, "fits the base model (LightGBM)"],
        [CALIB, "fits the calibrator (isotonic)"],
        [TEST, "outer test fold"]
      ];
      var lx = 12;
      items.forEach(function (it) {
        lg.appendChild(el("rect", { x: lx, y: 140, width: 11, height: 11, rx: 2, fill: it[0] }));
        var t = el("text", { x: lx + 16, y: 149, class: "nc-legend" });
        t.textContent = it[1];
        lg.appendChild(t);
        lx += 20 + it[1].length * 5.4;
      });

      // ---- readout ------------------------------------------------------
      btnUn.className = mode === "ungrouped" ? "nc-on" : "";
      btnGr.className = mode === "grouped" ? "nc-on" : "";

      if (mode === "ungrouped") {
        readout.innerHTML =
          "<strong>Inner split ignores <code>person_id</code>.</strong> " +
          "<span class='nc-bad'>" + straddlers.length + " of " + nMulti + "</span> multi-situation persons straddle the " +
          "base-model / calibrator boundary (" + straddlers.join(", ") + "): the isotonic map is fitted on rows " +
          "belonging to a person the base model has already trained on, where its scores are " +
          "optimistically sharp. <strong>Leak present</strong> \u2014 and note where it is <em>not</em>: the outer " +
          "test block is untouched, so the <em>reported</em> ECE (" + eceBefore + ") is still honest. " +
          "This defect does not inflate your number; it quietly ships a worse-shaped calibrator. " +
          "In the real labelled set that is <strong>" + realPersons + " persons / " + realRows + " rows</strong> " +
          "(25% of the data) exposed to the mis-split.";
      } else {
        readout.innerHTML =
          "<strong>Inner split respects <code>person_id</code>.</strong> " +
          "<span class='nc-good'>0 of " + nMulti + "</span> persons straddle the boundary \u2014 whole persons go to the " +
          "base model or to the calibrator, never both. The isotonic map is now fitted on scores from persons the " +
          "base model has never seen, which is exactly the regime it will face in production" +
          (eceAfter ? ", and the measured top-label ECE moves " + eceBefore + " \u2192 <strong>" + eceAfter + "</strong>." : ".");
      }
    }

    btnUn.addEventListener("click", function () { mode = "ungrouped"; render(); });
    btnGr.addEventListener("click", function () { mode = "grouped"; render(); });
    render();

    return {
      getMode: function () { return mode; },
      setMode: function (m) { mode = m; render(); }
    };
  }

  global.NestedCalibViz = { mount: mount };
})(window);
