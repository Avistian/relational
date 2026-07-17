/**
 * Greedy ensemble selection (Caruana et al. 2004) — Auto-sklearn's second extension over Auto-WEKA.
 *
 * After the CASH search leaves a POOL of trained models, don't just keep the single best. Build a
 * weighted ensemble greedily: start empty; repeatedly add (WITH replacement) the pool model whose
 * addition most improves the ENSEMBLE's validation AUC. Adding with replacement lets a strong model
 * get more weight; the result is a diverse blend that beats the single best on held-out data.
 *
 * REAL numbers (labs/_verify_l029.py, credit_g, seed 0): the single best-validation config was a
 * HistGB (TEST AUC 0.824). The greedy ensemble picked 10 distinct members across 3 algorithms
 * (LogReg ×5, HistGB ×4, ExtraTrees ×1 of 25 greedy picks) and scored TEST AUC 0.831 — +0.007 over
 * the single best, from diversity, not a new single winner.
 *
 * Two states via the toggle:
 *   - "Single best model": one bar (HistGB) at 0.824.
 *   - "Greedy ensemble":   the weighted blend at 0.831, with its composition.
 *
 * Usage: EnsembleSelectViz.mount(container, { caption })
 * Handle: { setMode("single" | "ensemble") }
 */
(function (global) {
  "use strict";

  var SINGLE = 0.824, ENS = 0.831;
  var COMPOSITION = [
    { algo: "LogReg", picks: 5, color: "#2e6fb0" },
    { algo: "HistGB", picks: 4, color: "#8e44ad" },
    { algo: "ExtraTrees", picks: 1, color: "#b07a2e" }
  ];
  var YMIN = 0.70, YMAX = 0.85;

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "ensemble-select-viz";

    var mode = "single";

    var ctl = document.createElement("div");
    ctl.className = "esv-ctl";
    var singleBtn = document.createElement("button");
    singleBtn.type = "button"; singleBtn.textContent = "Single best model";
    var ensBtn = document.createElement("button");
    ensBtn.type = "button"; ensBtn.textContent = "Greedy ensemble";
    ctl.appendChild(singleBtn); ctl.appendChild(ensBtn);
    container.appendChild(ctl);

    var svgNS = "http://www.w3.org/2000/svg";
    var W = 420, H = 260;
    var svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("width", "100%");
    svg.setAttribute("class", "esv-svg");
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "esv-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "esv-caption";
    cap.textContent = config.caption ||
      "Greedy ensemble selection over the search pool (credit_g, seed 0). The single best config vs the " +
      "weighted blend the greedy procedure builds. Height = held-out TEST ROC-AUC.";
    container.appendChild(cap);

    var PL = 46, PR = W - 14, PT = 20, PB = H - 40;
    function sy(v) { return PB - ((v - YMIN) / (YMAX - YMIN)) * (PB - PT); }
    function el(name, attrs) {
      var e = document.createElementNS(svgNS, name);
      for (var k in attrs) e.setAttribute(k, attrs[k]);
      return e;
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      svg.appendChild(el("rect", { x: PL, y: PT, width: PR - PL, height: PB - PT,
        fill: "var(--bg)", stroke: "var(--border)", "stroke-width": 1 }));
      [0.72, 0.76, 0.80, 0.84].forEach(function (v) {
        svg.appendChild(el("line", { x1: PL, y1: sy(v), x2: PR, y2: sy(v),
          stroke: "var(--border)", "stroke-width": 0.5, "stroke-dasharray": "2,3" }));
        svg.appendChild(el("text", { x: PL - 6, y: sy(v) + 3, "text-anchor": "end", class: "esv-tick" }))
          .textContent = v.toFixed(2);
      });

      // dashed reference line at the single-best score for comparison
      svg.appendChild(el("line", { x1: PL, y1: sy(SINGLE), x2: PR, y2: sy(SINGLE),
        stroke: "#b03a2e", "stroke-width": 1, "stroke-dasharray": "4,3" }));

      var bx = PL + 60, bw = 150;
      if (mode === "single") {
        svg.appendChild(el("rect", { x: bx, y: sy(SINGLE), width: bw, height: PB - sy(SINGLE),
          fill: "#8e44ad", opacity: 0.9 }));
        svg.appendChild(el("text", { x: bx + bw / 2, y: sy(SINGLE) - 6, "text-anchor": "middle", class: "esv-val" }))
          .textContent = SINGLE.toFixed(3);
        svg.appendChild(el("text", { x: bx + bw / 2, y: PB + 16, "text-anchor": "middle", class: "esv-lab" }))
          .textContent = "single best (HistGB)";
      } else {
        // stacked composition bar: segment heights proportional to greedy pick weight
        var total = COMPOSITION.reduce(function (s, c) { return s + c.picks; }, 0);
        var top = sy(ENS), full = PB - top, yoff = top;
        COMPOSITION.forEach(function (c) {
          var h = full * (c.picks / total);
          svg.appendChild(el("rect", { x: bx, y: yoff, width: bw, height: h, fill: c.color, opacity: 0.9,
            stroke: "var(--bg)", "stroke-width": 1 }));
          if (h > 14) {
            svg.appendChild(el("text", { x: bx + bw / 2, y: yoff + h / 2 + 3, "text-anchor": "middle", class: "esv-seg" }))
              .textContent = c.algo + " ×" + c.picks;
          }
          yoff += h;
        });
        svg.appendChild(el("text", { x: bx + bw / 2, y: top - 6, "text-anchor": "middle", class: "esv-val" }))
          .textContent = ENS.toFixed(3);
        svg.appendChild(el("text", { x: bx + bw / 2, y: PB + 16, "text-anchor": "middle", class: "esv-lab" }))
          .textContent = "greedy ensemble (blend)";
      }
      // annotate the single-best reference
      svg.appendChild(el("text", { x: PR - 2, y: sy(SINGLE) - 4, "text-anchor": "end", class: "esv-ref" }))
        .textContent = "single best 0.824";
    }

    function setMode(m) {
      mode = m;
      singleBtn.classList.toggle("esv-on", m === "single");
      ensBtn.classList.toggle("esv-on", m === "ensemble");
      if (m === "single") {
        readout.innerHTML = "The <strong>single best-validation</strong> config (a HistGB) scores " +
          "<strong>" + SINGLE.toFixed(3) + "</strong> on the held-out test set. Keeping only this one " +
          "throws away every other model the search already paid to train.";
      } else {
        readout.innerHTML = "Greedy ensemble selection (<strong>Caruana 2004</strong>): start empty, keep " +
          "adding the pool model that most improves <em>validation</em> AUC (with replacement). The blend " +
          "of <strong>3 algorithms</strong> (LogReg ×5, HistGB ×4, ExtraTrees ×1) scores <strong>" +
          ENS.toFixed(3) + "</strong> — <strong>+" + (ENS - SINGLE).toFixed(3) + "</strong> over the single " +
          "best. The gain is <em>diversity</em>, and it is free (models already trained).";
      }
      draw();
    }

    singleBtn.addEventListener("click", function () { setMode("single"); });
    ensBtn.addEventListener("click", function () { setMode("ensemble"); });

    setMode("single");
    return { setMode: setMode };
  }

  global.EnsembleSelectViz = { mount: mount };
})(window);
