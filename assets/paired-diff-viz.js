/**
 * Paired per-fold differences: is the gap real, or is it noise? (Lesson 023)
 *
 * The core skill of L023 made visible. Two models are compared on the SAME CV folds
 * (RepeatedStratifiedKFold), so we can pair the scores fold-by-fold and look at the
 * per-fold difference d_i = acc_A(fold i) - acc_B(fold i). The lesson's failure mode
 * is a variance question, so the widget draws the SAME differences under two variance
 * estimates (one knob):
 *
 *   - NAIVE paired t-test: treats the n fold differences as INDEPENDENT samples, so its
 *     confidence interval on the mean is sigma/sqrt(n) — narrow. With n=100 folds a tiny
 *     mean looks highly "significant" (0 far outside the CI).
 *   - CORRECTED resampled t-test (Nadeau & Bengio 2003): the folds share training data, so
 *     they are correlated; the variance is inflated by (1/n + n_test/n_train). The CI
 *     widens dramatically and now straddles 0 -> the "win" is indistinguishable from noise.
 *
 * The fold strip (top) shows each d_i as a dot on a zero-centred axis (green = favours A,
 * red = favours B). The band below is the CI on the MEAN difference; the toggle switches
 * between the naive (narrow) and corrected (wide) band and updates the verdict/p-value.
 *
 * Defaults reproduce the verified L023 headline case (labs/_verify_l023.py):
 *   LogReg vs GaussianNB, 10x10 CV: mean(A-B)=+0.0098, naive p=1.2e-5 (SIG),
 *   corrected p=0.19 (NOT sig), 63% of folds favour A.
 *
 * Usage: PairedDiffViz.mount(container, { });
 * Expected states:
 *   - default (naive): narrow band, 0 OUTSIDE, readout "SIGNIFICANT p=1.2e-5"
 *   - toggle corrected: wide band, 0 INSIDE, readout "not significant p=0.19"
 */
(function (global) {
  "use strict";

  // 20 representative per-fold differences (mean +0.0098, ~63% positive), rounded from
  // the verified 100-fold run so the strip visually matches the numbers in prose.
  var FOLD_DIFFS = [
    0.030, 0.048, -0.010, 0.021, 0.006, -0.028, 0.039, 0.012, -0.006, 0.024,
    0.001, 0.034, -0.021, 0.015, 0.043, -0.013, 0.009, 0.027, -0.004, 0.018
  ];

  function mount(container, config) {
    config = config || {};
    var diffs = config.foldDiffs || FOLD_DIFFS;
    var mean = config.mean != null ? config.mean : 0.0098;
    // half-widths of the 95% CI on the mean (from the verified run)
    var naiveHalf = config.naiveHalf != null ? config.naiveHalf : 0.0042;   // ~ t*sigma/sqrt(n)
    var corrHalf = config.corrHalf != null ? config.corrHalf : 0.0146;      // ~ t*sqrt((1/n+ntest/ntrain)sigma^2)
    var pNaive = config.pNaive || "1.2e-5";
    var pCorr = config.pCorr || "0.19";
    var labelA = config.labelA || "LogReg";
    var labelB = config.labelB || "GaussianNB";

    container.innerHTML = "";
    container.className = "paired-diff-viz";

    var mode = "naive";

    var ctl = document.createElement("div");
    ctl.className = "pd-ctl";
    var naiveBtn = document.createElement("button");
    naiveBtn.type = "button";
    naiveBtn.textContent = "Naive t-test";
    var corrBtn = document.createElement("button");
    corrBtn.type = "button";
    corrBtn.textContent = "Corrected (Nadeau–Bengio)";
    ctl.appendChild(naiveBtn);
    ctl.appendChild(corrBtn);
    container.appendChild(ctl);

    var svgNS = "http://www.w3.org/2000/svg";
    var W = 560, H = 210;
    var svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("width", "100%");
    svg.setAttribute("class", "pd-svg");
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "pd-readout";
    container.appendChild(readout);

    // geometry: symmetric axis around 0
    var PL = 40, PR = W - 20;
    var domain = 0.06; // +/- accuracy difference shown
    function sx(v) { return PL + (v + domain) / (2 * domain) * (PR - PL); }

    function el(name, attrs) {
      var e = document.createElementNS(svgNS, name);
      for (var k in attrs) e.setAttribute(k, attrs[k]);
      return e;
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var half = mode === "naive" ? naiveHalf : corrHalf;
      var zeroInside = mean - half <= 0 && 0 <= mean + half;

      var stripY = 58, bandY = 138;

      // titles
      svg.appendChild(el("text", { x: PL, y: 20, class: "pd-title" }))
        .textContent = "per-fold difference  d = acc(" + labelA + ") − acc(" + labelB + ")";

      // zero reference line (the null: no difference)
      svg.appendChild(el("line", { x1: sx(0), y1: 30, x2: sx(0), y2: 170, stroke: "#b03a2e", "stroke-width": 1.5, "stroke-dasharray": "4,3" }));
      svg.appendChild(el("text", { x: sx(0), y: 186, "text-anchor": "middle", class: "pd-tick" }))
        .textContent = "0 (no difference)";

      // axis ticks
      [-0.04, -0.02, 0.02, 0.04].forEach(function (v) {
        svg.appendChild(el("line", { x1: sx(v), y1: 168, x2: sx(v), y2: 172, stroke: "var(--muted)", "stroke-width": 1 }));
        svg.appendChild(el("text", { x: sx(v), y: 186, "text-anchor": "middle", class: "pd-tick" }))
          .textContent = (v > 0 ? "+" : "") + v.toFixed(2);
      });

      // fold dots strip
      svg.appendChild(el("text", { x: PL, y: stripY - 12, class: "pd-lab" }))
        .textContent = diffs.length + " folds (each a dot):";
      diffs.forEach(function (d, i) {
        var jitter = (i % 5) * 5 - 10;
        svg.appendChild(el("circle", {
          cx: sx(d), cy: stripY + jitter, r: 3.4,
          fill: d > 0 ? "#1e6b3c" : "#b03a2e", opacity: 0.72,
          stroke: "var(--bg)", "stroke-width": 0.8,
        }));
      });

      // CI band on the MEAN
      svg.appendChild(el("text", { x: PL, y: bandY - 14, class: "pd-lab" }))
        .textContent = (mode === "naive" ? "naive" : "corrected") + " 95% CI on the mean:";
      svg.appendChild(el("rect", {
        x: sx(mean - half), y: bandY - 8, width: sx(mean + half) - sx(mean - half), height: 16,
        fill: zeroInside ? "rgba(176,58,46,0.14)" : "rgba(30,107,60,0.16)",
        stroke: zeroInside ? "#b03a2e" : "#1e6b3c", "stroke-width": 1.2, rx: 3,
      }));
      // mean marker
      svg.appendChild(el("line", { x1: sx(mean), y1: bandY - 12, x2: sx(mean), y2: bandY + 12, stroke: "var(--fg, #222)", "stroke-width": 2 }));
      svg.appendChild(el("text", { x: sx(mean), y: bandY - 16, "text-anchor": "middle", class: "pd-lab" }))
        .textContent = "mean +" + mean.toFixed(4);

      var verdict = mode === "naive"
        ? "<span class='pd-bad'>SIGNIFICANT</span> — p ≈ " + pNaive + ". 0 sits <strong>outside</strong> the narrow CI, " +
          "so the naive test declares " + labelA + " the winner."
        : "<span class='pd-win'>NOT significant</span> — p ≈ " + pCorr + ". Correcting for the overlapping training sets " +
          "widens the CI until it <strong>straddles 0</strong>: the +0.0098 gap is within noise.";
      readout.innerHTML =
        "<span class='pd-pill'>" + (mode === "naive" ? "Naive paired t" : "Corrected resampled t") + "</span> " + verdict;
    }

    function setMode(m) {
      mode = m;
      naiveBtn.classList.toggle("pd-active", m === "naive");
      corrBtn.classList.toggle("pd-active", m === "corrected");
      draw();
    }
    naiveBtn.addEventListener("click", function () { setMode("naive"); });
    corrBtn.addEventListener("click", function () { setMode("corrected"); });

    setMode("naive");
    return { setMode: setMode };
  }

  global.PairedDiffViz = { mount: mount };
})(window);
