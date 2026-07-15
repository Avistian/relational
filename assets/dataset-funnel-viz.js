/**
 * The Grinsztajn benchmark dataset-selection funnel (Grinsztajn et al. 2022, §3).
 *
 * A benchmark is only fair if its DATASETS are curated with explicit, defensible
 * rules — otherwise "trees beat deep learning" could just be a dataset-selection
 * artifact. Grinsztajn applies a sequence of inclusion criteria to a large pool of
 * OpenML datasets, narrowing it to a suite where the tree-vs-DL question is
 * meaningful (medium-sized, heterogeneous, real, non-trivial tabular data).
 *
 * This funnel visualizes the criteria as successive filters. Counts are ILLUSTRATIVE
 * (schematic of the shrinking pool) — the load-bearing content is the criterion at
 * each stage, not the exact number. Click a stage to read what it removes and why.
 *
 * Usage: DatasetFunnelViz.mount(container, { caption })
 * Expected states:
 *   - Renders one bar per stage, widths shrinking top→bottom.
 *   - The final stage is highlighted as the benchmark suite.
 *   - Clicking a stage shows its criterion + rationale in the detail panel.
 */
(function (global) {
  "use strict";

  var STAGES = [
    { label: "All OpenML tabular datasets", n: 100,
      crit: "Starting pool.",
      why: "A large, unfiltered set of candidate datasets — most are unsuitable for a fair tabular tree-vs-DL comparison, so criteria are applied in turn." },
    { label: "Real-world, tabular data", n: 74,
      crit: "Drop artificial / simulated data and non-tabular signals (images, audio).",
      why: "Synthetic data can bake in a smoothness or structure that flatters one model family; the question is about real tabular data." },
    { label: "Not too small", n: 55,
      crit: "Require enough samples (≈ ≥ 3,000+); study a fixed 'medium' regime by truncating training size.",
      why: "Tiny datasets are dominated by variance; fixing the sample regime makes the comparison about the model, not the data budget." },
    { label: "Not too high-dimensional", n: 43,
      crit: "Drop very wide data (≈ > 500 features).",
      why: "Extreme p/n ratios are a different problem (regularization-dominated) and would confound the tabular comparison." },
    { label: "Heterogeneous columns", n: 34,
      crit: "Each column must be a distinct, meaningful feature (not pixels / repeated sensor channels).",
      why: "Homogeneous columns behave like perceptual data, where CNNs/MLPs have a natural edge — not the tabular setting the thesis is about." },
    { label: "Not too easy", n: 27,
      crit: "Remove datasets where a default simple model is already within a hair of a tuned GBT.",
      why: "If everything ties, the dataset can't discriminate methods — it adds noise, not signal, to the aggregate." },
    { label: "Benchmark suite", n: 20,
      crit: "The curated datasets (numerical-only and heterogeneous variants; classification + regression).",
      why: "On THIS suite the random-search budget curves are computed and averaged — the fair playing field.", final: true }
  ];

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "dataset-funnel-viz";

    var svgNS = "http://www.w3.org/2000/svg";
    function el(name, attrs) {
      var e = document.createElementNS(svgNS, name);
      for (var k in attrs) e.setAttribute(k, attrs[k]);
      return e;
    }

    var W = 460, rowH = 34, gap = 8, PT = 8;
    var H = PT + STAGES.length * (rowH + gap);
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", class: "df-svg" });
    container.appendChild(svg);

    var detail = document.createElement("div");
    detail.className = "df-detail";
    container.appendChild(detail);

    var cap = document.createElement("p");
    cap.className = "df-caption";
    cap.textContent = config.caption ||
      "Grinsztajn's inclusion criteria as successive filters (counts illustrative). " +
      "Click a stage to read the rule and why it matters — the fairness of the benchmark lives here.";
    container.appendChild(cap);

    var maxN = STAGES[0].n, cx = W / 2, selected = STAGES.length - 1;

    function showDetail(i) {
      var s = STAGES[i];
      detail.innerHTML =
        "<strong>" + s.label + "</strong> — <em>" + s.crit + "</em><br>" +
        "<span class='df-why'>" + s.why + "</span>";
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      STAGES.forEach(function (s, i) {
        var w = 90 + (s.n / maxN) * (W - 120);
        var x = cx - w / 2, y = PT + i * (rowH + gap);
        var isSel = i === selected;
        var g = el("g", { class: "df-row", style: "cursor:pointer" });
        g.appendChild(el("rect", {
          x: x, y: y, width: w, height: rowH, rx: 5,
          fill: s.final ? "#1e6b3c" : (isSel ? "#2e6fb0" : "var(--bg-soft)"),
          stroke: isSel ? "#2e6fb0" : "var(--border)", "stroke-width": isSel ? 2 : 1,
          opacity: s.final ? 0.92 : 1
        }));
        var t = el("text", { x: cx, y: y + rowH / 2 + 4, "text-anchor": "middle",
          class: "df-txt", fill: (s.final || isSel) ? "#fff" : "var(--fg, #222)" });
        t.textContent = s.label;
        g.appendChild(t);
        g.addEventListener("click", function () { selected = i; showDetail(i); draw(); });
        svg.appendChild(g);
        // arrow between stages
        if (i < STAGES.length - 1) {
          var ay = y + rowH, ay2 = y + rowH + gap;
          svg.appendChild(el("line", { x1: cx, y1: ay, x2: cx, y2: ay2,
            stroke: "var(--muted)", "stroke-width": 1 }));
        }
      });
    }

    draw();
    showDetail(selected);

    return { select: function (i) { selected = i; showDetail(i); draw(); } };
  }

  global.DatasetFunnelViz = { mount: mount };
})(window);
