/**
 * Uninformative-features bias — the ADD-junk experiment (Grinsztajn 2022 §5.3, Finding 2).
 *
 * Fix a small set of informative features; add k pure-noise (uninformative) columns and refit.
 * A tree/GBT does implicit feature selection (its greedy gain gate ignores junk), so it barely
 * moves; an MLP has no gate and (being rotationally invariant, Ng 2004) must spend samples learning
 * to ignore each junk direction — so it degrades fast. Read the curve LEFT→RIGHT as "adding junk"
 * and RIGHT→LEFT as "removing junk": the tree–MLP gap widens with junk and shrinks without it.
 *
 * The target here is deliberately SMOOTH, so at k=0 the MLP WINS (0.986 vs GBT 0.945) — this
 * isolates the junk-feature effect from the smoothness (L025) and rotation (L026) effects: junk
 * alone erodes the MLP's lead until the GBT overtakes it by k=100.
 *
 * REAL numbers from labs/_verify_l027.py (5 informative features, mean of 5 seeds, test accuracy).
 *
 * Usage: UninformativeAddViz.mount(container, { caption })
 * Expected states:
 *   - k=0:   MLP 0.986 vs GBT 0.945 (gap −0.040, MLP wins clean).
 *   - k=100: MLP 0.902 vs GBT 0.914 (gap +0.012, GBT overtakes) — MLP lost 0.084, GBT only 0.032.
 *   - slider marks k and reads out both accuracies + the gap.
 */
(function (global) {
  "use strict";

  var KS = [0, 5, 15, 30, 50, 100];
  var TREE = [0.8852, 0.8807, 0.8772, 0.8778, 0.8688, 0.8651];
  var GBT = [0.9452, 0.9393, 0.9316, 0.9236, 0.9209, 0.9136];
  var MLP = [0.9855, 0.9811, 0.9688, 0.9522, 0.9294, 0.9017];

  var C_TREE = "#8a6d3b", C_GBT = "#1e6b3c", C_MLP = "#2e6fb0";

  var svgNS = "http://www.w3.org/2000/svg";
  function el(name, attrs, text) {
    var e = document.createElementNS(svgNS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "uninformative-add-viz";

    var idx = 0;
    var N = KS.length;

    var ctl = document.createElement("div");
    ctl.className = "ua-ctl";
    var slabel = document.createElement("label");
    slabel.className = "ua-slabel";
    slabel.textContent = "uninformative features added, k = ";
    var slider = document.createElement("input");
    slider.type = "range"; slider.min = "0"; slider.max = String(N - 1); slider.step = "1"; slider.value = "0";
    slider.className = "ua-slider";
    var kval = document.createElement("span");
    kval.className = "ua-kval"; kval.textContent = "0";
    slabel.appendChild(slider); slabel.appendChild(kval);
    ctl.appendChild(slabel);
    container.appendChild(ctl);

    var W = 460, H = 300;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", class: "ua-svg" });
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "ua-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "ua-caption";
    cap.textContent = config.caption ||
      "Test accuracy as pure-noise columns are added to a fixed set of 5 informative features " +
      "(mean of 5 seeds). The MLP wins on clean data but degrades fast; the trees barely move.";
    container.appendChild(cap);

    var PL = 44, PR = W - 14, PT = 18, PB = H - 42;
    var YLO = 0.84, YHI = 1.0;
    function sx(i) { return PL + (i / (N - 1)) * (PR - PL); }
    function sy(v) { return PB - (v - YLO) / (YHI - YLO) * (PB - PT); }

    function line(arr, color) {
      var d = "M";
      for (var i = 0; i < N; i++) d += (i ? "L" : "") + sx(i).toFixed(1) + "," + sy(arr[i]).toFixed(1) + " ";
      return el("path", { d: d, fill: "none", stroke: color, "stroke-width": 2.3 });
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      svg.appendChild(el("rect", { x: PL, y: PT, width: PR - PL, height: PB - PT,
        fill: "var(--bg)", stroke: "var(--border)", "stroke-width": 1 }));

      [0.85, 0.9, 0.95, 1.0].forEach(function (v) {
        var yy = sy(v);
        svg.appendChild(el("line", { x1: PL, y1: yy, x2: PR, y2: yy,
          stroke: "var(--border)", "stroke-width": 0.5, opacity: 0.6 }));
        svg.appendChild(el("text", { x: PL - 5, y: yy + 3, "text-anchor": "end", class: "ua-tick" }, v.toFixed(2)));
      });

      KS.forEach(function (k, i) {
        var xx = sx(i);
        svg.appendChild(el("line", { x1: xx, y1: PB, x2: xx, y2: PB + 4, stroke: "var(--muted)", "stroke-width": 1 }));
        svg.appendChild(el("text", { x: xx, y: PB + 15, "text-anchor": "middle", class: "ua-tick" }, String(k)));
      });
      svg.appendChild(el("text", { x: (PL + PR) / 2, y: H - 4, "text-anchor": "middle", class: "ua-lab" },
        "number of uninformative features added (→ add junk · ← remove junk)"));
      svg.appendChild(el("text", { x: 12, y: (PT + PB) / 2, "text-anchor": "middle", class: "ua-lab",
        transform: "rotate(-90 12 " + ((PT + PB) / 2) + ")" }, "test accuracy"));

      var mx = sx(idx);
      svg.appendChild(el("line", { x1: mx, y1: PT, x2: mx, y2: PB,
        stroke: "var(--muted)", "stroke-width": 1, "stroke-dasharray": "3,3", opacity: 0.8 }));

      svg.appendChild(line(TREE, C_TREE));
      svg.appendChild(line(GBT, C_GBT));
      svg.appendChild(line(MLP, C_MLP));

      [[TREE, C_TREE], [GBT, C_GBT], [MLP, C_MLP]].forEach(function (pair) {
        svg.appendChild(el("circle", { cx: mx, cy: sy(pair[0][idx]), r: 4.5, fill: pair[1],
          stroke: "var(--bg)", "stroke-width": 1.5 }));
      });

      var lx = PL + 8;
      [["Tree", C_TREE], ["GBT", C_GBT], ["MLP", C_MLP]].forEach(function (pair, j) {
        svg.appendChild(el("rect", { x: lx, y: PT + 6 + j * 16, width: 10, height: 10, fill: pair[1] }));
        svg.appendChild(el("text", { x: lx + 14, y: PT + 15 + j * 16, class: "ua-leg" }, pair[1] === C_MLP ? "MLP (no gate)" : (pair[1] === C_GBT ? "GBT" : "Tree")));
      });

      var gap = GBT[idx] - MLP[idx];
      var sign = gap >= 0 ? "+" : "";
      readout.innerHTML =
        "<span class='ua-pill'>k = " + KS[idx] + " junk</span> " +
        "MLP <strong class='ua-mlp'>" + MLP[idx].toFixed(3) + "</strong> · " +
        "GBT <strong class='ua-gbt'>" + GBT[idx].toFixed(3) + "</strong> · " +
        "Tree <strong class='ua-tree'>" + TREE[idx].toFixed(3) + "</strong> · " +
        "GBT−MLP gap <strong>" + sign + gap.toFixed(3) + "</strong>" +
        (idx === 0
          ? " — clean data: the MLP wins (this target is smooth, so it is not the smoothness bias)."
          : (gap >= 0
            ? " — junk alone has handed the lead to the GBT; the MLP was not robust to it."
            : " — the MLP still leads, but its margin is shrinking as junk is added."));
    }

    slider.addEventListener("input", function () {
      idx = Number(slider.value);
      kval.textContent = String(KS[idx]);
      draw();
    });

    draw();
    return { setIdx: function (i) { idx = i; slider.value = String(i); kval.textContent = String(KS[i]); draw(); } };
  }

  global.UninformativeAddViz = { mount: mount };
})(window);
