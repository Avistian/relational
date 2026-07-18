/**
 * What a LEARNED entity embedding captures: a 2-D map where similar categories sit close together.
 *
 * Two views of the SAME mechanism (a learned embedding table, projected to 2-D):
 *   - "credit_g purpose" — REAL learned embedding from labs/_verify_l031.py (entity-embedding MLP,
 *     seed 0, 5-D `purpose` table PCA'd to 2-D). Points coloured by risk P(good|purpose), sized by count.
 *     Honest caption: on 1000 rows the geometry is loose — safe purposes (used car, radio/tv, retraining)
 *     drift together, but the signal is weak (the model ties one-hot: 0.774 vs 0.798, next section).
 *   - "Rossmann states (Guo & Berkhahn 2016)" — ILLUSTRATIVE coordinates reproducing the paper's headline
 *     finding: the learned 2-D projection of the German-state embedding recovers the geographic map. Shown
 *     to make concrete what embeddings capture *at scale* — the bridge Q4 is building toward.
 *
 * Usage: EmbeddingSpaceViz.mount(container, { caption })
 */
(function (global) {
  "use strict";

  // REAL credit_g purpose embedding (verify script, seed 0)
  var PURPOSE = [
    { name: "business", x: 0.184, y: -1.206, risk: 0.649, n: 97 },
    { name: "dom. appliance", x: -0.095, y: 0.327, risk: 0.667, n: 12 },
    { name: "education", x: 2.829, y: 2.624, risk: 0.560, n: 50 },
    { name: "furniture/eq.", x: -1.297, y: -0.787, risk: 0.680, n: 181 },
    { name: "new car", x: 2.013, y: -1.929, risk: 0.620, n: 234 },
    { name: "other", x: -1.119, y: 0.762, risk: 0.583, n: 12 },
    { name: "radio/tv", x: 0.093, y: -0.325, risk: 0.779, n: 280 },
    { name: "repairs", x: -2.978, y: 1.126, risk: 0.636, n: 22 },
    { name: "retraining", x: 0.307, y: -0.155, risk: 0.889, n: 9 },
    { name: "used car", x: 0.063, y: -0.437, risk: 0.835, n: 103 }
  ];

  // ILLUSTRATIVE German states in geographic layout (x = west→east, y = south→north), reproducing
  // Guo & Berkhahn (2016) Fig. — the learned state embedding recovered this map.
  var STATES = [
    { name: "Schleswig-H.", x: 0.45, y: 0.98 },
    { name: "Hamburg", x: 0.48, y: 0.88 },
    { name: "Meckl.-Vorp.", x: 0.70, y: 0.92 },
    { name: "Niedersachsen", x: 0.36, y: 0.76 },
    { name: "Berlin", x: 0.76, y: 0.80 },
    { name: "NRW", x: 0.20, y: 0.60 },
    { name: "Sachsen", x: 0.74, y: 0.56 },
    { name: "Hessen", x: 0.35, y: 0.48 },
    { name: "Rheinl.-Pfalz", x: 0.22, y: 0.40 },
    { name: "Bayern", x: 0.64, y: 0.22 },
    { name: "Baden-Württ.", x: 0.38, y: 0.20 }
  ];

  var svgNS = "http://www.w3.org/2000/svg";
  function el(name, attrs) {
    var e = document.createElementNS(svgNS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }
  function riskColor(v) {
    var t = Math.max(0, Math.min(1, (v - 0.55) / 0.35));
    var r = Math.round(176 + (30 - 176) * t);
    var g = Math.round(58 + (107 - 58) * t);
    var b = Math.round(46 + (60 - 46) * t);
    return "rgb(" + r + "," + g + "," + b + ")";
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "esp-viz";
    var view = "purpose";

    var ctl = document.createElement("div");
    ctl.className = "esp-ctl";
    [["purpose", "credit_g purpose (real)"], ["states", "Rossmann states (illustrative)"]].forEach(function (m) {
      var b = document.createElement("button");
      b.textContent = m[1];
      if (m[0] === view) b.className = "esp-on";
      b.addEventListener("click", function () {
        view = m[0];
        Array.prototype.forEach.call(ctl.children, function (c) { c.className = ""; });
        b.className = "esp-on";
        draw();
      });
      ctl.appendChild(b);
    });
    container.appendChild(ctl);

    var W = 460, H = 340;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", class: "esp-svg" });
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "esp-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "esp-caption";
    container.appendChild(cap);

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var PL = 30, PR = W - 14, PT = 16, PB = H - 26;
      svg.appendChild(el("rect", { x: PL, y: PT, width: PR - PL, height: PB - PT,
        fill: "var(--bg)", stroke: "var(--border)", "stroke-width": 1 }));

      var data = view === "purpose" ? PURPOSE : STATES;
      var xs = data.map(function (d) { return d.x; });
      var ys = data.map(function (d) { return d.y; });
      var xlo = Math.min.apply(null, xs), xhi = Math.max.apply(null, xs);
      var ylo = Math.min.apply(null, ys), yhi = Math.max.apply(null, ys);
      var padx = (xhi - xlo) * 0.15 || 1, pady = (yhi - ylo) * 0.15 || 1;
      xlo -= padx; xhi += padx; ylo -= pady; yhi += pady;
      function sx(x) { return PL + (x - xlo) / (xhi - xlo) * (PR - PL); }
      function sy(y) { return PB - (y - ylo) / (yhi - ylo) * (PB - PT); }

      // axes hint
      svg.appendChild(el("text", { x: (PL + PR) / 2, y: H - 8, "text-anchor": "middle", class: "esp-ax" }))
        .textContent = "learned embedding dimension 1";
      var yl = el("text", { x: 11, y: (PT + PB) / 2, "text-anchor": "middle", class: "esp-ax",
        transform: "rotate(-90 11 " + ((PT + PB) / 2) + ")" });
      yl.textContent = "dimension 2";
      svg.appendChild(yl);

      data.forEach(function (d) {
        var cx = sx(d.x), cy = sy(d.y);
        var r = view === "purpose" ? Math.max(4, Math.sqrt(d.n) * 0.9) : 7;
        var fill = view === "purpose" ? riskColor(d.risk) : "#2e6fb0";
        svg.appendChild(el("circle", { cx: cx, cy: cy, r: r, fill: fill, opacity: 0.82,
          stroke: "var(--bg)", "stroke-width": 1.5 }));
        var t = el("text", { x: cx + r + 2, y: cy + 3, class: "esp-pt" });
        t.textContent = d.name;
        svg.appendChild(t);
      });

      if (view === "purpose") {
        // legend for risk color
        svg.appendChild(el("text", { x: PR - 4, y: PT + 12, "text-anchor": "end", class: "esp-leg" }))
          .textContent = "colour = P(good), size = count";
        cap.textContent = "Real learned embedding of credit_g `purpose` (5-D, PCA→2-D). Colour = risk " +
          "P(good), size = #rows.";
        readout.innerHTML = "<strong>Honest read:</strong> the safe purposes (<em>used car, radio/tv, " +
          "retraining</em>, greener) drift toward one region and the riskier ones spread out — but on only " +
          "1000 rows the geometry is loose and noisy. The embedding <em>ties</em> one-hot here (next " +
          "section); its structure, not its accuracy, is the point.";
      } else {
        svg.appendChild(el("text", { x: PR - 4, y: PT + 12, "text-anchor": "end", class: "esp-leg" }))
          .textContent = "reproduces Guo & Berkhahn (2016)";
        cap.textContent = "Illustrative: the learned German-state embedding, projected to 2-D, recovers the " +
          "geographic map — a result nobody put in by hand.";
        readout.innerHTML = "<strong>Why it matters:</strong> trained only to predict store sales, the " +
          "embedding placed neighbouring states next to each other — it <em>discovered</em> that nearby " +
          "regions behave alike. That is a learned representation capturing real structure, and it grows " +
          "richer with scale — the bridge from tables toward relational learning.";
      }
    }

    draw();
    return { set: function (v) { view = v; draw(); } };
  }

  global.EmbeddingSpaceViz = { mount: mount };
})(window);
