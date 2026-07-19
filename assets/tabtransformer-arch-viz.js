/**
 * TabTransformer architecture data-flow (Huang et al. 2020, Fig. 1), as a stage-stepper.
 *
 * ONE mechanism — how a row flows through the network — with 4 stage buttons that highlight the
 * boxes/arrows for each stage and update the readout. Default ("whole net") shows the full figure.
 *
 * The figure it reproduces:
 *   categorical x_cat  -> column embedding  -> [ N x Transformer layers (self-attention + FFN) ]
 *                                            -> CONTEXTUAL embeddings -> flatten -\
 *   continuous  x_cont -> LayerNorm ----------------------------------------------> concat -> MLP -> y_hat
 *
 * Expected states:
 *   - stage "embed"   highlights the categorical inputs + column-embedding boxes
 *   - stage "trans"   highlights the Transformer stack + the "contextual" label
 *   - stage "bypass"  highlights the continuous inputs + LayerNorm (they SKIP the Transformer)
 *   - stage "head"    highlights concat + MLP + prediction
 *   - stage "all"     (default) no dimming; caption states the one-line summary
 *
 * Usage: TabTransformerArchViz.mount(container, {})
 */
(function (global) {
  "use strict";

  var svgNS = "http://www.w3.org/2000/svg";
  function el(name, attrs) {
    var e = document.createElementNS(svgNS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }

  // group key -> which stage(s) it belongs to (for highlight/dim)
  var STAGES = {
    all:    { label: "Whole net", groups: null },
    embed:  { label: "1 · Embed categoricals", groups: ["cat", "emb"] },
    trans:  { label: "2 · Transformer → contextual", groups: ["emb", "trans", "ctx"] },
    bypass: { label: "3 · Continuous bypass", groups: ["num", "ln"] },
    head:   { label: "4 · Concat + MLP head", groups: ["ctx", "ln", "concat", "mlp", "out"] }
  };

  var READOUT = {
    all: "<strong>The whole net.</strong> Categorical columns are embedded and passed through a stack of " +
      "Transformer self-attention layers that make them <em>contextual</em>; continuous columns skip the " +
      "Transformer (just normalised); the two are concatenated and an MLP predicts. The only new machinery " +
      "over L031 is the Transformer block in the middle.",
    embed: "<strong>Stage 1 — embed the categoricals.</strong> Each categorical column becomes a learned " +
      "dense vector — exactly the <em>entity embedding</em> of L031. With <em>m</em> categorical columns you " +
      "get <em>m</em> vectors of width <em>d</em>. So far this is nothing new.",
    trans: "<strong>Stage 2 — the Transformer makes them contextual.</strong> The <em>m</em> embeddings are " +
      "fed through <em>N</em> self-attention layers. Each column's output vector is now a blend of all the " +
      "columns in <em>this row</em> — a <strong>contextual embedding</strong>. This is the one idea the " +
      "lesson is about.",
    bypass: "<strong>Stage 3 — continuous features bypass the Transformer.</strong> Numeric columns are only " +
      "layer-normalised and set aside. Attention runs over categoricals only (a design choice of the 2020 " +
      "paper) — so numeric features never become contextual here.",
    head: "<strong>Stage 4 — concatenate and predict.</strong> The contextual embeddings are flattened " +
      "(<em>m·d</em> numbers), concatenated with the normalised continuous features, and an ordinary MLP " +
      "maps that to the prediction. The head is the same MLP you already know."
  };

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "tta-viz";
    var stage = "all";

    var ctl = document.createElement("div");
    ctl.className = "tta-ctl";
    Object.keys(STAGES).forEach(function (key) {
      var b = document.createElement("button");
      b.textContent = STAGES[key].label;
      if (key === stage) b.className = "tta-on";
      b.addEventListener("click", function () {
        stage = key;
        Array.prototype.forEach.call(ctl.children, function (c) { c.className = ""; });
        b.className = "tta-on";
        draw();
      });
      ctl.appendChild(b);
    });
    container.appendChild(ctl);

    var W = 640, H = 360;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", class: "tta-svg" });
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "tta-readout";
    container.appendChild(readout);

    // box registry: {group, x, y, w, h, lines:[...], fill}
    var GREEN = "#1e6b3c", BLUE = "#2e6fb0", GREY = "#6b7280";
    function boxes() {
      return [
        // categorical path (top)
        { group: "cat", x: 12,  y: 24,  w: 120, h: 58, fill: BLUE,
          lines: ["categorical x_cat", "purpose, checking,", "housing … (m cols)"] },
        { group: "emb", x: 168, y: 24,  w: 108, h: 58, fill: BLUE,
          lines: ["column embedding", "m vectors, dim d", "(entity embeddings)"] },
        { group: "trans", x: 312, y: 12, w: 132, h: 82, fill: GREEN,
          lines: ["N × Transformer", "self-attention + FFN", "(the new machinery)"] },
        { group: "ctx", x: 480, y: 24, w: 120, h: 58, fill: GREEN,
          lines: ["contextual", "embeddings", "flatten → m·d"] },
        // continuous path (bottom)
        { group: "num", x: 12,  y: 240, w: 120, h: 58, fill: GREY,
          lines: ["continuous x_cont", "age, duration,", "amount … (numeric)"] },
        { group: "ln", x: 240, y: 240, w: 120, h: 58, fill: GREY,
          lines: ["LayerNorm", "(bypasses the", "Transformer)"] },
        // merge
        { group: "concat", x: 452, y: 158, w: 92, h: 52, fill: "#7a5195",
          lines: ["concatenate", "context ⊕ numeric"] },
        { group: "mlp", x: 452, y: 240, w: 92, h: 58, fill: "#7a5195",
          lines: ["MLP head", "Linear→ReLU→…"] },
        { group: "out", x: 452, y: 316, w: 92, h: 30, fill: "#b03a2e",
          lines: ["prediction ŷ"] }
      ];
    }
    // arrows: [fromGroup, toGroup] drawn between box edges
    var ARROWS = [
      ["cat", "emb"], ["emb", "trans"], ["trans", "ctx"],
      ["ctx", "concat"], ["num", "ln"], ["ln", "concat"],
      ["concat", "mlp"], ["mlp", "out"]
    ];

    function active(group) {
      var g = STAGES[stage].groups;
      return g === null || g.indexOf(group) !== -1;
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var defs = el("defs", {});
      var mk = el("marker", { id: "tta-arrow", markerWidth: 8, markerHeight: 8, refX: 6, refY: 3,
        orient: "auto", markerUnits: "strokeWidth" });
      mk.appendChild(el("path", { d: "M0,0 L6,3 L0,6 Z", fill: "var(--muted)" }));
      defs.appendChild(mk);
      svg.appendChild(defs);

      var B = boxes(), byGroup = {};
      B.forEach(function (b) { byGroup[b.group] = b; });

      // arrows first (under boxes)
      ARROWS.forEach(function (a) {
        var f = byGroup[a[0]], t = byGroup[a[1]];
        var x1 = f.x + f.w, y1 = f.y + f.h / 2, x2 = t.x, y2 = t.y + t.h / 2;
        // route: if roughly same row draw straight, else elbow
        var on = active(a[0]) && active(a[1]);
        var col = on ? "var(--fg, #222)" : "var(--border)";
        var path;
        if (Math.abs(y1 - y2) < 8) {
          path = "M" + x1 + "," + y1 + " L" + (x2 - 2) + "," + y2;
        } else {
          var midx = (x1 + x2) / 2;
          path = "M" + x1 + "," + y1 + " L" + midx + "," + y1 + " L" + midx + "," + y2 + " L" + (x2 - 2) + "," + y2;
        }
        svg.appendChild(el("path", { d: path, fill: "none", stroke: col,
          "stroke-width": on ? 2 : 1.2, "marker-end": "url(#tta-arrow)", opacity: on ? 1 : 0.5 }));
      });

      B.forEach(function (b) {
        var on = active(b.group);
        var g = el("g", { opacity: on ? 1 : 0.28 });
        g.appendChild(el("rect", { x: b.x, y: b.y, width: b.w, height: b.h, rx: 7,
          fill: on ? b.fill : "var(--bg-soft)", stroke: on ? b.fill : "var(--border)",
          "stroke-width": on ? 0 : 1 }));
        b.lines.forEach(function (ln, i) {
          var t = el("text", { x: b.x + b.w / 2, y: b.y + (b.lines.length === 1 ? b.h / 2 + 4 : 17 + i * 15),
            "text-anchor": "middle", class: i === 0 ? "tta-boxhead" : "tta-boxsub",
            fill: on ? "#fff" : "var(--muted)" });
          t.textContent = ln;
          g.appendChild(t);
        });
        svg.appendChild(g);
      });

      // path labels
      var l1 = el("text", { x: 12, y: 16, class: "tta-lane" }); l1.textContent = "CATEGORICAL PATH →"; svg.appendChild(l1);
      var l2 = el("text", { x: 12, y: 232, class: "tta-lane" }); l2.textContent = "CONTINUOUS PATH →"; svg.appendChild(l2);

      readout.innerHTML = READOUT[stage];
    }

    draw();
    return { set: function (s) { stage = s; draw(); } };
  }

  global.TabTransformerArchViz = { mount: mount };
})(window);
