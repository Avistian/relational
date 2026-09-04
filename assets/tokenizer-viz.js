/**
 * FT-Transformer Feature Tokenizer data-flow (Gorishniy et al. 2021, Fig. 2), as a stage-stepper.
 *
 * ONE mechanism — how a row becomes tokens and flows to the [CLS] readout — with stage buttons that
 * highlight the relevant boxes/arrows and update the readout. The headline idea the widget makes visible:
 * EVERY feature becomes a token, NUMERICS INCLUDED, so numbers finally attend (TabTransformer did not).
 *
 * The figure it reproduces:
 *   numeric   x_j  -> token  T_j = b_j + x_j·W_j  \
 *   category  x_j  -> token  T_j = b_j + e_j[x_j]  >-- [CLS] ++ k tokens -> N× Transformer -> [CLS] -> head -> y
 *                                                  /
 *
 * Expected states:
 *   - stage "tok"     highlights the feature row + tokenizer (numeric AND categorical -> tokens)
 *   - stage "cls"     highlights the prepended [CLS] token
 *   - stage "attend"  highlights the Transformer stack (CLS attends to every token, numerics too)
 *   - stage "read"    highlights the final [CLS] -> head -> prediction
 *   - stage "all"     (default) no dimming; caption states the one-line summary
 *
 * Pure functions exposed for headless tests (labs/_viz_check_l046.js):
 *   TokenizerViz.numericToken(x, W, b)  -> element-wise x*W + b  (the affine numeric token)
 *   TokenizerViz.tokens()               -> ordered token descriptors ([CLS] first, then k features)
 *   TokenizerViz.features                -> the demo row
 *
 * Usage: TokenizerViz.mount(container, {})
 */
(function (global) {
  "use strict";

  var svgNS = "http://www.w3.org/2000/svg";
  function el(name, attrs) {
    var e = document.createElementNS(svgNS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }

  // A small mixed demo row (adult-like): numerics AND categoricals, so the "numerics are tokens too"
  // point is visible. kind drives the token formula shown.
  var FEATURES = [
    { name: "age", kind: "num", value: "39" },
    { name: "hours/wk", kind: "num", value: "40" },
    { name: "education", kind: "cat", value: "Bachelors" },
    { name: "occupation", kind: "cat", value: "Sales" },
    { name: "marital", kind: "cat", value: "Married" }
  ];

  // The affine numeric token: T_j = b_j + x_j * W_j, element-wise over the d embedding dims.
  function numericToken(x, W, b) {
    var out = [];
    for (var i = 0; i < W.length; i++) out.push(x * W[i] + b[i]);
    return out;
  }

  // Ordered token descriptors: [CLS] is always position 0, then one token per feature.
  function tokens() {
    var t = [{ name: "[CLS]", kind: "cls", value: "learned" }];
    FEATURES.forEach(function (f) { t.push({ name: f.name, kind: f.kind, value: f.value }); });
    return t;
  }

  function numFrac() {
    var n = FEATURES.filter(function (f) { return f.kind === "num"; }).length;
    return n / FEATURES.length;
  }

  var STAGES = {
    all:    { label: "Whole net", groups: null },
    tok:    { label: "1 · Tokenize every feature", groups: ["row", "tokenizer", "tok"] },
    cls:    { label: "2 · Prepend [CLS]", groups: ["cls"] },
    attend: { label: "3 · Transformer (all attend)", groups: ["cls", "tok", "trans"] },
    read:   { label: "4 · [CLS] → head → ŷ", groups: ["trans", "clsout", "head", "out"] }
  };

  var READOUT = {
    all: "<strong>The whole net.</strong> Each feature — <em>numeric and categorical alike</em> — becomes a " +
      "<em>d</em>-dim <strong>token</strong>. A learned <strong>[CLS]</strong> token is prepended, the " +
      "<em>k+1</em> tokens go through <em>N</em> Transformer layers, and the final [CLS] vector is read out " +
      "by the head. The one change from TabTransformer: <strong>numerics are tokenised too</strong>, so they attend.",
    tok: "<strong>Stage 1 — tokenize every feature.</strong> A <em>numeric</em> feature becomes " +
      "<code>T_j = b_j + x_j·W_j</code> (an affine map into <em>d</em> dims); a <em>categorical</em> feature " +
      "becomes <code>T_j = b_j + e_j[x_j]</code> (an embedding lookup + bias). <strong>Numerics are first-class " +
      "tokens here</strong> — the exact thing TabTransformer never did.",
    cls: "<strong>Stage 2 — prepend the [CLS] token.</strong> A single learned vector is added at position 0. " +
      "It carries no feature of its own; its job is to <em>collect</em> information from all the feature tokens " +
      "through attention, and become the row's summary (the BERT [CLS] trick).",
    attend: "<strong>Stage 3 — the Transformer lets everything attend.</strong> In each layer, every token " +
      "(including [CLS]) attends to every other token. Because numerics are tokens, a numeric feature now both " +
      "attends and is attended to — so a change in <code>age</code> can reshape the [CLS] summary. This is the " +
      "capability TabTransformer lacked.",
    read: "<strong>Stage 4 — read out [CLS] and predict.</strong> Only the <em>final</em> [CLS] vector is kept; " +
      "a small head (<code>LayerNorm → ReLU → Linear</code>) maps it to the prediction. The k feature tokens " +
      "have done their job — everything the head needs was pooled into [CLS]."
  };

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "tok-viz";
    var stage = "all";

    var ctl = document.createElement("div");
    ctl.className = "tok-ctl";
    Object.keys(STAGES).forEach(function (key) {
      var b = document.createElement("button");
      b.textContent = STAGES[key].label;
      if (key === stage) b.className = "tok-on";
      b.addEventListener("click", function () {
        stage = key;
        Array.prototype.forEach.call(ctl.children, function (c) { c.className = ""; });
        b.className = "tok-on";
        draw();
      });
      ctl.appendChild(b);
    });
    container.appendChild(ctl);

    var W = 660, H = 380;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", class: "tok-svg" });
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "tok-readout";
    container.appendChild(readout);

    var NUM = "#2e6fb0", CAT = "#1e6b3c", CLSC = "#b8860b", PURP = "#7a5195", RED = "#b03a2e";

    function active(group) {
      var g = STAGES[stage].groups;
      return g === null || g.indexOf(group) !== -1;
    }

    // token column geometry: 6 boxes ([CLS] + 5 features) laid out left→right
    var toks = tokens();
    var tW = 92, tH = 46, gap = 8, x0 = 14, tokY = 150;
    function tokX(i) { return x0 + i * (tW + gap); }

    function boxes() {
      var B = [];
      // the raw feature row (top) — one cell per FEATURE (no CLS)
      FEATURES.forEach(function (f, i) {
        B.push({ group: "row", x: tokX(i + 1), y: 20, w: tW, h: 40, fill: f.kind === "num" ? NUM : CAT,
          lines: [f.name, f.value + (f.kind === "num" ? "  (numeric)" : "  (categorical)")] });
      });
      // the tokenizer band label handled separately; token boxes (middle)
      toks.forEach(function (t, i) {
        var grp = t.kind === "cls" ? "cls" : "tok";
        var fill = t.kind === "cls" ? CLSC : (t.kind === "num" ? NUM : CAT);
        var formula = t.kind === "cls" ? "learned" :
          (t.kind === "num" ? "b + x·W" : "b + e[x]");
        B.push({ group: grp, x: tokX(i), y: tokY, w: tW, h: tH, fill: fill,
          lines: [t.name, formula], token: true, idx: i });
      });
      // Transformer stack (spans all tokens)
      B.push({ group: "trans", x: x0, y: 236, w: toks.length * (tW + gap) - gap, h: 44, fill: PURP,
        lines: ["N × Transformer  (every token attends to every token — numerics included)"] });
      // final CLS out + head + prediction
      B.push({ group: "clsout", x: x0, y: 300, w: tW, h: 40, fill: CLSC, lines: ["[CLS] out", "row summary"] });
      B.push({ group: "head", x: x0 + tW + 40, y: 300, w: 150, h: 40, fill: PURP,
        lines: ["head", "LN → ReLU → Linear"] });
      B.push({ group: "out", x: x0 + tW + 40 + 150 + 40, y: 305, w: 96, h: 30, fill: RED, lines: ["prediction ŷ"] });
      return B;
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var defs = el("defs", {});
      var mk = el("marker", { id: "tok-arrow", markerWidth: 8, markerHeight: 8, refX: 6, refY: 3,
        orient: "auto", markerUnits: "strokeWidth" });
      mk.appendChild(el("path", { d: "M0,0 L6,3 L0,6 Z", fill: "var(--muted)" }));
      defs.appendChild(mk);
      svg.appendChild(defs);

      var B = boxes();

      // vertical tokenizer arrows: feature row -> its token (only for the k features)
      FEATURES.forEach(function (f, i) {
        var on = active("row") || active("tok");
        var x = tokX(i + 1) + tW / 2;
        svg.appendChild(el("path", { d: "M" + x + ",60 L" + x + "," + (tokY - 2), fill: "none",
          stroke: on ? "var(--fg, #222)" : "var(--border)", "stroke-width": on ? 2 : 1.2,
          "marker-end": "url(#tok-arrow)", opacity: on ? 1 : 0.5 }));
      });
      // tokens -> transformer (down)
      toks.forEach(function (t, i) {
        var on = active("tok") || active("cls") || active("trans");
        var x = tokX(i) + tW / 2;
        svg.appendChild(el("path", { d: "M" + x + "," + (tokY + tH) + " L" + x + ",234", fill: "none",
          stroke: on ? "var(--fg, #222)" : "var(--border)", "stroke-width": on ? 1.6 : 1,
          "marker-end": "url(#tok-arrow)", opacity: on ? 1 : 0.4 }));
      });
      // transformer -> cls out -> head -> out (horizontal at bottom)
      var chain = [["trans", "clsout"], ["clsout", "head"], ["head", "out"]];
      var byGroup = {}; B.forEach(function (b) { if (!byGroup[b.group]) byGroup[b.group] = b; });
      chain.forEach(function (a) {
        var f = byGroup[a[0]], t = byGroup[a[1]];
        if (a[0] === "trans") {
          var xx = byGroup.clsout.x + byGroup.clsout.w / 2;
          var on0 = active("trans") || active("clsout");
          svg.appendChild(el("path", { d: "M" + xx + ",280 L" + xx + ",298", fill: "none",
            stroke: on0 ? "var(--fg, #222)" : "var(--border)", "stroke-width": on0 ? 1.6 : 1,
            "marker-end": "url(#tok-arrow)", opacity: on0 ? 1 : 0.4 }));
          return;
        }
        var on = active(a[0]) && active(a[1]);
        var x1 = f.x + f.w, y1 = f.y + f.h / 2, x2 = t.x, y2 = t.y + t.h / 2;
        svg.appendChild(el("path", { d: "M" + x1 + "," + y1 + " L" + (x2 - 2) + "," + y2, fill: "none",
          stroke: on ? "var(--fg, #222)" : "var(--border)", "stroke-width": on ? 2 : 1.2,
          "marker-end": "url(#tok-arrow)", opacity: on ? 1 : 0.5 }));
      });

      B.forEach(function (b) {
        var on = active(b.group);
        var g = el("g", { opacity: on ? 1 : 0.26 });
        g.appendChild(el("rect", { x: b.x, y: b.y, width: b.w, height: b.h, rx: 6,
          fill: on ? b.fill : "var(--bg-soft)", stroke: on ? b.fill : "var(--border)",
          "stroke-width": on ? 0 : 1, class: "tok-box" }));
        b.lines.forEach(function (ln, i) {
          var t = el("text", { x: b.x + b.w / 2, y: b.y + (b.lines.length === 1 ? b.h / 2 + 4 : 18 + i * 15),
            "text-anchor": "middle", class: i === 0 ? "tok-boxhead" : "tok-boxsub",
            fill: on ? "#fff" : "var(--muted)" });
          t.textContent = ln;
          g.appendChild(t);
        });
        svg.appendChild(g);
      });

      // lane labels
      var l1 = el("text", { x: x0, y: 14, class: "tok-lane" }); l1.textContent = "ROW (raw features)"; svg.appendChild(l1);
      var l2 = el("text", { x: x0, y: tokY - 6, class: "tok-lane" }); l2.textContent = "TOKENS  ([CLS] + one per feature)"; svg.appendChild(l2);

      readout.innerHTML = READOUT[stage];
    }

    draw();
    return { set: function (s) { stage = s; draw(); } };
  }

  global.TokenizerViz = {
    mount: mount,
    numericToken: numericToken,
    tokens: tokens,
    numFrac: numFrac,
    features: FEATURES
  };
})(window);
