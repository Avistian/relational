/**
 * TabNet encoder architecture (Arik & Pfister 2019, arXiv:1908.07442, Fig. 4a) as a stage-stepper.
 *
 * ONE mechanism — how a row flows through the encoder — with stage buttons that highlight the boxes and
 * arrows for each stage and print the paper equation that box realises. Default ("whole encoder") shows
 * the full figure undimmed. The point of the widget is to give the reader a MAP before the equations:
 * every symbol in the lesson's formulas (f, a[i], M[i], P[i], d[i], d_out, M_agg) is a labelled edge or
 * box here.
 *
 * The figure it reproduces (one decision step drawn, with the two feedback edges that make it a loop):
 *
 *   f -> BN -> feature transformer (step 0, unmasked) -> split -> a[0] ------------------------┐
 *                                                                                             v
 *   step i:      a[i-1], P[i-1] -> attentive transformer -> M[i] -> prior scale P[i] --(back)--┤
 *                                          |                                                   |
 *                       f -> [ mask M[i].f ] -> feature transformer -> split -> a[i] --(back)--┘
 *                                    |                                    \-> d[i] -> d_out -> y_hat
 *                                    \-> M_agg (eta-weighted masks, the global attribution)
 *
 * Geometry is verified against the from-scratch implementation in `labs/relkit/tabnet.py`
 * (`TabNetEncoder.forward`), which is the code the lesson's numbers come from — including the detail
 * the paper's figure states and prose skips: step 0 runs an UNMASKED feature transformer whose `a` half
 * seeds the first attentive transformer, and whose `d` half is discarded.
 *
 * Expected states:
 *   - stage "all"       (default) nothing dimmed; readout gives the one-line summary
 *   - stage "seed"      highlights f, BN, the step-0 feature transformer and its split (where a[0] is born)
 *   - stage "attend"    highlights the attentive transformer + both sources of a[i-1]; prints the mask eq
 *   - stage "mask"      highlights BN (the f bus) + the mask block; prints M[i] . f
 *   - stage "transform" highlights the mask, feature transformer and split; prints [d[i], a[i]] = f_i(...)
 *   - stage "prior"     highlights the attentive transformer + prior scale (the sequential feedback edge)
 *   - stage "outputs"   highlights split -> d_out -> y_hat AND mask -> M_agg (prediction + attribution)
 *
 * Usage: TabnetArchViz.mount(container, { caption })
 * api: { getStage(), setStage(k), stages() }
 */
(function (global) {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var W = 680, H = 418;

  // Palette: what a box DOES, not where it sits.
  var PLUMB = "#6b7280";   // input plumbing
  var PROC  = "#2e6fb0";   // processing (feature transformer, split)
  var ATTN  = "#1e6b3c";   // selection / attention
  var MEM   = "#b9770e";   // the prior scale — the memory that makes it sequential
  var OUT   = "#7a5195";   // aggregation / attribution
  var PRED  = "#b03a2e";   // the prediction

  var BOXES = [
    { key: "feat",   x: 12,  y: 26,  w: 120, h: 42, fill: PLUMB,
      lines: ["features  f", "B × D"] },
    { key: "bn",     x: 158, y: 26,  w: 44,  h: 42, fill: PLUMB,
      lines: ["BN"] },
    { key: "ft0",    x: 228, y: 26,  w: 140, h: 42, fill: PROC,
      lines: ["feature transformer", "step 0 · no mask yet"] },
    { key: "split0", x: 394, y: 26,  w: 62,  h: 42, fill: PROC,
      lines: ["split"] },

    { key: "att",    x: 76,  y: 110, w: 210, h: 54, fill: ATTN,
      lines: ["attentive transformer", "hᵢ(a[i−1]) → × P[i−1]", "→ sparsemax  ⇒  M[i]"] },
    { key: "prior",  x: 440, y: 110, w: 210, h: 54, fill: MEM,
      lines: ["prior scale", "P[i] = P[i−1] · (γ − M[i])", "what has been spent"] },
    { key: "mask",   x: 76,  y: 196, w: 118, h: 44, fill: ATTN,
      lines: ["mask", "M[i] ⊙ f"] },
    { key: "ft",     x: 232, y: 192, w: 160, h: 52, fill: PROC,
      lines: ["feature transformer", "2 shared + 2 step-dep", "FC → BN → GLU, √0.5"] },
    { key: "split",  x: 430, y: 192, w: 86,  h: 52, fill: PROC,
      lines: ["split", "d[i] | a[i]"] },

    { key: "magg",   x: 40,  y: 300, w: 306, h: 58, fill: OUT,
      lines: ["M_agg — global attribution", "Σᵢ η[i] · M[i], normalised", "η[i] = Σ ReLU(d[i])"] },
    { key: "agg",    x: 390, y: 300, w: 200, h: 46, fill: OUT,
      lines: ["d_out = Σᵢ ReLU(d[i])", "one term per step"] },
    { key: "final",  x: 390, y: 372, w: 200, h: 32, fill: PRED,
      lines: ["ŷ = W_final · d_out"] }
  ];

  // pts: polyline waypoints. from/to: the box groups whose activity lights the edge.
  // dash: the unmasked feature bus. head:false for a line that merges into another edge.
  var ARROWS = [
    { from: "feat",   to: "bn",     pts: [[132, 47], [154, 47]] },
    { from: "bn",     to: "ft0",    pts: [[202, 47], [224, 47]] },
    { from: "ft0",    to: "split0", pts: [[368, 47], [390, 47]] },

    // the same (batch-normalised) f enters every step's mask — it is never re-encoded
    { from: "bn",     to: "mask",   dash: true, label: "f", labelAt: [46, 150, "end"],
      pts: [[180, 68], [180, 74], [56, 74], [56, 218], [72, 218]] },

    // a[i-1]: from the step-0 split at the first step, from the previous step's split after that
    { from: "split0", to: "att",    head: false, label: "a[0]", labelAt: [431, 80, "start"],
      pts: [[425, 68], [425, 86]] },
    { from: "split",  to: "att",    label: "a[i]", labelAt: [656, 100, "end"],
      pts: [[516, 218], [662, 218], [662, 86], [140, 86], [140, 106]] },

    { from: "att",    to: "prior",  label: "M[i]", labelAt: [362, 131, "middle"],
      pts: [[286, 137], [436, 137]] },
    { from: "prior",  to: "att",    label: "P[i−1]", labelAt: [258, 118, "start"],
      pts: [[545, 110], [545, 100], [250, 100], [250, 106]] },

    { from: "att",    to: "mask",   label: "M[i]", labelAt: [141, 182, "start"],
      pts: [[135, 164], [135, 192]] },
    { from: "mask",   to: "ft",     pts: [[194, 218], [228, 218]] },
    { from: "ft",     to: "split",  pts: [[392, 218], [426, 218]] },

    { from: "split",  to: "agg",    label: "d[i]", labelAt: [479, 268, "start"],
      pts: [[473, 244], [473, 296]] },
    { from: "mask",   to: "magg",   label: "M[i], every step", labelAt: [106, 268, "start"],
      pts: [[100, 240], [100, 296]] },
    { from: "agg",    to: "final",  pts: [[490, 346], [490, 368]] }
  ];

  var STAGES = {
    all:       { label: "Whole encoder",          groups: null },
    seed:      { label: "1 · Input & seed",       groups: ["feat", "bn", "ft0", "split0"] },
    attend:    { label: "2 · Attentive transformer", groups: ["att", "split0", "split"] },
    mask:      { label: "3 · Feature masking",    groups: ["bn", "att", "mask"] },
    transform: { label: "4 · Feature transformer", groups: ["mask", "ft", "split"] },
    prior:     { label: "5 · Prior scale (the loop)", groups: ["att", "prior"] },
    outputs:   { label: "6 · Two outputs",        groups: ["mask", "split", "agg", "final", "magg"] }
  };

  var READOUT = {
    all:
      "<strong>The whole encoder.</strong> The same features <span class='tna-m'>f</span> are handed to " +
      "every decision step. A step does three things: its <em>attentive transformer</em> decides which " +
      "features it may look at (the mask <span class='tna-m'>M[i]</span>), the <em>mask</em> block " +
      "switches the rest off, and a <em>feature transformer</em> processes what survives and splits the " +
      "result into a piece for the answer (<span class='tna-m'>d[i]</span>) and a piece for the next " +
      "step's attention (<span class='tna-m'>a[i]</span>). Two edges close the loop and make the " +
      "attention <em>sequential</em>: <span class='tna-m'>a[i]</span> and the prior scale " +
      "<span class='tna-m'>P[i]</span>. Walk the six stages in order.",
    seed:
      "<strong>1 — input, and where the first mask comes from.</strong> The raw features get ordinary " +
      "batch normalisation (<em>not</em> ghost BN — that is used only inside the blocks), and the same " +
      "<span class='tna-m'>f</span> is passed to every step: TabNet never re-encodes the row. But the " +
      "first attentive transformer needs an <span class='tna-m'>a[i−1]</span> to read, and at " +
      "<span class='tna-m'>i = 1</span> there is no previous step. So step 0 runs a feature transformer " +
      "over the <strong>unmasked</strong> features and keeps only the <span class='tna-m'>a</span> half " +
      "of its split as <span class='tna-m'>a[0]</span>; its <span class='tna-m'>d</span> half is thrown " +
      "away. This is the piece of Fig. 4a the equations never mention.",
    attend:
      "<strong>2 — the attentive transformer computes the mask.</strong> " +
      "<span class='tna-eq'>M[i] = sparsemax( P[i−1] · h<sub>i</sub>(a[i−1]) )</span> " +
      "<span class='tna-m'>h<sub>i</sub></span> is one fully-connected layer plus batch norm, mapping the " +
      "attention vector <span class='tna-m'>a[i−1]</span> (width <span class='tna-m'>N_a</span>) to one " +
      "logit <em>per input feature</em>. Multiply those logits by the prior, project with sparsemax, and " +
      "you have a mask over the <span class='tna-m'>D</span> columns that sums to 1 and contains exact " +
      "zeros. Note the input: the mask for this step is decided by what the <em>previous</em> step " +
      "computed — at step 1, by the seed <span class='tna-m'>a[0]</span>.",
    mask:
      "<strong>3 — masking is multiplication, and that is the whole inductive bias.</strong> " +
      "<span class='tna-eq'>M[i] ⊙ f</span> Elementwise, feature by feature. A coordinate of " +
      "<span class='tna-m'>M[i]</span> that sparsemax set to exactly 0 removes that column from this " +
      "step completely — it cannot influence the step's output, and it receives no gradient at this " +
      "step. That is the tree-like ability an MLP lacks: capacity is not spent on columns the step " +
      "decided to ignore. Note the dashed edge — the features arriving here are the same " +
      "<span class='tna-m'>f</span> every step sees; only the mask changes.",
    transform:
      "<strong>4 — process what survived, then split it in two.</strong> " +
      "<span class='tna-eq'>[ d[i], a[i] ] = f<sub>i</sub>( M[i] ⊙ f )</span> " +
      "The feature transformer is four GLU blocks — 2 whose weights are <em>shared</em> across all steps " +
      "(every step sees the same columns, so low-level processing should not be relearned) and 2 that are " +
      "<em>step-dependent</em> — wired with <span class='tna-m'>√0.5</span>-scaled residuals. Its output " +
      "is split: <span class='tna-m'>d[i]</span> (width <span class='tna-m'>N_d</span>) goes to the " +
      "answer, <span class='tna-m'>a[i]</span> (width <span class='tna-m'>N_a</span>) goes back up to the " +
      "next step's attention. That second half is the first of the two loop edges.",
    prior:
      "<strong>5 — the prior scale: the memory that makes the attention sequential.</strong> " +
      "<span class='tna-eq'>P[i] = ∏<sub>j≤i</sub> ( γ − M[j] ),   P[0] = 1</span> " +
      "This is the second loop edge, and the one that matters most. The mask this step chose is folded " +
      "into a running record of how much each feature has been used, and that record multiplies the " +
      "<em>next</em> step's logits — so a feature already spent is discounted and sparsemax is more " +
      "likely to zero it out. Without this edge the steps would be independent and would all grab the " +
      "same strong columns: sparse, but not sequential. At <span class='tna-m'>γ = 1</span> a feature " +
      "used in full has prior exactly 0 and is banned from every later step.",
    outputs:
      "<strong>6 — two things come out of this network.</strong> A <em>prediction</em>: " +
      "<span class='tna-eq'>d_out = Σ<sub>i</sub> ReLU( d[i] ),   ŷ = W_final · d_out</span> " +
      "one additive term per decision step — the tree-like aggregation of Fig. 3, which is why a step " +
      "with all-negative <span class='tna-m'>d[i]</span> contributes nothing. And an " +
      "<em>attribution</em>, for free, from the same forward pass: " +
      "<span class='tna-eq'>M_agg = Σ<sub>i</sub> η[i] · M[i]  (normalised),   η[i] = Σ<sub>c</sub> ReLU( d[i]<sub>c</sub> )</span> " +
      "the masks summed across steps, each weighted by how much that step actually contributed. This is " +
      "the number the lesson reads as \"how much did the model use this feature\" — and the claim the " +
      "Syn2/Syn4 sections put on trial."
  };

  function el(name, attrs, text) {
    var e = document.createElementNS(SVGNS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "tna-viz";
    var stage = "all";

    var ctl = document.createElement("div");
    ctl.className = "tna-ctl";
    var buttons = {};
    Object.keys(STAGES).forEach(function (key) {
      var b = document.createElement("button");
      b.textContent = STAGES[key].label;
      b.addEventListener("click", function () { setStage(key); });
      buttons[key] = b;
      ctl.appendChild(b);
    });
    container.appendChild(ctl);

    var scroll = document.createElement("div");
    scroll.className = "tna-scroll";
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, class: "tna-svg",
      role: "img", "aria-label": "TabNet encoder architecture, Arik and Pfister 2019 Figure 4a" });
    scroll.appendChild(svg);
    container.appendChild(scroll);

    var readout = document.createElement("div");
    readout.className = "tna-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "tna-caption";
    cap.textContent = config.caption ||
      "TabNet's encoder (paper Fig. 4a), one decision step drawn. The step repeats N_steps times; the two " +
      "edges returning to the attentive transformer — a[i] and the prior scale P[i] — are what make the " +
      "attention sequential rather than merely sparse. Step through the stages to attach each of the " +
      "lesson's equations to the block that computes it.";
    container.appendChild(cap);

    var byKey = {};
    BOXES.forEach(function (b) { byKey[b.key] = b; });

    function active(key) {
      var g = STAGES[stage].groups;
      return g === null || g.indexOf(key) !== -1;
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      Object.keys(buttons).forEach(function (k) {
        buttons[k].className = k === stage ? "tna-on" : "";
      });

      var defs = el("defs", {});
      [["tna-arrow", "var(--muted)"], ["tna-arrow-on", "#222"]].forEach(function (m) {
        var mk = el("marker", { id: m[0], markerWidth: 9, markerHeight: 9, refX: 6.5, refY: 4.5,
          orient: "auto", markerUnits: "userSpaceOnUse" });
        mk.appendChild(el("path", { d: "M0.5,1.5 L7,4.5 L0.5,7.5 Z", fill: m[1] }));
        defs.appendChild(mk);
      });
      svg.appendChild(defs);

      // the decision step repeats: draw the frame first, behind everything
      svg.appendChild(el("rect", { x: 64, y: 92, width: 592, height: 160, rx: 10,
        fill: "none", stroke: "var(--border)", "stroke-width": 1, "stroke-dasharray": "5,4" }));
      svg.appendChild(el("text", { x: 656, y: 86, "text-anchor": "end", class: "tna-lane" },
        "ONE DECISION STEP i — REPEATS N_steps TIMES (PAPER: 3–10)"));
      svg.appendChild(el("text", { x: 12, y: 18, class: "tna-lane" },
        "INPUT, AND THE SEED FOR THE FIRST MASK"));
      svg.appendChild(el("text", { x: 12, y: 292, class: "tna-lane" }, "OUTPUTS"));

      ARROWS.forEach(function (a) {
        var on = active(a.from) && active(a.to);
        var d = a.pts.map(function (p, i) { return (i ? "L" : "M") + p[0] + "," + p[1]; }).join(" ");
        var attrs = {
          d: d, fill: "none", stroke: on ? "#222" : "var(--border)",
          "stroke-width": on ? 1.8 : 1.1, opacity: on ? 1 : 0.55
        };
        if (a.head !== false) attrs["marker-end"] = on ? "url(#tna-arrow-on)" : "url(#tna-arrow)";
        if (a.dash) attrs["stroke-dasharray"] = "4,3";
        svg.appendChild(el("path", attrs));
        if (a.head === false) {
          // a merge, not an endpoint: mark the junction so the edge does not read as a dead end
          var last = a.pts[a.pts.length - 1];
          svg.appendChild(el("circle", { cx: last[0], cy: last[1], r: 2.6,
            fill: on ? "#222" : "var(--border)" }));
        }
        if (a.label) {
          svg.appendChild(el("text", { x: a.labelAt[0], y: a.labelAt[1],
            "text-anchor": a.labelAt[2], class: "tna-edge" + (on ? " tna-edge-on" : "") }, a.label));
        }
      });

      BOXES.forEach(function (b) {
        var on = active(b.key);
        var g = el("g", { opacity: on ? 1 : 0.3 });
        g.appendChild(el("rect", { x: b.x, y: b.y, width: b.w, height: b.h, rx: 7,
          fill: on ? b.fill : "var(--bg-soft)", stroke: on ? b.fill : "var(--border)",
          "stroke-width": 1 }));
        var n = b.lines.length;
        b.lines.forEach(function (line, i) {
          var y = n === 1 ? b.y + b.h / 2 + 4 : b.y + (b.h - (n - 1) * 13) / 2 + 4 + i * 13;
          g.appendChild(el("text", { x: b.x + b.w / 2, y: y, "text-anchor": "middle",
            class: i === 0 ? "tna-head" : "tna-sub", fill: on ? "#fff" : "var(--muted)" }, line));
        });
        svg.appendChild(g);
      });

      readout.innerHTML = READOUT[stage];
    }

    function setStage(k) { if (STAGES[k]) { stage = k; draw(); } }

    draw();
    return {
      getStage: function () { return stage; },
      setStage: setStage,
      stages: function () { return Object.keys(STAGES); }
    };
  }

  global.TabnetArchViz = { mount: mount };
})(window);
