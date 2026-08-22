/**
 * TabNet's feature transformer, opened up (Arik & Pfister 2019, Fig. 4c) as a stage-stepper.
 *
 * The block that does TabNet's actual computing, and the one the lesson can only describe in prose:
 * four FC -> BN -> GLU layers, the first two SHARED across every decision step (and with the unmasked
 * step-0 transformer), the last two step-dependent, wired with sqrt(0.5)-scaled residuals, and split at
 * the end into the decision half d[i] and the attention half a[i].
 *
 *   M[i].f --[FC BN GLU  shared 1]--[FC BN GLU  shared 2]--(+)--[step-dep 1]--(+)--[step-dep 2]--(+)--split
 *                                        \___________________/     \_________/       \_________/
 *                                          residual, x sqrt(0.5)
 *
 * Verified against `labs/relkit/tabnet.py` (`FeatureTransformer.forward`): the FIRST block has no
 * residual because it is the one that changes width (D -> N_d + N_a); every later block is
 * `x = (x + block(x)) * sqrt(0.5)`.
 *
 * Expected states:
 *   - stage "all"      (default) nothing dimmed
 *   - stage "shared"   highlights the two shared layers (+ the input); readout explains WHY they are shared
 *   - stage "stepdep"  highlights the two step-dependent layers
 *   - stage "glu"      highlights the inset that opens one FC -> BN -> GLU layer (the doubled FC + gate)
 *   - stage "residual" highlights the three merge nodes and their bypass arcs, and only those
 *   - stage "split"    highlights the split into d[i] (N_d) and a[i] (N_a)
 *
 * Usage: TabnetFblockViz.mount(container, { caption })
 * api: { getStage(), setStage(k), stages() }
 */
(function (global) {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var W = 680, H = 256;

  var SHARED = "#2e6fb0";   // shared across steps
  var STEPD  = "#7a5195";   // step-dependent
  var NODE   = "#b9770e";   // the sqrt(0.5) merge
  var SPLIT  = "#1e6b3c";
  var PLUMB  = "#6b7280";

  var BOXES = [
    { key: "in",    x: 4,   y: 48, w: 76, h: 38, fill: PLUMB, grp: "in",
      lines: ["M[i] ⊙ f", "D wide"] },
    { key: "b1",    x: 94,  y: 44, w: 78, h: 46, fill: SHARED, grp: "b1",
      lines: ["FC → BN → GLU", "shared 1"] },
    { key: "b2",    x: 186, y: 44, w: 78, h: 46, fill: SHARED, grp: "b2",
      lines: ["FC → BN → GLU", "shared 2"] },
    { key: "b3",    x: 316, y: 44, w: 78, h: 46, fill: STEPD, grp: "b3",
      lines: ["FC → BN → GLU", "step-dep 1"] },
    { key: "b4",    x: 446, y: 44, w: 78, h: 46, fill: STEPD, grp: "b4",
      lines: ["FC → BN → GLU", "step-dep 2"] },
    { key: "split", x: 576, y: 44, w: 96, h: 46, fill: SPLIT, grp: "split",
      lines: ["split", "d[i] → answer", "a[i] → next mask"] }
  ];

  // the sqrt(0.5) merge points, between blocks
  var NODES = [
    { key: "m2", cx: 290, cy: 67, grp: "m2" },
    { key: "m3", cx: 420, cy: 67, grp: "m3" },
    { key: "m4", cx: 550, cy: 67, grp: "m4" }
  ];

  var ARROWS = [
    { from: "in", to: "b1", pts: [[80, 67], [92, 67]] },
    { from: "b1", to: "b2", pts: [[172, 67], [184, 67]] },
    { from: "b2", to: "m2", pts: [[264, 67], [276, 67]] },
    { from: "m2", to: "b3", pts: [[302, 67], [314, 67]] },
    { from: "b3", to: "m3", pts: [[394, 67], [406, 67]] },
    { from: "m3", to: "b4", pts: [[432, 67], [444, 67]] },
    { from: "b4", to: "m4", pts: [[524, 67], [536, 67]] },
    { from: "m4", to: "split", pts: [[562, 67], [574, 67]] },
    // the bypass arcs: each block after the first is added back to its own input, then rescaled
    { grp: "m2", pts: [[133, 90], [133, 124], [290, 124], [290, 81]] },
    { grp: "m3", pts: [[296, 77], [296, 142], [420, 142], [420, 81]] },
    { grp: "m4", pts: [[426, 77], [426, 160], [550, 160], [550, 81]] }
  ];

  // one FC -> BN -> GLU layer, opened up
  var INSET = [
    { x: 16,  y: 208, w: 78, h: 34, lines: ["FC: u → 2u"] },
    { x: 110, y: 208, w: 78, h: 34, lines: ["ghost BN"] },
    { x: 204, y: 208, w: 78, h: 34, lines: ["cut in half", "→ a, b"] },
    { x: 298, y: 208, w: 78, h: 34, lines: ["a · σ(b)", "→ u values"] }
  ];

  var STAGES = {
    all:      { label: "Whole block",        groups: null },
    shared:   { label: "1 · Shared layers",  groups: ["in", "b1", "b2"] },
    stepdep:  { label: "2 · Step-dependent", groups: ["b3", "b4"] },
    glu:      { label: "3 · Inside one layer (GLU)", groups: ["glu"] },
    residual: { label: "4 · The √0.5 residual", groups: ["m2", "m3", "m4"] },
    split:    { label: "5 · Split: d[i] and a[i]", groups: ["m4", "split"] }
  };

  var READOUT = {
    all:
      "<strong>The whole block.</strong> Four layers of <span class='tnf-m'>FC → BN → GLU</span>: the " +
      "first two use weights <em>shared</em> by every decision step, the last two are the step's own. " +
      "Each layer after the first is added back to its own input and rescaled by " +
      "<span class='tnf-m'>√0.5</span>. The output is cut in two — one half votes on the answer, the " +
      "other half tells the next step where to look. Walk the five stages.",
    shared:
      "<strong>1 — two shared layers, and the reason for them.</strong> Every decision step is handed the " +
      "<em>same</em> <span class='tnf-m'>D</span> features; only the mask in front of them differs. So " +
      "relearning low-level processing from scratch at each step would be a waste of parameters. These " +
      "two layers hold one set of weights used at every step — and by the unmasked step-0 transformer " +
      "that produces <span class='tnf-m'>a[0]</span> as well. This first layer is also the one that " +
      "changes the width, <span class='tnf-m'>D → N_d + N_a</span>, which is why it is the one layer with " +
      "no residual: there is nothing of matching width to add. The paper's ablation prefers this partial " +
      "sharing over both extremes — fully shared and fully independent are each worse.",
    stepdep:
      "<strong>2 — two step-dependent layers.</strong> These weights belong to this step alone, which is " +
      "what lets step 3 do something different with <span class='tnf-m'>debt</span> and " +
      "<span class='tnf-m'>hours</span> than step 1 did with <span class='tnf-m'>age</span> and " +
      "<span class='tnf-m'>income</span>. Shared layers give parameter efficiency; step-dependent layers " +
      "give the steps their individuality. The paper's default is 2 and 2 — and note what this means for " +
      "the parameter count: adding a decision step costs only two layers, not four.",
    glu:
      "<strong>3 — inside one layer: the gated linear unit.</strong> The fully-connected layer emits " +
      "<em>twice</em> the values the layer needs. They are cut in half into " +
      "<span class='tnf-m'>a</span> and <span class='tnf-m'>b</span>, and the layer returns " +
      "<span class='tnf-eq'>a · σ(b)</span> — half the activations act as a learned, per-value gate on " +
      "the other half, with <span class='tnf-m'>σ(b) ∈ (0,1)</span> deciding how much of each " +
      "<span class='tnf-m'>a</span> passes. A ReLU can only zero a value; a gate can attenuate it " +
      "smoothly and can be closed by <em>other</em> features. This is TabNet's non-linearity throughout, " +
      "and the paper's ablation shows the choice matters — swapping GLU for ReLU costs up to 3.1 accuracy " +
      "points. The BN here is <em>ghost</em> BN: normalisation over virtual sub-batches of size " +
      "<span class='tnf-m'>B_V</span>, because TabNet trains with batches so large that normalising them " +
      "whole averages away useful noise.",
    residual:
      "<strong>4 — why the residuals are multiplied by √0.5.</strong> " +
      "<span class='tnf-eq'>x ← ( x + block(x) ) · √0.5</span> " +
      "Adding two signals of similar variance gives you roughly <em>twice</em> the variance; do that at " +
      "every layer and activations grow geometrically as the block deepens. Multiplying by " +
      "<span class='tnf-m'>√0.5</span> divides the variance back by two, so it comes out where it went " +
      "in (Gehring et al. 2017). Same motive as the skip connections of L042 — keep a deep stack " +
      "trainable — but with the rescale made explicit instead of left to normalisation layers.",
    split:
      "<strong>5 — the split is what makes a step a step.</strong> " +
      "<span class='tnf-eq'>[ d[i], a[i] ] = f<sub>i</sub>( M[i] ⊙ f )</span> " +
      "The output vector is cut into <span class='tnf-m'>d[i]</span> (width " +
      "<span class='tnf-m'>N_d</span>), which goes through ReLU into the running sum that becomes the " +
      "prediction, and <span class='tnf-m'>a[i]</span> (width <span class='tnf-m'>N_a</span>), which is " +
      "handed to the <em>next</em> step's attentive transformer to choose the next mask. One tensor, two " +
      "jobs: answer and hand-off. The paper ties the two widths together in its search space " +
      "(<span class='tnf-m'>N_d = N_a</span>), and this split is the only channel through which one step " +
      "tells the next anything about the row."
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
    container.className = "tnf-viz";
    var stage = "all";

    var ctl = document.createElement("div");
    ctl.className = "tnf-ctl";
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
    scroll.className = "tnf-scroll";
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, class: "tnf-svg",
      role: "img", "aria-label": "TabNet feature transformer block, Arik and Pfister 2019 Figure 4c" });
    scroll.appendChild(svg);
    container.appendChild(scroll);

    var readout = document.createElement("div");
    readout.className = "tnf-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "tnf-caption";
    cap.textContent = config.caption ||
      "The feature transformer (paper Fig. 4c) — the part of TabNet that does the computing once the mask " +
      "has decided what it may see. Four gated layers, two of them shared with every other decision step, " +
      "residuals rescaled by √0.5, and a final split into the half that answers and the half that hands " +
      "off to the next step.";
    container.appendChild(cap);

    function active(g) {
      var groups = STAGES[stage].groups;
      return groups === null || groups.indexOf(g) !== -1;
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      Object.keys(buttons).forEach(function (k) {
        buttons[k].className = k === stage ? "tnf-on" : "";
      });

      var defs = el("defs", {});
      [["tnf-arrow", "var(--muted)"], ["tnf-arrow-on", "#222"]].forEach(function (m) {
        var mk = el("marker", { id: m[0], markerWidth: 9, markerHeight: 9, refX: 6.5, refY: 4.5,
          orient: "auto", markerUnits: "userSpaceOnUse" });
        mk.appendChild(el("path", { d: "M0.5,1.5 L7,4.5 L0.5,7.5 Z", fill: m[1] }));
        defs.appendChild(mk);
      });
      svg.appendChild(defs);

      // group frames: shared vs step-dependent
      [[86, 192, "SHARED — SAME WEIGHTS EVERY STEP", ["b1", "b2"]],
       [308, 250, "STEP-DEPENDENT — THIS STEP ONLY", ["b3", "b4"]]].forEach(function (fr) {
        var on = fr[3].some(active);
        svg.appendChild(el("rect", { x: fr[0], y: 32, width: fr[1], height: 70, rx: 8,
          fill: "none", stroke: "var(--border)", "stroke-width": 1, "stroke-dasharray": "5,4",
          opacity: on ? 1 : 0.45 }));
        svg.appendChild(el("text", { x: fr[0], y: 24, class: "tnf-lane" + (on ? " tnf-lane-on" : "") },
          fr[2]));
      });

      ARROWS.forEach(function (a) {
        var on = a.grp ? active(a.grp) : (active(a.from) && active(a.to));
        var d = a.pts.map(function (p, i) { return (i ? "L" : "M") + p[0] + "," + p[1]; }).join(" ");
        svg.appendChild(el("path", {
          d: d, fill: "none", stroke: on ? "#222" : "var(--border)",
          "stroke-width": on ? 1.8 : 1.1, opacity: on ? 1 : 0.5,
          "stroke-dasharray": a.grp ? "4,3" : "none",
          "marker-end": on ? "url(#tnf-arrow-on)" : "url(#tnf-arrow)"
        }));
      });

      // the one width change in the block — the reason the first layer has no residual
      svg.appendChild(el("text", { x: 4, y: 12, class: "tnf-note-svg" },
        "widths: D in → N_d + N_a after layer 1 (the one width change, hence the one layer with no residual)"));

      BOXES.forEach(function (b) {
        var on = active(b.grp);
        var g = el("g", { opacity: on ? 1 : 0.3 });
        g.appendChild(el("rect", { x: b.x, y: b.y, width: b.w, height: b.h, rx: 7,
          fill: on ? b.fill : "var(--bg-soft)", stroke: on ? b.fill : "var(--border)",
          "stroke-width": 1 }));
        var n = b.lines.length;
        b.lines.forEach(function (line, i) {
          var y = n === 1 ? b.y + b.h / 2 + 4 : b.y + (b.h - (n - 1) * 13) / 2 + 4 + i * 13;
          g.appendChild(el("text", { x: b.x + b.w / 2, y: y, "text-anchor": "middle",
            class: i === 0 ? "tnf-head" : "tnf-sub", fill: on ? "#fff" : "var(--muted)" }, line));
        });
        svg.appendChild(g);
      });

      NODES.forEach(function (nd) {
        var on = active(nd.grp);
        var g = el("g", { opacity: on ? 1 : 0.35 });
        g.appendChild(el("circle", { cx: nd.cx, cy: nd.cy, r: 12,
          fill: on ? NODE : "var(--bg-soft)", stroke: on ? NODE : "var(--border)", "stroke-width": 1 }));
        g.appendChild(el("text", { x: nd.cx, y: nd.cy + 4, "text-anchor": "middle",
          class: "tnf-plus", fill: on ? "#fff" : "var(--muted)" }, "+"));
        g.appendChild(el("text", { x: nd.cx, y: nd.cy - 19, "text-anchor": "middle",
          class: "tnf-scale" + (on ? " tnf-scale-on" : "") }, "×√0.5"));
        svg.appendChild(g);
      });

      // the GLU inset
      var gon = active("glu");
      var gi = el("g", { opacity: gon ? 1 : 0.3 });
      gi.appendChild(el("text", { x: 16, y: 200, class: "tnf-lane" + (gon ? " tnf-lane-on" : "") },
        "INSIDE ONE LAYER — THE GATE"));
      INSET.forEach(function (b, idx) {
        gi.appendChild(el("rect", { x: b.x, y: b.y, width: b.w, height: b.h, rx: 6,
          fill: gon ? "#3d7ebd" : "var(--bg-soft)", stroke: gon ? "#3d7ebd" : "var(--border)",
          "stroke-width": 1 }));
        var n = b.lines.length;
        b.lines.forEach(function (line, i) {
          var y = n === 1 ? b.y + b.h / 2 + 4 : b.y + (b.h - (n - 1) * 12) / 2 + 4 + i * 12;
          gi.appendChild(el("text", { x: b.x + b.w / 2, y: y, "text-anchor": "middle",
            class: i === 0 ? "tnf-head" : "tnf-sub", fill: gon ? "#fff" : "var(--muted)" }, line));
        });
        if (idx < INSET.length - 1) {
          gi.appendChild(el("path", { d: "M" + (b.x + b.w) + ",225 L" + (b.x + b.w + 12) + ",225",
            fill: "none", stroke: gon ? "#222" : "var(--border)", "stroke-width": gon ? 1.8 : 1.1,
            "marker-end": gon ? "url(#tnf-arrow-on)" : "url(#tnf-arrow)" }));
        }
      });
      gi.appendChild(el("text", { x: 388, y: 222, class: "tnf-note-svg" },
        "half the values gate the other half —"));
      gi.appendChild(el("text", { x: 388, y: 234, class: "tnf-note-svg" },
        "a ReLU can only zero a value, a gate can dim it"));
      svg.appendChild(gi);

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

  global.TabnetFblockViz = { mount: mount };
})(window);
