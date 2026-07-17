/**
 * ResNet block anatomy — the Gorishniy 2021 tabular ResNet block (§3.2).
 *
 * Shows the residual block as a vertical stack of operations with the SKIP path drawn as a
 * curved arrow that bypasses the block and lands on a "+" (addition) node:
 *
 *   ResNetBlock(x) = x + Dropout(Linear(Dropout(ReLU(Linear(BatchNorm(x))))))
 *
 * Toggle "skip connection" ON/OFF:
 *   - ON  (residual): the branch computes a small correction f(x); the output is x + f(x).
 *           The identity map is free (f can be ~0), so stacking many blocks cannot hurt.
 *   - OFF (plain MLP block): the output is just f(x); every added block must re-learn to
 *           preserve information, so deep plain stacks DEGRADE (train accuracy falls — an
 *           optimization problem, not overfitting).
 *
 * Pure diagram (no data); the depth consequence is quantified in depth-trainability-viz.js.
 *
 * Usage: ResNetBlockViz.mount(container, { caption })
 * Expected states:
 *   - default: skip ON — curved skip arrow + "+" node drawn; readout says "output = x + f(x)".
 *   - toggle OFF: skip arrow + "+" hidden, output routes straight from the branch; readout
 *                 says "output = f(x)" and warns about degradation with depth.
 */
(function (global) {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var BRANCH = "#2e6fb0", SKIP = "#1e6b3c", MUTE = "#8a94a6";

  // the operations inside the residual branch f(x), top to bottom
  var OPS = [
    { t: "BatchNorm", d: "normalize activations" },
    { t: "Linear", d: "d_main → d_hidden" },
    { t: "ReLU", d: "nonlinearity" },
    { t: "Dropout", d: "regularize" },
    { t: "Linear", d: "d_hidden → d_main" },
    { t: "Dropout", d: "regularize" }
  ];

  function el(name, attrs, text) {
    var e = document.createElementNS(SVGNS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "resnet-block-viz";

    var useSkip = true;

    var ctl = document.createElement("div");
    ctl.className = "rnb-ctl";
    var btn = document.createElement("button");
    btn.textContent = "Skip connection: ON";
    ctl.appendChild(btn);
    container.appendChild(ctl);

    var W = 420, H = 430;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", class: "rnb-svg" });
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "rnb-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "rnb-caption";
    cap.textContent = config.caption ||
      "One residual block. The branch f(x) is the same six operations either way; the skip " +
      "connection is the single line that turns a plain MLP block into a ResNet block.";
    container.appendChild(cap);

    var cx = 150;                 // branch column x-centre
    var boxW = 150, boxH = 40, gap = 14;
    var topY = 54;

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      // input node
      svg.appendChild(el("circle", { cx: cx, cy: 26, r: 12, fill: "var(--bg)", stroke: MUTE, "stroke-width": 1.5 }));
      svg.appendChild(el("text", { x: cx, y: 30, "text-anchor": "middle", class: "rnb-io" }, "x"));

      // branch boxes
      var y = topY;
      var centers = [];
      OPS.forEach(function (op, i) {
        // connector from previous
        var prevY = i === 0 ? 38 : y - gap;
        svg.appendChild(el("line", { x1: cx, y1: prevY, x2: cx, y2: y, stroke: BRANCH, "stroke-width": 1.6 }));
        svg.appendChild(el("rect", { x: cx - boxW / 2, y: y, width: boxW, height: boxH, rx: 5,
          fill: "var(--bg)", stroke: BRANCH, "stroke-width": 1.4 }));
        svg.appendChild(el("text", { x: cx, y: y + 17, "text-anchor": "middle", class: "rnb-op" }, op.t));
        svg.appendChild(el("text", { x: cx, y: y + 31, "text-anchor": "middle", class: "rnb-opd" }, op.d));
        centers.push(y + boxH / 2);
        y += boxH + gap;
      });
      var branchBottom = y - gap;   // bottom of last box

      // f(x) label on the branch
      svg.appendChild(el("text", { x: cx + boxW / 2 + 8, y: topY + 3 * (boxH + gap),
        "text-anchor": "start", class: "rnb-fx" }, "f(x)"));

      var plusY = branchBottom + 40;

      if (useSkip) {
        // "+" node
        svg.appendChild(el("line", { x1: cx, y1: branchBottom, x2: cx, y2: plusY - 13, stroke: BRANCH, "stroke-width": 1.6 }));
        svg.appendChild(el("circle", { cx: cx, cy: plusY, r: 13, fill: "var(--bg)", stroke: SKIP, "stroke-width": 2 }));
        svg.appendChild(el("text", { x: cx, y: plusY + 5, "text-anchor": "middle", class: "rnb-plus" }, "+"));
        // skip path: from input, out to the right, down, into the "+"
        var rx = cx + boxW / 2 + 55;
        svg.appendChild(el("path", {
          d: "M " + cx + " 26 H " + rx + " V " + plusY + " H " + (cx + 13),
          fill: "none", stroke: SKIP, "stroke-width": 2.2
        }));
        svg.appendChild(el("polygon", {
          points: (cx + 13) + "," + plusY + " " + (cx + 22) + "," + (plusY - 4) + " " + (cx + 22) + "," + (plusY + 4),
          fill: SKIP
        }));
        svg.appendChild(el("text", { x: rx + 6, y: (26 + plusY) / 2, "text-anchor": "start", class: "rnb-skip" }, "skip (identity)"));
        // output
        svg.appendChild(el("line", { x1: cx, y1: plusY + 13, x2: cx, y2: plusY + 40, stroke: SKIP, "stroke-width": 1.6 }));
        svg.appendChild(el("text", { x: cx, y: plusY + 58, "text-anchor": "middle", class: "rnb-io" }, "x + f(x)"));
      } else {
        // plain: branch output straight to output, no "+"
        svg.appendChild(el("line", { x1: cx, y1: branchBottom, x2: cx, y2: plusY + 40, stroke: BRANCH, "stroke-width": 1.6, "stroke-dasharray": "1,0" }));
        svg.appendChild(el("text", { x: cx, y: plusY + 58, "text-anchor": "middle", class: "rnb-io" }, "f(x)"));
        svg.appendChild(el("text", { x: cx + boxW / 2 + 20, y: plusY, "text-anchor": "start", class: "rnb-noskip" }, "no skip"));
      }

      readout.innerHTML = useSkip
        ? "<strong style='color:" + SKIP + "'>Residual block (skip ON).</strong> The branch learns a " +
          "<em>correction</em> f(x); the output is <code>x + f(x)</code>. Because f can settle near " +
          "<strong>zero</strong>, the block can act as the <strong>identity</strong> for free — so " +
          "stacking more blocks <strong>cannot</strong> make the network worse. Adding depth is safe."
        : "<strong style='color:" + BRANCH + "'>Plain MLP block (skip OFF).</strong> The output is just " +
          "<code>f(x)</code>. To preserve information, every added block must <em>re-learn</em> an " +
          "approximate identity — which is hard. Deep plain stacks <strong>degrade</strong>: their " +
          "<strong>training</strong> accuracy falls with depth (an optimization problem, not overfitting).";
    }

    btn.addEventListener("click", function () {
      useSkip = !useSkip;
      btn.textContent = "Skip connection: " + (useSkip ? "ON" : "OFF");
      btn.classList.toggle("rnb-off", !useSkip);
      draw();
    });

    draw();
    return { setSkip: function (v) { useSkip = v; btn.textContent = "Skip connection: " + (v ? "ON" : "OFF"); btn.classList.toggle("rnb-off", !v); draw(); } };
  }

  global.ResNetBlockViz = { mount: mount };
})(window);
