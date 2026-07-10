/**
 * Reproducibility-collapse visualizer (Kapoor & Narayanan 2022, civil-war study).
 *
 * Two side-by-side bars — a COMPLEX model (Random Forest) and LOGISTIC REGRESSION —
 * under a toggle: "leak on" vs "leak off". With an illegitimate leaked feature, the
 * complex model appears to crush LR; remove the leak and the apparent win collapses to
 * a tie. Verified numbers from labs/_verify_l022.py (sklearn 1.9.0, seed 0):
 *   leak on:  RF 0.935  LR 0.719  (gap +0.217)
 *   leak off: RF 0.712  LR 0.721  (gap -0.009)
 *
 * Plain <script> (file://-safe). Usage: ReproCollapseViz.mount(el, config?).
 *
 * Expected states (headless verification):
 *   - mounts an svg with 2 value bars + a toggle
 *   - default state = leak ON: RF bar taller than LR bar; readout shows +0.217
 *   - toggling to leak OFF: bars roughly equal; readout shows -0.009
 */
(function (global) {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var C_RF = "#8e44ad";  // complex model
  var C_LR = "#2e6fb0";  // logistic regression

  var DATA = {
    on:  { rf: 0.935, lr: 0.719 },
    off: { rf: 0.712, lr: 0.721 }
  };

  function el(tag, attrs) {
    var n = document.createElementNS(SVGNS, tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    return n;
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.classList.add("rcv");

    var W = 460, H = 240;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, class: "rcv-svg", width: "100%" });
    container.appendChild(svg);

    var bx = 70, by = 24, bw = 330, bh = 165; // plot box
    var maxV = 1.0;
    function yOf(v) { return by + bh - (v / maxV) * bh; }

    // y axis + gridlines at 0.5, 0.75, 1.0
    svg.appendChild(el("line", { x1: bx, y1: by, x2: bx, y2: by + bh, stroke: "#c9c9c9", "stroke-width": 1 }));
    svg.appendChild(el("line", { x1: bx, y1: by + bh, x2: bx + bw, y2: by + bh, stroke: "#c9c9c9", "stroke-width": 1 }));
    [0.5, 0.75, 1.0].forEach(function (g) {
      svg.appendChild(el("line", { x1: bx, y1: yOf(g), x2: bx + bw, y2: yOf(g), stroke: "#ececec", "stroke-width": 1 }));
      var tk = el("text", { x: bx - 8, y: yOf(g) + 4, class: "rcv-tick", "text-anchor": "end" });
      tk.textContent = g.toFixed(2);
      svg.appendChild(tk);
    });
    var yl = el("text", { x: bx - 40, y: by - 8, class: "rcv-axis" }); yl.textContent = "5-fold CV accuracy"; svg.appendChild(yl);

    // two bars
    var barW = 90;
    var xRF = bx + bw * 0.28 - barW / 2;
    var xLR = bx + bw * 0.72 - barW / 2;

    var barRF = el("rect", { x: xRF, y: yOf(DATA.on.rf), width: barW, height: (by + bh) - yOf(DATA.on.rf), fill: C_RF, "fill-opacity": 0.85 });
    var barLR = el("rect", { x: xLR, y: yOf(DATA.on.lr), width: barW, height: (by + bh) - yOf(DATA.on.lr), fill: C_LR, "fill-opacity": 0.85 });
    svg.appendChild(barRF);
    svg.appendChild(barLR);

    var valRF = el("text", { x: xRF + barW / 2, y: yOf(DATA.on.rf) - 6, class: "rcv-val", "text-anchor": "middle" });
    var valLR = el("text", { x: xLR + barW / 2, y: yOf(DATA.on.lr) - 6, class: "rcv-val", "text-anchor": "middle" });
    svg.appendChild(valRF);
    svg.appendChild(valLR);

    var labRF = el("text", { x: xRF + barW / 2, y: by + bh + 16, class: "rcv-lab", "text-anchor": "middle", fill: C_RF }); labRF.textContent = "Random Forest"; svg.appendChild(labRF);
    var labLR = el("text", { x: xLR + barW / 2, y: by + bh + 16, class: "rcv-lab", "text-anchor": "middle", fill: C_LR }); labLR.textContent = "Logistic Reg."; svg.appendChild(labLR);

    // controls
    var controls = document.createElement("div");
    controls.className = "rcv-controls";
    container.appendChild(controls);
    var btnOn = document.createElement("button");
    btnOn.type = "button"; btnOn.className = "rcv-btn active"; btnOn.textContent = "Leak ON (leaked feature in)";
    var btnOff = document.createElement("button");
    btnOff.type = "button"; btnOff.className = "rcv-btn"; btnOff.textContent = "Leak OFF (feature removed)";
    controls.appendChild(btnOn);
    controls.appendChild(btnOff);

    var readout = document.createElement("div");
    readout.className = "rcv-readout";
    container.appendChild(readout);

    function render(state) {
      var d = DATA[state];
      barRF.setAttribute("y", yOf(d.rf)); barRF.setAttribute("height", (by + bh) - yOf(d.rf));
      barLR.setAttribute("y", yOf(d.lr)); barLR.setAttribute("height", (by + bh) - yOf(d.lr));
      valRF.setAttribute("y", yOf(d.rf) - 6); valRF.textContent = d.rf.toFixed(3);
      valLR.setAttribute("y", yOf(d.lr) - 6); valLR.textContent = d.lr.toFixed(3);
      var gap = d.rf - d.lr;
      btnOn.classList.toggle("active", state === "on");
      btnOff.classList.toggle("active", state === "off");
      if (state === "on") {
        readout.innerHTML =
          "<strong>Leak ON.</strong> The complex model reads <strong style='color:" + C_RF + "'>0.935</strong> " +
          "vs LR <strong style='color:" + C_LR + "'>0.719</strong> — an apparent <strong>+" + gap.toFixed(3) +
          "</strong> win. A paper would headline: \u201ccomplex ML beats logistic regression by 22 points.\u201d";
      } else {
        readout.innerHTML =
          "<strong>Leak OFF.</strong> Remove the illegitimate feature and RF <strong style='color:" + C_RF + "'>0.712</strong> " +
          "\u2248 LR <strong style='color:" + C_LR + "'>0.721</strong> (gap <strong>" + gap.toFixed(3) +
          "</strong>). The apparent win was the leak, not the model \u2014 the reproducibility collapse.";
      }
    }

    btnOn.addEventListener("click", function () { render("on"); });
    btnOff.addEventListener("click", function () { render("off"); });
    render("on");

    return { render: render, data: DATA };
  }

  global.ReproCollapseViz = { mount: mount };
})(window);
