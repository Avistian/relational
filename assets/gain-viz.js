/**
 * XGBoost regularized split-gain visualizer (Chen & Guestrin 2016, §2).
 * A candidate split sends fixed gradient/hessian statistics into a Left and a
 * Right leaf. Two sliders control the regularizers:
 *   - lambda (L2 on leaf weights): shrinks each leaf weight  w* = -G/(H+λ)  toward 0
 *   - gamma  (complexity per leaf): a flat toll subtracted from the split gain
 *
 * Structure score of a node:  G^2 / (H + λ)      (higher = purer, more useful)
 * Split gain: ½[ score(L) + score(R) − score(parent) ] − γ
 *           = "children quality − parent quality − complexity toll"
 * This is the L011 ΔG idea (children minus parent) with regularization bolted on.
 *
 * Usage: GainViz.mount(container, { gL, hL, gR, hR, lambda, gamma, caption })
 * Expected states:
 *   - λ = 0, γ = 0: leaf weights largest; gain is the full raw gain; verdict KEEP.
 *   - raise λ: both leaf weights and every score shrink toward 0 (regularization).
 *   - raise γ past the raw gain: net gain goes negative → verdict flips to PRUNE.
 */
(function (global) {
  "use strict";

  function score(G, H, lam) { return (G * G) / (H + lam); }
  function weight(G, H, lam) { return -G / (H + lam); }

  function mount(container, config) {
    config = config || {};
    var gL = config.gL != null ? config.gL : -8;
    var hL = config.hL != null ? config.hL : 10;
    var gR = config.gR != null ? config.gR : 12;
    var hR = config.hR != null ? config.hR : 10;
    var G = gL + gR, H = hL + hR;

    container.innerHTML = "";
    container.className = "gain-viz tree-viz";

    var label = document.createElement("p");
    label.className = "tree-viz-intro";
    container.appendChild(label);

    var canvas = document.createElement("canvas");
    canvas.width = 360;
    canvas.height = 210;
    canvas.className = "tree-viz-canvas";
    container.appendChild(canvas);

    function slider(name, min, max, step, val) {
      var wrap = document.createElement("div");
      wrap.style.margin = "0.35rem 0";
      var lab = document.createElement("label");
      lab.style.cssText = "font-size:0.82rem;font-family:var(--font-mono);display:block;";
      var inp = document.createElement("input");
      inp.type = "range";
      inp.min = String(min); inp.max = String(max); inp.step = String(step);
      inp.value = String(val);
      inp.style.width = "100%";
      wrap.appendChild(lab);
      wrap.appendChild(inp);
      container.appendChild(wrap);
      return { input: inp, label: lab, name: name };
    }

    var lamS = slider("λ", 0, 20, 0.5, config.lambda != null ? config.lambda : 1);
    var gamS = slider("γ", 0, 15, 0.5, config.gamma != null ? config.gamma : 0);

    var readout = document.createElement("p");
    readout.className = "tree-viz-caption";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "tree-viz-caption";
    cap.textContent = config.caption ||
      "λ shrinks leaf weights; γ is a flat toll per split. Gain ≤ 0 ⇒ the split is pruned.";
    container.appendChild(cap);

    var ctx = canvas.getContext("2d");
    // max half-score at λ=0 for a stable y-scale.
    var maxHalf = 0.5 * (score(gL, hL, 0) + score(gR, hR, 0));

    function bar(x, wpx, half, color, txt) {
      var base = 170, maxh = 130;
      var hpx = Math.max(0, (half / maxHalf) * maxh);
      ctx.fillStyle = color;
      ctx.fillRect(x, base - hpx, wpx, hpx);
      ctx.fillStyle = "#34495e";
      ctx.font = "11px monospace";
      ctx.textAlign = "center";
      ctx.fillText(txt, x + wpx / 2, base + 15);
    }

    function draw() {
      var lam = Number(lamS.input.value);
      var gam = Number(gamS.input.value);
      lamS.label.textContent = "λ (L2 on leaf weight) = " + lam.toFixed(1);
      gamS.label.textContent = "γ (complexity per leaf) = " + gam.toFixed(1);

      var sP = score(G, H, lam);
      var sLv = score(gL, hL, lam);
      var sRv = score(gR, hR, lam);
      var halfP = 0.5 * sP;
      var halfChildren = 0.5 * (sLv + sRv);
      var rawGain = halfChildren - halfP;
      var netGain = rawGain - gam;
      var keep = netGain > 0;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#f8f9fa";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // three bars: parent half-score, children half-score, resulting gain
      bar(40, 60, halfP, "#95a5a6", "parent");
      bar(150, 60, halfChildren, "#2980b9", "L + R");
      bar(260, 60, Math.max(0, rawGain), keep ? "#1e6b3c" : "#c0392b", "gain");

      // gamma toll marker on the gain bar
      var base = 170, maxh = 130;
      var gamPx = (gam / maxHalf) * maxh;
      if (gam > 0) {
        var gainTop = base - Math.max(0, (rawGain / maxHalf) * maxh);
        ctx.strokeStyle = "#c0392b";
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(255, base - gamPx);
        ctx.lineTo(325, base - gamPx);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#c0392b";
        ctx.font = "10px monospace";
        ctx.textAlign = "left";
        ctx.fillText("γ", 327, base - gamPx + 3);
      }

      label.innerHTML =
        "Left leaf w* = <strong>" + weight(gL, hL, lam).toFixed(3) +
        "</strong> · Right leaf w* = <strong>" + weight(gR, hR, lam).toFixed(3) + "</strong>";
      readout.innerHTML =
        "raw gain ½[L+R−parent] = <strong>" + rawGain.toFixed(2) +
        "</strong> · net gain (−γ) = <strong>" + netGain.toFixed(2) +
        "</strong> · <span style='color:" + (keep ? "#1e6b3c" : "#c0392b") +
        ";font-weight:600'>" + (keep ? "SPLIT KEPT" : "SPLIT PRUNED") + "</span>";
    }

    lamS.input.addEventListener("input", draw);
    gamS.input.addEventListener("input", draw);
    draw();
  }

  global.GainViz = { mount: mount };
})(window);
