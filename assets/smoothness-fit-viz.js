/**
 * The smoothness inductive bias, seen in 1-D (Grinsztajn et al. 2022, §5.2 — "Finding 1").
 *
 * A single irregular target on [0,1] (a low-frequency trend + a high-frequency wiggle + a step).
 * Two learners are fit to noisy samples of it:
 *   - a depth-limited decision TREE  → piecewise-constant "staircase" that tracks the jags;
 *   - an MLP                          → a smooth curve that over-smooths the high-frequency part.
 *
 * The slider is the TARGET-SMOOTHING bandwidth h (Grinsztajn's operation): Gaussian-smoothing the
 * target erases its high-frequency structure. As h grows, the MLP's disadvantage vanishes — because
 * the thing the tree exploited (the jags) is gone. The readout shows each fit's MSE to the (smoothed)
 * target; the MLP/tree error ratio falls from ~5.3x (raw) to ~0.2x (smooth).
 *
 * All curves + MSEs are REAL, from labs/_verify_l025.py (300 samples, seed 0, tree max_leaf_nodes=28,
 * MLP 128x128). Nothing here is hand-drawn.
 *
 * Usage: SmoothnessFitViz.mount(container, { caption })
 * Expected states:
 *   - h=0.00 (raw):    tree MSE 0.013 vs MLP MSE 0.070  → MLP error 5.3x the tree's (tree wins).
 *   - h=0.045 (medium): tree 0.005 vs MLP 0.011          → 2.4x (gap shrinking).
 *   - h=0.12 (smooth):  tree 0.006 vs MLP 0.001          → 0.19x (MLP now WINS — its bias fits).
 */
(function (global) {
  "use strict";

  var GRID = [0,0.0125,0.025,0.0375,0.05,0.0625,0.075,0.0875,0.1,0.1125,0.125,0.1375,0.15,0.1625,0.175,0.1875,0.2,0.2125,0.225,0.2375,0.25,0.2625,0.275,0.2875,0.3,0.3125,0.325,0.3375,0.35,0.3625,0.375,0.3875,0.4,0.4125,0.425,0.4375,0.45,0.4625,0.475,0.4875,0.5,0.5125,0.525,0.5375,0.55,0.5625,0.575,0.5875,0.6,0.6125,0.625,0.6375,0.65,0.6625,0.675,0.6875,0.7,0.7125,0.725,0.7375,0.75,0.7625,0.775,0.7875,0.8,0.8125,0.825,0.8375,0.85,0.8625,0.875,0.8875,0.9,0.9125,0.925,0.9375,0.95,0.9625,0.975,0.9875,1.0];

  var STATES = [
    {
      key: "raw", h: 0.0, tree_mse: 0.0132, mlp_mse: 0.0699,
      target: [0.0,0.309,0.575,0.763,0.852,0.84,0.741,0.587,0.418,0.276,0.199,0.211,0.319,0.512,0.763,1.031,1.274,1.452,1.536,1.513,1.389,1.187,0.945,0.704,0.506,0.384,0.356,0.421,0.559,0.739,0.918,1.053,1.111,1.069,0.923,0.688,0.395,0.084,-0.201,-0.421,-0.55,-0.578,-0.514,-0.383,-0.223,0.523,0.615,0.624,0.535,0.354,0.103,-0.182,-0.459,-0.685,-0.826,-0.863,-0.796,-0.641,-0.431,-0.207,-0.011,0.119,0.161,0.108,-0.028,-0.217,-0.419,-0.593,-0.699,-0.71,-0.615,-0.423,-0.158,0.142,0.433,0.675,0.834,0.896,0.862,0.752,0.6],
      tree: [0.099,0.464,0.464,0.777,0.777,0.777,0.777,0.549,0.241,0.241,0.241,0.241,0.241,0.241,0.844,0.844,1.414,1.414,1.414,1.414,1.414,1.414,0.876,0.876,0.445,0.445,0.445,0.445,0.445,0.693,0.693,1.066,1.066,1.066,0.872,0.872,0.102,0.102,0.102,-0.434,-0.434,-0.434,-0.434,-0.434,-0.131,0.578,0.578,0.578,0.578,0.28,-0.058,-0.475,-0.475,-0.768,-0.768,-0.768,-0.768,-0.768,-0.358,-0.358,0.086,0.086,0.086,0.086,0.086,-0.294,-0.294,-0.646,-0.646,-0.646,-0.646,-0.369,-0.369,0.106,0.492,0.772,0.772,0.772,0.772,0.772,0.772],
      mlp: [0.23,0.255,0.28,0.305,0.33,0.355,0.381,0.406,0.431,0.456,0.481,0.507,0.537,0.586,0.639,0.693,0.746,0.8,0.853,0.871,0.863,0.855,0.847,0.838,0.83,0.822,0.814,0.807,0.802,0.796,0.791,0.785,0.779,0.774,0.758,0.463,0.183,-0.068,-0.29,-0.504,-0.634,-0.59,-0.463,-0.228,0.012,0.252,0.487,0.521,0.337,0.143,-0.051,-0.245,-0.439,-0.633,-0.8,-0.779,-0.7,-0.571,-0.442,-0.314,-0.185,-0.056,0.06,0.009,-0.097,-0.203,-0.308,-0.411,-0.515,-0.57,-0.373,-0.159,0.055,0.269,0.482,0.618,0.72,0.823,0.925,1.027,1.129]
    },
    {
      key: "medium", h: 0.045, tree_mse: 0.0045, mlp_mse: 0.0107,
      target: [0.486,0.52,0.548,0.567,0.576,0.573,0.559,0.54,0.523,0.517,0.529,0.567,0.631,0.718,0.818,0.92,1.01,1.077,1.113,1.112,1.077,1.015,0.936,0.854,0.781,0.727,0.696,0.69,0.703,0.724,0.742,0.745,0.721,0.665,0.577,0.461,0.328,0.191,0.065,-0.036,-0.101,-0.126,-0.111,-0.066,-0.004,0.057,0.101,0.115,0.093,0.034,-0.054,-0.158,-0.263,-0.353,-0.418,-0.45,-0.447,-0.415,-0.362,-0.301,-0.245,-0.204,-0.185,-0.191,-0.217,-0.256,-0.294,-0.321,-0.324,-0.295,-0.232,-0.138,-0.019,0.112,0.244,0.366,0.471,0.556,0.62,0.664,0.694],
      tree: [0.398,0.398,0.645,0.645,0.528,0.528,0.528,0.528,0.528,0.528,0.528,0.71,0.71,0.71,0.71,0.71,1.136,1.136,1.136,1.136,1.136,1.136,0.908,0.908,0.728,0.728,0.728,0.728,0.728,0.728,0.728,0.728,0.728,0.728,0.567,0.567,0.287,0.095,0.095,0.095,-0.192,-0.192,-0.192,-0.192,-0.029,-0.029,0.153,0.153,0.153,0.153,-0.085,-0.171,-0.171,-0.418,-0.418,-0.418,-0.418,-0.418,-0.418,-0.418,-0.199,-0.199,-0.199,-0.199,-0.199,-0.199,-0.323,-0.323,-0.323,-0.323,-0.323,-0.071,-0.071,0.13,0.182,0.355,0.355,0.523,0.63,0.63,0.63],
      mlp: [0.315,0.346,0.377,0.408,0.44,0.473,0.505,0.537,0.569,0.602,0.634,0.667,0.704,0.742,0.78,0.818,0.856,0.894,0.932,0.956,0.934,0.909,0.884,0.86,0.835,0.81,0.785,0.76,0.736,0.711,0.686,0.661,0.636,0.59,0.505,0.419,0.333,0.248,0.162,0.078,0.052,0.028,0.005,-0.018,-0.042,-0.065,-0.088,-0.111,-0.135,-0.158,-0.181,-0.204,-0.228,-0.252,-0.275,-0.299,-0.323,-0.345,-0.341,-0.331,-0.321,-0.311,-0.301,-0.292,-0.282,-0.273,-0.263,-0.254,-0.244,-0.226,-0.137,-0.046,0.045,0.136,0.227,0.318,0.408,0.497,0.583,0.668,0.754]
    },
    {
      key: "smooth", h: 0.12, tree_mse: 0.0064, mlp_mse: 0.0012,
      target: [0.587,0.597,0.608,0.619,0.631,0.644,0.657,0.671,0.684,0.698,0.712,0.725,0.737,0.749,0.759,0.768,0.775,0.78,0.782,0.782,0.778,0.772,0.762,0.749,0.732,0.713,0.69,0.665,0.636,0.605,0.572,0.537,0.501,0.463,0.424,0.385,0.345,0.306,0.267,0.229,0.192,0.155,0.121,0.087,0.055,0.024,-0.004,-0.032,-0.057,-0.081,-0.103,-0.122,-0.14,-0.156,-0.169,-0.18,-0.189,-0.195,-0.199,-0.199,-0.197,-0.193,-0.186,-0.176,-0.163,-0.149,-0.132,-0.113,-0.092,-0.07,-0.046,-0.02,0.006,0.033,0.06,0.088,0.117,0.145,0.173,0.201,0.228],
      tree: [0.621,0.621,0.621,0.621,0.621,0.621,0.621,0.621,0.621,0.621,0.902,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.72,0.638,0.638,0.827,0.827,0.511,0.511,0.511,0.511,0.511,0.511,0.351,0.351,0.351,0.351,0.351,0.228,0.103,0.103,0.103,0.103,0.013,0.09,0.09,-0.054,-0.445,-0.136,-0.136,-0.136,-0.136,-0.329,-0.204,-0.204,-0.204,-0.204,-0.068,-0.125,-0.125,-0.125,-0.125,-0.125,-0.125,-0.125,-0.125,-0.125,-0.125,-0.125,-0.125,-0.026,-0.026,-0.026,-0.026,0.125,0.125,0.125,0.201,0.201,0.403,0.081],
      mlp: [0.599,0.605,0.611,0.618,0.624,0.63,0.637,0.643,0.649,0.655,0.662,0.668,0.674,0.68,0.687,0.693,0.699,0.706,0.711,0.713,0.715,0.714,0.708,0.699,0.687,0.674,0.656,0.631,0.603,0.574,0.544,0.51,0.474,0.437,0.401,0.365,0.329,0.293,0.257,0.221,0.184,0.148,0.114,0.081,0.048,0.014,-0.017,-0.047,-0.078,-0.107,-0.133,-0.153,-0.167,-0.173,-0.171,-0.167,-0.163,-0.159,-0.155,-0.15,-0.146,-0.142,-0.138,-0.133,-0.128,-0.122,-0.113,-0.099,-0.085,-0.065,-0.04,-0.015,0.01,0.035,0.06,0.085,0.11,0.135,0.16,0.184,0.208]
    }
  ];

  var svgNS = "http://www.w3.org/2000/svg";
  function el(name, attrs) {
    var e = document.createElementNS(svgNS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "smoothness-fit-viz";

    var idx = 0; // 0 raw, 1 medium, 2 smooth

    var ctl = document.createElement("div");
    ctl.className = "sf-ctl";
    var slabel = document.createElement("label");
    slabel.className = "sf-slabel";
    slabel.textContent = "target smoothing h = ";
    var slider = document.createElement("input");
    slider.type = "range"; slider.min = "0"; slider.max = "2"; slider.step = "1"; slider.value = "0";
    slider.className = "sf-slider";
    var hval = document.createElement("span");
    hval.className = "sf-hval"; hval.textContent = "0.00 (raw)";
    slabel.appendChild(slider); slabel.appendChild(hval);
    ctl.appendChild(slabel);
    container.appendChild(ctl);

    var W = 470, H = 300;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", class: "sf-svg" });
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "sf-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "sf-caption";
    cap.textContent = config.caption ||
      "One irregular target (grey), a decision-tree staircase (green) and an MLP fit (blue). " +
      "Drag h to Gaussian-smooth the target: the tree's edge lives entirely in the high-frequency jags.";
    container.appendChild(cap);

    var PL = 40, PR = W - 12, PT = 16, PB = H - 40;
    var YLO = -0.9, YHI = 1.65;
    function sx(x) { return PL + x * (PR - PL); }
    function sy(v) { return PB - (v - YLO) / (YHI - YLO) * (PB - PT); }

    function path(arr, color, width, dash) {
      var d = "M";
      for (var i = 0; i < arr.length; i++) {
        d += (i ? "L" : "") + sx(GRID[i]).toFixed(1) + "," + sy(arr[i]).toFixed(1) + " ";
      }
      var a = { d: d, fill: "none", stroke: color, "stroke-width": width };
      if (dash) a["stroke-dasharray"] = dash;
      return el("path", a);
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var st = STATES[idx];

      svg.appendChild(el("rect", { x: PL, y: PT, width: PR - PL, height: PB - PT,
        fill: "var(--bg)", stroke: "var(--border)", "stroke-width": 1 }));

      [-0.5, 0, 0.5, 1.0, 1.5].forEach(function (v) {
        var yy = sy(v);
        svg.appendChild(el("line", { x1: PL, y1: yy, x2: PR, y2: yy,
          stroke: "var(--border)", "stroke-width": 0.5, opacity: 0.6 }));
        svg.appendChild(el("text", { x: PL - 5, y: yy + 3, "text-anchor": "end", class: "sf-tick" }))
          .textContent = v.toFixed(1);
      });

      svg.appendChild(path(st.target, "#8a8a8a", 3.0));
      svg.appendChild(path(st.tree, "#1e6b3c", 2.0));
      svg.appendChild(path(st.mlp, "#2e6fb0", 2.0, "5,3"));

      svg.appendChild(el("text", { x: (PL + PR) / 2, y: H - 6, "text-anchor": "middle", class: "sf-lab" }))
        .textContent = "feature x";

      var lx = PL + 8, ly = PT + 6;
      [["#8a8a8a", "target"], ["#1e6b3c", "tree (staircase)"], ["#2e6fb0", "MLP (smooth)"]].forEach(function (pair, i) {
        svg.appendChild(el("rect", { x: lx, y: ly + i * 15, width: 12, height: 4, fill: pair[0] }));
        svg.appendChild(el("text", { x: lx + 17, y: ly + i * 15 + 4, class: "sf-leg" })).textContent = pair[1];
      });

      var ratio = st.mlp_mse / st.tree_mse;
      var winner = st.mlp_mse < st.tree_mse ? "MLP" : "tree";
      readout.innerHTML =
        "<span class='sf-pill'>h = " + st.h.toFixed(3) + "</span> " +
        "tree MSE <strong class='sf-tree'>" + st.tree_mse.toFixed(4) + "</strong> · " +
        "MLP MSE <strong class='sf-mlp'>" + st.mlp_mse.toFixed(4) + "</strong> · " +
        "MLP/tree error <strong>" + ratio.toFixed(2) + "×</strong>" +
        (idx === 0
          ? " — on the raw jagged target the MLP over-smooths; the tree tracks the jags and wins."
          : idx === 2
            ? " — the target is now smooth, so the MLP's bias fits it and it edges the tree."
            : " — smoothing is erasing the jags; the tree's advantage is shrinking.");
    }

    slider.addEventListener("input", function () {
      idx = Number(slider.value);
      var st = STATES[idx];
      hval.textContent = st.h.toFixed(3) + (idx === 0 ? " (raw)" : idx === 2 ? " (smooth)" : "");
      draw();
    });

    draw();
    return { setState: function (i) { idx = i; slider.value = String(i); slider.input ? slider.input() : draw(); draw(); } };
  }

  global.SmoothnessFitViz = { mount: mount };
})(window);
