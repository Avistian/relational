/**
 * Reproduction probe: which knobs actually move the number? (Lesson 037)
 *
 * The pipeline is drawn as a chain of stages —
 *   data -> encode -> split -> fit -> predict -> OOF matrix -> fingerprint
 * — and each knob below perturbs exactly ONE stage. Selecting a knob re-runs the
 * (pre-measured) experiment: the perturbed stage lights up, the fingerprint chip
 * either matches the reference or does not, and the readout gives the measured
 * damage.
 *
 * Every number is measured, not illustrative. Source harnesses:
 *   labs/_repro_l037.py    threads x deterministic flag (9 configurations)
 *   labs/_repro2_l037.py   histogram mode, row order, dtype, model seed
 *   labs/_repro3_l037.py   splitter seed
 *   labs/_repro_env_l037.py  lightgbm 4.6.0 vs 4.5.0 under scikit-learn 1.9.0
 * Reference config: person-grouped 5-fold, LightGBM 400 trees, n_jobs=6, seed 0,
 * float32 input, lightgbm 4.6.0 -> OOF sha256 d2f0e4bf9b4fd761, mean log-loss
 * 1.632168, fold sd 0.035009.
 *
 * Plain <script> (file://-safe). Usage: ReproProbeViz.mount(el, config?)
 * config: { knobs } to override the table (defaults are the verified values).
 *
 * Expected states (headless verification):
 *   - mounts an svg chain of 6 stage boxes + one knob button per knob
 *   - default knob is "rerun": verdict "identical", fingerprint chip = reference
 *   - selecting "dtype": verdict "moves", chip differs, readout quotes 258 flips
 *   - selecting "libversion": verdict "crash", readout quotes the TypeError
 *   - selecting "splitseed": verdict "moves", readout quotes the 0.0166 range
 */
(function (global) {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var OK = "#1e6b3c";      // fingerprint matched
  var MOVED = "#b9770e";   // number changed
  var BROKE = "#b03a2e";   // did not run
  var DIM = "#94a3b8";

  var REF_SHA = "d2f0e4bf9b4fd761";
  var REF_LL = "1.632168";

  var STAGES = ["data", "encode", "split", "fit", "predict", "OOF matrix"];

  // verdict: "identical" | "moves" | "crash"
  var KNOBS = [
    {
      id: "rerun",
      label: "Run it again",
      stage: 5,
      changed: "nothing at all — same command, same minute",
      verdict: "identical",
      sha: REF_SHA,
      ll: REF_LL,
      detail:
        "Rung 1 (<em>repeatability</em>) passes: the OOF matrix is <strong>bit-for-bit</strong> the " +
        "same object, 5,587&nbsp;&times;&nbsp;5 float64 values hashing to the same digest. This is the " +
        "cheapest rung and the only one most projects ever test — usually by not testing it."
    },
    {
      id: "threads",
      label: "Thread count",
      stage: 3,
      changed: "n_jobs 6 &rarr; 1, 2 and 12 (four separate runs)",
      verdict: "identical",
      sha: REF_SHA,
      ll: REF_LL,
      detail:
        "<strong>Identical, all four.</strong> The knob that made the L036 audit affordable turns out " +
        "to be free of consequence for the model — it buys only wall time (best of 3 fits, quiet box): " +
        "9.81&nbsp;s at <code>n_jobs=1</code>, 6.01&nbsp;s at 2, <strong>3.52&nbsp;s at 6</strong>, and " +
        "13.74&nbsp;s at 12, where twelve threads contend over 4,470 training rows. You are allowed to " +
        "say this only because you ran it."
    },
    {
      id: "deterministic",
      label: "deterministic flag",
      stage: 3,
      changed: "deterministic=true + force_row_wise=true, at n_jobs 1/2/6/12",
      verdict: "identical",
      sha: REF_SHA,
      ll: REF_LL,
      detail:
        "LightGBM's own reproducibility switch — documented as ensuring &ldquo;stable results when " +
        "using the same data and the same parameters (and different <code>num_threads</code>)&rdquo; — " +
        "changes nothing here, because the result was already stable. It is not useless; it is " +
        "<em>insurance whose premium you can now price</em>: 4.58&nbsp;s against 3.52&nbsp;s per fit at " +
        "<code>n_jobs=6</code>, a 30&nbsp;% tax for a guarantee you already had."
    },
    {
      id: "histogram",
      label: "Histogram mode",
      stage: 3,
      changed: "force_row_wise=true vs force_col_wise=true",
      verdict: "identical",
      sha: REF_SHA,
      ll: REF_LL,
      detail:
        "This is the knob with the best story and the least effect. Left alone, LightGBM picks " +
        "between row-wise and column-wise histogram building by <em>timing both on your machine</em> " +
        "— so in principle the tree you get depends on how busy the box was. Forced each way, the " +
        "OOF matrix is byte-identical. A real hazard, absent from this pipeline."
    },
    {
      id: "roworder",
      label: "Training row order",
      stage: 2,
      changed: "each training fold's rows presented in a shuffled order",
      verdict: "identical",
      sha: REF_SHA,
      ll: REF_LL,
      detail:
        "Same rows, different sequence. Floating-point addition is not associative, so summing " +
        "gradients in another order <em>can</em> shift a split threshold — here it does not, because " +
        "LightGBM buckets each feature into at most 255 bins before summing anything, and the bin " +
        "totals are order-independent integers-worth of work."
    },
    {
      id: "modelseed",
      label: "Model seed",
      stage: 3,
      changed: "random_state 0 &rarr; 1, 2, 3, 4",
      verdict: "identical",
      sha: REF_SHA,
      ll: REF_LL,
      detail:
        "<strong>The seed is inert.</strong> LightGBM consults its RNG only when it samples — " +
        "<code>bagging_fraction&lt;1</code>, <code>feature_fraction&lt;1</code>, " +
        "<code>extra_trees</code> — and this configuration sets none of them, so five seeds give one " +
        "hash. &ldquo;All randomness is seeded&rdquo; is true here and tells you nothing."
    },
    {
      id: "dtype",
      label: "float32 &rarr; float64",
      stage: 1,
      changed: "the notebook's .astype(np.float32) removed; matrix stays float64",
      verdict: "moves",
      sha: "a4377f2a443dc970",
      ll: "1.633497",
      detail:
        "<strong>258 of 5,587 rows change their predicted class</strong> (4.6&nbsp;%), the largest " +
        "single-probability move is <strong>0.326</strong>, and folds 0, 1 and 4 stay bit-identical " +
        "while folds 2 and 3 diverge. Rounding to float32 nudges values across LightGBM's bin " +
        "boundaries, so a different tree is built. The aggregate barely notices: " +
        "<strong>&Delta; mean log-loss = +0.00133</strong> — 3.8&nbsp;% of one fold's " +
        "&sigma; (0.035), and <strong>42&nbsp;% of the 0.0032 margin that chose which model " +
        "shipped</strong>. One cast, in a notebook cell, in neither the README nor <code>src/</code>."
    },
    {
      id: "splitseed",
      label: "Splitter seed",
      stage: 2,
      changed: "StratifiedGroupKFold(shuffle=True, random_state=) 0 &rarr; 1..4",
      verdict: "moves",
      sha: "(five different matrices)",
      ll: "1.6191 – 1.6357",
      detail:
        "Nothing about the model changes; only <em>which persons land in which fold</em>. Mean " +
        "log-loss ranges over <strong>0.0166 nats</strong> across five draws — " +
        "<strong>5&times; the margin that decided the deployment</strong> and 12&times; the dtype " +
        "effect. Both seeds are the same literal <code>RANDOM_STATE = 0</code>; one of them does " +
        "nothing and the other is the largest controllable term in the report."
    },
    {
      id: "libversion",
      label: "One version older",
      stage: 3,
      changed: "lightgbm 4.6.0 &rarr; 4.5.0; every other package pinned identical",
      verdict: "crash",
      sha: "—",
      ll: "—",
      detail:
        "It does not produce a different number. <strong>It does not run.</strong> " +
        "<code>TypeError: check_X_y() got an unexpected keyword argument 'force_all_finite'</code> " +
        "— scikit-learn renamed that argument in 1.6 and deleted it in 1.8; LightGBM 4.5.0 still " +
        "calls it. Two packages, each perfectly valid under <code>lightgbm&gt;=4.0</code> and " +
        "<code>scikit-learn&gt;=1.5</code>, that cannot be in the same room."
    }
  ];

  function el(tag, attrs) {
    var n = document.createElementNS(SVGNS, tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    return n;
  }
  function txt(node, s) { node.textContent = s; return node; }

  function mount(container, config) {
    config = config || {};
    var knobs = config.knobs || KNOBS;
    container.innerHTML = "";
    container.classList.add("rp-viz");

    var ctl = document.createElement("div");
    ctl.className = "rp-ctl";
    container.appendChild(ctl);

    var svgHolder = document.createElement("div");
    container.appendChild(svgHolder);

    var readout = document.createElement("div");
    readout.className = "rp-readout";
    container.appendChild(readout);

    var current = knobs[0].id;
    var buttons = [];

    function colourFor(v) {
      return v === "identical" ? OK : v === "moves" ? MOVED : BROKE;
    }

    function draw() {
      var k = null;
      for (var i = 0; i < knobs.length; i++) if (knobs[i].id === current) k = knobs[i];
      if (!k) k = knobs[0];

      buttons.forEach(function (b, i) {
        b.classList.toggle("rp-on", knobs[i].id === current);
      });

      svgHolder.innerHTML = "";
      var W = 640, H = 132;
      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%" });
      svg.setAttribute("class", "rp-svg");

      var bw = 84, gap = 8, x0 = 14, y = 30;
      STAGES.forEach(function (name, i) {
        var x = x0 + i * (bw + gap);
        var hit = i === k.stage;
        svg.appendChild(el("rect", {
          x: x, y: y, width: bw, height: 34, rx: 5,
          fill: hit ? "#fdf3e3" : "#f8fafc",
          stroke: hit ? colourFor(k.verdict) : "#cbd5e1",
          "stroke-width": hit ? 2.5 : 1
        }));
        var t = el("text", {
          x: x + bw / 2, y: y + 21, "text-anchor": "middle", class: "rp-stage",
          fill: hit ? "#1f2937" : "#64748b"
        });
        svg.appendChild(txt(t, name));
        if (i < STAGES.length - 1) {
          svg.appendChild(el("line", {
            x1: x + bw, y1: y + 17, x2: x + bw + gap, y2: y + 17,
            stroke: "#cbd5e1", "stroke-width": 1.5
          }));
        }
        if (hit) {
          var mark = el("text", {
            x: x + bw / 2, y: y - 9, "text-anchor": "middle", class: "rp-mark",
            fill: colourFor(k.verdict)
          });
          svg.appendChild(txt(mark, "\u25bc knob acts here"));
        }
      });

      // fingerprint chips
      var fy = 92;
      var lab = el("text", { x: 14, y: fy + 15, class: "rp-lab" });
      svg.appendChild(txt(lab, "OOF fingerprint"));
      function chip(x, label, value, colour) {
        svg.appendChild(el("rect", {
          x: x, y: fy, width: 190, height: 22, rx: 4,
          fill: "#fff", stroke: colour, "stroke-width": 1.5
        }));
        var t1 = el("text", { x: x + 8, y: fy + 15, class: "rp-chip", fill: colour });
        svg.appendChild(txt(t1, label + " " + value));
      }
      chip(120, "reference", REF_SHA, DIM);
      chip(326, "this run  ", k.verdict === "crash" ? "did not run" : k.sha,
        colourFor(k.verdict));
      var verdictText = k.verdict === "identical" ? "\u2713 match"
        : k.verdict === "moves" ? "\u2260 differs" : "\u2717 error";
      var vt = el("text", { x: 530, y: fy + 15, class: "rp-verdict", fill: colourFor(k.verdict) });
      svg.appendChild(txt(vt, verdictText));

      svgHolder.appendChild(svg);

      var cls = k.verdict === "identical" ? "rp-ok" : k.verdict === "moves" ? "rp-moved" : "rp-broke";
      readout.innerHTML =
        "<p><span class='rp-pill " + cls + "'>" +
        (k.verdict === "identical" ? "IDENTICAL" : k.verdict === "moves" ? "NUMBER MOVES" : "DOES NOT RUN") +
        "</span> <strong>Changed:</strong> " + k.changed + "</p>" +
        "<p><strong>mean log-loss</strong> <code>" + k.ll + "</code> " +
        (k.verdict === "identical" ? "(reference <code>" + REF_LL + "</code>)" : "vs reference <code>" + REF_LL + "</code>") +
        "</p><p>" + k.detail + "</p>";
    }

    knobs.forEach(function (k) {
      var b = document.createElement("button");
      b.textContent = k.label;
      b.addEventListener("click", function () { current = k.id; draw(); });
      ctl.appendChild(b);
      buttons.push(b);
    });

    draw();
    return {
      select: function (id) { current = id; draw(); },
      getKnob: function () { return current; }
    };
  }

  global.ReproProbeViz = { mount: mount };
})(typeof window !== "undefined" ? window : this);
