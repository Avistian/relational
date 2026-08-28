/**
 * RTD (Replaced Token Detection) — TabTransformer's self-supervised pre-training pretext (Huang 2020, §3.3).
 *
 * ONE mechanism: turn UNLABELED rows into a supervised signal. With probability p, each categorical cell
 * is replaced by another value drawn uniformly from that column; a per-column detector must then flag
 * which cells were tampered with. Solving that forces the encoder to learn which category values COHERE
 * in a real row — the representation that later transfers to the real (labeled) task.
 *
 * The slider sets the replacement probability p. Replaced cells are highlighted; the readout shows the
 * detector's target mask and the subtlety that the *effective* replaced fraction is p·(1 − 1/card),
 * because a uniform redraw sometimes lands on the original value (which is then NOT a replacement).
 *
 * Values are a fixed illustrative adult-style row; the corruption is deterministic given (p, seed) so the
 * headless check can assert the exact mask. No labels are used anywhere — that is the whole point.
 *
 * Expected states:
 *   - p = 0    : nothing highlighted; detector target all-zero; "no signal without corruption".
 *   - p ~ 0.30 : ~2–3 of 8 cells highlighted; readout shows planned vs effective fraction.
 *   - p = 1    : every cell is redrawn, but ~1/card of them collide with the original and stay "real".
 *
 * Usage: RtdPretrainViz.mount(container, {})
 */
(function (global) {
  "use strict";

  // A fixed illustrative row of 8 categorical columns (adult-style), each with its candidate values.
  var COLUMNS = [
    { key: "workclass",   value: "Private",        levels: ["Private", "Self-emp", "Gov", "Never-worked"] },
    { key: "education",   value: "Bachelors",      levels: ["HS-grad", "Bachelors", "Masters", "Doctorate", "11th"] },
    { key: "marital",     value: "Married",        levels: ["Married", "Never-married", "Divorced", "Widowed"] },
    { key: "occupation",  value: "Tech-support",   levels: ["Tech-support", "Sales", "Exec", "Craft", "Farming", "Armed-Forces"] },
    { key: "relationship",value: "Husband",        levels: ["Husband", "Wife", "Own-child", "Unmarried"] },
    { key: "race",        value: "White",          levels: ["White", "Black", "Asian", "Other"] },
    { key: "sex",         value: "Male",           levels: ["Male", "Female"] },
    { key: "country",     value: "United-States",  levels: ["United-States", "Mexico", "India", "Germany", "Cuba"] }
  ];

  // Deterministic PRNG (mulberry32) so a given (p, seed) always yields the same corruption.
  function rng(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Pure: which cells are PLANNED for a replacement draw (before the collide-with-self check). Returns
  // an array of {draw: bool, newIdx: int} per column, deterministic in (p, seed).
  function corrupt(p, seed) {
    var r = rng(seed);
    return COLUMNS.map(function (col) {
      var draw = r() < p;                       // is this cell selected for corruption?
      var pick = Math.floor(r() * col.levels.length);   // uniform redraw over the column's values
      var origIdx = col.levels.indexOf(col.value);
      var newIdx = draw ? pick : origIdx;
      return { draw: draw, newIdx: newIdx, origIdx: origIdx,
               replaced: draw && newIdx !== origIdx };  // a real replacement only if the value changed
    });
  }

  // Pure: expected EFFECTIVE replaced fraction across columns = mean over columns of p·(1 − 1/card).
  function expectedReplacedFraction(p) {
    var s = 0;
    COLUMNS.forEach(function (c) { s += p * (1 - 1 / c.levels.length); });
    return s / COLUMNS.length;
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "rtd-viz";
    var p = 0.30;
    var seed = (config.seed == null ? 7 : config.seed);

    var ctl = document.createElement("div");
    ctl.className = "rtd-ctl";
    var lab = document.createElement("span");
    lab.className = "rtd-glab";
    lab.textContent = "replace probability ";
    var pval = document.createElement("b");
    pval.className = "rtd-mono";
    pval.textContent = "0.30";
    lab.appendChild(pval);
    var slider = document.createElement("input");
    slider.type = "range"; slider.min = "0"; slider.max = "100"; slider.value = "30";
    slider.className = "rtd-slider";
    slider.addEventListener("input", function () {
      p = (+slider.value) / 100;
      draw();
    });
    var reroll = document.createElement("button");
    reroll.textContent = "re-roll corruption";
    reroll.addEventListener("click", function () { seed = (seed * 1664525 + 1013904223) >>> 0; draw(); });
    ctl.appendChild(lab); ctl.appendChild(slider); ctl.appendChild(reroll);
    container.appendChild(ctl);

    var row = document.createElement("div");
    row.className = "rtd-row";
    container.appendChild(row);

    var readout = document.createElement("div");
    readout.className = "rtd-readout";
    container.appendChild(readout);

    function draw() {
      pval.textContent = p.toFixed(2);
      row.innerHTML = "";
      var state = corrupt(p, seed);
      var nReplaced = 0;
      COLUMNS.forEach(function (col, j) {
        var st = state[j];
        var cell = document.createElement("div");
        cell.className = "rtd-cell" + (st.replaced ? " rtd-rep" : "");
        var name = document.createElement("div");
        name.className = "rtd-cname";
        name.textContent = col.key;
        var val = document.createElement("div");
        val.className = "rtd-cval";
        val.textContent = col.levels[st.newIdx];
        var tag = document.createElement("div");
        tag.className = "rtd-ctag";
        tag.textContent = st.replaced ? "replaced" : "real";
        cell.appendChild(name); cell.appendChild(val); cell.appendChild(tag);
        row.appendChild(cell);
        if (st.replaced) nReplaced++;
      });
      var eff = (nReplaced / COLUMNS.length);
      var expected = expectedReplacedFraction(p);
      if (p === 0) {
        readout.innerHTML = "<strong>No corruption, no signal.</strong> At p = 0 every cell is real, so the " +
          "detector's target is all-zero and there is nothing to learn. The pretext task only exists once " +
          "we tamper with the row.";
      } else {
        readout.innerHTML = "<strong>The detector's job:</strong> from each column's <em>contextual</em> " +
          "embedding, output real/replaced for that cell. Here <b>" + nReplaced + " / " + COLUMNS.length +
          "</b> cells are actually replaced (" + (eff * 100).toFixed(0) + "%). Note the gap from the slider: " +
          "the <em>effective</em> replaced fraction is only p·(1 − 1/card) ≈ <b>" + (expected * 100).toFixed(0) +
          "%</b>, because a uniform redraw sometimes lands back on the original value — those cells stay " +
          "<em>real</em>. To flag the rest, the encoder must learn which category values <em>co-occur</em> " +
          "in a genuine row — a representation that needs <strong>no labels at all</strong>.";
      }
    }

    draw();
    return { set: function (v) { p = v; draw(); }, corrupt: corrupt,
             expectedReplacedFraction: expectedReplacedFraction, columns: COLUMNS };
  }

  global.RtdPretrainViz = { mount: mount, corrupt: corrupt,
                           expectedReplacedFraction: expectedReplacedFraction, columns: COLUMNS };
})(window);
