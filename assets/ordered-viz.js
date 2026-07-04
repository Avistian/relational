/**
 * Ordered vs greedy target statistics (CatBoost; Prokhorenkova et al. 2018, §3-4).
 *
 * A categorical column is replaced by a numeric "target statistic" (mean target
 * per category). There are two ways to compute the value for row i:
 *   - GREEDY (leaky): average the target over ALL rows of the same category,
 *     INCLUDING row i's own label. Row i's y leaks straight into its own
 *     feature -> "prediction shift" / target leakage, worst on rare categories.
 *   - ORDERED (CatBoost): fix a random permutation; encode row i using only the
 *     rows that appear BEFORE it in that permutation (its "prefix") plus a prior.
 *     Row i's own label is never used -> leakage-free, like a point-in-time join.
 *
 * The widget shows one fixed permutation of rows (category + target), a pointer
 * to the "current" row, and computes both encodings live. It highlights the
 * greedy leak (own row circled) vs the ordered prefix (only earlier same-category
 * rows). A reshuffle button shows the ordered value depends on the permutation —
 * which is why CatBoost averages several permutations.
 *
 * Usage: OrderedViz.mount(container, { caption })
 * Expected states:
 *   - pointer at row 1 (no prefix): ordered TS == prior (a·prior)/(a); greedy
 *     already uses the whole category (includes future + own rows).
 *   - moving the pointer right: ordered prefix grows; greedy is unchanged
 *     (it always sees the whole column, the leak).
 *   - reshuffle: ordered values change with order; greedy does not.
 */
(function (global) {
  "use strict";

  // Fixed toy data: category in {A,B,C}, binary target y. Overall mean = prior.
  var ROWS = [
    { cat: "A", y: 1 },
    { cat: "B", y: 0 },
    { cat: "A", y: 1 },
    { cat: "C", y: 0 },
    { cat: "B", y: 1 },
    { cat: "A", y: 0 },
    { cat: "C", y: 1 },
    { cat: "B", y: 0 },
  ];
  var PRIOR_A = 1.0; // smoothing strength (CatBoost's "a")

  function prior() {
    var s = 0;
    for (var i = 0; i < ROWS.length; i++) s += ROWS[i].y;
    return s / ROWS.length;
  }

  function greedyTS(order, pos) {
    // whole-category mean over ALL rows (including the current one) — the leak
    var row = ROWS[order[pos]];
    var sum = 0, cnt = 0, ownIncluded = false;
    for (var i = 0; i < ROWS.length; i++) {
      if (ROWS[i].cat === row.cat) { sum += ROWS[i].y; cnt++; }
    }
    ownIncluded = true;
    return { val: sum / cnt, cnt: cnt, ownIncluded: ownIncluded };
  }

  function orderedTS(order, pos, p) {
    // prefix-only mean: rows before `pos` in the permutation with the same cat
    var row = ROWS[order[pos]];
    var sum = 0, cnt = 0;
    for (var j = 0; j < pos; j++) {
      if (ROWS[order[j]].cat === row.cat) { sum += ROWS[order[j]].y; cnt++; }
    }
    return { val: (sum + PRIOR_A * p) / (cnt + PRIOR_A), cnt: cnt };
  }

  function shuffle(n, seed) {
    var arr = [];
    for (var i = 0; i < n; i++) arr.push(i);
    var s = seed || 1;
    for (var k = n - 1; k > 0; k--) {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      var j = s % (k + 1);
      var t = arr[k]; arr[k] = arr[j]; arr[j] = t;
    }
    return arr;
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "ordered-viz";

    var p = prior();
    var seed = 7;
    var order = shuffle(ROWS.length, seed);

    var intro = document.createElement("p");
    intro.className = "ov-intro";
    container.appendChild(intro);

    var table = document.createElement("table");
    table.className = "ov-table";
    container.appendChild(table);

    // controls
    var ctl = document.createElement("div");
    ctl.style.cssText = "margin:0.5rem 0;display:flex;gap:0.6rem;align-items:center;flex-wrap:wrap;";
    var lab = document.createElement("label");
    lab.style.cssText = "font-size:0.82rem;font-family:var(--font-mono);";
    var slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = String(ROWS.length - 1);
    slider.step = "1";
    slider.value = "4";
    slider.style.flex = "1";
    var shuf = document.createElement("button");
    shuf.type = "button";
    shuf.textContent = "Reshuffle permutation";
    shuf.style.cssText =
      "font-size:0.78rem;padding:0.3rem 0.6rem;border:1px solid var(--border);" +
      "border-radius:5px;background:var(--bg-soft);cursor:pointer;font-family:var(--font-sans);";
    ctl.appendChild(lab);
    ctl.appendChild(slider);
    ctl.appendChild(shuf);
    container.appendChild(ctl);

    var readout = document.createElement("div");
    readout.className = "ov-readout";
    container.appendChild(readout);

    var cap = document.createElement("p");
    cap.className = "ov-caption";
    cap.textContent = config.caption ||
      "Slide the pointer to a 'current' row. Greedy target encoding averages the " +
      "whole category — including the current row's own label (the leak, circled). " +
      "Ordered encoding uses only earlier rows in the permutation (the shaded prefix), " +
      "so the label never leaks. Reshuffle to see the ordered value depends on order.";
    container.appendChild(cap);

    function draw() {
      var pos = Number(slider.value);
      var curIdx = order[pos];
      var curRow = ROWS[curIdx];
      var g = greedyTS(order, pos);
      var o = orderedTS(order, pos, p);

      // which original-row indices are in the ordered prefix and same cat
      var prefixSameCat = {};
      for (var j = 0; j < pos; j++) {
        if (ROWS[order[j]].cat === curRow.cat) prefixSameCat[order[j]] = true;
      }

      // build table in PERMUTATION order
      var html = "<thead><tr><th>perm #</th><th>category</th><th>y</th><th>role</th></tr></thead><tbody>";
      for (var k = 0; k < order.length; k++) {
        var ri = order[k];
        var r = ROWS[ri];
        var cls = "", role = "";
        if (k === pos) { cls = "ov-cur"; role = "← current row (encode this)"; }
        else if (k < pos && r.cat === curRow.cat) { cls = "ov-prefix"; role = "ordered prefix (same cat, earlier)"; }
        else if (k < pos) { cls = "ov-past"; role = "earlier (other cat)"; }
        else { cls = "ov-future"; role = "later (ordered can't see)"; }
        var yCell = r.y;
        if (k === pos) yCell = "<span class='ov-leak'>" + r.y + "</span>"; // own label = greedy leak
        html += "<tr class='" + cls + "'><td class='n'>" + (k + 1) + "</td><td>" +
          r.cat + "</td><td class='n'>" + yCell + "</td><td class='ov-role'>" + role + "</td></tr>";
      }
      html += "</tbody>";
      table.innerHTML = html;

      lab.textContent = "current = perm #" + (pos + 1);
      intro.innerHTML = "Encoding category <strong>" + curRow.cat +
        "</strong> for the current row (its true label y = <strong>" + curRow.y +
        "</strong>). Prior (overall mean y) = <strong>" + p.toFixed(3) + "</strong>.";

      readout.innerHTML =
        "<div class='ov-card ov-card-bad'><div class='ov-h'>Greedy TS (leaky)</div>" +
        "mean y over <strong>all " + g.cnt + "</strong> rows of cat " + curRow.cat +
        " <em>(incl. this row)</em> = <strong>" + g.val.toFixed(3) + "</strong>" +
        (Math.abs(g.val - curRow.y) < 0.34 ? " — tracks its own y" : "") + "</div>" +
        "<div class='ov-card ov-card-good'><div class='ov-h'>Ordered TS (CatBoost)</div>" +
        "(&Sigma;y over <strong>" + o.cnt + "</strong> earlier cat-" + curRow.cat +
        " rows + a&middot;prior) / (" + o.cnt + " + a) = <strong>" + o.val.toFixed(3) +
        "</strong> <em>(own y unused)</em></div>";
    }

    slider.addEventListener("input", draw);
    shuf.addEventListener("click", function () {
      seed = (seed * 6364136223 + 1) & 0x7fffffff;
      order = shuffle(ROWS.length, seed);
      draw();
    });
    draw();
  }

  global.OrderedViz = { mount: mount };
})(window);
