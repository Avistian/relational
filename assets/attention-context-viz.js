/**
 * Self-attention turns context-FREE embeddings into CONTEXTUAL ones (the load-bearing idea of L032).
 *
 * ONE mechanism: pick a query column; the Transformer computes attention weights from that column to
 * every column in the SAME ROW, and the column's new (contextual) vector is the weighted blend of the
 * others' value vectors. Toggle "before / after" to see that the SAME column embedding changes once it
 * can see its neighbours — the tabular analogue of "bank" meaning river-bank vs money-bank in NLP.
 *
 * Numbers are ILLUSTRATIVE (a fixed 3-token toy), chosen so the blend is legible — not measured.
 *
 * Expected states:
 *   - default query = checking_status; "after" view selected.
 *   - "before" view: each column shows its own context-free embedding (no blending); attention row hidden.
 *   - "after" view: the query's contextual vector = Σ_j w[query][j] · value[j]; weights shown as a bar row
 *     summing to 1; the query attends most to itself + the most informative neighbour.
 *
 * Usage: AttentionContextViz.mount(container, {})
 */
(function (global) {
  "use strict";

  // 3-token toy row from credit_g. Each has a 4-dim context-free embedding (illustrative).
  var TOKENS = [
    { key: "purpose",  label: "purpose = new car",     emb: [0.8, 0.1, 0.6, 0.2] },
    { key: "checking", label: "checking = < 0",        emb: [0.2, 0.9, 0.3, 0.7] },
    { key: "housing",  label: "housing = rent",        emb: [0.5, 0.4, 0.2, 0.9] }
  ];
  // Illustrative attention weight matrix W[query][key] (rows sum to 1): each column attends to itself +
  // the risk-carrying "checking" column most.
  var ATTN = {
    purpose:  [0.55, 0.30, 0.15],
    checking: [0.15, 0.60, 0.25],
    housing:  [0.20, 0.45, 0.35]
  };
  var DIMCOL = ["#2e6fb0", "#1e6b3c", "#c47f17", "#7a5195"];

  function contextual(qKey) {
    var w = ATTN[qKey];
    var out = [0, 0, 0, 0];
    TOKENS.forEach(function (t, j) {
      for (var d = 0; d < 4; d++) out[d] += w[j] * t.emb[d];
    });
    return out;
  }

  function bars(vec, maxw) {
    // returns an inline-block set of vertical bars for a 4-dim vector
    var box = document.createElement("span");
    box.className = "atc-bars";
    vec.forEach(function (v, d) {
      var b = document.createElement("span");
      b.className = "atc-bar";
      b.style.height = Math.max(2, v * 34) + "px";
      b.style.background = DIMCOL[d];
      box.appendChild(b);
    });
    return box;
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "atc-viz";
    var view = "after";     // "before" | "after"
    var query = "checking";

    // view toggle
    var ctl = document.createElement("div");
    ctl.className = "atc-ctl";
    [["before", "Before Transformer (context-free)"], ["after", "After Transformer (contextual)"]].forEach(function (m) {
      var b = document.createElement("button");
      b.textContent = m[1];
      if (m[0] === view) b.className = "atc-on";
      b.addEventListener("click", function () {
        view = m[0];
        Array.prototype.forEach.call(ctl.children, function (c) { c.className = ""; });
        b.className = "atc-on";
        draw();
      });
      ctl.appendChild(b);
    });
    container.appendChild(ctl);

    var hint = document.createElement("p");
    hint.className = "atc-hint";
    container.appendChild(hint);

    var grid = document.createElement("div");
    grid.className = "atc-grid";
    container.appendChild(grid);

    var readout = document.createElement("div");
    readout.className = "atc-readout";
    container.appendChild(readout);

    function draw() {
      grid.innerHTML = "";
      hint.innerHTML = view === "after"
        ? "Click a column to make it the <strong>query</strong>. Its contextual vector is the weighted blend below."
        : "Each column carries its own embedding, unaware of the others — this is the L031 entity embedding.";

      TOKENS.forEach(function (t) {
        var card = document.createElement("div");
        card.className = "atc-card" + (view === "after" && t.key === query ? " atc-q" : "");
        if (view === "after") {
          card.style.cursor = "pointer";
          card.addEventListener("click", function () { query = t.key; draw(); });
        }
        var name = document.createElement("div");
        name.className = "atc-name";
        name.textContent = t.label;
        card.appendChild(name);

        var vec = (view === "after" && t.key === query) ? contextual(query) : t.emb;
        card.appendChild(bars(vec));

        var tag = document.createElement("div");
        tag.className = "atc-tag";
        tag.innerHTML = (view === "after" && t.key === query)
          ? "contextual (blended)"
          : "context-free";
        card.appendChild(tag);
        grid.appendChild(card);
      });

      if (view === "after") {
        var w = ATTN[query];
        var wrow = document.createElement("div");
        wrow.className = "atc-weights";
        wrow.innerHTML = "<span class='atc-wlab'>attention from <b>" +
          TOKENS.filter(function (t) { return t.key === query; })[0].label + "</b>:</span>";
        TOKENS.forEach(function (t, j) {
          var pill = document.createElement("span");
          pill.className = "atc-wpill";
          pill.innerHTML = t.key + " <b>" + w[j].toFixed(2) + "</b>";
          pill.style.opacity = 0.4 + 0.6 * w[j];
          wrow.appendChild(pill);
        });
        grid.appendChild(wrow);
        var qIdx = TOKENS.map(function (t) { return t.key; }).indexOf(query);
        var qLabel = TOKENS[qIdx].label;
        readout.innerHTML = "<strong>Contextual = weighted blend.</strong> The new vector for <em>" +
          qLabel + "</em> is Σ<sub>j</sub> w<sub>j</sub>·value<sub>j</sub> over the row — mostly itself (" +
          "self-weight " + w[qIdx].toFixed(2) + ") but bent toward the columns it attends to. Change the " +
          "query and the blend changes: the embedding of a column now <em>depends on</em> the rest of the row.";
      } else {
        readout.innerHTML = "<strong>Context-free.</strong> Before the Transformer, each column's vector is " +
          "fixed — the same <code>checking = &lt; 0</code> vector regardless of the other columns. Identical " +
          "to how L031's entity embeddings work. Switch to <em>after</em> to see attention blend them.";
      }
    }

    draw();
    return { set: function (v) { view = v; draw(); } };
  }

  global.AttentionContextViz = { mount: mount };
})(window);
