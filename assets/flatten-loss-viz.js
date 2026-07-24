/**
 * Flatten-loss / aggregation COLLISION: two entities with DIFFERENT neighbour
 * structure that flatten to the SAME fixed-width feature row.
 *
 * ONE mechanism — the lossiness of join+aggregate (Fey et al. 2024 §2, issue 4:
 * "by forcing data into a single table, information is aggregated into
 * lower-granularity features, thus losing out on valuable fine-grain signal").
 *
 * Left: two customers, each a small entity subgraph (customer -> 3 orders, drawn
 * left->right in TIME order, each order carrying an amount + a product id).
 * Middle: JOIN + AGGREGATE. Right: both collapse to the IDENTICAL aggregate row
 * (n_orders / total_spend / avg_basket / max_basket) — a collision. A model on the
 * flat table literally cannot tell the two customers apart.
 *
 * Data (baked; chosen so every count/sum/mean/max aggregate is identical):
 *   Ada (C_up): orders in time order  $10 (milk), $30 (bread), $50 (eggs)  -> RISING spend, 3 products
 *   Bo  (C_dn): orders in time order  $50 (wine), $30 (wine),  $10 (wine)  -> FALLING spend, 1 product
 *   Both -> n=3, total=90, avg=30, max=50  (IDENTICAL flat row)
 * They differ in exactly the structure aggregation throws away: temporal ORDER
 * (rising vs falling — an engaged vs churning trajectory), event/product IDENTITY
 * (3 distinct products vs 1), and higher-order paths (which products at all).
 *
 * Toggle "Reveal the lost structure": annotates each subgraph with its trend (up/down)
 * and distinct-product count, and shows the extra columns a STRUCTURE-KEEPING model
 * could recover (spend_trend, n_distinct_products) that break the tie.
 *
 * Expected states:
 *   - default: both subgraphs drawn; two IDENTICAL aggregate rows; banner names the collision
 *   - reveal:  trend glyphs + distinct-product badges appear; distinguishing columns shown
 *
 * Usage: FlattenLossViz.mount(container, {})
 */
(function (global) {
  "use strict";

  var svgNS = "http://www.w3.org/2000/svg";
  function el(name, attrs) {
    var e = document.createElementNS(svgNS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }
  function round1(x) { return Math.round(x * 10) / 10; }

  // Orders are listed in TIME order (oldest first).
  var ENTITIES = [
    { id: "Ada", cid: "C_up", trend: "rising", orders: [
      { ts: "Jan", amount: 10, product: "milk" },
      { ts: "Mar", amount: 30, product: "bread" },
      { ts: "May", amount: 50, product: "eggs" } ] },
    { id: "Bo", cid: "C_dn", trend: "falling", orders: [
      { ts: "Jan", amount: 50, product: "wine" },
      { ts: "Mar", amount: 30, product: "wine" },
      { ts: "May", amount: 10, product: "wine" } ] }
  ];

  function aggregate(e) {
    var amts = e.orders.map(function (o) { return o.amount; });
    var n = amts.length;
    var total = amts.reduce(function (s, a) { return s + a; }, 0);
    var products = {};
    e.orders.forEach(function (o) { products[o.product] = 1; });
    return {
      n_orders: n,
      total_spend: total,
      avg_basket: round1(total / n),
      max_basket: Math.max.apply(null, amts),
      // structure a flat aggregate throws away:
      spend_trend: amts[amts.length - 1] - amts[0] > 0 ? "up" : (amts[amts.length - 1] - amts[0] < 0 ? "down" : "flat"),
      n_distinct_products: Object.keys(products).length
    };
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "fl-viz";
    var revealed = false;

    var ctl = document.createElement("div");
    ctl.className = "fl-ctl";
    var revealBtn = document.createElement("button");
    revealBtn.textContent = "Reveal the lost structure";
    revealBtn.addEventListener("click", function () { revealed = !revealed; sync(); draw(); });
    ctl.appendChild(revealBtn);
    container.appendChild(ctl);

    function sync() { revealBtn.className = revealed ? "fl-on" : ""; }

    var W = 660, H = 340;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", class: "fl-svg" });
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "fl-readout";
    container.appendChild(readout);

    var BLUE = "#2e6fb0", GREEN = "#1e6b3c", RED = "#b03a2e", AMBER = "#b0632e";

    function box(x, y, w, h, fill, stroke, sw) {
      return el("rect", { x: x, y: y, width: w, height: h, rx: 6,
        fill: fill, stroke: stroke || fill, "stroke-width": sw || 1.4 });
    }
    function label(x, y, txt, cls, fill, anchor) {
      var t = el("text", { x: x, y: y, class: cls, "text-anchor": anchor || "start" });
      if (fill) t.setAttribute("fill", fill);
      t.textContent = txt; return t;
    }

    function drawEntity(e, topY) {
      var agg = aggregate(e);
      // customer node
      svg.appendChild(box(14, topY, 92, 44, BLUE));
      svg.appendChild(label(24, topY + 19, "customer", "fl-nsub", "#fff"));
      svg.appendChild(label(24, topY + 36, e.id, "fl-nhead", "#fff"));

      // order nodes, left->right in time order
      var ox0 = 132, ow = 78, ogap = 14, oy = topY - 2, oh = 48;
      e.orders.forEach(function (o, i) {
        var ox = ox0 + i * (ow + ogap);
        // connector customer -> order
        svg.appendChild(el("path", {
          d: "M106," + (topY + 22) + " L" + ox + "," + (oy + oh / 2),
          fill: "none", stroke: "var(--muted)", "stroke-width": 1.1 }));
        svg.appendChild(box(ox, oy, ow, oh, "#fff", GREEN, 1.3));
        svg.appendChild(label(ox + 8, oy + 15, o.ts + " · $" + o.amount, "fl-orow", GREEN));
        svg.appendChild(label(ox + 8, oy + 31, o.product, "fl-oprod",
          revealed ? AMBER : "var(--muted)"));
        if (i < e.orders.length - 1) {
          // time arrow between orders
          var ax = ox + ow, ax2 = ox + ow + ogap;
          svg.appendChild(el("path", { d: "M" + ax + "," + (oy + oh / 2) + " L" + ax2 + "," + (oy + oh / 2),
            fill: "none", stroke: "var(--border)", "stroke-width": 1.4, "marker-end": "url(#fl-tarrow)" }));
        }
      });
      // time axis caption
      svg.appendChild(label(132, oy + oh + 14, "time \u2192 (order the baskets arrived in)", "fl-axis", "var(--muted)"));

      if (revealed) {
        // trend glyph + distinct-product badge
        var tx = ox0 + 3 * (ow + ogap) + 6;
        var up = agg.spend_trend === "up";
        svg.appendChild(label(tx, oy + 16, up ? "spend \u2197 rising" : "spend \u2198 falling", "fl-trend",
          up ? GREEN : RED));
        svg.appendChild(label(tx, oy + 34, agg.n_distinct_products + " distinct product" +
          (agg.n_distinct_products === 1 ? "" : "s"), "fl-trend", AMBER));
      }
    }

    function drawFlatRow(agg, x, y, collide) {
      var cols = [
        { k: "n_orders", v: agg.n_orders },
        { k: "total_spend", v: agg.total_spend },
        { k: "avg_basket", v: agg.avg_basket },
        { k: "max_basket", v: agg.max_basket }
      ];
      var cw = 118, ch = 26;
      svg.appendChild(box(x, y, cw, ch * cols.length, collide ? "#fdecea" : "var(--bg-soft)",
        collide ? RED : "var(--border)", collide ? 1.6 : 0.8));
      cols.forEach(function (col, i) {
        var yy = y + i * ch;
        svg.appendChild(el("line", { x1: x, y1: yy, x2: x + cw, y2: yy,
          stroke: "var(--border)", "stroke-width": 0.6 }));
        svg.appendChild(label(x + 8, yy + ch / 2 + 4, col.k, "fl-fk", "var(--fg, #222)"));
        svg.appendChild(label(x + cw - 8, yy + ch / 2 + 4, String(col.v), "fl-fv", GREEN, "end"));
      });
      return y + cols.length * ch;
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      var defs = el("defs", {});
      var mk = el("marker", { id: "fl-arrow", markerWidth: 9, markerHeight: 9, refX: 7, refY: 3,
        orient: "auto", markerUnits: "strokeWidth" });
      mk.appendChild(el("path", { d: "M0,0 L7,3 L0,6 Z", fill: "var(--fg, #222)" }));
      defs.appendChild(mk);
      var tk = el("marker", { id: "fl-tarrow", markerWidth: 8, markerHeight: 8, refX: 6, refY: 3,
        orient: "auto", markerUnits: "strokeWidth" });
      tk.appendChild(el("path", { d: "M0,0 L6,3 L0,6 Z", fill: "var(--border)" }));
      defs.appendChild(tk);
      svg.appendChild(defs);

      svg.appendChild(label(14, 16, "RAW: two customers, different histories", "fl-lane"));
      drawEntity(ENTITIES[0], 40);
      drawEntity(ENTITIES[1], 150);

      // join+aggregate arrow
      svg.appendChild(el("path", { d: "M470,120 L520,120", fill: "none", stroke: "var(--fg, #222)",
        "stroke-width": 2.4, "marker-end": "url(#fl-arrow)" }));
      svg.appendChild(label(495, 108, "JOIN +", "fl-op", AMBER, "middle"));
      svg.appendChild(label(495, 138, "AGGREGATE", "fl-op", AMBER, "middle"));

      // flat rows (right)
      svg.appendChild(label(532, 40, "FLAT rows", "fl-lane"));
      var a0 = aggregate(ENTITIES[0]), a1 = aggregate(ENTITIES[1]);
      var same = a0.n_orders === a1.n_orders && a0.total_spend === a1.total_spend &&
                 a0.avg_basket === a1.avg_basket && a0.max_basket === a1.max_basket;
      svg.appendChild(label(532, 58, "Ada", "fl-rowid", BLUE));
      drawFlatRow(a0, 532, 64, same);
      svg.appendChild(label(532, 190, "Bo", "fl-rowid", BLUE));
      drawFlatRow(a1, 532, 196, same);
      if (same) {
        svg.appendChild(label(532, 316, "\u26a0 identical", "fl-collide", RED));
      }

      if (revealed) {
        readout.innerHTML =
          "<strong>What a graph would keep.</strong> The two customers are <em>not</em> the same: Ada's spend " +
          "is <span style='color:" + GREEN + "'>rising</span> across 3 <em>distinct</em> products (an engaged, " +
          "broadening customer); Bo's is <span style='color:" + RED + "'>falling</span> and stuck on a single " +
          "product (a likely churn). A model that keeps the relational structure can read a <code>spend_trend</code> " +
          "(up vs down) and <code>n_distinct_products</code> (3 vs 1) that split them apart. But those are exactly " +
          "the <strong>temporal order</strong>, <strong>event identity</strong>, and <strong>higher-order paths</strong> " +
          "the flatten discarded \u2014 you only recover them by hand-writing yet more aggregates, per task, forever. " +
          "RDL keeps the customer\u2192orders\u2192products edges as a graph and learns these end-to-end.";
      } else {
        readout.innerHTML =
          "<strong>An aggregation collision.</strong> Ada and Bo have <em>different</em> order histories \u2014 " +
          "different amounts <em>in a different order</em>, different products \u2014 yet <code>COUNT</code>, " +
          "<code>SUM</code>, <code>AVG</code> and <code>MAX</code> map both to the <strong>identical</strong> flat " +
          "row <code>n_orders=3, total_spend=90, avg_basket=30, max_basket=50</code>. On the flat table the two " +
          "customers are indistinguishable \u2014 any model, tuned however well, must give them the same prediction. " +
          "Aggregation is <em>lossy by construction</em> (Fey 2024 \u00a72): it forgets temporal order, event " +
          "identity and multi-hop paths. Press <em>Reveal the lost structure</em> to see what a graph keeps.";
      }
    }

    sync();
    draw();
    return {
      reveal: function (v) { revealed = v === undefined ? true : v; sync(); draw(); }
    };
  }

  global.FlattenLossViz = { mount: mount };
})(window);
