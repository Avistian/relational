/**
 * Join-and-flatten: how a customer's related orders collapse into ONE feature row.
 *
 * ONE mechanism — the flatten = join + aggregate. Left: a chosen customer's nested
 * orders (a small entity tree). Right: after "flatten", one row at the CUSTOMER grain
 * whose columns are AGGREGATES over that customer's orders (count, sum, mean). The
 * caption names what the aggregate keeps and what it silently discards (an L035 hook).
 *
 * Two customers under the same knob (same transform, different entity) show every
 * customer maps to the same fixed-width row regardless of how many orders it has.
 *
 * Data (baked; matches the L034 worked example, orders up to cutoff t):
 *   C1 (Ada):  3 orders, amounts [40, 55, 30] -> n=3, total=125, avg≈41.7, max=55
 *   C7 (Bo):   1 order,  amounts [120]         -> n=1, total=120, avg=120,  max=120
 *
 * Expected states:
 *   - default: customer C1, un-flattened (nested orders shown, right side empty/prompt)
 *   - toggle "flatten": right shows the single aggregated row for the current customer
 *   - customer switch: recomputes the tree + (if flattened) the aggregate row
 *
 * Usage: JoinFlattenViz.mount(container, {})
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

  var CUSTOMERS = {
    C1: { id: "C1", who: "Ada", segment: "retail", orders: [
      { oid: "o12", amount: 40 }, { oid: "o30", amount: 55 }, { oid: "o44", amount: 30 } ] },
    C7: { id: "C7", who: "Bo", segment: "retail", orders: [
      { oid: "o51", amount: 120 } ] }
  };

  function aggregate(c) {
    var n = c.orders.length;
    var total = c.orders.reduce(function (s, o) { return s + o.amount; }, 0);
    return {
      customer_id: c.id,
      segment: c.segment,
      n_orders: n,
      total_spend: total,
      avg_basket: round1(total / n),
      max_basket: c.orders.reduce(function (m, o) { return Math.max(m, o.amount); }, 0)
    };
  }

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "jf-viz";
    var cur = "C1";
    var flat = false;

    var ctl = document.createElement("div");
    ctl.className = "jf-ctl";

    var flatBtn = document.createElement("button");
    flatBtn.textContent = "Flatten @ customer grain";
    flatBtn.addEventListener("click", function () { flat = !flat; syncBtns(); draw(); });
    ctl.appendChild(flatBtn);

    var sep = document.createElement("span"); sep.className = "jf-sep"; sep.textContent = "customer:";
    ctl.appendChild(sep);

    ["C1", "C7"].forEach(function (id) {
      var b = document.createElement("button");
      b.className = "jf-cust";
      b.textContent = id + " (" + CUSTOMERS[id].who + ", " + CUSTOMERS[id].orders.length + " orders)";
      b.addEventListener("click", function () { cur = id; syncBtns(); draw(); });
      ctl.appendChild(b);
    });
    container.appendChild(ctl);

    function syncBtns() {
      flatBtn.className = flat ? "jf-on" : "";
      Array.prototype.forEach.call(ctl.children, function (ch) {
        if (ch.className && ch.className.indexOf("jf-cust") !== -1) {
          ch.className = "jf-cust" + (ch.textContent.indexOf(cur + " ") === 0 ? " jf-on" : "");
        }
      });
    }

    var W = 640, H = 300;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", class: "jf-svg" });
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "jf-readout";
    container.appendChild(readout);

    var BLUE = "#2e6fb0", GREEN = "#1e6b3c";

    function box(x, y, w, h, fill, stroke) {
      return el("rect", { x: x, y: y, width: w, height: h, rx: 6,
        fill: fill, stroke: stroke || fill, "stroke-width": 1.4 });
    }
    function label(x, y, txt, cls, fill, anchor) {
      var t = el("text", { x: x, y: y, class: cls, "text-anchor": anchor || "start" });
      if (fill) t.setAttribute("fill", fill);
      t.textContent = txt; return t;
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var c = CUSTOMERS[cur];
      var agg = aggregate(c);

      var defs = el("defs", {});
      var mk = el("marker", { id: "jf-arrow", markerWidth: 9, markerHeight: 9, refX: 7, refY: 3,
        orient: "auto", markerUnits: "strokeWidth" });
      mk.appendChild(el("path", { d: "M0,0 L7,3 L0,6 Z", fill: "var(--fg, #222)" }));
      defs.appendChild(mk);
      svg.appendChild(defs);

      // ----- LEFT: nested records (customer -> its orders) -----
      svg.appendChild(label(16, 20, "RAW: one customer, many orders", "jf-lane"));
      // customer node
      var cy = 40;
      svg.appendChild(box(16, cy, 150, 46, BLUE));
      svg.appendChild(label(26, cy + 19, "customer " + c.id, "jf-nhead", "#fff"));
      svg.appendChild(label(26, cy + 36, c.who + " · " + c.segment, "jf-nsub", "#fff"));
      // order children
      var ox = 40, oy0 = 118, oh = 34, ogap = 8;
      c.orders.forEach(function (o, i) {
        var oy = oy0 + i * (oh + ogap);
        // connector from customer to this order
        svg.appendChild(el("path", {
          d: "M" + (16 + 20) + "," + (cy + 46) + " L" + (16 + 20) + "," + (oy + oh / 2) + " L" + ox + "," + (oy + oh / 2),
          fill: "none", stroke: "var(--muted)", "stroke-width": 1.2 }));
        svg.appendChild(box(ox, oy, 150, oh, "#fff", GREEN));
        svg.appendChild(label(ox + 10, oy + 15, "order " + o.oid, "jf-orow", GREEN));
        svg.appendChild(label(ox + 10, oy + 28, "amount $" + o.amount, "jf-oamt", "var(--muted)"));
      });

      // ----- big join+aggregate arrow -----
      var midY = 150;
      var arrOn = flat;
      svg.appendChild(el("path", { d: "M212," + midY + " L300," + midY,
        fill: "none", stroke: arrOn ? "var(--fg, #222)" : "var(--border)",
        "stroke-width": arrOn ? 2.6 : 1.4, "marker-end": "url(#jf-arrow)", opacity: arrOn ? 1 : 0.6 }));
      svg.appendChild(label(256, midY - 10, "JOIN +", "jf-op", arrOn ? "#b0632e" : "var(--muted)", "middle"));
      svg.appendChild(label(256, midY + 22, "AGGREGATE", "jf-op", arrOn ? "#b0632e" : "var(--muted)", "middle"));

      // ----- RIGHT: the flattened row (or a prompt) -----
      svg.appendChild(label(324, 20, "FLAT: one row per customer", "jf-lane"));
      if (!flat) {
        svg.appendChild(label(324, 150, "Press \u201cFlatten\u201d \u2192", "jf-prompt", "var(--muted)"));
        readout.innerHTML =
          "<strong>The raw shape.</strong> Customer <code>" + c.id + "</code> has <strong>" +
          c.orders.length + " order" + (c.orders.length === 1 ? "" : "s") + "</strong>. A model wants " +
          "<em>one row per customer</em>, but the orders live in another table at a different grain. Press " +
          "<em>Flatten</em> to join the orders in and aggregate them into columns.";
        return;
      }

      // aggregated row: header + one data row of columns
      var cols = [
        { k: "customer_id", v: agg.customer_id, key: true },
        { k: "segment", v: agg.segment, key: false },
        { k: "n_orders", v: agg.n_orders, agg: true },
        { k: "total_spend", v: agg.total_spend, agg: true },
        { k: "avg_basket", v: agg.avg_basket, agg: true },
        { k: "max_basket", v: agg.max_basket, agg: true }
      ];
      var rx = 324, rw = 300, rowY = 60, ch = 30;
      cols.forEach(function (col, i) {
        var y = rowY + i * ch;
        svg.appendChild(el("rect", { x: rx, y: y, width: 150, height: ch,
          fill: "var(--bg-soft)", stroke: "var(--border)", "stroke-width": 0.8 }));
        svg.appendChild(el("rect", { x: rx + 150, y: y, width: rw - 150, height: ch,
          fill: col.agg ? "#eaf4ec" : "#fff", stroke: "var(--border)", "stroke-width": 0.8 }));
        svg.appendChild(label(rx + 8, y + ch / 2 + 4, col.k, "jf-fk2",
          col.agg ? GREEN : "var(--fg, #222)"));
        svg.appendChild(label(rx + 158, y + ch / 2 + 4, String(col.v), "jf-fv",
          col.agg ? GREEN : "var(--fg, #222)"));
      });
      svg.appendChild(label(rx, rowY + cols.length * ch + 16,
        "green = aggregate columns synthesised from the orders", "jf-note", "var(--muted)"));

      readout.innerHTML =
        "<strong>Flattened to the customer grain.</strong> The " + c.orders.length + " order" +
        (c.orders.length === 1 ? "" : "s") + " of <code>" + c.id + "</code> collapse into aggregate " +
        "columns: <code>n_orders=" + agg.n_orders + "</code>, <code>total_spend=" + agg.total_spend +
        "</code>, <code>avg_basket=" + agg.avg_basket + "</code>, <code>max_basket=" + agg.max_basket +
        "</code>. Every customer now yields the <em>same fixed-width row</em> no matter how many orders it " +
        "had \u2014 that uniformity is what a model needs. But notice what the mean and count <em>forget</em>: " +
        "<em>which</em> products, the <em>order</em> the baskets came in, the identity of each event. " +
        "Switch between C1 and C7 \u2014 same columns, and (as L035 shows) different histories can collapse to " +
        "the same numbers. That lost structure is the join's hidden cost.";
    }

    syncBtns();
    draw();
    return {
      flatten: function (v) { flat = v === undefined ? true : v; syncBtns(); draw(); },
      setCustomer: function (id) { cur = id; syncBtns(); draw(); }
    };
  }

  global.JoinFlattenViz = { mount: mount };
})(window);
