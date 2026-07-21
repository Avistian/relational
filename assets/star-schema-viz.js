/**
 * Star-schema anatomy: a 3-table relational schema (customers, orders, order_items)
 * drawn as boxes with PK/FK columns and the foreign-key edges between them.
 *
 * ONE mechanism — the STRUCTURE of a relational schema (fact/dimension, PK/FK, grain,
 * one-to-many cardinality) — with 3 relationship buttons that highlight one FK edge and
 * update the readout. Default ("whole schema") shows all edges undimmed.
 *
 * The schema it draws (the toy DB used throughout L034):
 *   customers(customer_id PK, signup_date, segment)
 *        ^ 1
 *        | customer_id (FK)   [one customer → many orders]
 *   orders(order_id PK, customer_id FK, order_ts, status)
 *        ^ 1
 *        | order_id (FK)      [one order → many line items]
 *   order_items(item_id PK, order_id FK, product_id FK, qty, price)
 *
 * Expected states:
 *   - rel "cust-ord"  highlights customers ← orders (customer_id), reads "1 : many"
 *   - rel "ord-item"  highlights orders ← order_items (order_id), reads "1 : many"
 *   - rel "item-prod" highlights the dangling product_id FK (dimension not drawn)
 *   - rel "all"       (default) all three edges lit; readout names fact vs dimension + grain
 *
 * Usage: StarSchemaViz.mount(container, {})
 */
(function (global) {
  "use strict";

  var svgNS = "http://www.w3.org/2000/svg";
  function el(name, attrs) {
    var e = document.createElementNS(svgNS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }

  // relationship key -> which edge id(s) to light + the readout
  var RELS = {
    all: {
      label: "Whole schema",
      edges: null,
      html: "<strong>Three tables, two joins.</strong> <code>orders</code> and <code>order_items</code> are " +
        "<em>fact / event</em> tables (they record things that happened, at a timestamp); <code>customers</code> " +
        "is a <em>dimension</em> (the entity we describe and want to predict for). Each arrow is a " +
        "<strong>foreign key</strong> pointing from the many-side back to the one-side's <strong>primary key</strong>. " +
        "The <em>grain</em> — what one row means — differs per table: one row = one customer, one order, or one " +
        "line item. Picking the grain of your <em>training table</em> is the first modelling decision."
    },
    "cust-ord": {
      label: "customers → orders",
      edges: ["cust-ord"],
      html: "<strong>customers 1 : ∞ orders.</strong> <code>orders.customer_id</code> is a foreign key into " +
        "<code>customers.customer_id</code> (its primary key). One customer has <em>many</em> orders — a " +
        "<strong>one-to-many</strong> relationship. This is the join you must aggregate: to get one row per " +
        "customer you have to collapse that customer's whole set of orders into columns."
    },
    "ord-item": {
      label: "orders → order_items",
      edges: ["ord-item"],
      html: "<strong>orders 1 : ∞ order_items.</strong> <code>order_items.order_id</code> is a foreign key into " +
        "<code>orders.order_id</code>. One order contains <em>many</em> line items — the lowest grain in the " +
        "schema. Reaching customer features from here is a <strong>two-hop</strong> path: customer → orders → items."
    },
    "item-prod": {
      label: "order_items → products",
      edges: ["item-prod"],
      html: "<strong>order_items → products (a fourth table, not drawn).</strong> <code>product_id</code> is a " +
        "foreign key into a <code>products</code> dimension. Real schemas fan out into many dimensions; the " +
        "point-in-time flatten has to decide how far along these paths to reach — every hop is a join you must " +
        "write and keep leak-free."
    }
  };

  function mount(container, config) {
    config = config || {};
    container.innerHTML = "";
    container.className = "sss-viz";
    var rel = "all";

    var ctl = document.createElement("div");
    ctl.className = "sss-ctl";
    Object.keys(RELS).forEach(function (key) {
      var b = document.createElement("button");
      b.textContent = RELS[key].label;
      if (key === rel) b.className = "sss-on";
      b.addEventListener("click", function () {
        rel = key;
        Array.prototype.forEach.call(ctl.children, function (c) { c.className = ""; });
        b.className = "sss-on";
        draw();
      });
      ctl.appendChild(b);
    });
    container.appendChild(ctl);

    var W = 640, H = 340;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", class: "sss-svg" });
    container.appendChild(svg);

    var readout = document.createElement("div");
    readout.className = "sss-readout";
    container.appendChild(readout);

    var BLUE = "#2e6fb0", GREEN = "#1e6b3c", GREY = "#6b7280";
    // tables: rows are column names; pk/fk flagged
    function tables() {
      return {
        customers: {
          name: "customers", x: 40, y: 26, w: 200, kind: "DIMENSION", fill: BLUE,
          grain: "one row = one customer",
          cols: [
            { name: "customer_id", tag: "PK" },
            { name: "signup_date", tag: "" },
            { name: "segment", tag: "" }
          ]
        },
        orders: {
          name: "orders", x: 40, y: 172, w: 200, kind: "FACT / EVENT", fill: GREEN,
          grain: "one row = one order",
          cols: [
            { name: "order_id", tag: "PK" },
            { name: "customer_id", tag: "FK" },
            { name: "order_ts", tag: "" },
            { name: "status", tag: "" }
          ]
        },
        order_items: {
          name: "order_items", x: 380, y: 172, w: 210, kind: "FACT / DETAIL", fill: GREEN,
          grain: "one row = one line item",
          cols: [
            { name: "item_id", tag: "PK" },
            { name: "order_id", tag: "FK" },
            { name: "product_id", tag: "FK" },
            { name: "qty", tag: "" },
            { name: "price", tag: "" }
          ]
        }
      };
    }

    var ROWH = 22, HEADH = 34;
    function tableHeight(t) { return HEADH + t.cols.length * ROWH; }
    function colY(t, i) { return t.y + HEADH + i * ROWH + ROWH / 2 + 4; }
    function colIndex(t, name) { for (var i = 0; i < t.cols.length; i++) if (t.cols[i].name === name) return i; return 0; }

    function active(edgeId) {
      var e = RELS[rel].edges;
      return e === null || e.indexOf(edgeId) !== -1;
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var defs = el("defs", {});
      var mk = el("marker", { id: "sss-arrow", markerWidth: 9, markerHeight: 9, refX: 7, refY: 3,
        orient: "auto", markerUnits: "strokeWidth" });
      mk.appendChild(el("path", { d: "M0,0 L7,3 L0,6 Z", fill: "var(--fg, #222)" }));
      defs.appendChild(mk);
      svg.appendChild(defs);

      var T = tables();

      // ---- edges (FK -> PK), drawn under the boxes ----
      // customers.customer_id (PK, one) <- orders.customer_id (FK, many)
      drawEdge("cust-ord",
        T.orders, "customer_id",
        T.customers, "customer_id");
      // orders.order_id (PK, one) <- order_items.order_id (FK, many)
      drawEdge("ord-item",
        T.order_items, "order_id",
        T.orders, "order_id");
      // order_items.product_id (FK) -> products (dangling, not drawn)
      (function () {
        var on = active("item-prod");
        var ft = T.order_items, i = colIndex(ft, "product_id");
        var x1 = ft.x + ft.w, y1 = colY(ft, i);
        var x2 = 620, y2 = y1;
        svg.appendChild(el("path", { d: "M" + x1 + "," + y1 + " L" + x2 + "," + y2,
          fill: "none", stroke: on ? "#b0632e" : "var(--border)", "stroke-dasharray": "4 3",
          "stroke-width": on ? 2.2 : 1, opacity: on ? 1 : 0.5 }));
        var tl = el("text", { x: 624, y: y2 + 4, class: "sss-dim", fill: on ? "#b0632e" : "var(--muted)" });
        tl.textContent = "→ products"; svg.appendChild(tl);
      })();

      // ---- table boxes ----
      Object.keys(T).forEach(function (key) { drawTable(T[key]); });

      readout.innerHTML = RELS[rel].html;
    }

    function drawEdge(id, factTable, factCol, dimTable, dimPk) {
      var on = active(id);
      var fi = colIndex(factTable, factCol), di = colIndex(dimTable, dimPk);
      var fy = colY(factTable, fi), dy = colY(dimTable, di);
      var fx = factTable.x, dx = dimTable.x;
      // route out the left of the fact col, up/over to the left of the dim PK
      var col = on ? "var(--fg, #222)" : "var(--border)";
      var lx = Math.min(fx, dx) - 22;
      var path = "M" + fx + "," + fy + " L" + lx + "," + fy + " L" + lx + "," + dy + " L" + dx + "," + dy;
      svg.appendChild(el("path", { d: path, fill: "none", stroke: col,
        "stroke-width": on ? 2.4 : 1.2, "marker-end": "url(#sss-arrow)", opacity: on ? 1 : 0.45 }));
      // cardinality glyphs: "∞" at the fact (many) end, "1" at the dim (one) end
      var many = el("text", { x: fx - 8, y: fy - 4, class: "sss-card", fill: col, "text-anchor": "end" });
      many.textContent = "∞"; svg.appendChild(many);
      var one = el("text", { x: dx - 8, y: dy - 4, class: "sss-card", fill: col, "text-anchor": "end" });
      one.textContent = "1"; svg.appendChild(one);
    }

    function drawTable(t) {
      var h = tableHeight(t);
      var g = el("g", {});
      // header
      g.appendChild(el("rect", { x: t.x, y: t.y, width: t.w, height: HEADH, rx: 6, fill: t.fill }));
      g.appendChild(el("rect", { x: t.x, y: t.y + HEADH - 8, width: t.w, height: 8, fill: t.fill }));
      var nm = el("text", { x: t.x + 10, y: t.y + 16, class: "sss-tname", fill: "#fff" });
      nm.textContent = t.name; g.appendChild(nm);
      var kd = el("text", { x: t.x + 10, y: t.y + 28, class: "sss-kind", fill: "#fff" });
      kd.textContent = t.kind; g.appendChild(kd);
      // body
      g.appendChild(el("rect", { x: t.x, y: t.y + HEADH, width: t.w, height: h - HEADH, rx: 0,
        fill: "#fff", stroke: t.fill, "stroke-width": 1.4 }));
      t.cols.forEach(function (c, i) {
        var y = t.y + HEADH + i * ROWH;
        if (i > 0) g.appendChild(el("line", { x1: t.x, y1: y, x2: t.x + t.w, y2: y,
          stroke: "var(--border)", "stroke-width": 0.7 }));
        var name = el("text", { x: t.x + 10, y: y + ROWH / 2 + 4, class: "sss-col" });
        name.textContent = c.name; g.appendChild(name);
        if (c.tag) {
          var badge = el("text", { x: t.x + t.w - 10, y: y + ROWH / 2 + 4,
            class: c.tag === "PK" ? "sss-pk" : "sss-fk", "text-anchor": "end" });
          badge.textContent = c.tag; g.appendChild(badge);
        }
      });
      // grain caption under the box
      var gr = el("text", { x: t.x + 4, y: t.y + h + 14, class: "sss-grain" });
      gr.textContent = t.grain; g.appendChild(gr);
      svg.appendChild(g);
    }

    draw();
    return { set: function (r) { rel = r; draw(); } };
  }

  global.StarSchemaViz = { mount: mount };
})(window);
