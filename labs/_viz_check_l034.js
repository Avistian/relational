// Headless mount checks for the L034 viz assets (no jsdom):
//   assets/star-schema-viz.js   assets/join-flatten-viz.js
// Verifies mount, the tables/edges/cardinality of the schema viz, and the
// raw->aggregate flatten of the join viz (columns + baked numbers).
const fs = require("fs");
const path = require("path");

function makeEl(tag) {
  const el = {
    tag, children: [], attrs: {}, style: {}, _text: "", _html: "", _cls: new Set(),
    listeners: {}, value: "", type: "",
    set className(v) { this._cls = new Set(String(v).split(/\s+/).filter(Boolean)); },
    get className() { return [...this._cls].join(" "); },
    classList: {
      add(...c) { c.forEach((x) => this._owner._cls.add(x)); },
      remove(...c) { c.forEach((x) => this._owner._cls.delete(x)); },
      toggle(c, on) { on ? this._owner._cls.add(c) : this._owner._cls.delete(c); },
      contains(c) { return this._owner._cls.has(c); },
    },
    set textContent(v) { this._text = String(v); },
    get textContent() { return this._text; },
    set innerHTML(v) { this._html = v; if (v === "") this.children = []; },
    get innerHTML() { return this._html; },
    setAttribute(k, v) { this.attrs[k] = v; },
    getAttribute(k) { return this.attrs[k]; },
    appendChild(c) { this.children.push(c); return c; },
    removeChild(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); return c; },
    get firstChild() { return this.children[0] || null; },
    addEventListener(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); },
    click() { (this.listeners.click || []).forEach((fn) => fn()); },
  };
  return el;
}
function bind(el) { el.classList._owner = el; return el; }
global.document = { createElement: (t) => bind(makeEl(t)), createElementNS: (_ns, t) => bind(makeEl(t)) };
global.window = {};

function load(f) { eval(fs.readFileSync(path.join(__dirname, "..", "assets", f), "utf8")); }
function walk(node, pred, acc) {
  (node.children || []).forEach((c) => { if (pred(c)) acc.push(c); walk(c, pred, acc); });
  return acc;
}
function textOf(svg) { return walk(svg, (x) => x.tag === "text", []).map((t) => t._text); }

let pass = true;
function check(name, cond) { console.log((cond ? "ok  " : "FAIL") + "  " + name); if (!cond) pass = false; }

// ---------- star-schema-viz ----------
load("star-schema-viz.js");
const SSS = global.window.StarSchemaViz;
let c1 = bind(makeEl("div"));
const sssApi = SSS.mount(c1, {});
let svg1 = c1.children.find((x) => x.tag === "svg");
check("SSS: mounts an svg + readout", !!svg1 && !!c1.children.find((x) => x._cls.has("sss-readout")));

let names1 = textOf(svg1);
check("SSS: draws all three tables", ["customers", "orders", "order_items"].every((n) => names1.includes(n)));
check("SSS: shows PK badges", names1.filter((t) => t === "PK").length >= 3);
check("SSS: shows FK badges", names1.filter((t) => t === "FK").length >= 3);
check("SSS: shows one-to-many cardinality glyphs", names1.includes("\u221e") && names1.includes("1"));
check("SSS: labels fact vs dimension", names1.includes("DIMENSION") && names1.some((t) => t.indexOf("FACT") === 0));
check("SSS: default readout names the grain", c1.children.find((x) => x._cls.has("sss-readout")).innerHTML.toLowerCase().includes("grain"));

// selecting a single relationship keeps the schema drawn
sssApi.set("cust-ord");
svg1 = c1.children.find((x) => x.tag === "svg");
check("SSS: cust-ord readout names one-to-many", c1.children.find((x) => x._cls.has("sss-readout")).innerHTML.includes("one-to-many"));

// ---------- join-flatten-viz ----------
load("join-flatten-viz.js");
const JF = global.window.JoinFlattenViz;
let c2 = bind(makeEl("div"));
const jfApi = JF.mount(c2, {});
let svg2 = c2.children.find((x) => x.tag === "svg");
let ro2 = c2.children.find((x) => x._cls.has("jf-readout"));
check("JF: mounts an svg + readout", !!svg2 && !!ro2);

// default (un-flattened): shows the 3 raw orders of C1, no aggregate row yet
let t2 = textOf(svg2);
check("JF: default shows C1's raw orders", t2.filter((t) => t.indexOf("order o") === 0).length === 3);
check("JF: default has no aggregate columns yet", !t2.includes("n_orders"));
check("JF: default readout is the raw-shape prompt", ro2.innerHTML.toLowerCase().includes("raw shape"));

// flatten C1 -> aggregate row with baked numbers (n=3, total=125, avg=41.7, max=55)
jfApi.flatten(true);
svg2 = c2.children.find((x) => x.tag === "svg");
ro2 = c2.children.find((x) => x._cls.has("jf-readout"));
t2 = textOf(svg2);
check("JF: flatten draws the aggregate columns", ["n_orders", "total_spend", "avg_basket", "max_basket"].every((k) => t2.includes(k)));
check("JF: C1 aggregates are correct (3/125/41.7/55)", t2.includes("3") && t2.includes("125") && t2.includes("41.7") && t2.includes("55"));
check("JF: readout notes information lost (L035 hook)", ro2.innerHTML.toLowerCase().includes("forget"));

// switch to C7 (1 order, $120) -> same columns, different numbers
jfApi.setCustomer("C7");
svg2 = c2.children.find((x) => x.tag === "svg");
t2 = textOf(svg2);
check("JF: C7 still yields the same fixed-width columns", ["n_orders", "avg_basket"].every((k) => t2.includes(k)));
check("JF: C7 aggregates are correct (n=1, total=120)", t2.includes("1") && t2.includes("120"));

console.log(pass ? "\nALL VIZ CHECKS PASS" : "\nVIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
