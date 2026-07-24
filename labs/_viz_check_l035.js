// Headless mount check for the L035 viz asset (no jsdom):
//   assets/flatten-loss-viz.js
// Verifies mount, that BOTH customers' raw histories are drawn, that they collapse
// to the IDENTICAL aggregate row (the collision: 3 / 90 / 30 / 50), and that the
// reveal exposes the distinguishing structure (trend + distinct products).
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

load("flatten-loss-viz.js");
const FL = global.window.FlattenLossViz;
let c = bind(makeEl("div"));
const api = FL.mount(c, {});
let svg = c.children.find((x) => x.tag === "svg");
let ro = c.children.find((x) => x._cls.has("fl-readout"));
check("FL: mounts an svg + readout", !!svg && !!ro);

let t = textOf(svg);
check("FL: draws both customers (Ada + Bo)", t.includes("Ada") && t.includes("Bo"));
check("FL: shows both raw order trails (Jan/Mar/May amounts)",
  t.some((s) => s.indexOf("Jan") === 0) && t.some((s) => s.indexOf("May") === 0));
check("FL: draws the aggregate columns", ["n_orders", "total_spend", "avg_basket", "max_basket"].every((k) => t.includes(k)));
// the collision: both rows carry the identical aggregate values
check("FL: aggregate values are the collision 3 / 90 / 30 / 50",
  t.filter((s) => s === "3").length >= 2 && t.filter((s) => s === "90").length >= 2 &&
  t.filter((s) => s === "30").length >= 2 && t.filter((s) => s === "50").length >= 2);
check("FL: flags the identical/collision row", t.some((s) => s.indexOf("identical") >= 0));
check("FL: default readout names the collision", ro.innerHTML.toLowerCase().includes("collision"));
check("FL: default readout does NOT yet reveal the trend", !ro.innerHTML.toLowerCase().includes("would keep"));

// reveal the lost structure
api.reveal(true);
svg = c.children.find((x) => x.tag === "svg");
ro = c.children.find((x) => x._cls.has("fl-readout"));
t = textOf(svg);
check("FL: reveal shows a rising trend for Ada", t.some((s) => s.indexOf("rising") >= 0));
check("FL: reveal shows a falling trend for Bo", t.some((s) => s.indexOf("falling") >= 0));
check("FL: reveal shows distinct-product counts (3 vs 1)",
  t.some((s) => s.indexOf("3 distinct") >= 0) && t.some((s) => s.indexOf("1 distinct") >= 0));
check("FL: reveal readout explains what a graph keeps", ro.innerHTML.toLowerCase().includes("what a graph would keep"));

console.log(pass ? "\nALL VIZ CHECKS PASS" : "\nVIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
