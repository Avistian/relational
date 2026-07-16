// Headless mount checks for assets/smoothness-fit-viz.js + assets/smoothness-gap-viz.js (no jsdom).
// Verifies mount, both curve sets, the slider states, the budget/smoothing marker + dots, and the
// readouts that carry the mechanism (MLP over-smooths the raw target; the gap closes when smoothed).
const fs = require("fs");
const path = require("path");

function makeEl(tag) {
  const el = {
    tag, children: [], attrs: {}, style: {}, _text: "", _html: "", _cls: new Set(),
    listeners: {}, value: "", type: "", min: "", max: "", step: "",
    set className(v) { this._cls = new Set(String(v).split(/\s+/).filter(Boolean)); },
    get className() { return [...this._cls].join(" "); },
    classList: {
      add(...c) { c.forEach((x) => this._owner._cls.add(x)); },
      remove(...c) { c.forEach((x) => this._owner._cls.delete(x)); },
      toggle(c, on) { on ? this._owner._cls.add(c) : this._owner._cls.delete(c); },
      contains(c) { return this._owner._cls.has(c); },
    },
    set textContent(v) { this._text = v; },
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
    input() { (this.listeners.input || []).forEach((fn) => fn()); },
  };
  return el;
}
function bind(el) { el.classList._owner = el; return el; }
global.document = { createElement: (t) => bind(makeEl(t)), createElementNS: (_ns, t) => bind(makeEl(t)) };
global.window = {};

function load(f) { eval(fs.readFileSync(path.join(__dirname, "..", "assets", f), "utf8")); }
load("smoothness-fit-viz.js");
load("smoothness-gap-viz.js");
const SFV = global.window.SmoothnessFitViz;
const SGV = global.window.SmoothnessGapViz;

function walk(node, pred, acc) {
  (node.children || []).forEach((c) => { if (pred(c)) acc.push(c); walk(c, pred, acc); });
  return acc;
}
let pass = true;
function check(name, cond) { console.log((cond ? "ok  " : "FAIL") + "  " + name); if (!cond) pass = false; }

// ---- smoothness-fit-viz (1-D staircase vs smooth) ----
let c = bind(makeEl("div"));
const fh = SFV.mount(c, {});
let svg = c.children.find((x) => x.tag === "svg");
check("SFV: mounts an svg", !!svg);
let paths = walk(svg, (x) => x.tag === "path", []);
check("SFV: three curves (target + tree + MLP)", paths.length === 3);
let ro = c.children.find((x) => x._cls.has("sf-readout"));
check("SFV: default h=0 raw, tree wins (MLP over-smooths)", ro.innerHTML.includes("h = 0.000") && ro.innerHTML.includes("over-smooth"));
check("SFV: default MLP/tree error ratio is large (5.30x)", ro.innerHTML.includes("5.30×"));
// slide to the smooth state
fh.setState(2);
check("SFV: smooth state — MLP now edges the tree", ro.innerHTML.includes("h = 0.120") && ro.innerHTML.includes("edges the tree"));
check("SFV: smooth state ratio < 1 (0.19x)", ro.innerHTML.includes("0.19×"));

// ---- smoothness-gap-viz (gap closes with smoothing) ----
c = bind(makeEl("div"));
const gh = SGV.mount(c, {});
svg = c.children.find((x) => x.tag === "svg");
check("SGV: mounts an svg", !!svg);
paths = walk(svg, (x) => x.tag === "path", []);
check("SGV: two curves (GBT + MLP)", paths.length === 2);
ro = c.children.find((x) => x._cls.has("sg-readout"));
check("SGV: default h=0, big positive gap (tree wins big)", ro.innerHTML.includes("h = 0.0") && ro.innerHTML.includes("tree wins big"));
check("SGV: default variance kept 100%", ro.innerHTML.includes("100%"));
let dots = walk(svg, (x) => x.tag === "circle", []);
check("SGV: two markers (GBT + MLP)", dots.length === 2);
// slide to h=1.0 (index 2) where the gap has closed
gh.setIdx(2);
check("SGV: at h=1.0 the gap has closed", ro.innerHTML.includes("h = 1.0") && ro.innerHTML.includes("gap has closed"));

console.log(pass ? "\nALL VIZ CHECKS PASS" : "\nVIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
