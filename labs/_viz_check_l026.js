// Headless mount checks for assets/rotation-splits-viz.js + assets/rotation-gap-viz.js (no jsdom).
// Verifies mount, the geometry states (straight splits vs staircase), the grouped bar chart, the
// reversal connectors, and the readouts that carry the mechanism (MLP invariant, tree collapses,
// ranking reverses).
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
load("rotation-splits-viz.js");
load("rotation-gap-viz.js");
const RSV = global.window.RotationSplitsViz;
const RGV = global.window.RotationGapViz;

function walk(node, pred, acc) {
  (node.children || []).forEach((c) => { if (pred(c)) acc.push(c); walk(c, pred, acc); });
  return acc;
}
let pass = true;
function check(name, cond) { console.log((cond ? "ok  " : "FAIL") + "  " + name); if (!cond) pass = false; }

// ---- rotation-splits-viz (geometry: axis-aligned splits vs staircase) ----
let c = bind(makeEl("div"));
const rh = RSV.mount(c, {});
let svg = c.children.find((x) => x.tag === "svg");
check("RSV: mounts an svg", !!svg);
let ro = c.children.find((x) => x._cls.has("rs-readout"));
check("RSV: default is original basis (axis-aligned)", ro.innerHTML.includes("Original basis") && ro.innerHTML.includes("axis-aligned"));
check("RSV: default tree accuracy 0.987 shown", ro.innerHTML.includes("0.987"));
let stair = walk(svg, (x) => x.tag === "polyline", []);
check("RSV: no staircase in original state", stair.length === 0);
let lines = walk(svg, (x) => x.tag === "line", []);
check("RSV: original has straight tree splits (>=2 lines)", lines.length >= 2);
// rotate
rh.setRotated(true);
check("RSV: rotated readout mentions staircase + drop 0.747", ro.innerHTML.includes("Rotated basis") && ro.innerHTML.includes("staircase") && ro.innerHTML.includes("0.747"));
stair = walk(svg, (x) => x.tag === "polyline", []);
check("RSV: rotated state draws a staircase polyline", stair.length === 1);
check("RSV: rotated readout says MLP invariant", ro.innerHTML.includes("invariant"));

// ---- rotation-gap-viz (grouped bars: reversal + MLP invariance) ----
c = bind(makeEl("div"));
const gh = RGV.mount(c, {});
svg = c.children.find((x) => x.tag === "svg");
check("RGV: mounts an svg", !!svg);
let rects = walk(svg, (x) => x.tag === "rect", []);
// 1 plot bg + 8 bars + 2 legend swatches = 11
check("RGV: 8 bars (4 models × original/rotated)", rects.length >= 11);
ro = c.children.find((x) => x._cls.has("rg-readout"));
check("RGV: readout notes ranking reverses", ro.innerHTML.includes("ranking reverses"));
check("RGV: readout notes MLP invariant (+0.008)", ro.innerHTML.includes("+0.008") && ro.innerHTML.includes("invariant"));
check("RGV: readout shows tree collapse 0.987 -> 0.747", ro.innerHTML.includes("0.987") && ro.innerHTML.includes("0.747"));
let dashed = walk(svg, (x) => x.tag === "line" && x.attrs["stroke-dasharray"], []);
check("RGV: no reversal connectors by default", dashed.length === 0);
gh.setReversal(true);
dashed = walk(svg, (x) => x.tag === "line" && x.attrs["stroke-dasharray"], []);
check("RGV: reversal toggle draws 2 connectors", dashed.length === 2);

console.log(pass ? "\nALL VIZ CHECKS PASS" : "\nVIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
