// Headless mount checks for assets/temporal-split-viz.js + assets/drift-viz.js (no jsdom).
// Verifies mount, circle/bar/element counts, the temporal cut line, reshuffle, and the
// drift slider rotating the rule arrow + updating the readout.
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
    set textContent(v) { this._text = v; },
    get textContent() { return this._text; },
    set innerHTML(v) { this._html = v; if (v === "") this.children = []; },
    get innerHTML() { return this._html; },
    setAttribute(k, v) { this.attrs[k] = v; },
    getAttribute(k) { return this.attrs[k]; },
    appendChild(c) { this.children.push(c); return c; },
    get firstChild() { return this.children[0] || null; },
    addEventListener(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); },
    click() { (this.listeners.click || []).forEach((fn) => fn()); },
    querySelectorAll() { return []; },
  };
  return el;
}
function bind(el) { el.classList._owner = el; return el; }
global.document = { createElement: (t) => bind(makeEl(t)), createElementNS: (_ns, t) => bind(makeEl(t)) };
global.window = {};

function load(f) { eval(fs.readFileSync(path.join(__dirname, "..", "assets", f), "utf8")); }
load("temporal-split-viz.js");
load("drift-viz.js");
const TSV = global.window.TemporalSplitViz;
const DV = global.window.DriftViz;

function fire(el, ev) { (el.listeners[ev] || []).forEach((fn) => fn()); }
function walk(node, pred, acc) {
  (node.children || []).forEach((c) => { if (pred(c)) acc.push(c); walk(c, pred, acc); });
  return acc;
}
let pass = true;
function check(name, cond) { console.log((cond ? "ok  " : "FAIL") + "  " + name); if (!cond) pass = false; }

// ---- temporal-split-viz ----
let c = bind(makeEl("div"));
const th = TSV.mount(c, { n: 60, testFrac: 0.2, seed: 7 });
const svg = c.children.find((x) => x.tag === "svg");
check("TSV: mounts an svg", !!svg);
const circles = walk(svg, (x) => x.tag === "circle", []);
// 60 rows x 2 strips + 2 legend dots = 122
check("TSV: 122 circles (60x2 rows + 2 legend)", circles.length === 122);
const dashed = walk(svg, (x) => x.tag === "line" && x.attrs["stroke-dasharray"], []);
check("TSV: exactly one dashed cut line (temporal strip)", dashed.length === 1);
const testDots = walk(svg, (x) => x.tag === "circle" && x.attrs.fill === "#b03a2e", []);
// round(60*0.2)=12 test in random strip + 12 in temporal strip + 1 legend = 25
check("TSV: 25 red/test dots (12 random + 12 temporal + 1 legend)", testDots.length === 25);
check("TSV: cut at index 48 (last 20% of 60)", th.getCut() === 48);
const maskBefore = th.getMask().join("");
const btn = c.children.find((x) => x.className && x.className.includes("tsv-controls")).children[0];
btn.click();
check("TSV: reshuffle changes the random mask", th.getMask().join("") !== maskBefore);
check("TSV: still 122 circles after reshuffle", walk(svg, (x) => x.tag === "circle", []).length === 122);

// ---- drift-viz ----
c = bind(makeEl("div"));
const dh = DV.mount(c, {});
const dsvg = c.children.find((x) => x.tag === "svg");
check("DV: mounts an svg", !!dsvg);
const rects = walk(dsvg, (x) => x.tag === "rect", []);
// 15 bars + 1 bucket highlight = 16
check("DV: 16 rects (15 bars + 1 highlight)", rects.length === 16);
const arrow = walk(dsvg, (x) => x.tag === "line" && x.attrs["stroke-width"] === 3, [])[0];
check("DV: rule arrow present", !!arrow);
const readout = c.children.find((x) => x.className && x.className.includes("dv-readout"));
const arrowAt0 = arrow.attrs.x2;
check("DV: bucket 0 arrow points along x0 (x2 == cx+R)", Math.abs(parseFloat(arrowAt0) - (460 + 66)) < 1e-6);
check("DV: readout shows x0=0.72 at bucket 0", readout.innerHTML.includes("0.72"));
const slider = walk(c, (x) => x.className && x.className.includes("dv-slider"), [])[0];
slider.value = "4"; fire(slider, "input");
check("DV: bucket 4 rotates arrow to x1 axis (x2 ~ cx)", Math.abs(parseFloat(arrow.attrs.x2) - 460) < 1e-3);
check("DV: readout updates to x1=0.71 at bucket 4", readout.innerHTML.includes("0.71"));

console.log(pass ? "\nALL VIZ CHECKS PASS" : "\nVIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
