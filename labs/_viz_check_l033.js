// Headless mount checks for the L033 viz asset (no jsdom):
//   assets/fe-returns-viz.js
// Verifies mount, the two real curves, the stop marker, the model toggles, the
// noise-band toggle, and the readout's within-noise / peak logic.
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
load("fe-returns-viz.js");
const FER = global.window.FeReturnsViz;

function walk(node, pred, acc) {
  (node.children || []).forEach((c) => { if (pred(c)) acc.push(c); walk(c, pred, acc); });
  return acc;
}
let pass = true;
function check(name, cond) { console.log((cond ? "ok  " : "FAIL") + "  " + name); if (!cond) pass = false; }

let c = bind(makeEl("div"));
const api = FER.mount(c, {});
let svg = c.children.find((x) => x.tag === "svg");
let ro = c.children.find((x) => x._cls.has("fer-readout"));
check("FER: mounts an svg + readout", !!svg && !!ro);

// two lines + two bands by default (both models, band on)
const isLine = (p) => Number(p.attrs["stroke-width"]) === 2.4;
const isBand = (p) => Number(p.attrs.opacity) === 0.12;
let paths = walk(svg, (x) => x.tag === "path", []);
check("FER: default draws two curves", paths.filter(isLine).length === 2);
check("FER: default draws two noise bands", paths.filter(isBand).length === 2);

// stop marker text present (class set via setAttribute -> attrs.class)
let stop = walk(svg, (x) => x.tag === "text" && x.attrs.class === "fer-stop", []);
check("FER: 'stop here' marker rendered", stop.length === 1 && stop[0]._text.toLowerCase().includes("stop"));

// default k = 3 (peak): readout flags peak + within noise for GBDT
check("FER: default k=3 readout names the peak", ro.innerHTML.toLowerCase().includes("peak"));
check("FER: default k=3 GBDT delta is within noise", ro.innerHTML.toLowerCase().includes("within noise"));

// move past the peak: k=8 should show GBDT below baseline (negative delta) + 'not paying off'
api.setK(8);
check("FER: k=8 readout says past the peak", ro.innerHTML.toLowerCase().includes("not paying off"));
check("FER: k=8 shows a negative GBDT delta", ro.innerHTML.includes("-0.0"));

// model toggles
api.setModel("gbdt");
paths = walk(c.children.find((x) => x.tag === "svg"), (x) => x.tag === "path", []);
check("FER: gbdt-only draws a single curve", paths.filter(isLine).length === 1);
api.setModel("linear");
ro = c.children.find((x) => x._cls.has("fer-readout"));
check("FER: linear-only readout mentions Linear", ro.innerHTML.includes("Linear"));

// band toggle removes the shaded bands
api.setModel("both");
api.toggleBand();
paths = walk(c.children.find((x) => x.tag === "svg"), (x) => x.tag === "path", []);
check("FER: toggling band off removes shaded bands", paths.filter(isBand).length === 0);

console.log(pass ? "\nALL VIZ CHECKS PASS" : "\nVIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
