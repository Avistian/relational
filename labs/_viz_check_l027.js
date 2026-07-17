// Headless mount checks for assets/uninformative-add-viz.js + assets/uninformative-mechanism-viz.js
// (no jsdom). Verifies mount, the accuracy curve states (MLP wins clean, GBT overtakes at k=100),
// the slider readout, and the gating panels (tree 117× gate, MLP 1.8× no-gate).
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
load("uninformative-add-viz.js");
load("uninformative-mechanism-viz.js");
const UAV = global.window.UninformativeAddViz;
const UMV = global.window.UninformativeMechanismViz;

function walk(node, pred, acc) {
  (node.children || []).forEach((c) => { if (pred(c)) acc.push(c); walk(c, pred, acc); });
  return acc;
}
let pass = true;
function check(name, cond) { console.log((cond ? "ok  " : "FAIL") + "  " + name); if (!cond) pass = false; }

// ---- uninformative-add-viz (accuracy curve) ----
let c = bind(makeEl("div"));
const ah = UAV.mount(c, {});
let svg = c.children.find((x) => x.tag === "svg");
check("UAV: mounts an svg", !!svg);
let paths = walk(svg, (x) => x.tag === "path", []);
check("UAV: draws 3 model lines", paths.length === 3);
let ro = c.children.find((x) => x._cls.has("ua-readout"));
check("UAV: default k=0, MLP wins clean (0.986)", ro.innerHTML.includes("k = 0") && ro.innerHTML.includes("0.986"));
check("UAV: default gap is MLP-favoured (−0.040)", ro.innerHTML.includes("-0.040"));
ah.setIdx(5);
check("UAV: k=100 shows GBT overtaking (gap +0.012)", ro.innerHTML.includes("k = 100") && ro.innerHTML.includes("+0.012"));
check("UAV: k=100 MLP dropped to 0.902", ro.innerHTML.includes("0.902"));

// ---- uninformative-mechanism-viz (gating panels) ----
c = bind(makeEl("div"));
UMV.mount(c, {});
svg = c.children.find((x) => x.tag === "svg");
check("UMV: mounts an svg", !!svg);
let rects = walk(svg, (x) => x.tag === "rect", []);
check("UMV: draws 4 bars (2 panels × informative/junk)", rects.length === 4);
let titles = walk(svg, (x) => x.tag === "text" && (x.attrs.class || "").includes("um-title"), []).map((t) => t._text);
check("UMV: labels both panels (tree gain + MLP weight)",
  titles.some((t) => /gain/i.test(t)) && titles.some((t) => /weight/i.test(t)));
let ratios = walk(svg, (x) => x.tag === "text" && (x.attrs.class || "").includes("um-ratio"), []).map((t) => t._text);
check("UMV: tree panel shows a ~118× gate", ratios.some((t) => /118×/.test(t)));
check("UMV: MLP panel shows a ~2× (no) gate", ratios.some((t) => /2×/.test(t)));
ro = c.children.find((x) => x._cls.has("um-readout"));
check("UMV: readout contrasts gate vs no gate", ro.innerHTML.includes("gate") && ro.innerHTML.includes("no gate"));

console.log(pass ? "\nALL VIZ CHECKS PASS" : "\nVIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
