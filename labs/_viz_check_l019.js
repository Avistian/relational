// Headless mount check for assets/biases-viz.js (no jsdom).
// Verifies mount, tile-count changes the checker rect count, the NN board carries the blur
// filter, and rotation mode produces the staircase polyline + verified readouts.
const fs = require("fs");
const path = require("path");

function makeEl(tag) {
  return {
    tag, children: [], attrs: {}, style: {}, _text: "", _html: "", _cls: new Set(),
    listeners: {},
    set className(v) { this._cls = new Set(String(v).split(/\s+/).filter(Boolean)); },
    get className() { return [...this._cls].join(" "); },
    classList: {
      add(...c) { c.forEach((x) => this._owner._cls.add(x)); },
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
    removeChild(c) { this.children = this.children.filter((x) => x !== c); return c; },
    get firstChild() { return this.children[0] || null; },
    addEventListener(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); },
    click() { (this.listeners.click || []).forEach((fn) => fn()); },
    querySelectorAll() { return []; },
  };
}
function bind(el) { el.classList._owner = el; return el; }

global.document = {
  createElement: (t) => bind(makeEl(t)),
  createElementNS: (_ns, t) => bind(makeEl(t)),
};
global.window = {};

const code = fs.readFileSync(path.join(__dirname, "..", "assets", "biases-viz.js"), "utf8");
eval(code);
const BiasesViz = global.window.BiasesViz;
if (!BiasesViz || typeof BiasesViz.mount !== "function") throw new Error("BiasesViz.mount missing");

const container = bind(makeEl("div"));
BiasesViz.mount(container, { caption: "test" });

const ctl = container.children.find((c) => c.className.includes("bv-ctl"));
const sub = container.children.find((c) => c.className.includes("bv-sub"));
const svg = container.children.find((c) => c.tag === "svg");
const readout = container.children.find((c) => c.className.includes("bv-readout"));
const [bIrreg, bRot] = ctl.children;

function walk(node, pred, acc) {
  (node.children || []).forEach((c) => { if (pred(c)) acc.push(c); walk(c, pred, acc); });
  return acc;
}
function countRects() {
  return walk(svg, (c) => c.tag === "rect" && (c.attrs.fill === "#2e6fb0" || c.attrs.fill === "#dce9f5"), []).length;
}
function hasBlurGroup() {
  return walk(svg, (c) => c.tag === "g" && c.attrs.filter === "url(#bias-blur)", []).length === 1;
}
function hasStaircase() {
  return walk(svg, (c) => c.tag === "polyline", []).length === 1;
}
function subBtn(txt) { return sub.children.find((b) => b.textContent === txt); }

let pass = true;
function check(name, cond) { console.log((cond ? "ok  " : "FAIL") + "  " + name); if (!cond) pass = false; }

check("mounts with svg + 2 mode buttons", !!svg && ctl.children.length === 2);
check("irregular 2x2: 8 checker rects (4/board)", countRects() === 8 && readout.innerHTML.includes("1.000"));
subBtn("8×8").click();
check("irregular 8x8: 128 checker rects", countRects() === 128 && readout.innerHTML.includes("0.837"));
check("NN board carries blur filter", hasBlurGroup());

bRot.click();
check("rotation axis-aligned: no staircase, readout 0.998", !hasStaircase() && readout.innerHTML.includes("0.998"));
subBtn("Rotated").click();
check("rotation rotated: staircase polyline, readout 0.831", hasStaircase() && readout.innerHTML.includes("0.831"));
bIrreg.click();
check("back to irregular keeps tiles=8 (128 rects)", countRects() === 128 && subBtn("8×8").className.includes("bv-on"));

console.log(pass ? "\nALL VIZ CHECKS PASS" : "\nVIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
