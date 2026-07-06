// Headless mount check for assets/stacking-viz.js (no jsdom).
// Minimal DOM shim: verifies mount renders, and OOF stepping / in-sample toggle
// drive the documented states.
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

const document = {
  createElement: (t) => bind(makeEl(t)),
  createElementNS: (_ns, t) => bind(makeEl(t)),
};
global.document = document;
global.window = {};

const code = fs.readFileSync(path.join(__dirname, "..", "assets", "stacking-viz.js"), "utf8");
eval(code);
const StackingViz = global.window.StackingViz;
if (!StackingViz || typeof StackingViz.mount !== "function") throw new Error("StackingViz.mount missing");

const container = bind(makeEl("div"));
StackingViz.mount(container, { caption: "test" });

// find buttons + svg + readout
const buttons = container.children.find((c) => c.className.includes("stk-ctl")).children;
const [bOof, bLeak, bNext, bReset] = buttons;
const svg = container.children.find((c) => c.tag === "svg");
const readout = container.children.find((c) => c.className.includes("stk-readout"));

function countMetaFilled() {
  // meta cells are rects with fill green (#d7ecdd) or red (#f7dcd7)
  return svg.children.filter(
    (c) => c.tag === "rect" && (c.attrs.fill === "#d7ecdd" || c.attrs.fill === "#f7dcd7")
  ).length;
}

let pass = true;
function check(name, cond) { console.log((cond ? "ok  " : "FAIL") + "  " + name); if (!cond) pass = false; }

check("mounts with svg + controls", !!svg && buttons.length === 4);
check("OOF start: fold 0 filled, 3/12", countMetaFilled() === 3 && readout.innerHTML.includes("fold <strong>0"));
bNext.click(); check("after 1 step: 6/12 filled", countMetaFilled() === 6);
bNext.click(); bNext.click();
check("after stepping all folds: 12/12 honest", countMetaFilled() === 12 && readout.innerHTML.includes("Honest"));
bLeak.click();
const allRed = svg.children.filter((c) => c.tag === "rect" && c.attrs.fill === "#f7dcd7").length;
check("in-sample: all 12 leak cells red instantly", allRed === 12 && readout.innerHTML.includes("Leak"));
bOof.click(); check("reset to OOF: back to 3/12", countMetaFilled() === 3);

console.log(pass ? "\nALL VIZ CHECKS PASS" : "\nVIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
