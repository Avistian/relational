// Headless mount checks for assets/benchmark-budget-viz.js + assets/dataset-funnel-viz.js (no jsdom).
// Verifies mount, both curves, the budget marker/dots, the raw/normalized toggle, and the
// funnel stage bars + click-to-detail interaction.
const fs = require("fs");
const path = require("path");

function makeEl(tag) {
  const el = {
    tag, children: [], attrs: {}, style: {}, _text: "", _html: "", _cls: new Set(),
    listeners: {}, value: "", type: "", min: "", max: "",
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
load("benchmark-budget-viz.js");
load("dataset-funnel-viz.js");
const BBV = global.window.BenchmarkBudgetViz;
const DFV = global.window.DatasetFunnelViz;

function walk(node, pred, acc) {
  (node.children || []).forEach((c) => { if (pred(c)) acc.push(c); walk(c, pred, acc); });
  return acc;
}
let pass = true;
function check(name, cond) { console.log((cond ? "ok  " : "FAIL") + "  " + name); if (!cond) pass = false; }

// ---- benchmark-budget-viz ----
let c = bind(makeEl("div"));
const bh = BBV.mount(c, {});
const svg = c.children.find((x) => x.tag === "svg");
check("BBV: mounts an svg", !!svg);
const paths = walk(svg, (x) => x.tag === "path", []);
check("BBV: two curves (GBT + NN)", paths.length === 2);
const readout = c.children.find((x) => x._cls.has("bb-readout"));
check("BBV: default k=1 shows GBT winning by default", readout.innerHTML.includes("k=1") && readout.innerHTML.includes("default"));
check("BBV: default gap positive (GBT>MLP)", readout.innerHTML.includes("gap"));
// move budget to 30
bh.setBudget(30);
check("BBV: k=30 shows fully tuned, GBT still on top", readout.innerHTML.includes("k=30") && readout.innerHTML.includes("still on top"));
// toggle normalized
const btns = walk(c, (x) => x.tag === "button", []);
btns.find((b) => b.textContent.includes("Normalized")).click();
check("BBV: normalized view pill", readout.innerHTML.includes("normalized"));
const dots = walk(svg, (x) => x.tag === "circle", []);
check("BBV: two budget dots", dots.length === 2);

// ---- dataset-funnel-viz ----
c = bind(makeEl("div"));
const dh = DFV.mount(c, {});
const dsvg = c.children.find((x) => x.tag === "svg");
check("DFV: mounts an svg", !!dsvg);
const rects = walk(dsvg, (x) => x.tag === "rect", []);
check("DFV: 7 stage bars", rects.length === 7);
const detail = c.children.find((x) => x._cls.has("df-detail"));
check("DFV: default detail shows the final suite", detail.innerHTML.includes("Benchmark suite"));
// select the 'not too easy' stage (index 5)
dh.select(5);
check("DFV: selecting a stage shows its criterion", detail.innerHTML.includes("Not too easy") && detail.innerHTML.includes("tuned GBT"));

console.log(pass ? "\nALL VIZ CHECKS PASS" : "\nVIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
