// Headless mount checks for the L028 viz assets (no jsdom):
//   assets/resnet-block-viz.js, assets/depth-trainability-viz.js, assets/baseline-bakeoff-viz.js
// Verifies mount, the skip ON/OFF states, the train/test depth curves + degradation readout, and
// the metric-toggle bake-off with its "no universal winner / tie" readout.
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
load("resnet-block-viz.js");
load("depth-trainability-viz.js");
load("baseline-bakeoff-viz.js");
const RNB = global.window.ResNetBlockViz;
const DTR = global.window.DepthTrainabilityViz;
const BKO = global.window.BaselineBakeoffViz;

function walk(node, pred, acc) {
  (node.children || []).forEach((c) => { if (pred(c)) acc.push(c); walk(c, pred, acc); });
  return acc;
}
let pass = true;
function check(name, cond) { console.log((cond ? "ok  " : "FAIL") + "  " + name); if (!cond) pass = false; }

// ---- resnet-block-viz ----
let c = bind(makeEl("div"));
const rh = RNB.mount(c, {});
let svg = c.children.find((x) => x.tag === "svg");
check("RNB: mounts an svg", !!svg);
let ro = c.children.find((x) => x._cls.has("rnb-readout"));
check("RNB: default skip ON -> output x + f(x)", ro.innerHTML.includes("x + f(x)") && ro.innerHTML.includes("identity"));
let texts = walk(svg, (x) => x.tag === "text", []);
check("RNB: draws BatchNorm op box", texts.some((t) => t._text === "BatchNorm"));
check("RNB: draws two Linear op boxes", texts.filter((t) => t._text === "Linear").length === 2);
let plus = texts.find((t) => t._text === "+");
check("RNB: skip ON draws a + (add) node", !!plus);
rh.setSkip(false);
check("RNB: skip OFF readout warns of degradation (train falls)", ro.innerHTML.includes("degrade") && ro.innerHTML.toLowerCase().includes("training"));
texts = walk(svg, (x) => x.tag === "text", []);
check("RNB: skip OFF removes the + node", !texts.some((t) => t._text === "+"));

// ---- depth-trainability-viz ----
c = bind(makeEl("div"));
const dh = DTR.mount(c, {});
svg = c.children.find((x) => x.tag === "svg");
check("DTR: mounts an svg", !!svg);
let polys = walk(svg, (x) => x.tag === "polyline", []);
check("DTR: default draws 2 test lines (plain + resnet)", polys.length === 2);
ro = c.children.find((x) => x._cls.has("dtr-readout"));
check("DTR: default readout shows plain degrades 0.917 -> 0.866", ro.innerHTML.includes("0.917") && ro.innerHTML.includes("0.866"));
check("DTR: default readout shows resnet flat 0.912 -> 0.899", ro.innerHTML.includes("0.912") && ro.innerHTML.includes("0.899"));
dh.setTrain(true);
polys = walk(svg, (x) => x.tag === "polyline", []);
check("DTR: train toggle adds 2 dashed lines (4 total)", polys.length === 4);
check("DTR: train readout shows degradation is optimization not overfitting", ro.innerHTML.includes("1.000 → 0.927") && ro.innerHTML.toLowerCase().includes("not overfitting"));

// ---- baseline-bakeoff-viz ----
c = bind(makeEl("div"));
const bh = BKO.mount(c, {});
svg = c.children.find((x) => x.tag === "svg");
check("BKO: mounts an svg", !!svg);
let rects = walk(svg, (x) => x.tag === "rect", []);
// 1 plot bg + 3 bars = 4
check("BKO: draws 3 metric bars", rects.length >= 4);
ro = c.children.find((x) => x._cls.has("bko-readout"));
check("BKO: default AUC readout shows the MLP≈ResNet tie + GBDT ahead", ro.innerHTML.includes("0.752") && ro.innerHTML.includes("0.743") && ro.innerHTML.includes("0.793"));
check("BKO: default AUC readout calls it a tie / not a headline winner", ro.innerHTML.toLowerCase().includes("tie") || ro.innerHTML.includes("overlap"));
bh.setMetric("acc");
check("BKO: accuracy readout shows acc numbers + no universal winner", ro.innerHTML.includes("0.734") && ro.innerHTML.includes("0.754") && ro.innerHTML.toLowerCase().includes("no universal winner"));

console.log(pass ? "\nALL VIZ CHECKS PASS" : "\nVIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
