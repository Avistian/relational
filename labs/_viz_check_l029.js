// Headless mount checks for the L029 viz assets (no jsdom):
//   assets/cash-search-viz.js, assets/ensemble-select-viz.js, assets/automl-bakeoff-viz.js
// Verifies mount, the CASH scatter + best-so-far readout, the single-vs-ensemble toggle, and the
// default-vs-tuned-vs-AutoML bake-off with its "tuning is the jump / AutoML ties" readout.
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
load("cash-search-viz.js");
load("ensemble-select-viz.js");
load("automl-bakeoff-viz.js");
const CSV = global.window.CashSearchViz;
const ESV = global.window.EnsembleSelectViz;
const ABO = global.window.AutomlBakeoffViz;

function walk(node, pred, acc) {
  (node.children || []).forEach((c) => { if (pred(c)) acc.push(c); walk(c, pred, acc); });
  return acc;
}
let pass = true;
function check(name, cond) { console.log((cond ? "ok  " : "FAIL") + "  " + name); if (!cond) pass = false; }

// ---- cash-search-viz ----
let c = bind(makeEl("div"));
const ch = CSV.mount(c, {});
let svg = c.children.find((x) => x.tag === "svg");
check("CSV: mounts an svg", !!svg);
let dots = walk(svg, (x) => x.tag === "circle", []);
check("CSV: draws all 40 evaluated configs", dots.length === 40);
let ro = c.children.find((x) => x._cls.has("csv-readout"));
check("CSV: default readout names CASH + the best-so-far climb", ro.innerHTML.includes("CASH") && ro.innerHTML.includes("0.796 → 0.817"));
check("CSV: default readout names the winning algorithm", ro.innerHTML.includes("HistGB"));
let best = walk(svg, (x) => x.tag === "path" && (x.attrs.class || "").includes("csv-best"), []);
check("CSV: draws a best-so-far step line", best.length === 1);
ch.setAlgo("histgb");
check("CSV: filter to HistGB reports its 11 configs", ro.innerHTML.includes("HistGB") && ro.innerHTML.includes("11"));

// ---- ensemble-select-viz ----
c = bind(makeEl("div"));
const eh = ESV.mount(c, {});
svg = c.children.find((x) => x.tag === "svg");
check("ESV: mounts an svg", !!svg);
ro = c.children.find((x) => x._cls.has("esv-readout"));
check("ESV: single mode shows the single-best 0.824", ro.innerHTML.includes("0.824"));
eh.setMode("ensemble");
check("ESV: ensemble mode shows 0.831 + the +0.007 gain + Caruana", ro.innerHTML.includes("0.831") && ro.innerHTML.includes("+0.007") && ro.innerHTML.includes("Caruana"));
let segTexts = walk(svg, (x) => x.tag === "text" && /×\d/.test(x._text), []);
check("ESV: ensemble draws the 3-algorithm blend composition", segTexts.length >= 2);

// ---- automl-bakeoff-viz ----
c = bind(makeEl("div"));
const ah = ABO.mount(c, {});
svg = c.children.find((x) => x.tag === "svg");
check("ABO: mounts an svg", !!svg);
let bars = walk(svg, (x) => x.tag === "rect" && (x.attrs.class || "").includes("abo-bar"), []);
check("ABO: draws 3 model bars", bars.length === 3);
ro = c.children.find((x) => x._cls.has("abo-readout"));
check("ABO: default readout shows the three means", ro.innerHTML.includes("0.775") && ro.innerHTML.includes("0.806") && ro.innerHTML.includes("0.803"));
check("ABO: default readout — tuning is the jump (+0.031), AutoML ties", ro.innerHTML.includes("+0.031") && ro.innerHTML.toLowerCase().includes("ties"));
ah.highlight("tuned");
check("ABO: highlight tuned shows its mean±sd", ro.innerHTML.includes("0.806") && ro.innerHTML.includes("0.016"));

console.log(pass ? "\nALL VIZ CHECKS PASS" : "\nVIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
