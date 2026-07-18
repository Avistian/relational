// Headless mount checks for the L031 viz assets (no jsdom):
//   assets/encoding-taxonomy-viz.js, assets/target-leak-viz.js, assets/embedding-space-viz.js
// Verifies mount, the encoding-scheme toggle + per-scheme readouts, the naive/oof leak toggle with real
// AUCs, and the credit_g/Rossmann embedding-space toggle.
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
load("encoding-taxonomy-viz.js");
load("target-leak-viz.js");
load("embedding-space-viz.js");
const ETX = global.window.EncodingTaxonomyViz;
const TLK = global.window.TargetLeakViz;
const ESP = global.window.EmbeddingSpaceViz;

function walk(node, pred, acc) {
  (node.children || []).forEach((c) => { if (pred(c)) acc.push(c); walk(c, pred, acc); });
  return acc;
}
let pass = true;
function check(name, cond) { console.log((cond ? "ok  " : "FAIL") + "  " + name); if (!cond) pass = false; }

// ---- encoding-taxonomy-viz ----
let c = bind(makeEl("div"));
const et = ETX.mount(c, {});
let ro = c.children.find((x) => x._cls.has("etx-readout"));
check("ETX: mounts with a readout", !!ro);
check("ETX: one-hot readout names the +54 column cost", ro.innerHTML.includes("+54"));
et.set("ordinal");
check("ETX: ordinal readout names the false order + 0.782→0.739 drop", ro.innerHTML.includes("false order") && ro.innerHTML.includes("0.739"));
et.set("target");
check("ETX: target readout is the smoothed mean + leak warning", ro.innerHTML.includes("leak") && ro.innerHTML.includes("1-D"));
et.set("embedding");
check("ETX: embedding readout names learned dense vector + Guo & Berkhahn", ro.innerHTML.includes("learned") && ro.innerHTML.includes("Berkhahn"));

// ---- target-leak-viz ----
c = bind(makeEl("div"));
const tl = TLK.mount(c, {});
ro = c.children.find((x) => x._cls.has("tlk-readout"));
check("TLK: naive readout shows the 0.891 leak AUC", ro.innerHTML.includes("0.891"));
let flags = walk(c, (x) => x._cls && x._cls.has("tlk-flag"), []);
check("TLK: naive rows flag 'its own label'", flags.length === 5);
tl.set("oof");
check("TLK: oof readout shows the honest 0.504 chance AUC", ro.innerHTML.includes("0.504"));
check("TLK: oof rows fall back to the 0.70 prior", walk(c, (x) => x._cls && x._cls.has("tlk-safe"), []).length === 5);

// ---- embedding-space-viz ----
c = bind(makeEl("div"));
const es = ESP.mount(c, {});
let svg = c.children.find((x) => x.tag === "svg");
check("ESP: mounts an svg", !!svg);
let dots = walk(svg, (x) => x.tag === "circle", []);
check("ESP: purpose view plots all 10 purpose levels", dots.length === 10);
ro = c.children.find((x) => x._cls.has("esp-readout"));
check("ESP: purpose readout is honest about the loose/tie geometry", ro.innerHTML.toLowerCase().includes("tie"));
es.set("states");
dots = walk(svg, (x) => x.tag === "circle", []);
check("ESP: states view plots the 11 German states", dots.length === 11);
check("ESP: states readout ties embeddings to learned structure/bridge", ro.innerHTML.toLowerCase().includes("representation"));

console.log(pass ? "\nALL VIZ CHECKS PASS" : "\nVIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
