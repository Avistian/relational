// Headless mount checks for the L032 viz assets (no jsdom):
//   assets/tabtransformer-arch-viz.js, assets/attention-context-viz.js
// Verifies mount, the 4-stage architecture stepper + per-stage readouts, and the attention
// before/after (context-free vs contextual) toggle with the 3-token blend + weight row.
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
load("tabtransformer-arch-viz.js");
load("attention-context-viz.js");
const TTA = global.window.TabTransformerArchViz;
const ATC = global.window.AttentionContextViz;

function walk(node, pred, acc) {
  (node.children || []).forEach((c) => { if (pred(c)) acc.push(c); walk(c, pred, acc); });
  return acc;
}
let pass = true;
function check(name, cond) { console.log((cond ? "ok  " : "FAIL") + "  " + name); if (!cond) pass = false; }

// ---- tabtransformer-arch-viz ----
let c = bind(makeEl("div"));
const ta = TTA.mount(c, {});
let svg = c.children.find((x) => x.tag === "svg");
let ro = c.children.find((x) => x._cls.has("tta-readout"));
check("TTA: mounts an svg + readout", !!svg && !!ro);
let rects = walk(svg, (x) => x.tag === "rect", []);
check("TTA: draws all 9 architecture boxes", rects.length === 9);
check("TTA: default readout summarises the whole net", ro.innerHTML.toLowerCase().includes("whole net"));
ta.set("embed");
check("TTA: embed stage names the L031 entity embedding", ro.innerHTML.toLowerCase().includes("entity embedding"));
ta.set("trans");
check("TTA: transformer stage names contextual embeddings", ro.innerHTML.toLowerCase().includes("contextual"));
ta.set("bypass");
check("TTA: bypass stage says continuous features skip the transformer", ro.innerHTML.toLowerCase().includes("bypass"));
ta.set("head");
check("TTA: head stage names concatenate + MLP", ro.innerHTML.toLowerCase().includes("concatenate") && ro.innerHTML.toLowerCase().includes("mlp"));

// ---- attention-context-viz ----
c = bind(makeEl("div"));
const at = ATC.mount(c, {});
ro = c.children.find((x) => x._cls.has("atc-readout"));
let grid = c.children.find((x) => x._cls.has("atc-grid"));
check("ATC: mounts a grid + readout", !!grid && !!ro);
let cards = walk(grid, (x) => x._cls && x._cls.has("atc-card"), []);
check("ATC: default (after) shows 3 column tokens", cards.length === 3);
check("ATC: exactly one query card highlighted", walk(grid, (x) => x._cls && x._cls.has("atc-q"), []).length === 1);
let weights = walk(grid, (x) => x._cls && x._cls.has("atc-weights"), []);
check("ATC: after view shows the attention weight row", weights.length === 1);
check("ATC: after readout explains the weighted blend", ro.innerHTML.toLowerCase().includes("weighted blend"));
at.set("before");
check("ATC: before view is context-free (no attention weight row)",
  walk(c, (x) => x._cls && x._cls.has("atc-weights"), []).length === 0);
check("ATC: before readout says context-free", ro.innerHTML.toLowerCase().includes("context-free"));

console.log(pass ? "\nALL VIZ CHECKS PASS" : "\nVIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
