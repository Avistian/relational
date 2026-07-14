// Headless mount checks for assets/paired-diff-viz.js + assets/cd-diagram-viz.js (no jsdom).
// Verifies mount, fold-dot count, the naive/corrected CI toggle + verdict, the CD-diagram
// nodes/join-bars, and the click-to-highlight interaction.
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
load("paired-diff-viz.js");
load("cd-diagram-viz.js");
const PDV = global.window.PairedDiffViz;
const CDV = global.window.CdDiagramViz;

function walk(node, pred, acc) {
  (node.children || []).forEach((c) => { if (pred(c)) acc.push(c); walk(c, pred, acc); });
  return acc;
}
let pass = true;
function check(name, cond) { console.log((cond ? "ok  " : "FAIL") + "  " + name); if (!cond) pass = false; }

// ---- paired-diff-viz ----
let c = bind(makeEl("div"));
const ph = PDV.mount(c, {});
const svg = c.children.find((x) => x.tag === "svg");
check("PDV: mounts an svg", !!svg);
const dots = walk(svg, (x) => x.tag === "circle", []);
check("PDV: 20 per-fold dots", dots.length === 20);
const readout = c.children.find((x) => x._cls.has("pd-readout"));
check("PDV: default (naive) verdict is SIGNIFICANT p=1.2e-5", readout.innerHTML.includes("SIGNIFICANT") && readout.innerHTML.includes("1.2e-5"));
check("PDV: default pill names the naive paired t", readout.innerHTML.includes("Naive paired t"));
const btns = walk(c, (x) => x.tag === "button", []);
const corrBtn = btns.find((b) => b.textContent.includes("Corrected"));
corrBtn.click();
check("PDV: toggling corrected flips to NOT significant p=0.19", readout.innerHTML.includes("NOT significant") && readout.innerHTML.includes("0.19"));
check("PDV: corrected pill names the corrected resampled t", readout.innerHTML.includes("Corrected resampled t"));
// switch back
btns.find((b) => b.textContent.includes("Naive")).click();
check("PDV: switching back restores the naive verdict", readout.innerHTML.includes("1.2e-5"));

// ---- cd-diagram-viz ----
c = bind(makeEl("div"));
const ch = CDV.mount(c, {});
const csvg = c.children.find((x) => x.tag === "svg");
check("CDV: mounts an svg", !!csvg);
const nodeDots = walk(csvg, (x) => x.tag === "circle", []);
check("CDV: 4 model nodes", nodeDots.length === 4);
const joinBars = walk(csvg, (x) => x.tag === "line" && x.attrs.stroke === "#555", []);
check("CDV: 2 join bars (LogReg–NB and NB–RF–HistGBDT)", joinBars.length === 2);
const cro = c.children.find((x) => x._cls.has("cdv-readout"));
check("CDV: readout shows CD = 1.354 and Friedman p", cro.innerHTML.includes("CD = 1.354") && cro.innerHTML.includes("4e-6"));
// click LogReg via API -> highlight non-different group
ch.select("LogReg");
check("CDV: selecting LogReg reports its non-different group", cro.innerHTML.includes("not") && cro.innerHTML.includes("LogReg"));

console.log(pass ? "\nALL VIZ CHECKS PASS" : "\nVIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
