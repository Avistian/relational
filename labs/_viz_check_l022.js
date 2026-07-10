// Headless mount checks for assets/leakage-taxonomy-viz.js + assets/repro-collapse-viz.js (no jsdom).
// Verifies mount, chip/family counts, detail-panel selection, and the repro-collapse toggle.
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
    get firstChild() { return this.children[0] || null; },
    addEventListener(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); },
    click() { (this.listeners.click || []).forEach((fn) => fn()); },
    querySelector() { return null; },
    querySelectorAll() { return []; },
  };
  return el;
}
function bind(el) { el.classList._owner = el; return el; }
global.document = { createElement: (t) => bind(makeEl(t)), createElementNS: (_ns, t) => bind(makeEl(t)) };
global.window = {};

function load(f) { eval(fs.readFileSync(path.join(__dirname, "..", "assets", f), "utf8")); }
load("leakage-taxonomy-viz.js");
load("repro-collapse-viz.js");
const LTV = global.window.LeakageTaxonomyViz;
const RCV = global.window.ReproCollapseViz;

function walk(node, pred, acc) {
  (node.children || []).forEach((c) => { if (pred(c)) acc.push(c); walk(c, pred, acc); });
  return acc;
}
let pass = true;
function check(name, cond) { console.log((cond ? "ok  " : "FAIL") + "  " + name); if (!cond) pass = false; }

// ---- leakage-taxonomy-viz ----
let c = bind(makeEl("div"));
const th = LTV.mount(c, {});
const fams = walk(c, (x) => x._cls.has("ltv-fam"), []);
check("LTV: 3 family blocks", fams.length === 3);
const chips = walk(c, (x) => x._cls.has("ltv-chip"), []);
check("LTV: 8 leak-type chips", chips.length === 8);
const detail = c.children.find((x) => x._cls.has("ltv-detail"));
check("LTV: detail defaults to L1.2", detail.innerHTML.includes("L1.2"));
// click the temporal-leakage chip (L3.1)
const l31 = th.chips.find((x) => x.code === "L3.1");
l31.el.click();
check("LTV: clicking L3.1 updates detail to temporal leakage", detail.innerHTML.includes("L3.1") && detail.innerHTML.toLowerCase().includes("temporal"));
check("LTV: L3.1 chip becomes active", l31.el._cls.has("active"));
check("LTV: L1.2 chip no longer active", th.chips.find((x) => x.code === "L1.2").el._cls.has("active") === false);

// ---- repro-collapse-viz ----
c = bind(makeEl("div"));
const rh = RCV.mount(c, {});
const svg = c.children.find((x) => x.tag === "svg");
check("RCV: mounts an svg", !!svg);
const bars = walk(svg, (x) => x.tag === "rect", []);
check("RCV: 2 value bars", bars.length === 2);
const readout = c.children.find((x) => x._cls.has("rcv-readout"));
check("RCV: default leak-ON readout shows +0.216 gap", readout.innerHTML.includes("0.216"));
check("RCV: default readout names the 0.935 / 0.719 numbers", readout.innerHTML.includes("0.935") && readout.innerHTML.includes("0.719"));
const btns = walk(c, (x) => x._cls.has("rcv-btn"), []);
const offBtn = btns.find((b) => b.textContent.includes("OFF"));
offBtn.click();
check("RCV: leak-OFF readout shows the -0.009 collapse", readout.innerHTML.includes("-0.009"));
check("RCV: leak-OFF readout names 0.712 / 0.721", readout.innerHTML.includes("0.712") && readout.innerHTML.includes("0.721"));

console.log(pass ? "\nALL VIZ CHECKS PASS" : "\nVIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
