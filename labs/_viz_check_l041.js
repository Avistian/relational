// Headless mount check for the L041 viz asset (no jsdom):
//   assets/tabular-dl-map-viz.js  (new — neural-tabular landscape map)
// Also asserts every .tdm- class the widget emits appears in the lesson
// stylesheet, and that the lesson mounts the pedagogy widgets (retrieval,
// predict, teachback) and uses the correct spacing bound (upTo: 41).
const fs = require("fs");
const path = require("path");

function makeEl(tag) {
  const el = {
    tag, children: [], attrs: {}, style: {}, _text: "", _html: "", _cls: new Set(),
    listeners: {}, value: "", type: "", checked: false, disabled: false,
    set className(v) { this._cls = new Set(String(v).split(/\s+/).filter(Boolean)); },
    get className() { return [...this._cls].join(" "); },
    classList: {
      add(...c) { c.forEach((x) => this._owner._cls.add(x)); },
      remove(...c) { c.forEach((x) => this._owner._cls.delete(x)); },
      toggle(c, on) { if (on) this._owner._cls.add(c); else this._owner._cls.delete(c); },
      contains(c) { return this._owner._cls.has(c); },
    },
    set textContent(v) { this._text = String(v); },
    get textContent() { return this._text; },
    set innerHTML(v) { this._html = v; if (v === "") this.children = []; },
    get innerHTML() { return this._html; },
    setAttribute(k, v) { this.attrs[k] = v; },
    getAttribute(k) { return this.attrs[k] === undefined ? null : this.attrs[k]; },
    setProperty(k, v) { this.attrs["style:" + k] = v; },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); },
    click() { (this.listeners.click || []).forEach((fn) => fn()); },
    querySelectorAll(sel) {
      const cls = sel.replace(/^\./, "");
      const out = [];
      walk(this, (x) => x._cls && x._cls.has(cls), out);
      return out;
    },
  };
  el.style.setProperty = function (k, v) { el.attrs["style:" + k] = v; };
  return el;
}
function bind(el) { el.classList._owner = el; return el; }
global.document = {
  createElement: (t) => bind(makeEl(t)),
  createElementNS: (_ns, t) => bind(makeEl(t)),
  createTextNode: (t) => ({ tag: "#text", _text: t, children: [] }),
};
global.window = {};

function load(f) { eval(fs.readFileSync(path.join(__dirname, "..", "assets", f), "utf8")); }
function walk(node, pred, acc) {
  (node.children || []).forEach((c) => { if (pred(c)) acc.push(c); walk(c, pred, acc); });
  return acc;
}

let pass = true;
function check(name, cond) { console.log((cond ? "ok  " : "FAIL") + "  " + name); if (!cond) pass = false; }

// --------------------------------------------------- tabular-dl-map-viz
load("tabular-dl-map-viz.js");
const TDM = global.window.TabularDLMapViz;
let c = bind(makeEl("div"));
const api = TDM.mount(c, {});
check("TDM: ten models", api.models().length === 10);
check("TDM: five families", api.families().length === 5);
check("TDM: default selection is resnet", api.getSel() === "resnet");
const readout = c.children[0].children.find((x) => x._cls.has("tdm-readout"));
check("TDM: default readout names ResNet baseline", readout.innerHTML.includes("baseline"));
check("TDM: default readout cites L042", readout.innerHTML.includes("L042"));

api.select("ft");
check("TDM: FT readout names Feature Tokenizer", readout.innerHTML.includes("Feature Tokenizer"));
check("TDM: FT readout names [CLS]", readout.innerHTML.includes("[CLS]"));
check("TDM: FT readout cites L046", readout.innerHTML.includes("L046"));

api.select("gbdt");
check("TDM: GBDT is the incumbent bar", readout.innerHTML.includes("bar"));
check("TDM: GBDT readout names Grinsztajn / no universal", readout.innerHTML.includes("Grinsztajn"));

const ftBtn = walk(c, (x) => x.tag === "button" && x.getAttribute("data-id") === "tabtransformer", [])[0];
ftBtn.click();
check("TDM: clicking a chip selects it", api.getSel() === "tabtransformer");
check("TDM: TabTransformer readout notes categoricals-only", readout.innerHTML.includes("categoricals"));

// ------------------------------------------- lesson <-> viz CSS coupling
const lesson = fs.readFileSync(
  path.join(__dirname, "..", "lessons", "0041-deep-tabular-landscape.html"), "utf8");
const tdmClasses = ["tdm-viz", "tdm-legend", "tdm-key", "tdm-dot", "tdm-chips", "tdm-chip",
  "tdm-on", "tdm-chip-lab", "tdm-chip-lsn", "tdm-readout", "tdm-r-head", "tdm-r-badge",
  "tdm-r-lesson", "tdm-r-title", "tdm-r-ev"];
tdmClasses.forEach((cls) => check("lesson stylesheet defines ." + cls, lesson.includes("." + cls)));
check("lesson mounts the dl-map widget", lesson.includes('id="dl-map"'));
check("lesson mounts the warm-up", lesson.includes('id="warmup"'));
check("lesson mounts the predict widget", lesson.includes('id="predict-winner"'));
check("lesson mounts the teachback widget", lesson.includes('id="teachback-baseline"'));
check("lesson warm-up draws from lessons before 41", lesson.includes("upTo: 41"));
check("lesson cites the primary reading arXiv id", lesson.includes("2106.11959"));
check("lesson defines fig-caption used in prose", lesson.includes(".fig-caption"));

console.log(pass ? "\nALL VIZ CHECKS PASS" : "\nVIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
