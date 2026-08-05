// Headless mount check for the L042 viz asset (no jsdom):
//   assets/protocol-bakeoff-viz.js  (new — shared-protocol fair bake-off)
// Also asserts the lesson mounts the reused viz (resnet-block, depth-trainability),
// the pedagogy widgets (retrieval upTo:42, predict, teachback), and defines every
// .pbo- / .rnb- / .dtr- class the widgets emit.
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

// --------------------------------------------------- protocol-bakeoff-viz
load("protocol-bakeoff-viz.js");
const PBO = global.window.ProtocolBakeoffViz;
let c = bind(makeEl("div"));
const api = PBO.mount(c, {});
const readout = walk(c, (x) => x._cls && x._cls.has("pbo-readout"), [])[0];

check("PBO: three baselines", api.models().length === 3);
check("PBO: models are mlp/resnet/gbdt",
  ["mlp", "resnet", "gbdt"].every((m) => api.models().includes(m)));
check("PBO: default selection is resnet", api.getSel() === "resnet");
check("PBO: default readout names the strong simple baseline",
  readout.innerHTML.toLowerCase().includes("strong simple baseline"));

api.select("gbdt");
check("PBO: GBDT readout names it the incumbent", readout.innerHTML.toLowerCase().includes("incumbent"));
check("PBO: GBDT readout names no universal winner",
  readout.innerHTML.toLowerCase().includes("no universal winner"));

api.select("mlp");
check("PBO: MLP readout names the simplest net floor",
  readout.innerHTML.toLowerCase().includes("simplest net floor"));

// click a model name to select it
const gbdtName = walk(c, (x) => x.tag === "text" && x.getAttribute("data-id") === "gbdt", [])[0];
gbdtName.click();
check("PBO: clicking a model name selects it", api.getSel() === "gbdt");

// the shared-protocol banner is present
const banner = walk(c, (x) => x._cls && x._cls.has("pbo-banner"), [])[0];
check("PBO: shared-protocol banner present", !!banner && banner.innerHTML.includes("shared protocol"));

// ------------------------------------------- lesson <-> viz CSS coupling
const lesson = fs.readFileSync(
  path.join(__dirname, "..", "lessons", "0042-mlp-resnet-baselines.html"), "utf8");

const pboClasses = ["pbo-viz", "pbo-banner", "pbo-svg", "pbo-tick", "pbo-axis", "pbo-name",
  "pbo-name-on", "pbo-val", "pbo-readout", "pbo-r-head", "pbo-r-badge", "pbo-r-role",
  "pbo-r-score", "pbo-r-space", "pbo-caption"];
pboClasses.forEach((cls) => check("lesson stylesheet defines ." + cls, lesson.includes("." + cls)));

// reused viz CSS must also be present (resnet-block + depth-trainability)
["rnb-svg", "rnb-readout", "dtr-svg", "dtr-readout", "dtr-leg"].forEach(
  (cls) => check("lesson stylesheet defines reused ." + cls, lesson.includes("." + cls)));

check("lesson mounts the bakeoff widget", lesson.includes('id="bakeoff"'));
check("lesson mounts the resnet-block widget", lesson.includes('id="resnet-block"'));
check("lesson mounts the depth widget", lesson.includes('id="depth"'));
check("lesson mounts the warm-up", lesson.includes('id="warmup"'));
check("lesson mounts the predict widget", lesson.includes('id="predict-resnet"'));
check("lesson mounts the teachback widget", lesson.includes('id="teachback-baseline"'));
check("lesson warm-up draws from lessons before 42", lesson.includes("upTo: 42"));
check("lesson cites the primary reading arXiv id", lesson.includes("2106.11959"));
check("lesson loads the new viz script", lesson.includes("protocol-bakeoff-viz.js"));

// ---------------------------------- multi-dataset rigor content (NOTES #22, #23)
check("lesson has the cross-dataset section", lesson.includes('id="across-datasets"'));
check("lesson reports mean ranks", /mean rank/i.test(lesson));
check("lesson reports the Friedman test", /friedman/i.test(lesson) && lesson.includes("0.039"));
check("lesson flags the numeric-skew / biased sample caveat", /numeric-skew|numeric-heavy|biased sample/i.test(lesson));
check("lesson cites Grinsztajn's ~45-dataset benchmark", lesson.includes("2207.08815") && /45[- ]dataset|~45/i.test(lesson));
check("lesson teaches build-from-scratch + rtdl validation", /validation point|let the library check you|validate/i.test(lesson) && lesson.includes("rtdl"));
check("lesson mounts the multi-dataset rigor quiz", lesson.includes('id="quiz4"') && lesson.includes('getElementById("quiz4")'));
check("lesson mounts the rtdl-validation quiz", lesson.includes('id="quiz5"') && lesson.includes('getElementById("quiz5")'));
check("lesson names the four verified datasets", ["credit_g", "diabetes", "blood_transfusion", "kc1"].every((d) => lesson.includes(d)));

console.log(pass ? "\nALL VIZ CHECKS PASS" : "\nVIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
