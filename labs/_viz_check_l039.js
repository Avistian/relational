// Headless mount check for the L039 viz assets (no jsdom):
//   assets/y1-arc-viz.js         (new — Year 1 as four argument moves)
//   assets/trees-frontier-viz.js (new — win / exhausted / frontier zones)
//   assets/biases-viz.js         (reused from L019 — irregular + rotation preview)
//   assets/checklist.js          (reused — synthesis-essay checklist)
// Every number quoted in a readout must match a prior verified lesson figure;
// L039 introduces no new bake-offs. Also asserts every .ya- / .tf- class the
// widgets emit appears in the lesson stylesheet.
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
      // minimal: support ".ya-chip" / ".tf-pt"
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

// -------------------------------------------------------------- y1-arc-viz
load("y1-arc-viz.js");
const YA = global.window.Y1ArcViz;
let c = bind(makeEl("div"));
const yaApi = YA.mount(c, {});
check("YA: mounts legend + bands + readout", c.children.length === 3);
check("YA: four quarters", yaApi.quarters().length === 4);
check("YA: twelve milestone chips", yaApi.chips().length === 12);
check("YA: default selection is q3-biases", yaApi.getSel() === "q3-biases");
const yaReadout = c.children.find((x) => x._cls.has("ya-readout"));
check("YA: default readout names three inductive biases",
  yaReadout.innerHTML.includes("Smoothness") || yaReadout.innerHTML.includes("three inductive"));
check("YA: default readout quotes rotation collapse 0.987", yaReadout.innerHTML.includes("0.987"));

yaApi.select("q4-join");
check("YA: q4-join quotes aggregation collision P(churn)=0.502",
  yaReadout.innerHTML.includes("0.502"));
check("YA: q4-join names Ada/Bo",
  yaReadout.innerHTML.includes("Ada") && yaReadout.innerHTML.includes("Bo"));

yaApi.select("q4-cred");
check("YA: q4-cred quotes 0.0032-nat winner's curse", yaReadout.innerHTML.includes("0.0032"));
check("YA: q4-cred quotes major revision verdict", yaReadout.innerHTML.toLowerCase().includes("major revision"));

yaApi.select("q1-checkpoint");
check("YA: q1-checkpoint quotes HistGBDT CV-PR 0.470", yaReadout.innerHTML.includes("0.470"));

// click a chip
const chipBtn = walk(c, (x) => x.tag === "button" && x.getAttribute("data-id") === "q2-boosters", [])[0];
chipBtn.click();
check("YA: clicking a chip selects it", yaApi.getSel() === "q2-boosters");

// -------------------------------------------------------- trees-frontier-viz
load("trees-frontier-viz.js");
const TF = global.window.TreesFrontierViz;
let c2 = bind(makeEl("div"));
const tfApi = TF.mount(c2, {});
check("TF: three zones", tfApi.zones().join(",") === "WIN,EXHAUSTED,FRONTIER");
check("TF: nine points", tfApi.points().length === 9);
check("TF: default selection is win-smooth", tfApi.getSel() === "win-smooth");
const tfReadout = c2.children.find((x) => x._cls.has("tf-readout"));
check("TF: default quotes checkerboard 0.969 / 0.837",
  tfReadout.innerHTML.includes("0.969") && tfReadout.innerHTML.includes("0.837"));

tfApi.select("ex-automl");
check("TF: AutoML point quotes tuned XGB 0.806 vs AutoML 0.803",
  tfReadout.innerHTML.includes("0.806") && tfReadout.innerHTML.includes("0.803"));

tfApi.select("fr-collision");
check("TF: collision quotes identical row n=3/total=90",
  tfReadout.innerHTML.includes("n=3/total=90") || tfReadout.innerHTML.includes("n=3/total=90/avg=30"));
check("TF: collision quotes P(churn)=0.502", tfReadout.innerHTML.includes("0.502"));

tfApi.select("fr-thesis");
check("TF: thesis point keeps the open burden open",
  tfReadout.innerHTML.includes("NOT yet shown") || tfReadout.innerHTML.includes("not yet shown") ||
  tfReadout.innerHTML.includes("has NOT yet"));

const ptBtn = walk(c2, (x) => x.tag === "button" && x.getAttribute("data-id") === "win-rotate", [])[0];
ptBtn.click();
check("TF: clicking a point selects it", tfApi.getSel() === "win-rotate");
check("TF: rotation quotes 0.987→0.747",
  tfReadout.innerHTML.includes("0.987") && tfReadout.innerHTML.includes("0.747"));

// ---------------------------------------------------------------- biases-viz
load("biases-viz.js");
const BV = global.window.BiasesViz;
let c3 = bind(makeEl("div"));
BV.mount(c3, { caption: "test" });
check("BV: mounts biases-viz container", c3.className === "biases-viz");
check("BV: has mode controls + svg + readout", c3.children.length >= 4);

// ----------------------------------------------------------------- checklist
load("checklist.js");
const CL = global.window.Checklist;
let c4 = bind(makeEl("div"));
CL.mount(c4, {
  title: "Synthesis-essay checklist",
  items: [{ label: "<strong>Claim.</strong> One falsifiable sentence?", hint: "working claim" }],
  done: "That is a synthesis.",
});
check("CL: mounts the essay checklist", c4.children.length > 0);

// -------------------------------------------- lesson <-> viz CSS coupling
const lesson = fs.readFileSync(
  path.join(__dirname, "..", "lessons", "0039-year-1-synthesis-essay.html"), "utf8");
const yaClasses = ["ya-viz", "ya-legend", "ya-key", "ya-dot", "ya-bands", "ya-band", "ya-band-head",
  "ya-band-title", "ya-band-blurb", "ya-chips", "ya-chip", "ya-on", "ya-chip-lab", "ya-readout",
  "ya-r-head", "ya-r-badge", "ya-r-title", "ya-r-ev", "ya-r-essay"];
yaClasses.forEach((cls) => check("lesson stylesheet defines ." + cls, lesson.includes("." + cls)));
const tfClasses = ["tf-viz", "tf-grid", "tf-zone", "tf-zone-head", "tf-zone-lab", "tf-zone-tag",
  "tf-points", "tf-pt", "tf-on", "tf-readout", "tf-r-head", "tf-r-badge", "tf-r-title",
  "tf-r-ev", "tf-r-essay"];
tfClasses.forEach((cls) => check("lesson stylesheet defines ." + cls, lesson.includes("." + cls)));
check("lesson mounts the y1-arc widget", lesson.includes('id="y1-arc"'));
check("lesson mounts the trees-frontier widget", lesson.includes('id="trees-frontier"'));
check("lesson mounts the biases widget", lesson.includes('id="biases-viz"'));
check("lesson mounts the essay checklist", lesson.includes('id="essay-checklist"'));
check("lesson warm-up draws from lessons before 39", lesson.includes("upTo: 39"));
check("lesson does not invent a new bake-off number (no L039 verify harness claim)",
  !lesson.includes("_verify_l039"));

console.log(pass ? "\nALL VIZ CHECKS PASS" : "\nVIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
