// Headless mount check for the L038 viz assets (no jsdom):
//   assets/review-triage-viz.js    (new — triage findings on conclusion-impact x artifact-severity)
//   assets/leakage-taxonomy-viz.js (reused from L022 — the 8-type leakage sweep)
//   assets/checklist.js            (reused — the peer-review checklist deliverable)
// Every number asserted in a readout must match a value MEASURED in L036/L037 and recorded in
// the learning records; L038 introduces no new numbers, so the check is that the widget quotes the
// prior measurements faithfully. Also asserts every .rt- class the widget emits appears in the
// lesson's stylesheet, so the viz and the lesson cannot drift apart silently.
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
      toggle(c, on) { on ? this._owner._cls.add(c) : this._owner._cls.delete(c); },
      contains(c) { return this._owner._cls.has(c); },
    },
    set textContent(v) { this._text = String(v); },
    get textContent() { return this._text; },
    set innerHTML(v) { this._html = v; if (v === "") this.children = []; },
    get innerHTML() { return this._html; },
    setAttribute(k, v) { this.attrs[k] = v; },
    getAttribute(k) { return this.attrs[k] === undefined ? null : this.attrs[k]; },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); },
    click() { (this.listeners.click || []).forEach((fn) => fn()); },
  };
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

// ---------------------------------------------------------- review-triage-viz
load("review-triage-viz.js");
const RT = global.window.ReviewTriageViz;
let c = bind(makeEl("div"));
const api = RT.mount(c, {});
const legend = c.children.find((x) => x._cls.has("rt-legend"));
const holder = c.children.find((x) => x._cls.has("rt-svg-holder"));
const readout = c.children.find((x) => x._cls.has("rt-readout"));

check("RT: mounts legend + svg holder + readout", !!legend && !!holder && !!readout && c.children.length === 3);
check("RT: legend has one key per axis (4)", walk(legend, (x) => x._cls.has("rt-key"), []).length === 4);
const svg = holder.children.find((x) => x.tag === "svg");
check("RT: draws 7 defect points", walk(svg, (x) => x.tag === "circle", []).length === 7);
check("RT: draws 2 quadrant guide lines", walk(svg, (x) => x.tag === "line", []).length === 2);
check("RT: points() returns 7 codes", api.points().length === 7);

check("RT: default selection is the winner's-curse blocker T1", api.getSel() === "T1");
check("RT: T1 readout quotes the measured 0.0032-nat margin", readout.innerHTML.includes("0.0032"));
check("RT: T1 readout quotes the corrected p = 0.75", readout.innerHTML.includes("0.75"));
check("RT: T1 is graded HIGH conclusion-impact", readout.innerHTML.includes("HIGH on the conclusion"));

api.select("L1");
check("RT: L1 (ungrouped inner split) is a leakage finding", readout.innerHTML.includes("Leakage"));
check("RT: L1 quotes the grouped re-measure 1.4232", readout.innerHTML.includes("1.4232"));
check("RT: L1 is LOW conclusion-impact but HIGH artifact severity",
  readout.innerHTML.includes("LOW on the conclusion") && readout.innerHTML.includes("HIGH in the artifact"));

api.select("M1");
check("RT: M1 quotes the 0.0332 / 0.018 estimator ambiguity",
  readout.innerHTML.includes("0.0332") && readout.innerHTML.includes("0.018"));

api.select("T2");
check("RT: T2 (float32) ties the +0.00133 to 42% of the margin",
  readout.innerHTML.includes("+0.00133") && readout.innerHTML.includes("42 %"));

api.select("R1");
check("RT: R1 names the lightgbm/scikit-learn constraint crash",
  readout.innerHTML.includes("force_all_finite"));

api.select("R2");
check("RT: R2 quotes the 0.0166-nat splitter-seed spread", readout.innerHTML.includes("0.0166"));

// clicking a point selects it
const g = walk(svg, (x) => x.tag === "g" && x.getAttribute && x.getAttribute("data-code") === "M2", [])[0];
g.click();
check("RT: clicking a point selects it", api.getSel() === "M2");
check("RT: M2 (slice gate) quotes the 0.1071 noise floor", readout.innerHTML.includes("0.1071"));

// ------------------------------------------------------ leakage-taxonomy-viz
load("leakage-taxonomy-viz.js");
const LTV = global.window.LeakageTaxonomyViz;
let c2 = bind(makeEl("div"));
const ltvApi = LTV.mount(c2, {});
check("LTV: mounts 8 leak-type chips across 3 families", ltvApi.chips.length === 8);
const detail = c2.children.find((x) => x._cls.has("ltv-detail"));
check("LTV: default detail is L1.2 (preprocessing on all data)", detail.innerHTML.includes("L1.2"));
ltvApi.select("L3.2");
check("LTV: selecting L3.2 shows the group non-independence type", detail.innerHTML.includes("L3.2"));

// ------------------------------------------------------------------- checklist
load("checklist.js");
const CL = global.window.Checklist;
let c3 = bind(makeEl("div"));
CL.mount(c3, {
  title: "The peer-review checklist",
  items: [{ label: "<strong>Leakage.</strong> Have I walked all 8 leak types?", hint: "L022" }],
  done: "That is the review.",
});
check("CL: mounts the peer-review checklist", c3.children.length > 0);

// ------------------------------------------------- lesson <-> viz CSS coupling
const lesson = fs.readFileSync(
  path.join(__dirname, "..", "lessons", "0038-peer-review-your-evaluation.html"), "utf8");
const rtClasses = ["rt-viz", "rt-legend", "rt-key", "rt-dot", "rt-svg-holder", "rt-svg", "rt-guide",
  "rt-axis", "rt-quad", "rt-pt", "rt-circ", "rt-active", "rt-code", "rt-readout",
  "rt-r-head", "rt-r-badge", "rt-r-title", "rt-r-origin", "rt-r-ev", "rt-r-req"];
rtClasses.forEach((cls) => check("lesson stylesheet defines ." + cls, lesson.includes("." + cls)));
const ltvClasses = ["ltv-grid", "ltv-fam", "ltv-chip", "ltv-code", "ltv-detail", "ltv-d-code"];
ltvClasses.forEach((cls) => check("lesson stylesheet defines ." + cls, lesson.includes("." + cls)));
check("lesson mounts the review-triage widget", lesson.includes('id="review-triage"'));
check("lesson mounts the leakage-taxonomy widget", lesson.includes('id="leakage-taxonomy"'));
check("lesson mounts the review checklist", lesson.includes('id="review-checklist"'));
check("lesson warm-up draws from lessons before 38", lesson.includes("upTo: 38"));

console.log(pass ? "\nALL VIZ CHECKS PASS" : "\nVIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
