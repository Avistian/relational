// Headless mount check for the L036 viz assets (no jsdom):
//   assets/nested-calib-viz.js   (new)
//   assets/pipeline-viz.js       (reused — transductive preprocessing)
//   assets/paired-diff-viz.js    (reused — was the winner picked by noise?)
//   assets/checklist.js          (reused — the audit rubric)
// Verifies the nested-calibration widget's two modes (ungrouped inner split =>
// straddling persons + "leak present" but an honest reported number; grouped =>
// zero straddlers), and that the reused widgets accept the L036 config.
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
function walk(node, pred, acc) {
  (node.children || []).forEach((c) => { if (pred(c)) acc.push(c); walk(c, pred, acc); });
  return acc;
}
function textOf(root) { return walk(root, (x) => x.tag === "text", []).map((t) => t._text); }
function rects(root) { return walk(root, (x) => x.tag === "rect", []); }
// Row squares are 16px; the legend swatches are 11px, so filter by width to count
// data cells only.
function rowSquares(root, fill) {
  return rects(root).filter((r) => r.attrs.fill === fill && Number(r.attrs.width) === 16);
}
// Collect all visible strings without JSON.stringify (the fake DOM is circular).
function allText(root) {
  const out = [];
  (function rec(n) {
    if (!n || typeof n !== "object") return;
    if (n._text) out.push(n._text);
    if (n._html) out.push(n._html);
    (n.children || []).forEach(rec);
  })(root);
  return out.join(" | ");
}

let pass = true;
function check(name, cond) { console.log((cond ? "ok  " : "FAIL") + "  " + name); if (!cond) pass = false; }

// ---------------------------------------------------------------- nested-calib
load("nested-calib-viz.js");
const NC = global.window.NestedCalibViz;
let c = bind(makeEl("div"));
const api = NC.mount(c, { eceBefore: "0.0363", eceAfter: "0.0331" });
let svg = c.children.find((x) => x.tag === "svg");
let ro = c.children.find((x) => x._cls.has("nc-readout"));
const btns = c.children.find((x) => x._cls.has("nc-ctl")).children;

check("NC: mounts svg + readout + two mode buttons", !!svg && !!ro && btns.length === 2);
check("NC: default mode is the library default (ungrouped)", api.getMode() === "ungrouped");

let t = textOf(svg);
check("NC: labels the outer training block as person-grouped",
  t.some((s) => s.includes("TRAINING BLOCK") && s.includes("person-grouped")));
check("NC: labels the outer test block as held out",
  t.some((s) => s.includes("TEST BLOCK") && s.includes("never touched")));
check("NC: legend names base model, calibrator and outer test",
  t.some((s) => s.includes("base model")) && t.some((s) => s.includes("calibrator")) &&
  t.some((s) => s.includes("outer test")));

// three straddling persons, each flagged with a warning glyph
let flagged = t.filter((s) => s.indexOf("\u26a0") >= 0);
check("NC: ungrouped flags exactly 3 straddling persons", flagged.length === 3);
check("NC: the straddlers are P1, P3, P9",
  ["P1", "P3", "P9"].every((p) => flagged.some((s) => s.indexOf(p) === 0)));
check("NC: ungrouped readout reports 3 of 4", ro.innerHTML.includes("3 of 4"));
check("NC: ungrouped readout says the leak is present", ro.innerHTML.includes("Leak present"));
check("NC: ungrouped readout stresses the reported number stays honest",
  ro.innerHTML.includes("does not inflate your number") && ro.innerHTML.includes("0.0363"));
check("NC: ungrouped readout quotes the real exposure (675 persons / 1411 rows)",
  ro.innerHTML.includes("675 persons") && ro.innerHTML.includes("1411 rows"));

// count coloured squares: outer-test rows must be red in BOTH modes
check("NC: 4 outer-test rows drawn in test colour", rowSquares(svg, "#b03a2e").length === 4);
check("NC: 13 training-block rows split between base model and calibrator",
  rowSquares(svg, "#2e6fb0").length + rowSquares(svg, "#c8871b").length === 13);
check("NC: ungrouped sends 4 rows to the calibrator", rowSquares(svg, "#c8871b").length === 4);

// ---- grouped mode
btns[1].click();
svg = c.children.find((x) => x.tag === "svg");
ro = c.children.find((x) => x._cls.has("nc-readout"));
t = textOf(svg);
check("NC: toggles to grouped mode", api.getMode() === "grouped");
check("NC: grouped flags no straddling persons", t.filter((s) => s.indexOf("\u26a0") >= 0).length === 0);
check("NC: grouped readout reports 0 of 4", ro.innerHTML.includes("0 of 4"));
check("NC: grouped readout quotes the re-measured ECE", ro.innerHTML.includes("0.0331"));
check("NC: grouped keeps the 4 outer-test rows red", rowSquares(svg, "#b03a2e").length === 4);
check("NC: grouped still splits 13 training rows",
  rowSquares(svg, "#2e6fb0").length + rowSquares(svg, "#c8871b").length === 13);

// back to default
btns[0].click();
check("NC: toggles back to ungrouped", api.getMode() === "ungrouped");

// ---------------------------------------------------------------- pipeline-viz
load("pipeline-viz.js");
const PV = global.window.PipelineViz;
let c2 = bind(makeEl("div"));
PV.mount(c2, { nSamples: 10, nSplits: 5 });
check("PV: mounts (reused for the transductive-encoder finding)", c2.children.length > 0);
check("PV: exposes both fit regimes as buttons",
  walk(c2, (x) => x.tag === "button", []).length >= 2);

// -------------------------------------------------------------- paired-diff-viz
load("paired-diff-viz.js");
const PD = global.window.PairedDiffViz;
let c3 = bind(makeEl("div"));
const pdApi = PD.mount(c3, {
  foldDiffs: [0.0116, -0.0267, 0.0047, -0.0022, -0.0037],
  mean: -0.00325, naiveHalf: 0.01791, corrHalf: 0.02686,
  pNaive: "0.64", pCorr: "0.75", labelA: "M2a", labelB: "M1",
  metricLabel: "log-loss", domain: 0.04, ticks: [-0.03, -0.015, 0.015, 0.03],
  verdictNaive: "NOT significant — p = 0.64; 0 sits inside the CI.",
  verdictCorrected: "Still NOT significant — p = 0.75.",
});
check("PD: mounts with the homework's own 5 fold deltas", c3.children.length > 0);
check("PD: labels the two shipped candidates",
  allText(c3).includes("M2a") && allText(c3).includes("M1"));
check("PD: uses the log-loss metric label, not accuracy",
  allText(c3).includes("log-loss(M2a)") && !allText(c3).includes("acc(M2a)"));
check("PD: shows the negative mean with an explicit minus sign",
  allText(c3).includes("mean \u22120.003"));
check("PD: naive verdict is the L036 override, not L023's SIGNIFICANT prose",
  allText(c3).includes("NOT significant \u2014 p = 0.64") && !allText(c3).includes("SIGNIFICANT</span>"));
pdApi.setMode("corrected");
check("PD: corrected verdict is the L036 override", allText(c3).includes("Still NOT significant \u2014 p = 0.75"));

// ------------------------------------------------------------------- checklist
load("checklist.js");
const CL = global.window.Checklist;
let c4 = bind(makeEl("div"));
CL.mount(c4, {
  title: "Leakage-spine audit",
  items: [{ label: "Are groups respected in EVERY split, including library-internal ones?", hint: "L004" }],
  done: "Audit complete.",
});
check("CL: mounts the audit rubric", c4.children.length > 0);
check("CL: renders the rubric title", allText(c4).includes("Leakage-spine audit"));

console.log(pass ? "\nALL VIZ CHECKS PASS" : "\nVIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
