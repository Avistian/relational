// Headless mount check for the L040 viz assets (no jsdom):
//   assets/exit-verdict-viz.js  (new — BEAT / TIE / EXPLAIN fork)
//   assets/exit-gates-viz.js    (new — six exit protocol gates)
//   assets/biases-viz.js        (reused — three-bias geometry preview)
//   assets/checklist.js         (reused — exit rubric)
// Every number quoted in a readout must match a prior verified lesson figure;
// L040 introduces no new bake-off. Also asserts every .ev- / .eg- class the
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

// -------------------------------------------------------- exit-verdict-viz
load("exit-verdict-viz.js");
const EV = global.window.ExitVerdictViz;
let c = bind(makeEl("div"));
const evApi = EV.mount(c, {});
check("EV: three forks", evApi.forks().join(",") === "BEAT,TIE,EXPLAIN");
check("EV: nine points", evApi.points().length === 9);
check("EV: default selection is tie-adult", evApi.getSel() === "tie-adult");
const evReadout = c.children[0].children.find((x) => x._cls.has("ev-readout"));
check("EV: default quotes ref 0.9282", evReadout.innerHTML.includes("0.9282"));
check("EV: default quotes OOF corr 0.997", evReadout.innerHTML.includes("0.997"));

evApi.select("beat-gap");
check("EV: beat-gap quotes L030 corrected p=0.64", evReadout.innerHTML.includes("0.64"));
check("EV: beat-gap quotes ±0.002 or 0.002 noise",
  evReadout.innerHTML.includes("0.002") || evReadout.innerHTML.includes("±0.002"));

evApi.select("ex-biases");
check("EV: ex-biases names inductive biases / Grinsztajn",
  evReadout.innerHTML.includes("inductive") || evReadout.innerHTML.includes("Grinsztajn"));

evApi.select("ex-frontier");
check("EV: frontier quotes P=0.502 collision", evReadout.innerHTML.includes("0.502"));

const ptBtn = walk(c, (x) => x.tag === "button" && x.getAttribute("data-id") === "tie-credit", [])[0];
ptBtn.click();
check("EV: clicking a point selects it", evApi.getSel() === "tie-credit");
check("EV: credit_g point quotes +0.008", evReadout.innerHTML.includes("0.008"));

// ---------------------------------------------------------- exit-gates-viz
load("exit-gates-viz.js");
const EG = global.window.ExitGatesViz;
let c2 = bind(makeEl("div"));
const egApi = EG.mount(c2, {});
check("EG: six gates", egApi.gates().length === 6);
check("EG: default selection is g-regen", egApi.getSel() === "g-regen");
const egReadout = c2.children[0].children.find((x) => x._cls.has("eg-readout"));
check("EG: default quotes 0.9282 regenerable band", egReadout.innerHTML.includes("0.9282"));

egApi.select("g-verdict");
check("EG: verdict gate quotes ±0.002 or 0.002",
  egReadout.innerHTML.includes("0.002"));
check("EG: verdict gate mentions winner's curse / 0.0032",
  egReadout.innerHTML.includes("0.0032"));

egApi.select("g-essay");
check("EG: essay gate names STAND or REVISE",
  egReadout.innerHTML.includes("STAND") && egReadout.innerHTML.includes("REVISE"));

const chipBtn = walk(c2, (x) => x.tag === "button" && x.getAttribute("data-id") === "g-biases", [])[0];
chipBtn.click();
check("EG: clicking a chip selects it", egApi.getSel() === "g-biases");
check("EG: biases gate names Grinsztajn / three",
  egReadout.innerHTML.includes("Grinsztajn") || egReadout.innerHTML.includes("three"));

// ---------------------------------------------------------------- biases-viz
load("biases-viz.js");
const BV = global.window.BiasesViz;
let c3 = bind(makeEl("div"));
BV.mount(c3, { caption: "test" });
check("BV: mounts biases-viz container", c3.className === "biases-viz");

// ----------------------------------------------------------------- checklist
load("checklist.js");
const CL = global.window.Checklist;
let c4 = bind(makeEl("div"));
CL.mount(c4, {
  title: "Year 1 exit rubric",
  items: [{ label: "<strong>Fork.</strong> BEAT / TIE / FAIL classified?", hint: "noise band" }],
  done: "Year 1 is closed.",
});
check("CL: mounts the exit checklist", c4.children.length > 0);

// -------------------------------------------- lesson <-> viz CSS coupling
const lesson = fs.readFileSync(
  path.join(__dirname, "..", "lessons", "0040-year-1-exit-exam.html"), "utf8");
const evClasses = ["ev-viz", "ev-grid", "ev-zone", "ev-zone-head", "ev-zone-lab", "ev-zone-tag",
  "ev-points", "ev-pt", "ev-on", "ev-readout", "ev-r-head", "ev-r-badge", "ev-r-title",
  "ev-r-ev", "ev-r-essay"];
evClasses.forEach((cls) => check("lesson stylesheet defines ." + cls, lesson.includes("." + cls)));
const egClasses = ["eg-viz", "eg-legend", "eg-key", "eg-dot", "eg-chips", "eg-chip", "eg-on",
  "eg-chip-lab", "eg-readout", "eg-r-head", "eg-r-badge", "eg-r-title", "eg-r-ev", "eg-r-fail"];
egClasses.forEach((cls) => check("lesson stylesheet defines ." + cls, lesson.includes("." + cls)));
check("lesson mounts the exit-gates widget", lesson.includes('id="exit-gates"'));
check("lesson mounts the exit-verdict widget", lesson.includes('id="exit-verdict"'));
check("lesson mounts the biases widget", lesson.includes('id="biases-viz"'));
check("lesson mounts the exit checklist", lesson.includes('id="exit-checklist"'));
check("lesson warm-up draws from lessons before 40", lesson.includes("upTo: 40"));
check("lesson does not invent a new bake-off verify harness",
  !lesson.includes("_verify_l040"));

console.log(pass ? "\nALL VIZ CHECKS PASS" : "\nVIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
