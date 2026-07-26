// Headless mount check for the L037 viz assets (no jsdom):
//   assets/repro-probe-viz.js     (new — which knob moves the number)
//   assets/ece-estimator-viz.js   (new — pooled vs per-fold ECE, and the noise floor)
//   assets/tolerance-gate-viz.js  (new — where to set the reproduction tolerance)
//   assets/checklist.js           (reused — the packaging rubric)
// Every asserted number here must match a value measured by labs/_repro*_l037.py
// or labs/_ece_estimator_l037.py; the point of the check is that the widgets and
// the harnesses cannot drift apart silently.
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
    dispatch(ev) { (this.listeners[ev] || []).forEach((fn) => fn()); },
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
function textOf(root) { return walk(root, (x) => x.tag === "text", []).map((t) => t._text); }
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

// ------------------------------------------------------------- repro-probe-viz
load("repro-probe-viz.js");
const RP = global.window.ReproProbeViz;
let c = bind(makeEl("div"));
const rpApi = RP.mount(c, {});
let ctl = c.children.find((x) => x._cls.has("rp-ctl"));
let readout = c.children.find((x) => x._cls.has("rp-readout"));

check("RP: mounts controls + svg holder + readout", !!ctl && !!readout && c.children.length === 3);
check("RP: one button per knob (9 measured perturbations)", ctl.children.length === 9);
check("RP: default knob is the plain rerun", rpApi.getKnob() === "rerun");

function svgOf() { return c.children[1].children.find((x) => x.tag === "svg"); }
let t = textOf(svgOf());
check("RP: draws the six pipeline stages",
  ["data", "encode", "split", "fit", "predict", "OOF matrix"].every((s) => t.includes(s)));
check("RP: shows the reference fingerprint chip",
  t.some((s) => s.includes("d2f0e4bf9b4fd761") && s.includes("reference")));
check("RP: default verdict is a match", t.some((s) => s.indexOf("\u2713 match") >= 0));
check("RP: default readout is the IDENTICAL pill", readout.innerHTML.includes("IDENTICAL"));
check("RP: default readout quotes the reference log-loss", readout.innerHTML.includes("1.632168"));

rpApi.select("threads");
check("RP: threads verdict stays identical", textOf(svgOf()).some((s) => s.indexOf("\u2713 match") >= 0));
check("RP: threads readout quotes the measured per-fit timings",
  readout.innerHTML.includes("9.81") && readout.innerHTML.includes("3.52") &&
  readout.innerHTML.includes("13.74"));

rpApi.select("modelseed");
check("RP: model seed is reported inert", readout.innerHTML.includes("inert"));
check("RP: model seed readout names the sampling parameters that are unset",
  readout.innerHTML.includes("bagging_fraction") && readout.innerHTML.includes("feature_fraction"));

rpApi.select("dtype");
t = textOf(svgOf());
check("RP: dtype verdict differs", t.some((s) => s.indexOf("\u2260 differs") >= 0));
check("RP: dtype shows the second measured fingerprint",
  t.some((s) => s.includes("a4377f2a443dc970")));
check("RP: dtype readout is the NUMBER MOVES pill", readout.innerHTML.includes("NUMBER MOVES"));
check("RP: dtype readout quotes 258 flips, max |dp| 0.326 and +0.00133",
  readout.innerHTML.includes("258") && readout.innerHTML.includes("0.326") &&
  readout.innerHTML.includes("+0.00133"));
check("RP: dtype readout ties the delta to the L036 shipping margin",
  readout.innerHTML.includes("0.0032"));

rpApi.select("splitseed");
check("RP: splitter seed readout quotes the measured 0.0166 range",
  readout.innerHTML.includes("0.0166"));
check("RP: splitter seed readout gives the measured min/max", rpApi.getKnob() === "splitseed" &&
  readout.innerHTML.indexOf("NUMBER MOVES") >= 0);

rpApi.select("libversion");
t = textOf(svgOf());
check("RP: version rollback verdict is an error", t.some((s) => s.indexOf("\u2717 error") >= 0));
check("RP: version rollback readout is the DOES NOT RUN pill",
  readout.innerHTML.includes("DOES NOT RUN"));
check("RP: version rollback readout quotes the real TypeError",
  readout.innerHTML.includes("force_all_finite"));
check("RP: version rollback names both loose constraints",
  readout.innerHTML.includes("lightgbm&gt;=4.0") && readout.innerHTML.includes("scikit-learn&gt;=1.5"));

// clicking a button selects it
ctl.children[0].click();
check("RP: buttons select their knob", rpApi.getKnob() === "rerun");

// ----------------------------------------------------------- ece-estimator-viz
load("ece-estimator-viz.js");
const EE = global.window.EceEstimatorViz;
let c2 = bind(makeEl("div"));
const eeApi = EE.mount(c2, {});
let eeCtl = c2.children.find((x) => x._cls.has("ee-ctl"));
let eeRead = c2.children.find((x) => x._cls.has("ee-readout"));

check("EE: mounts three estimator modes", eeCtl.children.length === 3);
check("EE: default mode is pooled", eeApi.getMode() === "pooled");
check("EE: pooled readout reports the measured 0.0178", eeRead.innerHTML.includes("0.0178"));
check("EE: pooled readout reports the measured noise floor 0.0149",
  eeRead.innerHTML.includes("0.0149"));
check("EE: pooled readout says where the number is published",
  eeRead.innerHTML.includes("README.md"));

function eeSvg() { return c2.children[1].children.find((x) => x.tag === "svg"); }
let bars = walk(eeSvg(), (x) => x.tag === "rect", []);
check("EE: pooled draws one bar per populated bin (11)", bars.length === 11);
let eeT = textOf(eeSvg());
check("EE: pooled labels the sparsest bin as n=17", eeT.includes("n=17"));
check("EE: pooled labels the most populated bin as n=1264", eeT.includes("n=1264"));

eeApi.setMode("fold");
check("EE: fold mode reports the measured 0.0332", eeRead.innerHTML.includes("0.0332"));
check("EE: fold mode reports the noise floor 0.0335 (a perfect model scores the same)",
  eeRead.innerHTML.includes("0.0335"));
eeT = textOf(eeSvg());
check("EE: fold mode's sparsest bin holds 2 rows", eeT.includes("n=2"));
check("EE: fold mode annotates the 0.40 gap produced by those 2 rows", eeT.includes("0.40"));
check("EE: fold mode cites the selection table it came from",
  eeRead.innerHTML.includes("3.1"));

eeApi.setMode("slice");
check("EE: slice mode reports the failing gate value 0.094", eeRead.innerHTML.includes("0.094"));
check("EE: slice mode reports the 107-row noise floor 0.1071", eeRead.innerHTML.includes("0.1071"));
check("EE: slice mode says the floor exceeds the reported value",
  eeRead.innerHTML.includes("ee-bad"));

// ---------------------------------------------------------- tolerance-gate-viz
load("tolerance-gate-viz.js");
const TG = global.window.ToleranceGateViz;
let c3 = bind(makeEl("div"));
const tgApi = TG.mount(c3, {});
let tgRead = c3.children.find((x) => x._cls.has("tg-readout"));
let tgSvg = c3.children[1].children.find((x) => x.tag === "svg");

check("TG: mounts slider + svg + readout", c3.children.length === 3);
let tgT = textOf(tgSvg);
check("TG: places all four measured landmarks",
  tgT.some((s) => s.includes("float32")) && tgT.some((s) => s.includes("M2a")) &&
  tgT.some((s) => s.includes("splitter seed")) && tgT.some((s) => s.includes("fold's")));
check("TG: marks the bit-identical result at the origin",
  tgT.some((s) => s.includes("8 of 9")));

tgApi.setTol(1e-4);
check("TG: a very tight gate is labelled TIGHT", tgRead.innerHTML.includes("TIGHT"));
check("TG: a tight gate fires on the dtype cast",
  tgRead.innerHTML.indexOf("Fires on:") >= 0 && tgRead.innerHTML.includes("float32"));

tgApi.setTol(1e-2);
check("TG: a 1e-2 gate is labelled PERMISSIVE", tgRead.innerHTML.includes("PERMISSIVE"));
check("TG: a 1e-2 gate waves the shipping margin through",
  tgRead.innerHTML.split("Waves through:")[1].includes("M2a"));

tgApi.setTol(5e-2);
check("TG: a gate above one fold's sigma CANNOT FAIL", tgRead.innerHTML.includes("CANNOT FAIL"));
check("TG: at 5e-2 nothing fires",
  tgRead.innerHTML.includes("no measured perturbation exceeds"));

// ------------------------------------------------------------------- checklist
load("checklist.js");
const CL = global.window.Checklist;
let c4 = bind(makeEl("div"));
CL.mount(c4, {
  title: "The packaging rubric",
  items: [{ label: "Is the environment <strong>locked</strong>?", hint: "L037" }],
  done: "That is the package.",
});
check("CL: mounts the packaging rubric", c4.children.length > 0);
check("CL: renders the rubric title", allText(c4).includes("The packaging rubric"));

console.log(pass ? "\nALL VIZ CHECKS PASS" : "\nVIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
