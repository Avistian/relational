// Headless mount check for the L043 viz asset (no jsdom):
//   assets/tabnet-mask-viz.js  (new — sequential attention: masks, prior scale, aggregate)
// Also asserts the lesson mounts the pedagogy widgets (retrieval upTo:43, two predicts, teachback),
// defines every .tnm- class the widget emits, and states the verified L043 numbers.
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

const near = (a, b, tol) => Math.abs(a - b) < (tol === undefined ? 1e-9 : tol);

// --------------------------------------------------- tabnet-mask-viz
load("tabnet-mask-viz.js");
const TNM = global.window.TabnetMaskViz;
const c = bind(makeEl("div"));
const api = TNM.mount(c, {});
const readout = walk(c, (x) => x._cls && x._cls.has("tnm-readout"), [])[0];

check("TNM: default step is 1 (index 0)", api.getStep() === 0);
check("TNM: default gamma is 1.5", near(api.getGamma(), 1.5, 1e-9));

let st = api.state();
check("TNM: three decision steps", st.masks.length === 3);

// every mask is a sparsemax output: non-negative and sums to 1
const sums = st.masks.map((m) => m.reduce((a, b) => a + b, 0));
check("TNM: every mask sums to 1", sums.every((s) => near(s, 1, 1e-9)));
check("TNM: masks are non-negative", st.masks.every((m) => m.every((v) => v >= 0)));
check("TNM: masks contain EXACT zeros (sparsemax, not softmax)",
  st.masks.every((m) => m.some((v) => v === 0)));

// P[0] is all ones — nothing spent yet
check("TNM: prior at step 1 is all ones", st.priors[0].every((p) => p === 1));
// P[1] = gamma - M[0] exactly: untouched features sit at gamma, spent ones strictly below it
const spentAtStep1 = st.masks[0].map((v) => v > 0);
check("TNM: step-2 prior is exactly gamma - M[0]",
  st.priors[1].every((p, j) => near(p, 1.5 - st.masks[0][j], 1e-9)));
check("TNM: step-2 prior is below gamma exactly where step 1 selected",
  st.priors[1].every((p, j) => (spentAtStep1[j] ? p < 1.5 - 1e-9 : near(p, 1.5, 1e-9))));

// the sequential point: different steps select different features
const top = (m) => m.indexOf(Math.max(...m));
check("TNM: steps select different features (division of labour)",
  new Set(st.masks.map(top)).size === 3);

// gamma = 1: the leftover budget is exactly (1 - M), so full use bans and partial use suppresses
api.setGamma(1.0);
st = api.state();
check("TNM: gamma = 1 -> prior is exactly 1 - M (full use would ban)",
  st.priors[1].every((p, j) => near(p, 1 - st.masks[0][j], 1e-9)));
check("TNM: gamma = 1 leaves untouched features at prior 1",
  st.priors[1].every((p, j) => (st.masks[0][j] > 0 ? p < 1 : near(p, 1, 1e-9))));
const suppression1 = Math.min(...st.priors[1].filter((p, j) => st.masks[0][j] > 0));

// higher gamma relaxes the suppression: the spent feature keeps a larger share of its budget
api.setGamma(2.5);
st = api.state();
const suppression25 = Math.min(...st.priors[1].filter((p, j) => st.masks[0][j] > 0));
check("TNM: higher gamma suppresses a spent feature LESS", suppression25 > suppression1);
check("TNM: no feature is permanently banned at high gamma", st.priors[1].every((p) => p > 0));

// aggregate is a normalised distribution
check("TNM: aggregate M_agg sums to 1", near(st.agg.reduce((a, b) => a + b, 0), 1, 1e-9));

// readout reacts to the step
api.setGamma(1.5);
api.setStep(0);
check("TNM: step-1 readout says nothing has been spent yet",
  /nothing has been spent/i.test(readout.innerHTML));
api.setStep(1);
check("TNM: step-2 readout explains the prior", /prior/i.test(readout.innerHTML));
check("TNM: readout names sparsemax vs softmax",
  /sparsemax, not softmax/i.test(readout.innerHTML));
api.setStep(2);
check("TNM: step-3 readout mentions M_agg", /M_agg/.test(readout.innerHTML));

// step buttons are clickable and select
const btns = walk(c, (x) => x.tag === "button", []);
check("TNM: one button per decision step", btns.length === 3);
btns[0].click();
check("TNM: clicking a step button selects it", api.getStep() === 0);

// ------------------------------------------- lesson <-> viz CSS coupling
const lesson = fs.readFileSync(path.join(__dirname, "..", "lessons", "0043-tabnet.html"), "utf8");

const tnmClasses = ["tnm-viz", "tnm-ctl", "tnm-glab", "tnm-slider", "tnm-svg", "tnm-feat",
  "tnm-feat-on", "tnm-rowlab", "tnm-cell", "tnm-cell-on", "tnm-axis", "tnm-readout",
  "tnm-mono", "tnm-note", "tnm-caption"];
tnmClasses.forEach((cls) => check("lesson stylesheet defines ." + cls, lesson.includes("." + cls)));
check("lesson stylesheet defines .tnm-on (active step button)", lesson.includes(".tnm-on"));

check("lesson mounts the mask widget", lesson.includes('id="masks"'));
check("lesson mounts the warm-up", lesson.includes('id="warmup"'));
check("lesson warm-up draws from lessons before 43", lesson.includes("upTo: 43"));
check("lesson mounts the Syn4 predict widget", lesson.includes('id="predict-syn4"'));
check("lesson mounts the bake-off predict widget", lesson.includes('id="predict-bakeoff"'));
check("lesson mounts the teachback widget", lesson.includes('id="teachback-attention"'));
check("lesson loads the new viz script", lesson.includes("tabnet-mask-viz.js"));
check("lesson cites the primary reading arXiv id", lesson.includes("1908.07442"));
check("lesson cites sparsemax", lesson.includes("1602.02068"));
check("lesson cites the synthetic-data source", lesson.includes("1802.07814"));

// ---------------------------------- mechanism content must all be explained (#17 thoroughness)
["sparsemax", "prior scale", "relaxation", "GLU", "ghost", "virtual batch",
  "decision step", "instance-wise"].forEach(
  (t) => check("lesson explains: " + t, new RegExp(t, "i").test(lesson)));
check("lesson gives the sparsemax algorithm (tau threshold)", /τ\(z\)/.test(lesson));
check("lesson gives the mask equation", /M\[i\]\s*=\s*sparsemax/.test(lesson));
check("lesson gives the prior-scale equation", /P\[i\]\s*=\s*∏/.test(lesson));
check("lesson gives the sparsity penalty", /L_sparse/.test(lesson));
check("lesson gives the decision aggregation", /d_out\s*=\s*Σ/.test(lesson));

// ---------------------------------- verified numbers (must match labs/_verify_l043_results.json)
const res = JSON.parse(fs.readFileSync(path.join(__dirname, "_verify_l043_results.json"), "utf8"));
const bk = res.bakeoff;
check("verified: TabNet mean rank 2.50 in results", near(bk.mean_rank.tabnet, 2.5, 1e-9));
check("lesson states TabNet's mean rank 2.50", lesson.includes("2.50"));
check("lesson states MLP mean rank 1.75", lesson.includes("1.75"));
check("lesson states ResNet mean rank 2.00", lesson.includes("2.00"));
check("lesson states the Friedman p = 0.127", lesson.includes("0.127") && /friedman/i.test(lesson));
check("results Friedman p is 0.127", near(bk.friedman.p, 0.127, 1e-9));
check("lesson states credit_g TabNet 0.748", lesson.includes("0.748"));
check("results credit_g TabNet is 0.748", near(bk.per_dataset.credit_g.tabnet.mean, 0.748, 5e-4));
check("lesson states the Syn2 mask mass 76.8%", lesson.includes("76.8"));
check("results Syn2 mass on relevant is 0.768", near(res.mask_reading.syn2.mass_on_relevant, 0.768, 5e-4));
check("lesson states the Syn4 partial recovery 15.6% / 97.9%",
  lesson.includes("15.6") && lesson.includes("97.9"));
check("lesson names the four verified datasets",
  ["credit_g", "diabetes", "blood_transfusion", "kc1"].every((d) => lesson.includes(d)));

// ---------------------------------- honesty content (#20 / #22 / #23)
check("lesson reports the reference-validation FAILURE openly",
  /outside tolerance/i.test(lesson) && lesson.includes("0.053"));
check("lesson reports the exact sparsemax validation", /2\.3\s*×\s*10⁻⁷|2\.3e-07/i.test(lesson));
check("lesson names both refuted hypotheses (training length + LR schedule)",
  /training length/i.test(lesson) && /learning-rate schedule|StepLR/i.test(lesson));
check("lesson says the discrepancy is unexplained", /unexplained|not yet identified/i.test(lesson));
check("lesson separates 'did not clear the bar' from 'significantly worse'",
  /did not clear the bar/i.test(lesson) && /significantly worse/i.test(lesson));
check("lesson flags four datasets as a demonstration, not the verdict",
  /demonstration, not the verdict/i.test(lesson));
check("lesson cites the large benchmarks for the strong claim",
  lesson.includes("2106.11959") && lesson.includes("2207.08815"));
check("lesson notes the paper needed 10M samples for sharp masks", /10M/.test(lesson));
check("lesson notes the paper's own KDD appendix tie", /Appendix A/.test(lesson));

console.log(pass ? "\nALL L043 VIZ CHECKS PASS" : "\nL043 VIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
