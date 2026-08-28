// Headless mount check for the L045 viz assets (no jsdom):
//   assets/rtd-pretrain-viz.js       (new — the RTD self-supervision pretext)
//   assets/label-efficiency-viz.js   (new — the pre-training payoff curve)
// Reused from L032: tabtransformer-arch-viz.js, attention-context-viz.js (mount smoke + lesson coupling).
// Also asserts the lesson mounts the pedagogy widgets (retrieval upTo:45, predicts, teachback), defines
// every .rtd-/.leff- class the widgets emit, and states the verified L045 numbers.
const fs = require("fs");
const path = require("path");

function makeEl(tag) {
  const el = {
    tag, children: [], attrs: {}, style: {}, dataset: {}, _text: "", _html: "", _cls: new Set(),
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
    appendChild(c) { this.children.push(c); return c; },
    get firstChild() { return this.children.length ? this.children[0] : null; },
    removeChild(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); return c; },
    addEventListener(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); },
    click() { (this.listeners.click || []).forEach((fn) => fn()); },
    querySelector(sel) { const r = this.querySelectorAll(sel); return r.length ? r[0] : null; },
    querySelectorAll(sel) {
      const out = [];
      if (sel.charAt(0) === "#") walk(this, (x) => x.attrs && x.attrs.id === sel.slice(1), out);
      else if (sel.charAt(0) === ".") walk(this, (x) => x._cls && x._cls.has(sel.slice(1)), out);
      else walk(this, (x) => x.tag === sel, out);
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

// --------------------------------------------------- rtd-pretrain-viz
load("rtd-pretrain-viz.js");
const RTD = global.window.RtdPretrainViz;
check("rtd: p=0 corrupts nothing (no signal without corruption)",
  RTD.corrupt(0, 7).every((s) => !s.replaced && !s.draw));
check("rtd: a replaced cell has draw=true AND value actually changed",
  RTD.corrupt(1, 7).filter((s) => s.replaced).every((s) => s.draw && s.newIdx !== s.origIdx));
check("rtd: some drawn cells collide with the original and are NOT counted replaced (p·(1-1/card))",
  RTD.corrupt(1, 3).some((s) => s.draw && !s.replaced));
check("rtd: corruption is deterministic in (p, seed)",
  JSON.stringify(RTD.corrupt(0.4, 11)) === JSON.stringify(RTD.corrupt(0.4, 11)));
check("rtd: higher p replaces at least as many cells (monotone in expectation)",
  RTD.corrupt(0.8, 5).filter((s) => s.replaced).length >= RTD.corrupt(0.1, 5).filter((s) => s.replaced).length);
// effective fraction = mean_j p*(1 - 1/card_j)
const expFrac = RTD.expectedReplacedFraction(0.3);
const manual = RTD.columns.reduce((a, c) => a + 0.3 * (1 - 1 / c.levels.length), 0) / RTD.columns.length;
check("rtd: expectedReplacedFraction = mean p·(1-1/card), strictly below p", near(expFrac, manual, 1e-12) && expFrac < 0.3);
check("rtd: mount returns an api and paints 8 cells", (() => {
  const c = global.document.createElement("div"); const api = RTD.mount(c, { seed: 7 });
  return c.querySelectorAll(".rtd-cell").length === 8 && typeof api.set === "function";
})());

// --------------------------------------------------- label-efficiency-viz
load("label-efficiency-viz.js");
const LEFF = global.window.LabelEfficiencyViz;
check("leff: two measured anchors (0.03 and 0.10)", LEFF.data.length === 2 &&
  near(LEFF.data[0].frac, 0.03) && near(LEFF.data[1].frac, 0.10));
check("leff: lift at 3% is +0.008 (verified)", near(LEFF.liftAt(0.03), 0.008, 1e-9));
check("leff: lift at 10% is +0.001 (verified)", near(LEFF.liftAt(0.10), 0.001, 1e-9));
check("leff: pre-training beats scratch at BOTH fractions", LEFF.data.every((d) => d.pretrain >= d.scratch));
check("leff: the lift is LARGER at fewer labels (3% > 10%)", LEFF.liftAt(0.03) > LEFF.liftAt(0.10));
check("leff: 3% lift is all-seeds-positive, 10% is not (matches verify)",
  LEFF.data[0].allpos === true && LEFF.data[1].allpos === false);
check("leff: mount paints an svg with polylines + a readout", (() => {
  const c = global.document.createElement("div"); LEFF.mount(c, {});
  return c.querySelectorAll("polyline").length === 2 && c.querySelectorAll(".leff-readout").length === 1;
})());

// ------------------------------------------- lesson <-> viz CSS + mount coupling
const lesson = fs.readFileSync(path.join(__dirname, "..", "lessons", "0045-tabtransformer.html"), "utf8");
["rtd-viz", "rtd-ctl", "rtd-glab", "rtd-mono", "rtd-slider", "rtd-row", "rtd-cell", "rtd-rep",
  "rtd-cname", "rtd-cval", "rtd-ctag", "rtd-readout"].forEach((cls) =>
  check("lesson stylesheet defines ." + cls, lesson.includes("." + cls)));
["leff-viz", "leff-ctl", "leff-glab", "leff-mono", "leff-slider", "leff-svg", "leff-axis", "leff-leg",
  "leff-readout", "leff-pos"].forEach((cls) =>
  check("lesson stylesheet defines ." + cls, lesson.includes("." + cls)));
// reused L032 viz styles must be present too (arch .tta-, contextual .atc-)
["tta-viz", "tta-svg", "atc-viz", "atc-grid", "atc-card"].forEach((cls) =>
  check("lesson stylesheet defines reused ." + cls, lesson.includes("." + cls)));

check("lesson mounts the warm-up drawing from lessons before 45", lesson.includes('id="warmup"') && lesson.includes("upTo: 45"));
check("lesson mounts the architecture viz (reused)", lesson.includes('id="arch"') && lesson.includes("tabtransformer-arch-viz.js"));
check("lesson mounts the contextual viz (reused)", lesson.includes('id="contextual"') && lesson.includes("attention-context-viz.js"));
check("lesson mounts the RTD viz (new)", lesson.includes('id="rtd"') && lesson.includes("rtd-pretrain-viz.js"));
check("lesson mounts the label-efficiency viz (new)", lesson.includes('id="leff"') && lesson.includes("label-efficiency-viz.js"));
check("lesson mounts the contextual predict", lesson.includes('id="predict-contextual"'));
check("lesson mounts the ssl predict", lesson.includes('id="predict-ssl"'));
check("lesson mounts the teachback", lesson.includes('id="teachback-ssl"'));
check("lesson cites the primary reading arXiv id (TabTransformer)", lesson.includes("2012.06678"));
check("lesson cites the attention paper arXiv id", lesson.includes("1706.03762"));

// ---------------------------------- mechanism content must all be explained (#17 thoroughness)
["contextual", "context-free", "self-attention", "replaced token detection", "RTD", "pre-train",
  "fine-tun", "n_layers", "numeric", "LayerNorm"].forEach((t) =>
  check("lesson explains: " + t, new RegExp(t, "i").test(lesson)));

// ---------------------------------- verified numbers (must match labs/_verify_l045_results.json)
const res = JSON.parse(fs.readFileSync(path.join(__dirname, "_verify_l045_results.json"), "utf8"));
const bk = res.bakeoff, ss = res.semisupervised.summary, mech = res.mechanism;
check("results: TabTransformer mean rank 2.33", near(bk.mean_rank.tabtransformer, 2.33, 1e-9));
check("results: context-free mean rank 2.67", near(bk.mean_rank.context_free, 2.67, 1e-9));
check("results: CatBoost mean rank 1.00 (wins all)", near(bk.mean_rank.catboost, 1.0, 1e-9));
check("results: Friedman p is 0.097", near(bk.friedman.p, 0.097, 1e-9));
check("results: contextual beats context-free 2/3", bk.contextual_beats_contextfree === "2/3");
check("results: TabTransformer beats CatBoost 0/3", bk.tabtransformer_beats_catboost === "0/3");
check("results: attention validated to machine precision (sdpa)", mech.sdpa_max_abs_delta < 1e-12);
check("results: attention validated to machine precision (mha)", mech.mha_max_abs_delta < 1e-12);
check("results: contextual moves, context-free does not", mech.contextual_move_attention > 0.1 && mech.contextual_move_contextfree < 1e-9);
check("results: semi-sup lift +0.008 at 3%, all seeds positive", near(ss["0.03"].lift, 0.008, 1e-9) && ss["0.03"].lift_all_seeds_positive === true);
check("results: semi-sup lift +0.001 at 10%", near(ss["0.10"].lift, 0.001, 1e-9));

// lesson states the verified numbers
check("lesson states the mean ranks 2.33 / 2.67 / 1.00",
  lesson.includes("2.33") && lesson.includes("2.67") && lesson.includes("1.00"));
check("lesson states the Friedman p = 0.097", lesson.includes("0.097") && /friedman/i.test(lesson));
check("lesson names the three verified datasets",
  ["credit_g", "adult", "churn"].every((d) => lesson.includes(d)));
check("lesson states the +0.008 lift at 3% labels", lesson.includes("+0.008") && /3\s*%|3&nbsp;%/.test(lesson));
check("lesson states contextual beats context-free 2 of 3", /2 of 3|2\/3/.test(lesson));
check("lesson quotes the machine-precision attention deltas", /10⁻¹⁶|10\^-16|1e-16/.test(lesson));

// ---------------------------------- honesty content (#20 / #22 / #23)
check("lesson states TabTransformer only MATCHES / loses to trees on flat tables",
  /matches?/i.test(lesson) && /CatBoost wins all/i.test(lesson));
check("lesson flags the run as a demonstration, not the paper's benchmark", /demonstration/i.test(lesson));
check("lesson names the numeric-bypass limitation and FT-Transformer as the fix",
  /bypass/i.test(lesson) && /FT-Transformer/i.test(lesson));
check("lesson keeps the honest 'lift is small at this scale' note", /small at this/i.test(lesson));

console.log(pass ? "\nALL L045 VIZ CHECKS PASS" : "\nL045 VIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
