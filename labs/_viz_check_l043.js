// Headless mount check for the L043 viz assets (no jsdom):
//   assets/tabnet-mask-viz.js    (sequential attention: masks, prior scale, aggregate)
//   assets/tabnet-arch-viz.js    (the encoder data flow, paper Fig. 4a)
//   assets/sparsemax-viz.js      (sparsemax as a solved-for threshold, vs softmax)
//   assets/tabnet-fblock-viz.js  (the feature transformer opened up, paper Fig. 4c)
// The two hand-laid-out diagrams also get GEOMETRY checks — every box inside the viewBox and no two
// boxes overlapping — because SVG coordinates written by hand are exactly the thing that silently rots.
// Also asserts the lesson mounts the pedagogy widgets (retrieval upTo:43, two predicts, teachback),
// defines every viz class the widgets emit, and states the verified L043 numbers.
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
    get firstChild() { return this.children[0]; },
    removeChild(c) {
      const i = this.children.indexOf(c);
      if (i >= 0) this.children.splice(i, 1);
      return c;
    },
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

// --------------------------------------------------- shared helpers for the diagram assets
const texts = (root) => walk(root, (x) => x.tag === "text", []).map((t) => t._text);
const cls = (root, name) =>
  walk(root, (x) => x.tag !== undefined && x.attrs && x.attrs.class === name, []);

/** Every filled box must sit inside the viewBox, and no two of them may overlap. */
function geometry(prefix, root) {
  const svg = walk(root, (x) => x.tag === "svg", [])[0];
  const vb = svg.getAttribute("viewBox").split(/\s+/).map(Number);
  const boxes = walk(root, (x) => x.tag === "rect", [])
    .filter((r) => r.getAttribute("fill") !== "none")
    .map((r) => ({
      x: +r.getAttribute("x"), y: +r.getAttribute("y"),
      w: +r.getAttribute("width"), h: +r.getAttribute("height")
    }));
  check(prefix + ": boxes were drawn", boxes.length > 0);
  check(prefix + ": every box is inside the viewBox",
    boxes.every((b) => b.x >= 0 && b.y >= 0 && b.x + b.w <= vb[2] && b.y + b.h <= vb[3]));
  let clash = null;
  for (let i = 0; i < boxes.length && !clash; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i], b = boxes[j];
      if (a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h) {
        clash = JSON.stringify([a, b]);
        break;
      }
    }
  }
  check(prefix + ": no two boxes overlap" + (clash ? " — " + clash : ""), clash === null);
  // labels must be inside the frame too, or they are clipped away
  const labels = walk(root, (x) => x.tag === "text", []);
  check(prefix + ": every label is inside the viewBox",
    labels.every((t) => {
      const x = +t.getAttribute("x"), y = +t.getAttribute("y");
      return x >= 0 && x <= vb[2] && y >= 0 && y <= vb[3];
    }));
}

// --------------------------------------------------- tabnet-arch-viz (paper Fig. 4a)
load("tabnet-arch-viz.js");
const TNA = global.window.TabnetArchViz;
const ac = bind(makeEl("div"));
const aapi = TNA.mount(ac, {});
const aread = walk(ac, (x) => x._cls && x._cls.has("tna-readout"), [])[0];

check("TNA: seven stages, whole-encoder first",
  aapi.stages().join(",") === "all,seed,attend,mask,transform,prior,outputs");
check("TNA: default stage is the whole encoder", aapi.getStage() === "all");
geometry("TNA", ac);

const atext = texts(ac);
[["attentive transformer", "the block that computes the mask"],
 ["prior scale", "the memory that makes it sequential"],
 ["mask", "the multiplicative selection"],
 ["feature transformer", "the block that computes"],
 ["split", "d[i] / a[i]"],
 ["d_out = Σᵢ ReLU(d[i])", "the decision aggregation"],
 ["ŷ = W_final · d_out", "the output mapping"],
 ["M_agg — global attribution", "the interpretability output"]].forEach((b) =>
  check("TNA: diagram shows " + b[1], atext.includes(b[0])));
["f", "a[0]", "a[i]", "M[i]", "P[i−1]", "d[i]"].forEach((e) =>
  check("TNA: edge labelled " + e, atext.includes(e)));
check("TNA: the step is marked as repeating N_steps times",
  atext.some((t) => /REPEATS N_steps TIMES/.test(t)));

check("TNA: default readout summarises the whole encoder", /whole encoder/i.test(aread.innerHTML));
aapi.setStage("seed");
check("TNA: seed stage explains where a[0] comes from",
  /a\[0\]/.test(aread.innerHTML) && /unmasked/i.test(aread.innerHTML));
aapi.setStage("attend");
check("TNA: attend stage prints the mask equation",
  /M\[i\] = sparsemax\( P\[i−1\] · h/.test(aread.innerHTML));
aapi.setStage("mask");
check("TNA: mask stage prints the masking and says a zero kills the gradient",
  /M\[i\] ⊙ f/.test(aread.innerHTML) && /no gradient/i.test(aread.innerHTML));
aapi.setStage("transform");
check("TNA: transform stage prints the split equation",
  /\[ d\[i\], a\[i\] \] = f/.test(aread.innerHTML));
aapi.setStage("prior");
check("TNA: prior stage prints the prior-scale product",
  /P\[i\] = ∏/.test(aread.innerHTML) && /γ = 1/.test(aread.innerHTML));
aapi.setStage("outputs");
check("TNA: outputs stage prints BOTH outputs",
  /d_out = Σ/.test(aread.innerHTML) && /M_agg = Σ/.test(aread.innerHTML));
check("TNA: outputs stage defines the step weight eta", /η\[i\] = Σ/.test(aread.innerHTML));
aapi.setStage("nonsense");
check("TNA: an unknown stage is ignored", aapi.getStage() === "outputs");

const abtns = walk(ac, (x) => x.tag === "button", []);
check("TNA: one button per stage", abtns.length === 7);
abtns[0].click();
check("TNA: clicking a stage button selects it", aapi.getStage() === "all");

// --------------------------------------------------- sparsemax-viz
load("sparsemax-viz.js");
const SPX = global.window.SparsemaxViz;
const sc = bind(makeEl("div"));
const sapi = SPX.mount(sc, {});
const sread = walk(sc, (x) => x._cls && x._cls.has("spx-readout"), [])[0];

check("SPX: default spread is 1.0x", near(sapi.getSpread(), 1.0, 1e-9));
let sst = sapi.state();
const sum = (a) => a.reduce((x, y) => x + y, 0);

check("SPX: default state has 4 survivors", sst.k === 4);
check("SPX: default state has 4 exact zeros", sst.zeros === 4);
check("SPX: sparsemax output sums to 1", near(sum(sst.mask), 1, 1e-9));
check("SPX: softmax output sums to 1", near(sum(sst.soft), 1, 1e-9));
check("SPX: sparsemax mask matches max(z - tau, 0) exactly",
  sst.mask.every((v, j) => near(v, Math.max(sst.z[j] - sst.tau, 0), 1e-12)));
check("SPX: the survivors are exactly the logits above tau",
  sst.mask.filter((v) => v > 0).length === sst.z.filter((z) => z > sst.tau).length);
check("SPX: softmax never produces a zero", sst.soft.every((v) => v > 0));
check("SPX: sparsemax is lower-entropy than softmax on the same logits",
  sst.maskEntropy < sst.softEntropy);
check("SPX: softmax entropy is below the ln D ceiling", sst.softEntropy < Math.log(8) + 1e-9);

sapi.setSpread(0.2);
sst = sapi.state();
check("SPX: flat logits -> every feature survives", sst.k === 8 && sst.zeros === 0);
check("SPX: flat logits -> tau goes negative (a dense mask)", sst.tau < 0);
check("SPX: flat mask still sums to 1", near(sum(sst.mask), 1, 1e-9));
check("SPX: flat readout says sparsity is a property of the logits",
  /negative/i.test(sread.innerHTML) && /separated/i.test(sread.innerHTML));

sapi.setSpread(3.0);
sst = sapi.state();
check("SPX: peaked logits -> 2 survivors, 6 zeros", sst.k === 2 && sst.zeros === 6);
check("SPX: peaked mask still sums to 1", near(sum(sst.mask), 1, 1e-9));
check("SPX: peaked softmax still has no zeros", sst.soft.every((v) => v > 0));
check("SPX: sparsemax stays lower-entropy at every spread", sst.maskEntropy < sst.softEntropy);
check("SPX: readout ties mask entropy to L_sparse", /L_sparse/.test(sread.innerHTML));
check("SPX: readout says softmax cannot switch a feature off",
  /exp\(z\) > 0/.test(sread.innerHTML));
check("SPX: readout notes both operators are shift-invariant",
  /shift-invariant/i.test(sread.innerHTML));

// monotonicity: more separated logits can only ever mean fewer survivors
let prevK = 9;
for (let s = 0.2; s <= 3.0001; s += 0.1) {
  sapi.setSpread(s);
  const k = sapi.state().k;
  if (k > prevK) { prevK = -1; break; }
  prevK = k;
}
check("SPX: survivor count is non-increasing as the logits separate", prevK !== -1);

const sbtns = walk(sc, (x) => x.tag === "button", []);
check("SPX: three preset buttons", sbtns.length === 3);
sbtns[0].click();
check("SPX: the 'flat' preset sets spread 0.2", near(sapi.getSpread(), 0.2, 1e-9));

// --------------------------------------------------- tabnet-fblock-viz (paper Fig. 4c)
load("tabnet-fblock-viz.js");
const TNF = global.window.TabnetFblockViz;
const fc = bind(makeEl("div"));
const fapi = TNF.mount(fc, {});
const fread = walk(fc, (x) => x._cls && x._cls.has("tnf-readout"), [])[0];

check("TNF: six stages, whole-block first",
  fapi.stages().join(",") === "all,shared,stepdep,glu,residual,split");
check("TNF: default stage is the whole block", fapi.getStage() === "all");
geometry("TNF", fc);

const ftext = texts(fc);
check("TNF: four FC -> BN -> GLU layers are drawn",
  ftext.filter((t) => t === "FC → BN → GLU").length === 4);
["shared 1", "shared 2", "step-dep 1", "step-dep 2"].forEach((l) =>
  check("TNF: layer labelled " + l, ftext.includes(l)));
check("TNF: three sqrt(0.5) merge nodes (one per residual)",
  ftext.filter((t) => t === "×√0.5").length === 3);
check("TNF: the merge nodes are additions", ftext.filter((t) => t === "+").length === 3);
check("TNF: the split names both halves",
  ftext.includes("d[i] → answer") && ftext.includes("a[i] → next mask"));
check("TNF: the GLU inset opens the layer",
  ftext.includes("FC: u → 2u") && ftext.includes("a · σ(b)") && ftext.includes("ghost BN"));
check("TNF: the one width change is annotated",
  ftext.some((t) => /D in → N_d \+ N_a after layer 1/.test(t)));

fapi.setStage("shared");
check("TNF: shared stage explains why the weights are shared",
  /same/i.test(fread.innerHTML) && /a\[0\]/.test(fread.innerHTML));
check("TNF: shared stage explains why the first layer has no residual",
  /no residual/i.test(fread.innerHTML) && /D → N_d \+ N_a/.test(fread.innerHTML));
fapi.setStage("stepdep");
check("TNF: step-dependent stage contrasts with the shared layers",
  /step-dependent/i.test(fread.innerHTML));
fapi.setStage("glu");
check("TNF: glu stage prints the gate", /a · σ\(b\)/.test(fread.innerHTML));
check("TNF: glu stage explains ghost BN and the virtual batch size",
  /ghost/i.test(fread.innerHTML) && /B_V/.test(fread.innerHTML));
fapi.setStage("residual");
check("TNF: residual stage prints the rescaled residual",
  /\( x \+ block\(x\) \) · √0\.5/.test(fread.innerHTML));
check("TNF: residual stage gives the variance argument", /variance/i.test(fread.innerHTML));
fapi.setStage("split");
check("TNF: split stage prints the split equation and both widths",
  /\[ d\[i\], a\[i\] \] = f/.test(fread.innerHTML) && /N_d/.test(fread.innerHTML) &&
  /N_a/.test(fread.innerHTML));

const fbtns = walk(fc, (x) => x.tag === "button", []);
check("TNF: one button per stage", fbtns.length === 6);
fbtns[0].click();
check("TNF: clicking a stage button selects it", fapi.getStage() === "all");

// ------------------------------------------- lesson <-> viz CSS coupling
const lesson = fs.readFileSync(path.join(__dirname, "..", "lessons", "0043-tabnet.html"), "utf8");

const tnmClasses = ["tnm-viz", "tnm-ctl", "tnm-glab", "tnm-slider", "tnm-svg", "tnm-feat",
  "tnm-feat-on", "tnm-rowlab", "tnm-cell", "tnm-cell-on", "tnm-axis", "tnm-readout",
  "tnm-mono", "tnm-note", "tnm-caption"];
tnmClasses.forEach((c) => check("lesson stylesheet defines ." + c, lesson.includes("." + c)));
check("lesson stylesheet defines .tnm-on (active step button)", lesson.includes(".tnm-on"));

["tna-viz", "tna-ctl", "tna-on", "tna-scroll", "tna-svg", "tna-head", "tna-sub", "tna-lane",
 "tna-edge", "tna-edge-on", "tna-readout", "tna-m", "tna-eq", "tna-caption",
 "spx-viz", "spx-ctl", "spx-on", "spx-scroll", "spx-svg", "spx-feat", "spx-feat-on", "spx-rowlab",
 "spx-tau", "spx-val", "spx-cell", "spx-cell-on", "spx-axis", "spx-lab", "spx-slider",
 "spx-readout", "spx-m", "spx-note", "spx-caption",
 "tnf-viz", "tnf-ctl", "tnf-on", "tnf-scroll", "tnf-svg", "tnf-head", "tnf-sub", "tnf-lane",
 "tnf-lane-on", "tnf-plus", "tnf-scale", "tnf-scale-on", "tnf-note-svg", "tnf-readout", "tnf-m",
 "tnf-eq", "tnf-caption"].forEach((c) =>
  check("lesson stylesheet defines ." + c, lesson.includes("." + c)));

check("lesson mounts the architecture diagram", lesson.includes('id="arch"'));
check("lesson mounts the sparsemax widget", lesson.includes('id="sparsemax"'));
check("lesson mounts the feature-transformer diagram", lesson.includes('id="fblock"'));
check("lesson loads all four viz scripts",
  ["tabnet-arch-viz.js", "sparsemax-viz.js", "tabnet-fblock-viz.js", "tabnet-mask-viz.js"]
    .every((f) => lesson.includes(f)));
check("lesson puts the architecture map BEFORE the equations",
  lesson.indexOf('id="arch"') < lesson.indexOf("M[i] = sparsemax"));
check("lesson puts the sparsemax picture next to Algorithm 1",
  lesson.indexOf("τ(z)") < lesson.indexOf('id="sparsemax"') &&
  lesson.indexOf('id="sparsemax"') < lesson.indexOf('id="fblock"'));
check("lesson credits the paper figures the diagrams redraw",
  /Fig\.\s*4a/.test(lesson) && /Fig\.\s*4c/.test(lesson));
check("lesson explains the a[0] seed the equations omit",
  /Where <code>a\[0\]<\/code> comes from/.test(lesson));
check("lesson says the diagrams were checked against relkit.tabnet",
  /labs\/relkit\/tabnet\.py/.test(lesson));

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
