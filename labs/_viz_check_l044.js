// Headless mount check for the L044 viz asset (no jsdom):
//   assets/node-tree-viz.js  (new — a differentiable oblivious tree: entmoid split + outer-product routing)
// Also asserts the lesson mounts the pedagogy widgets (retrieval upTo:44, predict, teachback),
// defines every .nod- class the widget emits, and states the verified L044 numbers.
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
    addEventListener(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); },
    click() { (this.listeners.click || []).forEach((fn) => fn()); },
    querySelector(sel) { const r = this.querySelectorAll(sel); return r.length ? r[0] : null; },
    querySelectorAll(sel) {
      const out = [];
      if (sel.charAt(0) === "#") {
        const id = sel.slice(1);
        walk(this, (x) => x.attrs && x.attrs.id === id, out);
      } else if (sel.charAt(0) === ".") {
        const cls = sel.slice(1);
        walk(this, (x) => x._cls && x._cls.has(cls), out);
      } else {
        walk(this, (x) => x.tag === sel, out);
      }
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
const sum = (a) => a.reduce((x, y) => x + y, 0);
const argmax = (a) => a.indexOf(Math.max(...a));

// --------------------------------------------------- node-tree-viz
load("node-tree-viz.js");
const NTV = global.window.NodeTreeViz;

// (1) entmoid = two-class 1.5-entmax: P(right). Test its defining properties.
const em = NTV.entmoid15;
check("entmoid: 0 -> 0.5 (indifference)", near(em(0), 0.5, 1e-9));
check("entmoid: is antisymmetric  em(-x) = 1 - em(x)",
  [0.3, 0.9, 1.7, 2.5].every((x) => near(em(-x), 1 - em(x), 1e-12)));
check("entmoid: stays in [0,1]",
  [-5, -2, -0.5, 0, 0.5, 2, 5].every((x) => em(x) >= 0 && em(x) <= 1));
check("entmoid: SATURATES to exact 1 for a decisive positive gap (|x| >= sqrt 8)", em(3) === 1);
check("entmoid: SATURATES to exact 0 for a decisive negative gap", em(-3) === 0);
check("entmoid: monotonically increasing", em(0.5) > em(0) && em(1.5) > em(0.5));

// (2) outer-product routing is a real distribution over 2^3 leaves.
const lw = NTV.leafWeights;
const w1 = lw([0.7, 0.3, 0.9]);
check("routing: exactly 8 leaves (depth 3)", w1.length === 8);
check("routing: weights sum to 1 (a distribution)", near(sum(w1), 1, 1e-12));
check("routing: weights non-negative", w1.every((v) => v >= 0));
// leaf 5 = 101b -> right,left,right -> c0*(1-c1)*c2
check("routing: a leaf weight is the PRODUCT of its per-level choices (outer product)",
  near(w1[5], 0.7 * (1 - 0.3) * 0.9, 1e-12));
// a HARD tree (choices in {0,1}) puts all mass on ONE leaf
const wh = lw([1, 0, 1]);
check("routing: hard choices {0,1} give a ONE-HOT leaf (ordinary tree)",
  wh.filter((v) => v === 1).length === 1 && wh.filter((v) => v === 0).length === 7);

// (3) the pure forward pass (model): the invariants the widget renders
const Z = [1.2, -0.8, 0.4];               // the widget's default row
const m1 = NTV.model(Z, 1.0, false);
check("model: default routing sums to 1", near(sum(m1.weights), 1, 1e-12));
check("model: default routing is SOFT (spread, not one-hot)", Math.max(...m1.weights) < 0.95);
check("model: output = Σ wₖ·Rₖ over the fixed responses",
  near(m1.output, m1.weights.reduce((acc, wk, k) => acc + wk * NTV.responses[k], 0), 1e-12));

// tau -> 0.1 : routing collapses onto the argmax leaf, matching the hard tree
const mSmall = NTV.model(Z, 0.1, false);
check("model: small tau makes routing ~one-hot", Math.max(...mSmall.weights) > 0.99);
const mHard = NTV.model(Z, 1.0, true);
check("model: the collapsed soft leaf is the SAME as the hard-tree argmax leaf",
  argmax(mSmall.weights) === argmax(mHard.weights));

// tau -> 3 : routing flattens toward uniform (max weight shrinks)
const flatMax = Math.max(...NTV.model(Z, 3.0, false).weights);
const sharpMax = Math.max(...NTV.model(Z, 0.5, false).weights);
check("model: larger tau flattens the leaf distribution (less peaky)", flatMax < sharpMax);

// hard toggle is a genuine one-hot decision tree
check("model: hard tree routes all mass to ONE leaf",
  mHard.weights.filter((v) => v === 1).length === 1);
check("model: hard tree is exactly the sign-of-gap decision",
  mHard.choices.every((ci, i) => ci === (Z[i] >= 0 ? 1 : 0)));

// ------------------------------------------- lesson <-> viz CSS coupling
const lesson = fs.readFileSync(path.join(__dirname, "..", "lessons", "0044-node.html"), "utf8");
const nodClasses = ["nod-viz", "nod-ctl", "nod-glab", "nod-slider", "nod-zslider", "nod-mono",
  "nod-levels", "nod-lrow", "nod-lname", "nod-lval", "nod-svg", "nod-leaflab", "nod-leafw",
  "nod-leafr", "nod-axis", "nod-readout", "nod-note"];
nodClasses.forEach((cls) => check("lesson stylesheet defines ." + cls, lesson.includes("." + cls)));
check("lesson stylesheet defines .nod-on (active hard-tree button)", lesson.includes(".nod-on"));

check("lesson mounts the tree widget", lesson.includes('id="tree"'));
check("lesson mounts the warm-up", lesson.includes('id="warmup"'));
check("lesson warm-up draws from lessons before 44", lesson.includes("upTo: 44"));
check("lesson mounts the CatBoost predict widget", lesson.includes('id="predict-catboost"'));
check("lesson mounts the teachback widget", lesson.includes('id="teachback-diff"'));
check("lesson loads the new viz script", lesson.includes("node-tree-viz.js"));
check("lesson cites the primary reading arXiv id (NODE)", lesson.includes("1909.06312"));
check("lesson cites the entmax paper arXiv id", lesson.includes("1905.05702"));

// ---------------------------------- mechanism content must all be explained (#17 thoroughness)
["oblivious", "entmax15", "entmoid", "outer product", "DenseNet", "compose", "temperature",
  "argmax"].forEach((t) => check("lesson explains: " + t, new RegExp(t, "i").test(lesson)));

// ---------------------------------- verified numbers (must match labs/_verify_l044_results.json)
const res = JSON.parse(fs.readFileSync(path.join(__dirname, "_verify_l044_results.json"), "utf8"));
const bk = res.bakeoff;
check("results: NODE mean rank 3.50 (last)", near(bk.mean_rank.node, 3.5, 1e-9));
check("results: CatBoost mean rank 2.50", near(bk.mean_rank.catboost, 2.5, 1e-9));
check("results: Friedman p is 0.308", near(bk.friedman.p, 0.308, 1e-9));
check("results: NODE beats CatBoost on 1/4", bk.node_beats_catboost === "1/4");
check("results: mechanism validated to machine precision (entmax15)",
  res.mechanism.entmax15_max_abs_delta < 1e-12);
check("results: mechanism validated to machine precision (entmoid)",
  res.mechanism.entmoid15_max_abs_delta < 1e-12);
check("lesson states NODE's mean rank 3.50 (behind the baselines)", lesson.includes("3.50"));
check("lesson states CatBoost mean rank 2.50", lesson.includes("2.50"));
check("lesson states MLP/ResNet mean rank 2.00", lesson.includes("2.00"));
check("lesson states the Friedman p = 0.308", lesson.includes("0.308") && /friedman/i.test(lesson));
check("lesson names the four verified datasets",
  ["credit_g", "diabetes", "blood_transfusion", "kc1"].every((d) => lesson.includes(d)));
check("lesson states the ~70x training-cost gap", /70×|70x|~?70/.test(lesson) && lesson.includes("60 s"));
check("lesson quotes the machine-precision mechanism deltas", lesson.includes("5.6") && lesson.includes("3.3"));

// ---------------------------------- honesty content (#20 / #22 / #23)
check("lesson separates 'did not clear the bar' from 'significantly worse'",
  /did not clear/i.test(lesson) && /significantly worse/i.test(lesson));
check("lesson flags four datasets as a demonstration, not the verdict",
  /demonstration/i.test(lesson));
check("lesson makes the honest point that value is COMPOSITION not flat-table accuracy",
  /compose|composition/i.test(lesson));

console.log(pass ? "\nALL L044 VIZ CHECKS PASS" : "\nL044 VIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
