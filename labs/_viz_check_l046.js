// Headless mount check for the L046 viz assets (no jsdom):
//   assets/tokenizer-viz.js   (new — the FT-Transformer feature tokenizer + [CLS] data-flow)
// Reused from L032/L045: tabtransformer-arch-viz.js (mount smoke + lesson coupling, shown for contrast).
// Also asserts the lesson mounts the pedagogy widgets (retrieval upTo:46, predict, teachback), defines
// every .tok- class the widget emits, and states the verified L046 numbers.
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

// --------------------------------------------------- tokenizer-viz
load("tokenizer-viz.js");
const TOK = global.window.TokenizerViz;

// the numeric token is affine: T = b + x*W (element-wise). Worked example b=[1,5], W=[2,0].
check("tok: numericToken(3,[2,0],[1,5]) = [7,5] (affine b + x*W)",
  JSON.stringify(TOK.numericToken(3, [2, 0], [1, 5])) === JSON.stringify([7, 5]));
check("tok: bumping x by 1 moves the token by exactly W (affine)",
  (() => { const a = TOK.numericToken(3, [2, 0], [1, 5]), b = TOK.numericToken(4, [2, 0], [1, 5]);
    return near(b[0] - a[0], 2) && near(b[1] - a[1], 0); })());
check("tok: [CLS] is token 0, then one token per feature",
  TOK.tokens()[0].kind === "cls" && TOK.tokens().length === TOK.features.length + 1);
check("tok: the demo row has both numeric AND categorical features",
  TOK.features.some((f) => f.kind === "num") && TOK.features.some((f) => f.kind === "cat"));
check("tok: numFrac is 2/5 = 0.4 for the demo row", near(TOK.numFrac(), 0.4, 1e-9));
check("tok: mount paints an svg with token boxes + a readout, and set() switches stage", (() => {
  // NOTE: the fake DOM maps class only via className/classList, not setAttribute('class') on svg nodes,
  // so we count svg children by TAG (rect) and the readout div by its className.
  const c = global.document.createElement("div"); const api = TOK.mount(c, {});
  const rects = c.querySelectorAll("rect").length;
  const ro = c.querySelectorAll(".tok-readout").length;
  if (typeof api.set !== "function") return false;
  api.set("tok");   // must not throw and must repaint
  return rects >= TOK.tokens().length && ro === 1 && c.querySelectorAll("rect").length >= TOK.tokens().length;
})());

// --------------------------------------------------- tabtransformer-arch-viz (reused, for contrast)
load("tabtransformer-arch-viz.js");
const TTA = global.window.TabTransformerArchViz;
check("tta (reused): mount returns an api with set()", (() => {
  const c = global.document.createElement("div"); const api = TTA.mount(c, {});
  return typeof api.set === "function" && c.querySelectorAll("svg").length === 1;
})());

// ------------------------------------------- lesson <-> viz CSS + mount coupling
const lesson = fs.readFileSync(path.join(__dirname, "..", "lessons", "0046-ft-transformer.html"), "utf8");
["tok-viz", "tok-ctl", "tok-svg", "tok-box", "tok-boxhead", "tok-boxsub", "tok-lane", "tok-readout",
  "tok-on"].forEach((cls) =>
  check("lesson stylesheet defines ." + cls, lesson.includes("." + cls)));
["tta-viz", "tta-svg", "tta-boxhead", "tta-lane", "tta-readout"].forEach((cls) =>
  check("lesson stylesheet defines reused ." + cls, lesson.includes("." + cls)));

check("lesson mounts the warm-up drawing from lessons before 46", lesson.includes('id="warmup"') && lesson.includes("upTo: 46"));
check("lesson mounts the tokenizer viz (new)", lesson.includes('id="tokenizer"') && lesson.includes("tokenizer-viz.js"));
check("lesson mounts the TabTransformer arch viz (reused, contrast)", lesson.includes('id="tabt-arch"') && lesson.includes("tabtransformer-arch-viz.js"));
check("lesson mounts the bake-off predict", lesson.includes('id="predict-bakeoff"'));
check("lesson mounts the tokenizer teachback", lesson.includes('id="teachback-tokenizer"'));
check("lesson cites the primary reading arXiv id (FT-Transformer / Gorishniy)", lesson.includes("2106.11959"));
check("lesson cites the attention paper arXiv id", lesson.includes("1706.03762"));

// ---------------------------------- mechanism content must all be explained (#17 thoroughness)
["feature tokenizer", "\\[CLS\\]", "numeric", "categorical", "affine", "attend", "TabTransformer",
  "PreNorm", "readout", "num-frac"].forEach((t) =>
  check("lesson explains: " + t.replace("\\\\", ""), new RegExp(t, "i").test(lesson)));

// ---------------------------------- verified numbers (must match labs/_verify_l046_results.json)
const res = JSON.parse(fs.readFileSync(path.join(__dirname, "_verify_l046_results.json"), "utf8"));
const bk = res.bakeoff, mech = res.mechanism;
check("results: FT-Transformer mean rank 2.50", near(bk.mean_rank.ft_transformer, 2.5, 1e-9));
check("results: MLP mean rank 2.75", near(bk.mean_rank.mlp, 2.75, 1e-9));
check("results: TabTransformer mean rank 3.75", near(bk.mean_rank.tabtransformer, 3.75, 1e-9));
check("results: CatBoost mean rank 1.00 (wins all)", near(bk.mean_rank.catboost, 1.0, 1e-9));
check("results: Friedman p is 0.026", near(bk.friedman.p, 0.026, 1e-9));
check("results: FT-T beats TabTransformer 3/4", bk.ft_beats_tabtransformer === "3/4");
check("results: FT-T beats CatBoost 0/4", bk.ft_beats_catboost === "0/4");
check("results: FT-T is the best neural model", bk.ft_is_best_neural === true);
check("results: attention validated to machine precision", mech.sdpa_max_abs_delta < 1e-12);
check("results: numeric change moves FT-T [CLS], not TabTransformer",
  mech.ft_cls_move_numeric > 0.1 && mech.tabtransformer_ctx_move_numeric < 1e-9);

// lesson states the verified numbers
check("lesson states the mean ranks 2.50 / 2.75 / 3.75 / 1.00",
  lesson.includes("2.50") && lesson.includes("2.75") && lesson.includes("3.75") && lesson.includes("1.00"));
check("lesson states the Friedman p = 0.026", lesson.includes("0.026") && /friedman/i.test(lesson));
check("lesson names the four verified datasets",
  ["credit_g", "adult", "churn", "phoneme"].every((d) => lesson.includes(d)));
check("lesson states FT-T beats TabTransformer 3 of 4", /3 of 4|3\/4/.test(lesson));
check("lesson states the numerics-attend move 0.438 vs 0.0",
  lesson.includes("0.438") && /0\.0\b/.test(lesson));

// ---------------------------------- honesty content (#20 / #22 / #23 / #25)
check("lesson states FT-T is the strongest single neural baseline", /strongest single neural/i.test(lesson));
check("lesson states CatBoost still wins all tables (loses to trees)",
  /wins all|wins .* 4|0 of 4|0\/4/i.test(lesson) && /CatBoost/.test(lesson));
check("lesson flags the run as a demonstration, not the paper's benchmark", /demonstration/i.test(lesson));
check("lesson keeps the paper-results / INCOMPARABLE honesty note",
  /INCOMPARABLE/.test(lesson) && /cited, not reproduced/i.test(lesson));
check("lesson names the numeric-bypass fix vs TabTransformer", /bypass/i.test(lesson) && /TabTransformer/.test(lesson));

console.log(pass ? "\nALL L046 VIZ CHECKS PASS" : "\nL046 VIZ CHECKS FAILED");
process.exit(pass ? 0 : 1);
