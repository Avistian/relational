// Headless checks for the Q2-retrospective pedagogy assets (no jsdom):
//   assets/retrieval-pool.js   — pool integrity
//   assets/retrieval-bank.js   — spacing, count, interleaving, Leitner transitions
//   assets/predict.js          — commit-then-reveal
// Mirrors the DOM-stub style of labs/_viz_check_l019.js, with dataset + class-aware querySelectorAll.
const fs = require("fs");
const path = require("path");

function makeEl(tag) {
  const el = {
    tag, children: [], attrs: {}, style: {}, dataset: {}, _text: "", _html: "", _cls: new Set(),
    listeners: {}, disabled: false, type: "",
    set className(v) { this._cls = new Set(String(v).split(/\s+/).filter(Boolean)); },
    get className() { return [...this._cls].join(" "); },
    classList: {
      add(...c) { c.forEach((x) => this._owner._cls.add(x)); },
      remove(...c) { c.forEach((x) => this._owner._cls.delete(x)); },
      toggle(c, on) { on ? this._owner._cls.add(c) : this._owner._cls.delete(c); },
      contains(c) { return this._owner._cls.has(c); },
    },
    set textContent(v) { this._text = v; },
    get textContent() { return this._text; },
    set innerHTML(v) { this._html = v; if (v === "") this.children = []; },
    get innerHTML() { return this._html; },
    setAttribute(k, v) { this.attrs[k] = v; },
    getAttribute(k) { return this.attrs[k]; },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); },
    click() { (this.listeners.click || []).forEach((fn) => fn()); },
    querySelectorAll(sel) {
      const out = [];
      if (sel && sel[0] === ".") {
        const cls = sel.slice(1);
        (function walk(n) {
          (n.children || []).forEach((c) => { if (c._cls && c._cls.has(cls)) out.push(c); walk(c); });
        })(this);
      }
      return out;
    },
  };
  return el;
}
function bind(el) { el.classList._owner = el; return el; }
global.document = { createElement: (t) => bind(makeEl(t)), createElementNS: (_n, t) => bind(makeEl(t)) };
global.window = {};

function load(f) { eval(fs.readFileSync(path.join(__dirname, "..", "assets", f), "utf8")); }
load("retrieval-pool.js");
load("retrieval-bank.js");
load("predict.js");
const POOL = global.window.RETRIEVAL_POOL;
const RetrievalBank = global.window.RetrievalBank;
const Predict = global.window.Predict;

let pass = true;
function check(name, cond) { console.log((cond ? "ok  " : "FAIL") + "  " + name); if (!cond) pass = false; }

function collect(node, cls, out) {
  (node.children || []).forEach((c) => { if (c._cls && c._cls.has(cls)) out.push(c); collect(c, cls, out); });
  return out;
}
function memStorage() {
  const m = {};
  return { getItem: (k) => (k in m ? m[k] : null), setItem: (k, v) => { m[k] = String(v); }, _m: m };
}
const NOW = 1_700_000 * 86400000; // fixed day
const TODAY = Math.floor(NOW / 86400000);

// ---- pool integrity ----
check("pool loaded (>= 20 items)", Array.isArray(POOL) && POOL.length >= 20);
const ids = POOL.map((q) => q.id);
check("ids are unique", new Set(ids).size === ids.length);
check("every item has correct option present, >=2 options, explain", POOL.every((q) =>
  q.options && q.options.length >= 2 && q.options.some((o) => o.value === q.correct) && !!q.explain));
check("every item tagged with lesson + quarter + concept", POOL.every((q) => q.lesson && q.quarter && q.concept));
check("has misconception-tagged items (feeds G)", POOL.some((q) => q.misconception === true));

// ---- spacing: only earlier lessons appear ----
let c = bind(makeEl("div"));
let res = RetrievalBank.mount(c, { upTo: 12, count: 3, storage: memStorage(), now: NOW, shuffle: false });
check("spacing: all selected items are from lessons < upTo", res.items.every((q) => q.lesson < 12));
check("count respected (3 items when enough eligible)", res.items.length === 3);
check("interleave: no two adjacent share the same concept", res.items.every((q, i) =>
  i === 0 || q.concept !== res.items[i - 1].concept));

// ---- Leitner transitions on answer ----
const store = memStorage();
c = bind(makeEl("div"));
res = RetrievalBank.mount(c, { upTo: 20, count: 3, storage: store, now: NOW, shuffle: false });
const items = res.items;
// find the option buttons per rendered .rb-item, click correct on first, wrong on second
const rbItems = collect(c, "rb-item", []);
function clickValue(itemEl, value) {
  collect(itemEl, "quiz-option", []).forEach((b) => { if (b.dataset.value === value) b.click(); });
}
clickValue(rbItems[0], items[0].correct);
const wrongVal = items[1].options.find((o) => o.value !== items[1].correct).value;
clickValue(rbItems[1], wrongVal);

const st = JSON.parse(store.getItem("rdl-retrieval"));
check("correct answer promotes to box 2", st[items[0].id].box === 2);
check("correct answer schedules due = today + 1", st[items[0].id].due === TODAY + 1);
check("wrong answer resets to box 1", st[items[1].id].box === 1);
check("wrong answer schedules due = today (returns soon)", st[items[1].id].due === TODAY + 0);
check("seen counter incremented", st[items[0].id].seen === 1 && st[items[1].id].seen === 1);

// ---- due-first selection: a not-yet-due item should be deprioritised ----
const store2 = memStorage();
// mark l001-forcing as far-future (not due); everything else due at 0
store2.setItem("rdl-retrieval", JSON.stringify({ "l001-forcing": { box: 5, due: TODAY + 100, seen: 3 } }));
c = bind(makeEl("div"));
res = RetrievalBank.mount(c, { upTo: 20, count: 3, storage: store2, now: NOW, shuffle: false });
check("due-first: not-yet-due item excluded when due items exist", !res.items.some((q) => q.id === "l001-forcing"));

// ---- empty case ----
c = bind(makeEl("div"));
res = RetrievalBank.mount(c, { upTo: 1, count: 3, storage: memStorage(), now: NOW });
check("no eligible items -> empty (graceful)", res.items.length === 0);

// ---- storage fallback (no localStorage available) ----
let threw = false;
try {
  c = bind(makeEl("div"));
  RetrievalBank.mount(c, { upTo: 20, count: 2, now: NOW }); // no storage injected, no window.localStorage
} catch (e) { threw = true; }
check("degrades gracefully without localStorage", !threw);

// ---- predict widget ----
c = bind(makeEl("div"));
const handle = Predict.mount(c, {
  prompt: "Who wins on clean features?",
  options: [{ label: "The MLP wins here", value: "mlp" }, { label: "The tree wins here", value: "tree" }],
  correct: "mlp",
  reveal: "The MLP won 0.965 vs 0.936.",
  shuffle: false,
});
const pOpts = collect(c, "predict-option", []);
const revealBtn = collect(c, "predict-reveal", [])[0];
const outcome = collect(c, "predict-outcome", [])[0];
check("predict: reveal disabled before commit", revealBtn.disabled === true);
pOpts.find((b) => b.dataset.value === "tree").click(); // commit a WRONG prediction
check("predict: commit records choice", handle.getCommitted() === "tree");
check("predict: reveal enabled after commit", revealBtn.disabled === false);
pOpts.find((b) => b.dataset.value === "mlp").click(); // second commit ignored
check("predict: commit is locked (no re-commit)", handle.getCommitted() === "tree");
revealBtn.click();
check("predict: reveal shows outcome text + miss framing", outcome._text.includes("missed") && outcome._text.includes("0.965"));
check("predict: correct option marked truth", pOpts.find((b) => b.dataset.value === "mlp")._cls.has("truth"));
revealBtn.click(); // idempotent
check("predict: reveal is idempotent", outcome._text.includes("0.965"));

console.log(pass ? "\nALL PEDAGOGY CHECKS PASS" : "\nPEDAGOGY CHECKS FAILED");
process.exit(pass ? 0 : 1);
