---
name: lab-authoring
description: Author mid-difficulty lab notebooks with PROVIDED/TODO/CHECK/EXIT cells, paper-reproduction structure, dataset tiers, and agent scoring when the user completes a lab.
---

Use when creating, retrofitting, or reviewing lab notebooks in `labs/`.

## Notebook quality is a delivery requirement (user, 2026-09-05)

Compare each new notebook with the strongest recent lessons. Existence, length, and a passing
training run do not establish teaching quality. Before delivery:

- Apply the [visual teaching standard](../lesson-visuals/SKILL.md#visual-teaching-standard--raised-after-l048-user-2026-09-05)
  to notebook figures: show the operation and its intermediate values, select explanatory
  snapshots, and inspect readability at notebook display width. The user asked for visuals
  to improve beyond L048; more images or passing geometry checks alone do not satisfy that.
- Make the notebook understandable without reopening the lesson: define terms, give a worked
  example, and connect each paper element to the code that implements it.
- When introducing a new model, include a dedicated **Model architecture** section with an
  embedded end-to-end diagram and explanation, following the
  [architecture requirement](../lesson-visuals/SKILL.md#model-architecture-requirement-user-2026-09-05).
  A diagram of one internal block alone does not satisfy the requirement.
- Embed portable mechanism diagrams beside the relevant explanation. For Colab, use inline
  markdown `data:image/png;base64,...` images; Jupyter `attachment:` links do not render there
  (L047 regression, colabtools #3836). Check payloads and the target frontend explicitly;
  successful nbconvert rendering alone does not establish Colab compatibility.
  Captions must match the exact pictured state and
  distinguish synthetic illustrations from measured results.
- Split the visible canonical implementation into coherent, annotated chunks. Preserve the
  student's live functions; verify the trained model actually calls them. Avoid a single wall
  containing the entire architecture and trainer.
- Give substantive mid-difficulty tasks, immediate checks with diagnostic messages, predictions
  before interventions, and written interpretation/retrieval prompts. Do not leak completed TODOs.
- Present experiments with readable tables, per-seed points, uncertainty, ranks/CD, and an
  explanation of what each result permits. Raw JSON is an artifact, not the main lesson display.
- For an unexecuted student notebook, include clearly labeled author-reference evidence when it
  helps teach result interpretation. Never represent it as the student's current kernel output.
- Execute the solution, inspect the figures, regenerate prepared HTML, and check embedded images,
  links, student blanks, and source/result provenance. Record any unperformed browser check honestly.

## The lab ships WITH the lesson (NOTES standard #21 — read first)

**A lesson is not "created" or "published" until its lab notebook exists.** When the task is "create
lesson N", the `Lab` column in `CURRICULUM.md` is a deliverable, not a suggestion — the lab
`labs/NNNN-<slug>.ipynb` ships in the **same session** as the lesson HTML. Do not defer it, do not leave
`labPath: null` on a lesson that trains/reads/benchmarks/implements anything.

- **`labPath: null` is allowed ONLY for genuine writing lessons** with no code deliverable (essay /
  peer-review / synthesis / checkpoint-essay — e.g. L038, L039). "Environment setup" or "forward-pass
  only" is **not** an exemption.
- **Build-before-done checklist (every non-writing lesson):**
  1. `labs/_verify_lNNN.py` run; numbers recorded (or a Modal job — standard #20).
  2. `labs/_build_lNNN.py` → student `labs/NNNN-*.ipynb` (PROVIDED/TODO/CHECK/EXIT, concept recap,
     `@colab-bootstrap` first cell) **and** `labs/solutions/NNNN-*.ipynb`.
  3. Execute the solution with nbconvert so the numbers are real; render `labs/html/NNNN-*.html`.
  4. `lessons/manifest.json` `labPath` set; `notebooks.html` renders it (auto from manifest).
  5. Any new lab dependency added to `requirements-labs.txt`.
  6. The lesson HTML's inline "Lab" section points at the notebook.
  7. **If the core source is a paper (standard #25):** the notebook *inlines* the from-scratch
     architecture as a PROVIDED cell (not `from relkit.X import TheModel` as the only copy the
     student can read); a post-EXIT **paper-results** cell + `modal/l0NN_paper_repro.py` exist
     whenever the lab downscaled. Completion criterion: a hostile reader can scroll the notebook
     and see the forward pass / train loop, and has a Modal or Colab command that trains closer
     to the paper's table.
- **Reconcile numbers:** if `_verify` contradicts numbers already written into the lesson/viz/dossier,
  fix the lesson to the verified numbers (honesty rule, standard #20) — never ship the borrowed story.
- **Miss of record (do not repeat):** L042 ("Train ResNet baseline") was first published `labPath: null`
  with only an inline lab; retrofitted the same day into a full rtdl training notebook.

## Paper-mirror doctrine (NOTES standard #24 — read with #18/#22/#23/#20)

When the lesson's core source is a paper, **always try to mirror that paper from scratch** — implementation,
datasets/protocol, and reproducibility — so the student learns from the paper itself, not a loosely
inspired toy. State the mirror scope in the lab intro. Narrow exceptions: genuine writing lessons
(`labPath: null`), or a lesson whose skill is *using a tool/API* (e.g. L041).

### (A) Implementation from scratch (#18 + #22)

The load-bearing model/mechanism is **implemented from scratch** — the student writes the forward pass /
update / algorithm. A reference library (**rtdl** for MLP/ResNet/FT-Transformer; the canonical impl for
other topics) is a **validation point, not the teacher**:

- Preferred: same architecture + copied weights → assert `torch.allclose` on outputs.
- Practical: train both under the same protocol → assert `|Δ metric| < tol` (e.g. 0.03) over seeds.
- Only peripheral boilerplate is imported. Promote reusable from-scratch models into `labs/relkit/`
  (e.g. `relkit/nets.py`) so Modal / `_verify` / later labs share one canonical file.
- **Visible in the notebook (standard #25):** the student notebook must **inline** that canonical
  file as a PROVIDED cell (`relkit.paper_repro.inline_source`, skipping the names the student
  writes in TODOs). `from relkit.tabnet import TabNetEncoder` is a *checker / harness* import, not
  an acceptable way to present the paper's architecture. If the bake-off only trains a packaged
  model, the implementation is hidden — that is a miss.
- **Exception:** a lesson whose *skill is the tool itself* (e.g. L041 "set up rtdl") may use it directly —
  say so in the intro.

### (B) Datasets — prefer the paper's (#7 + #23 + #24)

- **Default:** use the paper's own datasets, splits, and preprocessing when open and affordable (#20).
- **If not:** closest Tier-A/B substitute or documented subsample; **name the paper datasets you skip**
  and which claim therefore cannot be reproduced here. Never label a substitute bake-off as "we
  reproduced Paper X" without that gap.
- Comparative claims still need **multiple datasets + rank stats** (#23):
  - **≥3 real datasets** (small OpenML tables keep it CPU-cheap: credit_g/diabetes/blood_transfusion/kc1/
    phoneme are registered in `relkit.data`; or subsample larger ones and *state it*).
  - **Per-dataset mean ± std over ≥3 seeds** — CIs, never a bare point estimate.
  - **Cross-dataset summary = mean ranks + Friedman test + a Nemenyi critical-difference diagram**
    (reuse `assets/cd-diagram-viz.js`; method from L023/L030), not one t-test.
  - When compute can't reach the N the claim needs, **cite the big benchmark** for the strong version
    (Grinsztajn 2022 ~45 datasets; Gorishniy 2021; TabArena) and separate "verified here on k" from
    "established in the literature on N". A single-dataset number is a *demonstration*, never the evidence.
- Document tier, dataset keys, and paper↔lab mapping in the lab intro (`labs/data/README.md`).

### (C) Reproducibility contract (#20 + L037 pattern + #24)

Every paper-mirror lab's intro + EXIT must make the run regenerable:

| Requirement | Concrete |
|-------------|----------|
| Seeds | Fix model **and** splitter seeds; call out both (L037: splitter seed moved the metric more than model seed) |
| Versions | Claim numbers against `requirements-labs.txt` / lockfile; note library pins that matter |
| Protocol | Identical across arms; downscales = different experiment (honesty rule #20) |
| Verify harness | `_verify_lNNN.py` (+ Modal if needed) → committed `_*_results.json` |
| EXIT target | Paper metric within stated tolerance **or** honest fail / protocol-deviation note |

Harness reuse (`relkit/` loaders/CV/metrics) is encouraged; importing the *model* from a library **or
from relkit** instead of showing it in the notebook is not (#22 / #25).

### (D) Paper-results scale-up — required after every paper-mirror lab (#25)

The learning lab may downscale so it fits in minutes. That lab is a *different experiment* from the
paper's table. **Do not stop there.** After EXIT, every paper-mirror lab ships a **NEXT STEP** that
tries harder to reproduce the paper's *results*:

1. **Same from-scratch code** the student just read (inlined), not a different library model.
2. **Paper dataset / HPs / metric** when open and affordable; otherwise a documented closer-to-paper
   run with protocol deviations listed.
3. **If local CPU cannot train it:** ship **both** of
   - `modal/l0NN_paper_repro.py` (unattended; `modal run --detach ... --preset closer`)
   - a Colab-gated cell in the same notebook (`RUN_PAPER_REPRO = True` on a T4).
   Optional stretch does **not** satisfy this. A comment that says "the paper used more trees" does
   **not** satisfy this.
4. **Conclusion ledger** (`relkit.paper_repro.format_ledger`): three buckets —
   *verified here* / *paper claim* / *scale-up run* — with MATCH · CLOSE · FAIL · INCOMPARABLE ·
   NOT_RUN · DIRECTION_*. Mixing bucket 1 with bucket 2 is how the student learns the wrong
   conclusion (TabNet "lost" on four tiny tables ≠ the paper was wrong).

Until the scale-up has been run, the paper claim stays **cited, not reproduced**. That is an honest
state.

Presets on every harness: `smoke` (seconds, CI) · `closer` (T4, tens of minutes) · `paper` (hours,
paper HPs). Default the notebook cell to `closer` gated OFF.

## Cell convention

| Tag | Student sees | Agent writes |
|-----|--------------|--------------|
| **PROVIDED** | Complete code — run only | Imports, data load, helpers, leaky baseline for contrast |
| **TODO** | Blanks (`____`, `# TODO`) | 2–4 focused blanks per task on the *skill* being practised |
| **CHECK** | Auto assertions — do not edit | Immediate feedback |
| **EXIT TICKET** | Final deliverable print | Verifiable summary the user pastes to teacher |

**Never:** complete solution in TODO code cells or markdown hints. Hints describe *what* to achieve, not *how* to code it.

**Teacher solutions:** store in `labs/solutions/NNNN-<slug>.ipynb` (gitignored). Student notebooks stay blank.

## Difficulty — mid zone

- Not easy: student must write non-trivial lines (not copy-paste from markdown).
- Not hard: peripheral boilerplate is PROVIDED; one skill per task.
- Optional **stretch** cell at end (ungraded) for fast finishers.

## Dataset tiers

| Tier | When | Examples |
|------|------|----------|
| **A — Real, small, open** | Default Q2+ training/eval labs | OpenML (`fetch_openml`), UCI; see `labs/data/README.md` |
| **B — Real, relational** | Y3+ / RelBench | RelBench tasks via PyTorch Frame |
| **C — Synthetic** | Mechanism isolation only | MCAR/MAR/MNAR demos, pure-noise leak demos |

Document tier and rationale in the lab intro markdown.

## Compute budget (from 2026-07-25 — NOTES standard #20)

Authoring box: **12-core CPU, ~15 GB RAM, no GPU.** Two cloud paths with different operators:

| Path | Operator | Use for | Limits |
|------|----------|---------|--------|
| **Modal** (Starter, $0) | the **agent**, unattended via CLI token | the lesson's *verified* numbers when CPU is too slow | $30/mo credits ≈ 50 T4-h; CPU ~free |
| **Colab** (free) | the **user**, in a browser | labs where the *student* needs a GPU | T4, ~15–30 GPU-h/wk, 12 h cap, ~90 min idle disconnect |

Modal is **already set up and smoke-tested** — harness `modal/common.py`, runbook `modal/README.md`.
Colab is wired into every lab (`labs/_colab.py` bootstrap + `notebooks.html` link) and is the **only**
run-anywhere path (Binder was dropped). **Never design a lab that needs an unattended long run on free
Colab** — the idle timeout keys on browser-tab interaction, not on whether code is running.

**Escalate in this order, never skip a rung silently:**

1. **Check thread oversubscription first** — the real cause every time so far (L031 HistGB 21 s → 0.28 s with
   `OMP_NUM_THREADS=1`; L036 LightGBM 210 s → 9.2 s at `n_jobs=6`). Free, usually 10–20×.
2. **Shrink the measurement, not the claim** — fewer trees/epochs/folds, *identical across arms*, stated in
   the intro (L036: 120 trees vs 400).
3. **Change the scope, not the paper** — forward-only / key-parts / gradual-across-labs (standard #18; L032).
4. **Tier C synthetic** or a **torch-free stand-in** (L030 `MLPClassifier` for the ResNet).
5. **Modal** for an agent-side verified number; commit the launch script and quote the hardware in the record.
6. **Colab-GPU for the lab** — label it "GPU recommended — open in Colab", keep it under ~20 min on a T4,
   checkpoint to Drive, and ship a smaller CPU config that still passes the CHECK cells.
7. **Paper-results scale-up (standard #25)** — after EXIT, a closer-to-paper run of the *same from-scratch
   code* via Modal *and* a Colab-gated cell, with a conclusion ledger. Rung 2 (shrink the learning lab)
   is a sequel, not a skip: the student still gets a command that trains nearer the paper's table.

**Honesty rule:** a downscaled run is a *different experiment*. State the config in the lab intro and the
learning record; never quote a shrunken result as the paper's; if the affordable version cannot support the
claim, say the claim is unsupported rather than shipping a toy that looks like evidence. After shrinking,
still ship the paper-results track (#25) — shrinking is how the *learning lab* fits, not permission to
treat EXIT ranks as the paper.

## Paper-reproduction labs (Q2 onward)

Four blocks per reproduction lab:

1. **Paper step** — numbered algorithm step from the paper (with section ref)
2. **Crucial fragment** — student implements one non-obvious function (split gain, ICL batch layout, etc.)
3. **Harness** — `import relkit` for data loaders, CV, metrics, leakage-safe pipelines. **Not** for
   the paper's model: that source is inlined into the notebook (#25).
4. **Reproduction target** — metric within tolerance of paper, or documented honest fail
5. **Scale-up next step** — Modal script + Colab-gated cell + conclusion ledger (#25), whenever the
   lab downscaled

## Labs implement the paper (from L032 — standard #18; elevated by #24)

A lab's crucial content is a **faithful (if minimal) implementation of the lesson's core paper**, not a
generic sklearn/toy exercise. Labs must be **very informative**. Standard **#24** elevates this: mirror
the paper on **implementation + datasets + reproducibility**, not architecture alone.

### Decide the implementation scope (state it in the lab intro)

| Scope | When | Example |
|-------|------|---------|
| **Whole model** | The architecture is small enough to build + run end-to-end in one lab | MLP, single attention block, a from-scratch boosting loop |
| **Key parts** | The full model is too large for one sitting — implement the load-bearing mechanism(s), PROVIDE/borrow the rest | scaled dot-product + transformer block, with a plain MLP head |
| **Gradual across labs** | One paper spans **several lessons** — split its implementation so each lab lands one coherent, runnable piece aligned with its lesson | **TabTransformer (Huang 2020):** L032 = architecture + forward pass (real data, no training); Y2 L045 = training + semi-supervised pre-training + benchmarking |

For a multi-lesson paper, note the split in **each** lab's intro and in the learning record so the arc is
legible (which piece this lab lands, what an earlier/later lab covers).

### "Very informative" concretely

- **Annotate each implementation cell with the paper element it realises** — figure / section / equation
  ref (e.g. "Fig. 1 Transformer layer", "Vaswani §3.2 eq. 1").
- **Minimal PROVIDED scaffolding** — the student writes the *load-bearing* code (attention equation,
  residual wiring, the concat before the head), not boilerplate. Still mid-zone (2–4 focused blanks/task).
- **Real data + a runnable result** over toys wherever feasible (Tier A; use **torch** now that it is
  installed) — even a forward-pass-only lab should run on real rows and inspect a real intermediate.
- **EXIT / reproduction target ties back to the paper's actual claim.**
- **Length is not the constraint** (standard #17) — a paper-implementation lab may be longer than a
  mechanism lab.

Reference implementation: **L032** (`labs/_build_l032.py`) — a faithful torch TabTransformer *architecture*
(scaled dot-product self-attention → transformer block with residuals + FFN + LayerNorm → contextual
embeddings → concat continuous → MLP head), forward-run on real `credit_g`, with training deferred to L045.

## Agent review when user says *lab done*

Score 0–2 per axis (max 10):

| Axis | 0 | 1 | 2 |
|------|---|---|---|
| **Correctness** | EXIT/CHECK fail | Partial pass | All CHECK + EXIT clean |
| **Leakage discipline** | Obvious leak | Minor doubt | Pipeline/fold boundaries correct |
| **Conceptual takeaway** | Missing/wrong | Vague | One accurate sentence |
| **Mid-zone effort** | Copy-paste or blank | Too easy or stuck | Appropriate challenge |
| **Reproduction (if applicable)** | Not attempted | Wrong metric | Within tolerance or honest fail documented |

Return: total /10, one concrete improvement, log notable gaps in `NOTES.md` or a learning record.

## Template

Follow `labs/LAB-TEMPLATE.ipynb` and `labs/README.md`.

## Introductory content — required (from L011 onward)

Labs are not a silent worksheet. The lesson HTML teaches; the notebook **bridges** lesson → hands-on practice. Every new lab must include markdown that a student can follow **without reopening the lesson** for the core idea.

### Required sections (markdown cells)

1. **Header** — lesson link, skill, exit criteria, how PROVIDED/TODO/CHECK/EXIT works, environment (see template).
2. **Concept recap** (after header, before setup) — 3–6 short paragraphs:
   - Restate the *one skill* in plain language
   - Define key terms and formulas used in the lab (with LaTeX or code-style math)
   - One **worked micro-example** on toy numbers (not the lab's answer — e.g. Gini on `[0,0,1,1]`, not the German-credit split)
   - Link back to the lesson HTML for the full viz/reading
3. **Before each task** — a markdown cell with:
   - **Goal** — one sentence: what you will produce
   - **Why it matters** — tie to mission/thesis or the lesson's failure mode
   - **Hint boundary** — describe *what* to compute, never the completed code

### Balance with mid-zone difficulty

- Intro markdown **explains concepts**; TODO cells still hold the **implementation** blanks.
- Do not paste the solution into markdown or prefilled TODO code.
- A formula in the recap is fine; the student still writes the Python that implements it.
- Target: ~30% of notebook cells are explanatory markdown; the rest are PROVIDED/TODO/CHECK/EXIT.

### Retrofit rule

When publishing a new lesson, if the matching lab lacks a concept recap, add one before marking the unit published. Retrofit the current lesson's lab when the user flags thin intros (see `NOTES.md` Preferences).
