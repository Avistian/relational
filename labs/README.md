# Labs

Each lesson's exit-ticket lab ships as a Jupyter notebook here, named `NNNN-<slug>.ipynb`
to match its lesson (`lessons/NNNN-<slug>.html`).

## The fill-in convention

Every lab follows [`LAB-TEMPLATE.ipynb`](./LAB-TEMPLATE.ipynb):

| Cell tag | Meaning |
|----------|---------|
| **PROVIDED** | Complete boilerplate (imports, data, helpers). Just run it. |
| **TODO** | Has blanks — `____` or `# TODO`. You fill these in. This is where the learning happens. |
| **CHECK** | Auto-runs assertions / prints so you get *immediate* feedback. Don't edit. |
| **EXIT TICKET** | The final deliverable cell. When it prints cleanly, the lab is done. |

Design intent: blanks sit only on the **skill** being practised; everything peripheral is
provided so working memory stays on the one idea (see `../NOTES.md` preferences and the
`teach` skill's fluency-vs-storage-strength note). CHECK cells make the feedback loop tight.

**Introductory content (from L011 onward):** each lab opens with a **concept recap**
(terms, formulas, one toy worked example) and a short **goal / why** block before every task.
The notebook should bridge lesson → practice without forcing the student to re-read the HTML
for core definitions. See `.agents/skills/lab-authoring/SKILL.md` § Introductory content.

**No prefilled answers:** TODO code cells use `____` only — never the completed solution.
Teacher copies with filled answers live in [`solutions/`](./solutions/) (gitignored).

**Visible paper implementations (standard #25):** paper-mirror labs (L043 TabNet, L044 NODE, L045 TabTransformer, and later) **inline** the from-scratch encoder as a PROVIDED cell. You can scroll the notebook and read Ghost BN, ODST, the Transformer stack, and the train loop. `labs/relkit/*.py` remains the canonical file for Modal / `_verify`; the notebook is a readable copy that keeps the functions you wrote in TODOs. Data loaders, CV, and metrics still `import` from `relkit`.

**Paper-results after EXIT (standard #25):** a downscaled bake-off is a *different experiment* from the paper's table. After the EXIT ticket, those labs ship a **NEXT STEP** cell (`RUN_PAPER_REPRO = False` until you attach a GPU) plus `modal/l0NN_paper_repro.py`. Paste the printed ledger (MATCH / CLOSE / FAIL / INCOMPARABLE / DIRECTION_*), not the EXIT ranks, as the paper claim.

**Agent scoring:** when you say *lab done*, your teacher scores the lab with the rubric in
`.agents/skills/lab-authoring/SKILL.md`.

## Environment (one-time setup)

From the repo root (`relational/`):

```bash
bash labs/setup-env.sh
```

This creates `.venv/`, installs [`requirements-labs.txt`](../requirements-labs.txt), and registers the Jupyter kernel **Relational Labs (.venv)**.

**In Cursor / VS Code:** open a lab notebook → **Select Kernel** → choose either:
- **Relational Labs (.venv)** (Jupyter kernel), or
- the interpreter at `.venv/bin/python`

**In a terminal:**

```bash
source .venv/bin/activate
jupyter lab    # then open labs/NNNN-<slug>.ipynb
```

Re-run `bash labs/setup-env.sh` after pulling new dependencies.

## Running

## Paper-mirror doctrine (standard #24) and visible implementations (#25)

When a lesson's core source is a paper, the lab **mirrors that paper from scratch** — not a loosely
inspired toy. Three axes (full rule in `NOTES.md` Preferences #24/#25 and
`.agents/skills/lab-authoring/SKILL.md`):

1. **Implementation** — write the paper's load-bearing mechanism; libraries only validate. The rest of
   the encoder is **inlined into the notebook** (#25) — not only `from relkit.X import TheModel`.
2. **Datasets** — prefer the paper's own data/splits; document substitutes and honest gaps.
3. **Reproducibility** — fixed seeds, versions, verify harness + results JSON; EXIT ties to the
   paper metric or records an honest fail.
4. **Paper-results scale-up** — after EXIT, a closer-to-paper run (Modal + Colab-gated cell) and a
   conclusion ledger. Until that run finishes, the paper claim stays *cited, not reproduced*.

## Reproduction labs build incrementally (harness reuse)

Concept labs (like `0006`) are self-contained. **Paper-reproduction / experiment labs**
(RelBench baselines, GBDT/RealMLP/TabM, RDL) must build on what earlier labs already wrote —
a cumulative, reusable *harness*, not isolated one-offs:

- Promote shared **data loaders, CV/eval, leakage-safe pipelines, metrics** into `labs/relkit/`.
  Reuse that harness; still implement the *model* from scratch and **show it in the notebook**
  (#22 / #24 / #25). Importing the paper's encoder from `relkit` is for checkers / Modal / `_verify`,
  not the only copy the student can read.
- Each lab leaves the harness stronger and better-tested for the next.

This keeps the thesis baselines trustworthy by the time results matter.

## Index

- `0006-missingness.ipynb` — classify MCAR/MAR/MNAR; show complete-case bias and the
  missing-indicator fix (Lesson 006).
- `0007-class-imbalance.ipynb` — accuracy paradox, the SMOTE-before-CV leak vs the imblearn
  Pipeline fix, and leak-free class weights (Lesson 007).
- `0008-metrics-calibration.ipynb` — ROC-AUC vs PR-AUC against the prevalence baseline, and
  reliability-curve + Brier calibration with CalibratedClassifierCV (Lesson 008).
- `0009-feature-engineering.ipynb` — ratio, cyclical datetime, leak-free target encoding (Lesson 009).
- `0010-baseline-checkpoint.ipynb` — Q1 capstone reproducible baseline (Lesson 010).
- `0011-decision-trees-partitions.ipynb` — tree splits on OpenML credit data (Lesson 011).
- `0012-bagging-random-forest.ipynb` — RF vs single tree, OOB, variance drop (Lesson 012).

## View / run in the browser

- **View:** static HTML renders live in `labs/html/` (regenerate with `bash scripts/render_notebooks.sh`).
- **Run:** the [Notebooks page](../notebooks.html) links each lab to **Colab**, the canonical
  run-anywhere path — the `@colab-bootstrap` first cell clones the repo and installs
  `requirements-labs.txt` (full lab stack incl. torch), and a free T4 is available under
  *Runtime → Change runtime type*. Heavy authoring jobs go to Modal instead (`modal/README.md`).

## Lesson 049: claim audit

See [the reproduction guide](l049-reproduction.md) for exact data/source scope, local commands, resumable larger runs, and the distinction between forward fidelity and paper-score replication.


### L056 · TabArena benchmark literacy
`0056-tabarena-benchmark-literacy.ipynb` reconstructs a four-method leaderboard audit from checksum-verified published scores. Four live evaluator TODOs, five inline figures, 51 real datasets; seconds of CPU computation, no training. Begin with split 0 then analyze every released outer split. See `l056-reproduction.md` for exact scope and unrun full-table/training tracks.

<!-- FOUNDATION-058-070:begin -->
## Lessons 058–070 · benchmark evidence and tabular foundation models

Each integer unit has a standalone student notebook and prepared HTML. Start at L058 and
follow retrieval → input → live implementation → CHECK → evidence audit → EXIT. The tutor
reviews the submitted artifact and explanation; completed author runs do not establish learner mastery.

`_run_foundation.py --lesson N --preset smoke|lab|closer --output PATH` regenerates the
declared experiment. `--lesson 70 --current` runs explicitly pinned current-version arms.
Historical packages use isolated subprocess imports. Full original paper protocols are not
implemented; `--preset paper` fails honestly. See `lNNN-reproduction.md`, `_sources_foundation.json`
and `_delivery_foundation_results.json`. Student TODOs remain blank; local solutions follow the
workspace's ignored `labs/solutions/` convention.
<!-- FOUNDATION-058-070:end -->

## Expanded explanations and architecture studies, lessons 047–070

All 24 packages include additional derivations, worked traces and computational reference aids.
The [architecture gallery](html/architecture-review/index.html) collects sixteen redesigned
model studies; each shows the forward path, a worked internal operator, shapes and variant scope.
The same figures are embedded portably in the notebooks. Phone readers can open the responsive
HTML study from each figure caption.

Authoring sources and regeneration/check commands are recorded in the
[revision review](../reviews/lessons-047-070-depth-and-architecture.md).
`_depth_delivery_results.json` records content and code/output integrity;
`_depth_browser_results.json` records the separate browser checks. These checks do not establish
live Colab compatibility or full paper reproduction.

- L073: [When SSL actually helps](0073-when-ssl-helps.ipynb) — nested label budgets, matched fine-tuning controls and crossover audits.

- L074: [CARTE cross-table transfer](0074-carte-cross-table-transfer.ipynb) — real FastText/YAGO embeddings, graph attention, matched scratch controls.

- L075: [PyTorch Frame row encoder](0075-pytorch-frame-row-encoder.ipynb) — five-stype API trace, train-only materialization, real credit_g row vectors; no accuracy claim.

- L076: [Encoder → predictor stack](0076-encoder-predictor-stack.ipynb) — full two-table teaching composition; separate historical RelBench replay and explicit NOT_RUN ledger.

- L077: [Single-table ceiling](0077-single-table-ceiling.ipynb) — complete five-seed synthetic reproduction, exact information bound and tabular feature-repair control.

- L078: [Message passing](0078-message-passing-preview.ipynb) — hand aggregation plus the full100-seed Cora GCN experiment with visible code and pinned bytes.

- L079: [Decision guide](0079-neural-tabular-decision-guide.ipynb) — one-page writing deliverable plus complete frozen-prediction audit. [Reproduction](l079-reproduction.md).

- L080: [Year 2 exit exam](0080-year-2-exit-exam.ipynb) — fresh four-family comparison, random + temporal, cold teach-back and rubric. [Reproduce](l080-reproduction.md).

- L081: [MPNN framework](0081-mpnn-framework.ipynb) — routing, permutation tests, full sparse molecular GG-NN and [reconstruction protocol](l081-reproduction.md).

- L082: [GCN](0082-gcn.ipynb) — exact normalization, sparse propagation, masked gradients and [full Cora experiment](l082-reproduction.md).

- L083: [GraphSAGE](0083-graphsage.ipynb) — three live TODOs, inductive audit and [full PPI reproduction contract](l083-reproduction.md).

- L084: [GAT](0084-gat.ipynb) — visible multi-head attention, three live TODOs and [full Cora reproduction contract](l084-reproduction.md).

- L086: [PyG fundamentals](0086-pyg-fundamentals.ipynb) — [full reproduction guide](l086-reproduction.md), real sampling and operator/gradient parity.

- L087: [Link prediction](0087-link-prediction.ipynb) — [full baseline reproduction guide](l087-reproduction.md), visible decoder, leakage checks and candidate-defined MRR/Hits.

- L088: [Graph classification / GIN](0088-graph-classification.ipynb) — [MUTAG reproduction contract](l088-reproduction.md), full inline implementation, WL limit and readout intervention.

- L089: [Sampling at scale / Cluster-GCN](0089-sampling-at-scale.ipynb) — [reproduction contract](l089-reproduction.md), full inline model/trainer and12-run PPI batching intervention. Full paper execution remains NOT_RUN.

- L091: [R-GCN](0091-r-gcn.ipynb) — [AIFB reproduction contract](l091-reproduction.md), ten full target runs, visible basis/message/loss tasks and a separate basis-sharing extension.

- **L093 HGT:** `0093-hgt.ipynb`; three live tasks, visible source-conditioned model and full OAG trainer. `_run_l093.py` provides smoke/teaching/paper/release tracks. Read `l093-reproduction.md` before interpreting scores.

- **L094 HIN survey:** `0094-hin-survey.ipynb`; taxonomy, route composition, protocol eligibility and complete NN statistics audit. No new model training. See `l094-reproduction.md`.

- **L095 Bipartite graphs:** `0095-bipartite-graphs.ipynb`; full ML-100K release audit and five-fold graph-walk recommendation, typed IDs, reverse-edge leakage checks and full-catalog ranking. See `l095-reproduction.md`.

- **L096 SQL FK semantics:** `0096-multi-relational-data.ipynb`; Tier C complete four-table schema→HeteroData experiment, three live tasks, 33-database SQL oracle. See `l096-reproduction.md`.

- **L097 Negative sampling:** `0097-negative-sampling.ipynb`; complete ML-100K 45-fit uniform/degree/hard ablation, typed train-only exclusion, visible BPR trainer and fixed full-catalog ranking. Three live tasks; separate sampled-candidate diagnostic. Historical BPR paper reproduction remains NOT_RUN. See `l097-reproduction.md`.

- **L098 Heterogeneous mini-batching:** `0098-hetero-mini-batching.ipynb`; real native sampling on a three-type synthetic graph, 96 independent output/gradient audits, temporal isolation, nine training fits. Full protocol in `l098-reproduction.md`; RelBench benchmark NOT_RUN.

### Lesson099 · R-GCN versus HGT

[Student](0099-rgcn-vs-hgt.ipynb) · [Solution](solutions/0099-rgcn-vs-hgt.ipynb) · [Read lab](html/0099-rgcn-vs-hgt.html) · [Protocol](l099-reproduction.md). Full24-fit ACM comparison, attention ablation, fresh ten-run AIFB port; HGT CS NOT_RUN. Three live tasks and written attribution defense.

### Lesson100 · Heterogeneous checkpoint

[Student](0100-heterogeneous-gnn-checkpoint.ipynb) · [Solution](solutions/0100-heterogeneous-gnn-checkpoint.ipynb) · [Read lab](html/0100-heterogeneous-gnn-checkpoint.html) · [Protocol](l100-reproduction.md). Native3-type ACM experiment,24fits, typed-ID and gradient audits, three live tasks and written defense. Full visible AIFB/CS appendices; CS training NOT_RUN.

### Lesson133 · Heterogeneous convolution on REG

[Student](0133-hetero-conv-reg.ipynb) · [Executed solution](html/0133-hetero-conv-reg.html) · [Protocol](l133-reproduction.md). Three live layer tasks, two sums, per-relation root terms, empty/absent semantics and fresh five-seed full-data F1 reproduction. Complete visible model/trainer and separate source/score evidence. Whole-paper parity remains NOT_ESTABLISHED.

### Lesson134 · Training at scale

[Student](0134-training-at-scale.ipynb) · [Executed reference](html/0134-training-at-scale.html) · [Solution](solutions/0134-training-at-scale.ipynb) · [Reproduction contract](l134-reproduction.md).
Three live functions govern typed frontier bounds, temporal query ownership and aggregate timing. Fresh full selected F1 reproduction and a separate full-topology rel-stack systems workload; full rel-stack benchmark and historical identity are not claimed. Default notebook independently replays recorded evidence; optional full training has a separate GPU gate.

### Lesson136 · Leaderboard literacy

[Student](0136-leaderboard-literacy.ipynb) · [Executed reference](html/0136-leaderboard-literacy.html) · [Solution](solutions/0136-leaderboard-literacy.ipynb) · [Reproduction contract](l136-reproduction.md).
Three live audit functions govern query identity, normalized error and complete-board coverage. Full nine-task archive replay for three entries plus five fresh historical RDL fits. Default notebook directly rescores F1 submissions and author training evidence; separate full-evaluation and full-training gates both have executed validation. Archive replay does not establish top-entry search/training identity.

### Lesson146 · GNN versus graph transformer

[Student](0146-gnn-vs-graph-transformer.ipynb) · [Executed reference](html/0146-gnn-vs-graph-transformer.html) · [Solution](solutions/0146-gnn-vs-graph-transformer.ipynb) · [Reproduction contract](l146-reproduction.md).
Three live contracts govern query-owned context sampling, validation selection and paired keyed errors. Six complete full-data course fits are independently scored; the nine-config published-protocol search remains separately blocked. Full visible models and trainer, portable figures, explicit default author-evidence checks and optional full-course gate. Author execution does not establish learner mastery.

- L149: [0149-weakest-relbench-tasks.ipynb](0149-weakest-relbench-tasks.ipynb) — weakness catalog, compatible ranking and supported-slice diagnosis. Full RDL/FE lanes and [protocol](l149-reproduction.md).


### Lesson152 · Regression portfolio

[Student](0152-regression-portfolio.ipynb) · [Prepared notebook](html/0152-regression-portfolio.html) · [Solution](solutions/0152-regression-portfolio.ipynb) · [Protocol](l152-reproduction.md).
Three live tasks: key-aligned MAE/RMSE/bias, tie-aware median diagnostics and complete five-seed evidence. Full model/trainer visible; default author-evidence audit and optional complete fresh five-seed gate are separate. Learning requires the written defense.


### Lesson154 — Portfolio synthesis

[Student notebook](0154-portfolio-synthesis.ipynb) · [Worked notebook](html/0154-portfolio-synthesis.html) · [Protocol](l154-reproduction.md) · [Generated report](evidence/l154/report.md). CPU-only, standalone replay of55frozen inputs and61,148prediction rows. Three live functions gate seed summaries, comparable differences and coverage. No new training or cloud spend; recommendation remains INCOMPLETE.

- [Lesson155: manual features vs RDL](0155-compare-manual-fe.ipynb): paired real query losses, prospective human-effort logs, full released FE/GNN gates. [Protocol](l155-reproduction.md).

- [Lesson156: temporal leakage audit](0156-temporal-leakage-audit.ipynb): three live owner-time,label-window and sign-off functions. Complete offline evidence replay plus both full GPU lanes. [Protocol](l156-reproduction.md) · [Reference execution](html/0156-temporal-leakage-audit.html).

- [Lesson157: open-source contribution](0157-open-source-contribution.ipynb): live integrity, complete-run and bounded-claim functions; [standalone package](releases/l157-f1-audit.zip), [protocol](l157-reproduction.md), [executed solution](html/0157-open-source-contribution.html). Full selected experiment separate from public publication.

## Lesson158 · Year4 synthesis
[Student](0158-year-4-synthesis.ipynb) · [Executed solution](html/0158-year-4-synthesis.html) · [Protocol](l158-reproduction.md) · [Essay template](l158-essay-template.md). Full selected CPU evidence replay; no new training. Written defense pending.

## Lesson 159 · Pre-training objectives
[Student](0159-foundation-model-preview.ipynb) · [Executed solution](html/0159-foundation-model-preview.html) · [Protocol](l159-reproduction.md) · [Brief template](l159-vision-brief-template.md). Three live masking/loss/freezing tasks and complete synthetic two-stage training. Historical BART+GCN benchmark NOT_RUN; this is not a foundation-model training run.


### Lesson 168 · Cross-database generalization

[Student](0168-cross-database-generalization.ipynb) · [Executed solution](html/0168-cross-database-generalization.html) · [Reproduction](l168-reproduction.md). Default notebook independently audits all60raw runs across F1(reused) and trial(fresh). Separate complete30-evaluation fresh-inference lane. No automatic paid dispatch. Learner PENDING_WRITTEN_DEFENSE.

### Lesson 169 · Scaling laws & open questions

[Student](0169-scaling-laws-open-questions.ipynb) · [Executed reference](html/0169-scaling-laws-open-questions.html) · [Protocol](l169-reproduction.md) · [Gap map](l169-gap-template.md). Default portable notebook independently replays all300selected evaluations:240fresh +60reused512-context runs, across two tasks/five sizes/three models/ten seeds. Three live learner contracts drive support verification, curve aggregation and evidence classification. Fresh inference is explicit and separately budgeted. Pretraining law NOT_ESTABLISHED; fresh pretraining/whole paper NOT_RUN; learner PENDING_WRITTEN_DEFENSE.
