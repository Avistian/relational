# Year 1 — Tabular Foundations · Lesson Decomposition

Only **Q4 (031–040)** is decomposed here: Q1 (001–010) and Q2 (011–020) are published, and Q3
(021–030) already has adequate one-line rows plus the Grinsztajn spine previewed in L019. Q4 is the
consolidation quarter — it teaches almost **no new architecture**; it converts the tabular skills
into a reusable, auditable baseline package and plants the *relational* seed that Years 2–4 grow.

**Q4 goal:** turn "I can tune a tree" into "I ship a leak-free, documented, reproducible baseline
and I can name exactly what flattening a database throws away." This is the setup for the whole thesis.

**Papers:** Guo & Berkhahn 2016 (entity embeddings) · Huang et al. 2020 (TabTransformer, `2012.06678`) ·
Domingos 2012 (A Few Useful Things) · Kimball (star schema, book) · Fey et al. 2024 (RDL, PMLR v235) §2.

---

### 031 · Embeddings for categoricals — *Guo & Berkhahn 2016 (entity embeddings); target-encoding recap*
- **Skill** — represent a high-cardinality categorical as a learned dense vector, and state why an
  embedding table is a lookup an MLP can learn end-to-end (vs one-hot / target encoding).
- **Teach** — one-hot blows up + is orthogonal (no similarity); entity embedding = trainable row in a
  `(n_categories × d)` matrix; geometry captures similarity (Guo & Berkhahn's Rossmann store map);
  target/mean encoding is a *fixed* 1-D embedding and leaks without out-of-fold discipline (callback L016).
- **Lab** — Tier A (OpenML categorical-rich, e.g. `adult`/`credit_g`) · crucial fragment: build a tiny
  `nn.Embedding`-style lookup by hand (numpy) + fit a target encoder *inside* CV folds → compare downstream
  logistic AUC of one-hot vs OOF-target vs learned embedding. Deliverable: EXIT table of the three.
- **Viz** — new `embedding-viz.js`: 2-D scatter of category vectors (start random → drift so similar
  categories cluster); toggle one-hot (orthogonal axes) vs learned (clustered).
- **Bridge** — callback L016 ordered TS / leak; forward to Y2 L045 (TabTransformer *contextual*
  embeddings) and Y4 L125 (PyTorch Frame row encoder). Thesis: embeddings are how raw cells enter a network.

### 032 · TabTransformer preview — *Huang et al. 2020, ◆ `2012.06678`*
- **Skill** — read a tabular-transformer architecture figure and explain what "contextual categorical
  embeddings" add over a static embedding table.
- **Teach** — column embeddings → Transformer blocks over categorical tokens → concat with continuous →
  MLP head; "contextual" = a category's vector depends on the *other* columns in the row; numeric columns
  bypass attention (a known limitation later fixed by FT-Transformer). Read the figure, not the training.
- **Lab** — Tier C · no training: annotate the architecture (a provided diagram/notebook markdown) — label
  where embeddings, attention, and the head live; predict which column types get contextualized. Deliverable:
  a 5-line "what attention buys here" note. (Light unit — reading, not reproduction.)
- **Viz** — reuse `embedding-viz.js` (static → context-shifted vector) or a static figure; no new asset needed.
- **Bridge** — sets up Y2 Q1 (L045 TabTransformer proper, L046 FT-Transformer which tokenizes *numeric*
  features too). Honest note: TabTransformer rarely beats a tuned GBDT — it's a stepping stone, not SOTA.

### 033 · When to stop feature engineering — *Domingos 2012, "A Few Useful Things to Know about ML"*
- **Skill** — allocate a fixed modeling budget between feature engineering and model tuning, and justify
  the split with the "feature engineering is the key / but has diminishing returns" trade-off.
- **Teach** — Domingos: "more data beats a cleverer algorithm," feature engineering is where most gains
  live *but* returns diminish and overfitting risk grows; the relational thesis reframes this — RDL tries to
  *learn* the aggregations manual FE produces (callback L009 DFS bridge).
- **Lab** — Tier A · crucial fragment: build a small FE-budget experiment — measure CV-AUC after 0, 3, 5, 10
  hand features on a tuned tree; plot the diminishing-returns curve. Deliverable: the curve + "where I'd stop."
- **Viz** — reuse `feature-viz.js`; optionally a simple returns curve (matplotlib in-notebook, not a JS asset).
- **Bridge** — callback L009/L010; forward to L035 (what joins destroy) and Y4 L129/L155 (RDL vs manual FE
  user study — the human-effort ratio Domingos is implicitly about).

### 034 · Relational data without RDL — *Kimball star schema & joins (summary)*
- **Skill** — read a 3-table schema (fact + dimensions, primary/foreign keys) and write the join +
  point-in-time aggregation that flattens it into one design matrix.
- **Teach** — star schema vocabulary (fact, dimension, grain, PK/FK); a "row" for ML = one entity at one
  time; the flatten = join then aggregate neighbor tables into columns; PIT correctness (only pre-timestamp
  rows, callback L002 leakage).
- **Lab** — Tier C · crucial fragment: given 3 toy tables (customers, orders, order_items), write the SQL/pandas
  join + a leak-free aggregate (e.g. `n_orders_before_t`, `avg_basket_before_t`). Deliverable: the flat table
  + one sentence on the grain.
- **Viz** — reuse `leakage-viz.js` (PIT join sketch); extend caption to show FK edges between the 3 tables.
- **Bridge** — this *is* the single-table paradigm the thesis critiques; L035 quantifies what it costs;
  Y3/Y4 turn these same PK/FK edges into a graph instead of columns.

### 035 · What joins destroy — *Fey et al. 2024 (RDL) §2 (feature-engineering cost), ★ preview*
- **Skill** — enumerate the structural information a flatten-then-aggregate pipeline discards (cardinality,
  identity, higher-order paths, temporal order within a neighbor set).
- **Teach** — aggregation is lossy: `mean(basket)` forgets *which* items and *how many*; multi-hop
  relations (customer→orders→items→product→category) collapse; the same join must be redone per task; Fey's
  argument that PK/FK links carry predictive signal FE throws away. This is the intellectual pivot of Year 1.
- **Lab** — Tier C · crucial fragment: take the L034 flat table and show a case where two entities with
  *different* neighbor structure map to the *same* feature vector (aggregation collision). Deliverable: the
  colliding pair + "what a graph would keep."
- **Viz** — new `flatten-loss-viz.js`: left = small entity graph with distinct neighborhoods; right = both
  collapse to identical feature rows; toggle shows the information lost.
- **Bridge** — closes Y1's conceptual arc (L001 single-table assumption → here, why it's a *choice* with a
  cost). Directly seeds Y3 message passing and Y4 REG. Core of the Y1 exit essay.

### 036 · Revisit your homework pipeline — *your `homework/report.md` (audit)*
- **Skill** — audit an existing ML pipeline against the Q1–Q3 leakage-spine checklist and find its real defects.
- **Teach** — no new concept; apply `assets/checklist.js` rubric (split hygiene, per-fold preprocessing,
  missingness handling, metric choice, tuning honesty) to the user's own prior work.
- **Lab** — Tier A (user's own data) · crucial fragment: run the checklist, fix one real leak, re-measure.
  Deliverable: before/after CV numbers + the defect found.
- **Viz** — reuse `checklist.js` + `pipeline-viz.js`.
- **Bridge** — callbacks L002–L010; personalizes the exit exam; feeds L037 (package the fixed pipeline).

### 037 · Document a baseline package — *— (engineering unit)*
- **Skill** — package the Q1–Q4 baseline into a reproducible, importable script/module with fixed seeds and a
  one-command run.
- **Teach** — reproducibility contract (pinned deps, seed, data hash, saved metrics); promote notebook code
  into `labs/relkit/` (callback: the incremental-relkit rule); a baseline is an artifact you *rerun*, not retype.
- **Lab** — Tier A · crucial fragment: refactor the L036 pipeline into `relkit`-style functions + a
  `run_baseline.py` that prints the metric table. Deliverable: the script + a README stub.
- **Viz** — none (code artifact).
- **Bridge** — this harness is the battle-tested baseline the RDL thesis will be measured against for 5 more
  years (callback lab-authoring incremental rule; forward to every Y2+ reproduction lab).

### 038 · Peer review your evaluation — *— (checklist unit)*
- **Skill** — critique an evaluation (yours or a published table) for leakage, unfair tuning budget, and
  metric mismatch — the reviewer stance.
- **Teach** — the "fair comparison" contract (same splits, same tuning budget, same metric, reported
  variance); Demšar-style paired comparison recap (callback L023 planned); steel-manning the baseline.
- **Lab** — Tier A · crucial fragment: given a deliberately-flawed provided results table, list every unfair
  comparison and rewrite the protocol. Deliverable: the corrected protocol.
- **Viz** — reuse `checklist.js` (evaluation-fairness variant).
- **Bridge** — the reviewer discipline every later reproduction and the Y6 research project depend on;
  forward to Y2 L059 (validation-set overfitting) and Y4 error-analysis lessons.

### 039 · Year 1 synthesis essay — *— (writing unit)*
- **Skill** — write a tight argument for *what trees beat and why*, integrating the three Grinsztajn biases,
  the tuning/FE budget, and the flatten cost.
- **Teach** — synthesis only: three biases (L019) + honest-baseline discipline (L014/L018) + flatten loss
  (L035); the thesis framed honestly ("beat a strong baseline whose biases fit flat data").
- **Lab** — writing deliverable (not a notebook): ~1 page, must name the honest conditions each tree
  advantage flips.
- **Viz** — reuse `biases-viz.js` as the essay's anchor figure.
- **Bridge** — dress rehearsal for the exit exam; the argument recurs at every year boundary.

### 040 · **Year 1 exit exam** — *all Y1 papers · Deliverable-based*
- **Deliverable** — on a fresh flat task: (a) reproduce a *tuned* tree baseline with leak-free CV, (b) beat
  it with another model **or** explain with evidence why you can't, (c) a written account of Grinsztajn's
  three biases with the conditions each flips.
- **Exit criterion (from CURRICULUM)** — reproducible tuned tree baseline + written understanding of the
  three biases. Score against that, not a new concept.
- **Bridge** — gate to Year 2. The baseline harness (L037) and the "what joins destroy" argument (L035) are
  the two artifacts carried forward. Do not advance without passing.

**Optional (◆, only after exit):** SHAP `1705.07874` (feature attribution reused on every later model);
Molnar *Interpretable ML* 3rd ed. (PDP/ICE/SHAP for auditing the baseline).
