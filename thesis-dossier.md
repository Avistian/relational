# Thesis Dossier

The living, skeptic-facing argument for the mission's contrarian bet. This is not lesson notes — it is the
**case**: claims, the evidence each lesson adds, and (just as important) the **counter-evidence**. By the
time results matter (Y3–Y6), this should read as an honest brief a reviewer could not easily dismiss.

**Update ritual.** After each lesson, add one line to the Evidence Ledger: what the lesson contributed
*for* the thesis, *against* it, or *to the honest bar* it must clear. Revise the Current Verdict when the
balance shifts. Never delete counter-evidence — a thesis that only accumulates supporting points is
propaganda, not a case. Related: `MISSION.md` (why), `NOTES.md` (teaching standards), lesson "thesis
bridge" callouts (the raw material).

---

## The thesis

> Relational deep learning (RDL) and foundation relational models are **undervalued**: by learning
> directly over a database's relational structure instead of a hand-flattened single table, they can
> unlock predictive value the dominant single-table paradigm systematically discards.

### Sub-claims (each must be defended, not assumed)

- **C1 — Flattening is lossy.** Collapsing a relational database into one design matrix throws away
  structure (entity identity, shared groups, event sequences, many-to-many links) that carries signal.
- **C2 — The loss is *recoverable* by learning over structure.** A model that operates on the relational
  graph can exploit what flattening discarded, end-to-end, without hand-crafted joins.
- **C3 — The gain is *real and fair*.** The advantage survives honest evaluation — temporal splits, no
  leakage, and a genuinely strong single-table baseline (tuned GBDT + leak-free stacked ensemble), not a
  strawman.
- **C4 — The field undervalues this.** Relative to attention/compute spent elsewhere, the relational
  frontier is neglected given its potential.

---

## Evidence ledger

Legend: **[FOR]** supports a sub-claim · **[BAR]** raises the honest baseline the thesis must beat ·
**[AGAINST]** genuine counter-evidence.

| Lesson | Contribution | Type | Bears on |
|--------|--------------|------|----------|
| L055 | TabReD split audit: three checksum-verified release tasks, same-size random/time comparison. Classification errors rise without winner changes; Sberbank point-estimate winner changes while errors fall. Availability, validation and dataset-unit uncertainty remain requirements for a future RDL claim. Local INCOMPARABLE to Figure 2; no relational model tested. | BAR | C3 |
| L056 | TabArena audit: four frozen methods on 51 IID datasets and 816 outer splits; tuning/ensembling regime, competitor pool and equal-dataset weighting are part of any ranking claim. Cached-score reanalysis and evaluator parity pass; full paper Elo and fresh training are not reproduced. This raises the baseline comparison standard without testing relational advantage or future-period deployment. | BAR | C3 |
| L001 | RDL learns over the relational entity graph; flattening is the lossy step it replaces (Fey 2024). | FOR | C1, C2 |
| L002 | Point-in-time flattening is not just lossy but leakage-prone — the manual pipeline is fragile. | FOR | C1 |
| L004 | Honest evaluation needs nested CV / grouped splits — the discipline any thesis claim must meet. | BAR | C3 |
| L009 | Manual relational features (DFS/Featuretools) already recover *some* structure by hand — RDL's premise is to learn this end-to-end. | FOR | C2 |
| L010 | A reproducible single-table baseline (leak-free HistGBDT) is strong and cheap — the floor is high. | BAR | C3 |
| L011–L016 | GBDTs (XGBoost/LightGBM/CatBoost) are stubborn, well-engineered baselines; tuning a strong default barely moves it. | BAR | C3 |
| L018 | The real single-table bar is a *leak-free stacked ensemble* of diverse tuned models, not one default. | BAR | C3 |
| L019 | Grinsztajn: on *typical* tabular data trees win via three inductive biases; on clean data an MLP won — DL is not weak at tables. | AGAINST | C3, C4 |
| L020 | Q2 checkpoint: a sensible default GBDT reproduces published results; a big "win" should trigger leak suspicion. Bridge: the flat `adult` table discards employer identity, shared households, job sequence a model over the source DB could exploit. | BAR + FOR | C1, C3 |
| L021 | Random splits are optimistic on drifting data (random-CV 0.846 vs temporal 0.758); honest eval needs temporal splits. TabReD: on real industrial data, time-based splits change rankings and shrink XGBoost's margin. RelBench/RDL evaluate with strict time cutoffs by construction. | BAR + FOR | C3, C1 |
| L022 | Kapoor & Narayanan: leakage across 17 fields / 329 papers; a leaked feature makes a complex model appear to crush LR (demo: gap +0.217) but the win collapses to a tie (−0.009) once removed. A big relational-vs-GBDT margin is therefore a *leak hypothesis first*; every reported RDL gain must travel with a provenance/leakage audit (model info sheet), and RelBench's point-in-time cutoffs are the structural defence. | BAR | C3 |
| L023 | Demšar: a reported gap is a random variable, so an RDL "win" needs a significance test — and the obvious one lies. A naive paired t-test on CV folds is anticonservative (demo: +0.0098 gap, naive p=1.2e−5 vs corrected p=0.19); the fix is the corrected resampled t-test on one dataset and Friedman + Nemenyi CD across many. C3 is won by a gap a skeptic's *test* cannot dissolve, never by a bigger mean. | BAR | C3 |
| L024 | Grinsztajn benchmark: the fair way to compare model families is a random-search *budget curve* (default + ceiling), on curated datasets, normalized per dataset — not one tuned peak. GBTs beat NN families at every budget and are stronger defaults (repro: GBT vs MLP on credit-g, gap +0.062 default → +0.015 tuned, never closing). This raises the bar (a strong, cheap-to-tune GBT default). But the whole contest lives *inside the single-table world* — the benchmark curates away relational structure, so beating a GBT on Grinsztajn's terms is table stakes; the thesis (C1) attacks the flattening that happens before either contestant sees the data. | BAR + FOR | C3, C1 |
| L025 | Grinsztajn §5.2 (Finding 1): the deepest reason MLPs lose to trees is the **smoothness (spectral) bias** — gradient descent fits smooth functions easily and irregular ones poorly, while tabular targets are jagged. Proof by ablation: Gaussian-smoothing the target collapses the GBT-vs-MLP gap in lockstep with the variance removed (repro: +0.33 R² → ~0; the MLP even edges ahead once the target is smooth). This raises the bar (any RDL model ending in an MLP head over a flattened row inherits this bias, so a naïve RDL model can lose to a GBT for exactly this reason). But it also hints at the opening: the irregularity that trips an MLP on a flat table often *is* relational structure crushed into a column (e.g. "days since last order" is a jagged function of an entity's event history) — a model reading the history as structure may represent it natively instead of fighting its own smoothness bias. | BAR + FOR | C3, C1 |
| L026 | Grinsztajn §5.4 (Finding 3): the **rotation bias**. A tree is not rotationally invariant (axis-aligned splits attend to each meaningful column); an MLP/ResNet is (`W·(Qx)=(WQ)·x`). A random, *lossless* rotation collapses the tree and leaves the MLP unmoved, **reversing the ranking** (repro: tree 0.987→0.747, MLP 0.862→0.869, +0.008). This cuts both ways for the thesis. BAR: an RDL model that ends by feeding a flattened row into an MLP head inherits rotation invariance and is handicapped on exactly the tabular signal a tree exploits — "GNN then MLP" is not a free win. FOR: the reason invariance is *bad* is that **columns carry individual meaning** — the original basis is privileged — and a relational database is the maximal version of "the structure is meaningful" (which entity, which foreign key, which event). The instinct "don't rotate a table's axes" is the same one as "don't dissolve a database's schema"; and the flat-table fix (numeric-feature embeddings in SAINT / FT-Transformer that *break* invariance) is a small echo of giving the model back the structure the naïve representation discarded. | BAR + FOR | C3, C1 |
| L027 | Grinsztajn §5.3 (Finding 2): the **uninformative-features bias**. Trees are robust to junk columns via greedy, gain-gated split selection (implicit feature selection); MLPs are not — every feature enters the first layer and, being rotationally invariant, an MLP needs ≥ linearly more samples per junk feature (Ng 2004), leaking capacity onto noise. Proof: on a *smooth* target where the MLP wins clean (0.986 vs GBT 0.945), adding 100 pure-noise columns costs the MLP 0.084 vs the GBT 0.032, reversing the ranking; the gate is visible as root-split gain ~118× higher on informative than junk. BAR: a naïve RDL model that pools a graph into a flat vector and feeds an MLP head inherits this fragility, and databases are full of near-junk columns (surrogate keys, audit fields, denormalized dupes) — so "GNN then MLP" does not get the tree's implicit selection for free. FOR: the deeper reading is that a good model should *select structure, not drown in it* — message passing that learns which foreign-key paths carry signal is implicit feature selection over the schema, the tree's instinct scaled to the whole database; the flat-table fix (learned attention/embeddings in SAINT/FT-Transformer) is a learned gate recovering what a tree gets for free. | BAR + FOR | C3, C1 |
| L029 | Feurer 2015 (Auto-sklearn): AutoML automates the **CASH** search (which algorithm + its hyperparameters, selected by validation) plus meta-learning warm-start and Caruana ensemble selection — the strongest form of "just automate the single-table pipeline." Repro on credit_g: the big jump is tuning *at all* (default XGB 0.775 → tuned 0.806, +0.031); a 4-algorithm AutoML with ensembling then only **ties** the tuned XGB (0.803, bands overlap) at far higher compute (the greedy ensemble did add a free +0.007 over the single best config). BAR: an honest single-table baseline can now be "whatever AutoML finds", foreclosing the objection that an RDL win merely beat a lazily-tuned model. FOR (indirect, C1/C2/C4): AutoML searches *models and knobs*, never the **feature representation** — it explicitly does no domain feature engineering and tunes on top of an already-flattened table, so it cannot recover what the join/flatten step discarded; and that a many-algorithm AutoML only ties a tuned GBDT shows the returns to cleverer single-table *search* are nearly exhausted, so the remaining upside must come from a better *representation* (learning over relational structure) — exactly the thesis. | BAR + FOR | C3, C1, C4 |
| L028 | Gorishniy 2021: the honest neural baselines are a *properly-tuned* MLP and **ResNet** (pre-activation residual blocks; the skip makes the identity free, fixing the degradation problem so depth stops hurting — repro: same net, plain test 0.917→0.866 & TRAIN 1.000→0.927 over depth 1→32, ResNet holds ~0.90). Their central finding is methodological: once you compare against these baselines, much prior tabular-DL "progress" evaporates. BAR: the single-table bar an RDL result must clear is now a tuned GBDT **and** a tuned ResNet — an RDL "win" over a weak neural baseline would be exactly the mistake Gorishniy exposes; and on small categorical `credit_g` the GBDT still leads (0.793) with MLP (0.752) ≈ ResNet (0.743) tied, so the neural machinery does not repeal L024–L027. FOR (indirect): this is the machinery the thesis is *built from* — the RDL stack is encoder → message passing → a residual-MLP head — so owning the honest baseline and the residual block is a prerequisite for making (and defending) any relational win; and the inductive-bias debts (L025–L027) a flat ResNet still carries are exactly what reading structure natively is meant to pay off. | BAR + FOR | C3, C1 |
| L030 | **Q3 checkpoint** — the whole evaluation-rigor toolkit assembled into one **benchmark report**: deployment-matched split (L021), leakage audit (L022), a random-search budget curve over a tuned GBDT + honest neural baseline + AutoML bar (L024/L028/L029), a corrected resampled significance test (L023), and an inductive-bias explanation (L025–L027). Repro on credit_g: GBDT led the honest MLP baseline by only **+0.0081 ROC-AUC** over 25 paired folds, and the corrected resampled t-test gave **p=0.64** (naive p=0.22) → **no significant winner**. BAR: this is the complete, exact instrument any RDL claim must clear — a fair, temporal, leak-audited, significance-tested, bias-explained one-pager that, crucially, *reports a tie when the data says tie*. FOR (indirect, C3/C4): the checkpoint's honesty is the thesis's credibility reserve — a program that reports "no significant winner" on credit_g today is one whose eventual "RDL beats the incumbent" verdict a skeptic can trust; and the tie confirms (with L029) that returns to single-table *search/architecture* are nearly exhausted, so the open upside is representational. | BAR + FOR | C3, C4 |
| L031 | Guo & Berkhahn 2016 (Entity Embeddings) — Q4 opener, the first **learned representation**. The four categorical encodings and their trade-offs (repro on credit_g: ordinal's false order drops the linear model 0.782→0.739 but not the GBDT 0.778→0.774; OOF target ties one-hot at 20 cols vs 61; naive target encoding of a signal-free id **leaks** 0.891 AUC, fixed to 0.504 out-of-fold). An entity-embedding MLP **ties** a *fair* one-hot MLP (0.774 vs 0.798; an undertrained baseline 0.728 would fake a +0.07 win — the L028 trap, live). FOR (C4, C1): entity embeddings *are* the atom of RDL — target encoding is a 1-D learned embedding, entity embeddings the d-dim generalisation, and the highest-cardinality categoricals are the **foreign keys** that point into other tables, which one-hot cannot touch and naive target encoding leaks on; a learned embedding of an entity id trained end-to-end is exactly how a relational model represents a customer/product. BAR: the tie on credit_g shows a learned representation buys nothing on a *small flat* table — the payoff is structural, only visible at scale and with relational reuse, which is the thesis restated (the value single-table misses is structural). | FOR + BAR | C4, C1 |
| L032 | Huang et al. 2020 (TabTransformer) — preview: the per-column entity embeddings (L031) are passed through Transformer **self-attention** (`softmax(Q·Kᵀ/√d)·V`) to become **contextual** embeddings (each categorical column's vector now depends on the other columns in the same row); continuous features bypass, then concat → MLP head. Honest verdict: it **matches** tree ensembles on supervised tabular (the +1.0% is over other DEEP methods, not trees), with genuine wins only in robustness to noise/missingness and a +2.1% **semi-supervised** lift from unlabeled data. AGAINST-leaning BAR: another deep architecture that ties, not beats, a tuned GBDT on a flat table (the L028/L029/L030/L031 pattern) — attention over a row's columns is more single-table cleverness and buys no accuracy, so an RDL win cannot lean on "attention is powerful" alone. FOR (C1, C2, C4): a contextual embedding is a **weighted aggregate of related vectors** — the exact operation a GNN uses to update an entity from its neighbours; TabTransformer attends *within a row* (over columns), while RDL applies the same operation *across rows in related tables* (over an entity's foreign-key neighbours). The tie is the thesis: no new structural information enters within one table, so the untapped value is the cross-table structure — and the semi-supervised pre-training foreshadows the relational foundation models of Y5. | BAR + FOR | C3, C1, C4 |
| L033 | Domingos 2012 (*A Few Useful Things to Know about ML*) — the essay's three load-bearing claims turned into a controlled experiment: "feature engineering is the key", "more data beats a cleverer algorithm", "overfitting has many faces". Repro on credit_g with the **model held fixed** (HistGB), adding hand features one at a time in a fixed order: CV ROC-AUC **peaks at just k=3** (0.7911 vs 0.7865 baseline, **+0.0046 — inside the ±0.032 CV band**, not significant by L023) then **declines to 0.7659 below baseline** by k=8 (the overfitting tax); a linear model drifts only +0.006, also within noise. FOR (C1, C4 — the load-bearing entry): this **quantifies the ceiling** the whole thesis rests on — the value extractable by *reshaping one table* is nearly exhausted against a competent model, so the returns to manual single-table feature effort have effectively **gone to zero (then negative)**. The features that would still pay are aggregates **across related tables** (a customer's 90-day average, prior-default count) — which Deep Feature Synthesis (L009) hand-builds and RDL aims to **learn end-to-end**; "the returns moved across the join." The human-effort ratio Domingos implicitly measures is exactly what Year 4's manual-FE-vs-RDL studies test. BAR: honesty guard — the correct read of a +0.005 bump inside a ±0.03 band is "no measurable FE gain", so any future "RDL adds features that pay" claim must clear the same noise-band / significance test, not celebrate a within-noise bump. | FOR + BAR | C1, C4, C3 |
| L034 | Kimball dimensional modeling (star schema & joins) — makes the flatten **literal**: real data is a relational schema (fact/event tables + dimensions, linked by PK/FK), and to feed any Q1–Q3 model you must **choose a grain** (one entity at one prediction time), **join** neighbour tables via foreign keys, and **aggregate** the one-to-many rows into fixed-width columns — guarded by a point-in-time filter (`order_ts < t`) or the future leaks (demo on the toy DB: dropping the guard moved C1 from n=3/total=125 to n=4/total=1124). FOR (C1, C2): this *is* the single-table paradigm the thesis critiques, now shown to be (a) a **hand-built, per-task pipeline** of DFS-style aggregates rather than a given, (b) **lossy by construction** — the mean/count discard the one-to-many cardinality, event order, identity, and multi-hop paths — and (c) **leakage-prone at the feature step**, re-imposing PIT discipline on every aggregate and every re-flatten. These are exactly the costs RDL claims to remove by keeping the PK/FK edges as a **graph** and learning the aggregations end-to-end (Y3 message passing, Y4 REG). BAR: the demonstration is still *conceptual/mechanical* — flattening is shown to be a choice with a cost, but no result yet shows a model **beating the fair bar by keeping structure** (that is L035's setup and the Y1-exit → Y3–Y4 burden). | FOR | C1, C2 |
| L035 | Fey et al. 2024 §1–2 (★ preview) — the flatten's cost measured. Join+aggregate is a **lossy (surjective) map**, proven by an **aggregation collision**: two customers with different histories (Ada rising $10→$30→$50 over 3 products; Bo falling $50→$30→$10 on 1 product) flatten to the **byte-identical** row `n=3/total=90/avg=30/max=50`, so a fitted classifier gives them the **same** P(churn)=0.502 although their true labels differ (0 vs 1) — an *information* loss before any model, not a capacity/tuning/leakage problem. The discarded structure has four names — **cardinality, event identity, temporal order, higher-order paths** — matching Fey §2's five issues with manual feature engineering, of which issue (4) ("forcing data into a single table aggregates into lower-granularity features, thus losing fine-grain signal") is the load-bearing claim. FOR (C1, C2 — the pivot of Year 1): this is the thesis's central mechanism made concrete and runnable — the single-table paradigm doesn't just cost effort (L033) or risk leakage (L034), it **destroys recoverable signal**, and hand-built recovery (spend_trend restores order +40/−40; n_distinct_products restores identity 3/1) is an **unbounded per-task treadmill** (a third customer Zoe collides again), which is exactly why keeping the DB as its **relational entity graph** (row=node, PK/FK=edge) and learning aggregations end-to-end is the proposed escape. BAR (honesty guard): still a *demonstration of cost*, not a win — no result yet shows a graph model **recovering** the discarded structure to beat the honest single-table bar (tuned GBDT + ResNet + AutoML, L028–L030); that is the Y1-exit essay's argument and the Y3–Y4 empirical burden. A future "RDL keeps signal the flatten loses" claim must clear that fair bar, not merely exhibit a collision. | FOR + BAR | C1, C2 |
| L039 | **Year 1 synthesis essay** — turns L001–L038 into a single falsifiable claim a hostile reader can grade. Grants Grinsztajn's flat-table result, explains it with the three inductive biases, documents the Q4 **exhaustion cascade** (honest nets / AutoML / embeddings / TabTransformer / hand FE repeatedly tie or fail), names **boundary conditions** (smooth / rotated / low-junk; silence after a lossy join), and ends on the **open burden** (no fair-bar RDL win yet). FOR (C1, C4): absorbs the skeptic's strongest objection instead of dodging it — "trees win on flat tables" becomes the *setup* for "the unpaid upside sits across the join." BAR: the essay's credibility coda binds every comparative sentence to the L038 peer-review checklist (two pipelines, one standard), so an eventual RDL claim inherits the same immune system. | FOR + BAR | C1, C3, C4 |
| L040 | **Year 1 exit exam** — closes the year on the curriculum's two deliverables: a **regenerable** XGBoost baseline under the L020 fair protocol on OpenML `adult`, plus a **written** account of Grinsztajn's three inductive biases with numbers and flip conditions. The experimental fork is BEAT / TIE / EXPLAIN against a disclosed ±0.002 ROC-AUC noise band; the modal honest outcome (L020 evidence of record: ref 0.9282 vs LGBM 0.9296) is a **TIE**, and a TIE plus explanation is a full pass. FOR (C3, C4): the exit institutionalises "matching the stubborn flat bar is success" — so an eventual RDL win cannot be faked by soft-selling tiny deltas (M48) or by demanding a leaderboard scalp Year 1 never promised (M49). BAR: STAND/REVISE binds the regenerable number to the L039 claim without inventing a fair-bar RDL win; the open burden stays open as Year 2 begins. | FOR + BAR | C3, C4, C1 |
| L041 | **Deep-tabular landscape & rtdl** (Gorishniy et al. 2021 ★) — opens Year 2 by making the *neural* half of the single-table bar strong and honest. The paper's contribution is methodological: the field lacked a **strong simple baseline** (a tuned ResNet, which alone matches many prior "novel" architectures) and a **shared tuning protocol**, so earlier "DL beats trees" claims were unfair (an HP-budget gap, L038). It establishes ResNet as the baseline to run first and **FT-Transformer** (Feature Tokenizer over *all* features + [CLS] + Transformer) as the strong universal DL model, and its honest headline is **no universal winner** — a tuned GBDT still wins on a large share of datasets (M50). BAR: the neural single-table opponent an RDL result must beat is now the strongest fair one (FT-Transformer via rtdl), not a strawman — beating a weak net would be as worthless as beating an undertuned XGBoost. FOR (indirect, C4): the Feature Tokenizer's per-entity embeddings and attention are the machinery an RDL encoder is built from. | BAR | C3, C4 |
| L042 | **MLP & ResNet baselines — do these first** (Gorishniy 2021 §3.2) — turns L041's map into a trained skill: build the ResNet **from scratch** (the L028 residual block, promoted to `relkit.nets`; skip makes the identity free so depth stops degrading — He 2015) and **validate it against rtdl** (|Δ| ROC-AUC = 0.000), then tune both nets under a **shared protocol** (shared frame = same split/metric/search-budget/validation-selection; only the per-model search space differs) and read the result across **several datasets**. Two load-bearing disciplines: the **baseline-first rule** (M51 — a tuned ResNet alone matches many "novel" models, run it *before* the fancy one; fairness is equal *budget*, not equal knobs, L038) and **multi-dataset rigor** — no comparative conclusion from one table. Verified instance (L042 evidence of record — from-scratch models validated vs rtdl, `labs/_verify_l042.py`): credit_g is a within-noise tie (MLP **0.802** ≈ ResNet **0.790** ≈ GBDT **0.780**); across four small tables the nets rank ahead (mean ranks 1.25/1.75 vs 3.00, Friedman p=0.039), but that set is numeric-skewed — a *demonstration*, not proof "nets beat trees". The representative **no universal winner** is Grinsztajn 2022's ~45 datasets (L024/L041). BAR: the neural half of the single-table bar is now *trainable, fair, and from-scratch*, so an eventual RDL win over it cannot be dismissed as beating an undertuned or buggy net. FOR (indirect, C4): the residual-MLP head trained here is a literal component of the RDL stack (encoder → message passing → residual-MLP head, Years 3–5). | BAR | C3, C4 |
| L043 | **TabNet — sequential attention** (Arik &amp; Pfister 2019 ★) — the first novel architecture put through L042's bar, built **from scratch** (sparsemax, attentive transformer, prior scale, feature transformer; sparsemax **validated against `pytorch_tabnet`**, max \|Δ\| = 2.4e-07). Two mechanisms worth keeping: **sparsemax** gives exact zeros, so a mask is a genuine *selection* rather than a weighting; the **prior scale** `P[i] = ∏(γ − M[j])` is what makes the attention *sequential* rather than merely sparse (M54). **AGAINST-the-hype / BAR**: held to the baseline-first rule under one shared frame on four small tables, TabNet ranks **behind** the tuned simple baselines — mean ranks TabNet **2.50**, MLP **1.75**, ResNet **2.00**, GBDT 3.75, Friedman **p = 0.127** (`labs/_verify_l043.py`). Stated precisely (M55): p > 0.05 licenses only "cannot distinguish on this sample", *not* "significantly worse" — but the burden of proof is the new model's, and it was **not met**. The paper's own Appendix A (KDD) has TabNet tying or trailing XGBoost/CatBoost. **Interpretability, verified rather than assumed** (M53): on the paper's own generators, `M_agg` recovers **global** relevance cleanly (Syn2: top-4 exact, 76.8% of mass on the truth) but only **partially** recovers **instance-wise** relevance (Syn4: switch found at 0.118, yet 15.6% vs 97.9% of rows favour their own group; the paper used 10M not 10k samples for sharp masks). Honest open item: from-scratch TabNet **outscored** the reference end-to-end (credit_g 0.748 vs 0.694) and two hypotheses — training length and LR schedule — were tested and **refuted**, so the gap stands **unexplained** and is recorded as such. FOR (indirect, C4): *instance-wise* feature selection is the single-table shadow of what RDL does structurally — different rows genuinely need different context. | BAR | C3, C4 |
| L049 | Source claims, copied-weight fidelity, fixed-budget scores and temporal transfer are separate evidence. Neither a close score nor a local rank reversal reproduces a published benchmark. | BAR | C3 |
| L050 | A fixed three-task, two-candidate comparison gives FT-T two wins and XGBoost one; Friedman p=.368. Model parity and protocol discipline establish a baseline bar, not relational superiority. | BAR | C3 |
| L051 | On three released numeric tasks, fixed-recipe rotation changes model ranks while information remains invertible; noise has finite-sample exceptions. Original ranks have Friedman p=.717. Controlled interventions strengthen baseline diagnosis; they do not establish relational superiority or reproduce the paper search curves. | BAR | C3 |
| L195 | Complete replay of 33,650 stored predictions; strong engineered-relational comparator, conditional uncertainty crossing zero, small task-specific RDB-PFN gain, and all 21 unrun RDBLearn task results preserved. Distinguishes relational signal, learned processing, fair robustness and economic undervaluation. | AGAINST universal necessity + BAR | C1–C4 |

---

## The honest bar (what "beating the incumbent" requires)

Assembled from Q1–Q2. To make the thesis legible to a skeptic, an RDL result must:

1. Use a **fair-comparison contract** (fixed data, split, metric, tuning budget, preprocessing scope; L020).
2. Beat a **tuned** GBDT *and* a **leak-free OOF stacked ensemble** (L018), not a single default.
3. Hold under **temporal / grouped splits** with no leakage (L002–L005), not just random IID.
4. Report the **gap size and verdict honestly**; a suspiciously large win implies a leak or an unfair
   reference (L020).
5. **Prove the gap is not noise** with a correct significance test — a corrected resampled t-test on one
   dataset, or a Friedman + Nemenyi rank test (CD diagram) across tasks — plus an effect size, not a bare
   mean (L023).

---

## Skeptic's strongest objections (and our current answer)

- **"Trees already win on tabular data — why bother?"** (Grinsztajn, L019). *Answer so far:* that result is
  about *single-table* data whose biases fit trees; the thesis is that the single-table *representation*
  discards relational structure, a different axis. Not yet demonstrated — this is the Y3–Y4 burden.
- **"Just flatten harder / engineer more features."** *Answer so far:* DFS/Featuretools show manual
  recovery is possible but ad hoc and leakage-prone (L002, L009); the bet is that learning it end-to-end
  beats hand-crafting. Undemonstrated at scale yet.
- **"Modern tabular nets (RealMLP/TabM/TabPFN) already close the gap."** *Answer so far:* acknowledged —
  they narrow the single-table tree–DL gap, which is *orthogonal* to exploiting cross-table structure. To
  be tested against on the relational frontier, not dodged.
- **"Maybe RDL's reported wins are just leakage too."** (Kapoor & Narayanan, L022). *Answer so far:* the
  right worry, and the reason every RDL result in this program must ship a leakage audit (model info sheet)
  and lean on RelBench's structural point-in-time cutoffs. A suspiciously large win is treated as a leak
  hypothesis before a method hypothesis — the thesis is only credible if it survives that scrutiny.

---

## Current verdict — L195 stress-test (2026-10-02)

**Narrow the claim; broad superiority and undervaluation remain unestablished.** Complete saved-evidence replay rescored all 33,650 L149/L182 predictions and authenticated all 21 L194 task rows. L149's basic GNN has test MAE 4.123071 versus 3.948917 for relational feature engineering; GNN advantage −0.174155, conditional 95% driver interval [−0.449621,+0.085850]. This crosses zero and establishes neither superiority nor equivalence. L182's RDB-PFN minus TabICL mean is +.004369 AUROC, positive 6/10 support draws on one task. Both comparisons use relational information in both arms. Replaying them creates no independent replication. L194 has zero fresh task results; its preprocessing gate is a reproducibility obstacle, not measured performance counter-evidence.

Retain C1 as a task- and representation-dependent possibility. C2 remains a testable hypothesis, not a claim that learned relational encoders are necessary. C3 needs matched information, strong baselines and new held-out tasks/times. C4 requires cost, utility and adoption evidence absent from these scores. Earlier language below asserting exhausted single-table returns from a few ties, or inevitable useful loss from flattening, is too broad; retain it as historical reasoning rather than the current conclusion. Mission unchanged; learner defense pending.

[Full falsification brief](labs/evidence/l195/falsification-brief.md) · [Lesson](lessons/0195-thesis-stress-test.html).

---

## Historical verdict (updated 2026-07-29, after L040 / Year 1 exit exam)

**Undecided, and honestly so.** Q3 completed the *instrument* rather than the *case*: the dossier now
owns the full honest bar (C3) — not just a strong incumbent (Q2) but the whole apparatus that certifies
any "A beats B" claim (temporal splits L021, leakage audit L022, corrected significance L023, budget-curve
benchmark L024, the three inductive-bias explanations L025–L027, an honest neural baseline L028, the AutoML
ceiling L029), assembled into one defensible benchmark report (L030). Two Q3 findings sharpen the case
*for* the thesis, indirectly: (a) AutoML only *ties* a tuned GBDT (L029) and (b) on credit_g the GBDT vs
neural gap is *not significant* (L030) — together showing the returns to cleverer single-table
search/architecture are nearly exhausted, so the remaining upside is representational (C4). Q4 now opens
that representational front: L031 introduces the first *learned representation* (entity embeddings) and
shows it too *ties* one-hot on a small flat table — deflationary on its face, but exactly the thesis
restated (a learned representation buys nothing without structure/scale to exploit), and it names the
concrete seam the bet lives on: the high-cardinality **foreign keys** one-hot cannot touch and naive
target encoding leaks on are precisely what an entity embedding — the atom of RDL — is built to represent.
L032 (TabTransformer preview) sharpens this: it adds *self-attention* over a row's columns to make the
embeddings **contextual**, and — for the fifth time (L028/L029/L030/L031) — only *matches* a tuned GBDT on
a flat table, its headline gain being over other deep methods, not trees. The pattern is now unmistakable:
returns to single-table cleverness (search, architecture, representation, attention) are exhausted. But the
same lesson supplies the thesis's clearest mechanism yet — a contextual embedding is a weighted aggregate
of related vectors, i.e. exactly the message-passing/attention a GNN performs; TabTransformer attends
*within a row*, and RDL applies the identical operation *across rows in related tables* via foreign keys.
The bet is precise: the value is not in attending harder over one table's columns, but in unleashing that
same aggregation over the relational graph.
L033 (Domingos 2012) now *quantifies the ceiling* directly: with the model held fixed, hand-crafted features
on credit_g peak at a mere 3 features (+0.005, inside the ±0.03 noise band → not significant) and then go
**negative** (0.766, below the no-feature baseline) — manual single-table feature effort has effectively
zero, then negative, marginal return against a competent model. This is the deflationary Q4 pattern stated
in its bluntest form (returns to reshaping *one* table are exhausted), and simultaneously the sharpest
setup for the bet: the features that would still pay are relational aggregates *across* tables (the DFS
operations of L009), which is precisely what RDL proposes to learn end-to-end — "the returns moved across
the join," and the human-effort ratio is Year 4's explicit test.
L034 (Kimball star schema & joins) now makes that join *literal* and, with it, the thesis's target concrete:
the flat design matrix every Q1–Q3 model consumed is not given but **manufactured** — pick a grain, join
foreign keys, aggregate the one-to-many rows, and re-impose a point-in-time filter on every aggregate (drop
it and a toy customer's spend jumps 125→1124 as future orders leak in). That manufacturing is a per-task,
hand-written pipeline of DFS-style primitives that is **lossy by design** (the mean/count throw away
cardinality, event order, identity, multi-hop paths) — so the thesis's C1 ("flattening is the lossy step RDL
replaces") is no longer a slogan but a mechanism the learner can now build and audit. This sharpens the open
burden rather than discharging it: L034 shows the *cost* of flattening, L035 now *quantifies* the discarded
structure, and only Y3–Y4 can show a model **recovering** it to beat the fair bar.
L035 (Fey 2024 §1–2, the Year-1 pivot) turns "lossy by construction" from an assertion into a demonstration:
join+aggregate is a lossy map, and an **aggregation collision** proves the loss can be *total* — Ada (rising
spend, 3 products) and Bo (falling spend, 1 product) flatten to the identical row `n=3/total=90/avg=30/max=50`,
so a fitted model returns the same P(churn)=0.502 for both even though their labels are 0 and 1. The single
table cannot express the difference, so the failure is upstream of every model — it is *information*, not
capacity, tuning, or leakage. This is the sharpest statement yet of C1 (flattening is the lossy step RDL
replaces) and it names the four dimensions destroyed (cardinality, identity, temporal order, multi-hop paths)
against Fey's issue (4). Crucially it does **not** discharge the burden: hand-built recovery works one
collision at a time (spend_trend, n_distinct_products) but is an unbounded per-task treadmill, and no result
yet shows a graph model *recovering* the lost structure to beat the honest bar (L028–L030). The arc L001→L035
is now complete on the *diagnostic* side — the single-table assumption is exposed as a manufactured, lossy
choice — and the *constructive* side (a model that keeps structure and wins fairly) is exactly what Years 3–6
must deliver.
L036 contributes nothing to C1–C2 and everything to the **credibility precondition** underneath them. The
thesis's eventual claim ("RDL beats a fair single-table bar") is only worth reading if the person making it
audits their *own* pipelines as hard as they audit the baselines they intend to beat — so this lesson turns
the L001–L035 diagnostic apparatus on the learner's real submission and finds four defects in work that was
already careful: an inner calibration split that silently drops the person grouping the outer split enforces
(degrades the shipped artifact, leaves the reported metric honest — re-measured 1.4248→1.4232 log-loss,
0.0363→0.0360 ECE, both far inside the 0.039 fold σ); a shipped winner chosen by argmin over five correlated
folds on 0.0032 nats, 8 % of one fold's std, that flips to the runner-up when a single fold is dropped
(changes the decision, every number correct); preprocessing fit on all 119,498 rows including every test fold
(transductive, so the CV number is not a deployment number); and no event timestamp anywhere in the schema,
so a system deployed forward in time can only be evaluated random-in-time (a declarable limitation, not a
bug). The transferable instrument is the **consequence-class triage** — inflates the number / degrades the
artifact / changes the decision / can only be declared — which is precisely what will be demanded of the
Y3–Y4 RDL-vs-GBDT comparison: a leak found there must be priced, not merely announced, and a win inside the
fold noise band is not a win. Note also which finding is *not* leakage at all: the report's only
"significant" claim (M1−M0, naive p = 0.0146) dies under Nadeau–Bengio correction (0.0514) and Holm over its
four tests (0.0583), while the effect itself (5/5 folds, 0.055 nats) survives — the exact discipline L023/L030
demanded, now applied where it costs the learner something.

L037 finishes the precondition L036 opened, and it does so by *measuring* the thing everyone asserts. A
comparison is only evidence if it can be regenerated, so the lesson probes nine one-knob perturbations of the
learner's own pipeline against a hash of the full out-of-fold matrix. **Eight were bit-identical** — thread
counts 1–12, LightGBM's `deterministic` flag, row-wise vs column-wise histogram building, shuffled training
row order, and the **model seed**, which is inert here because this configuration never samples. The one that
moved was an undocumented `.astype(np.float32)` living in a notebook cell: **258 of 5,587 predicted classes
change** (max Δp = 0.326) for a mean log-loss shift of only **+0.00133** — which is nonetheless **42 % of the
0.0032 margin that chose which model shipped** (L036). Two more results bear directly on the honest bar. The
*same literal* `RANDOM_STATE = 0`, handed to the splitter instead of the model, spans **0.0166 nats across
five fold draws — 5× that margin**, making the fold draw the largest controllable term in the report and any
future RDL-vs-GBDT delta smaller than it uninterpretable. And rolling LightGBM back one minor version does not
change the number, it **crashes**: `lightgbm 4.5.0` + `scikit-learn 1.9.0`, both satisfying this workspace's
own constraints, raise `TypeError: check_X_y() got an unexpected keyword argument 'force_all_finite'`. The
transferable instruments are the **run manifest** (the run describes itself), the **output fingerprint**
(strictly stronger than agreeing on a summary metric), the **estimator of record** (the same ECE reads 0.0332
per-fold and 0.0178 pooled, a 1.87× spread that is about the ruler, not the model), and the **noise floor**
(what a perfectly-calibrated control scores at that n — which exposes one ship-gate in the submission that no
model could pass). Y3–Y4 will compare an RDL model to this baseline; every one of those instruments is what
makes the comparison mean something rather than merely happen.

L038 closes the credibility arc by converting the self-audit stance into the *reviewer's* stance — the one a
skeptic will actually adopt when the thesis's comparative claim ("RDL beats a fair single-table bar") lands on
their desk. It is a method lesson, not a new measurement: it re-reads every verified L036/L037 finding through
a peer-review checklist across three axes (leakage, tuning, metrics) and, crucially, triages them on **two
independent axes** — *conclusion-impact* vs *artifact severity* — that a novice collapses into one. Applied to
the learner's own submission the verdict is **major revision**: sound engineering, but all three headline
claims are overstated (the 0.0032-nat selection fails a corrected test and flips on one dropped fold; "the
ECE" is 0.0332 or 0.018 depending on an unnamed estimator, with one gate below its noise floor; "reproducible"
is true-but-inert and ships no lockfile). The load-bearing point *for the thesis* is the discipline it names
explicitly: a comparative claim needs **two** pipelines reviewed to **one** standard, and the baseline you
want to beat is the one you are least motivated to scrutinise — so the review that makes an eventual RDL win
credible is the hostile review of the GBDT baseline, run first, to the exact standard a reviewer would demand
of the model you love. This is the immune system the Y1 exit (L040) and every Y3–Y4 RelBench comparison run
on.

L039 (Year 1 synthesis essay) turns the whole arc into a **written claim a hostile reader can grade**. It does
not add a new bake-off; it forces the learner to *absorb* the skeptic's strongest objection ("trees already
win on tabular — why bother with RDL?") instead of dodging it. The essay's working claim grants the flat-table
result, explains it with Grinsztajn's three inductive biases, shows via the Q4 **exhaustion cascade**
(L028–L033) that further single-table cleverness has repeatedly tied or failed, names the **boundary
conditions** (smooth / rotated / low-junk; silence about signal a lossy join destroyed), and ends on the
**open burden**: Year 1 demonstrated the *cost* of flattening (L034–L035) and built the honest bar plus the
credibility apparatus, but has **not** yet shown a relational model recovering discarded structure to beat
that bar. The transferable instrument is the genre itself — synthesis ≠ recap — and the peer-review coda that
binds every comparative sentence to L038's checklist. L040 will ask the learner to beat XGBoost on a flat task
*or explain why not*; this essay is the explanation they will stand on or revise.

L040 (Year 1 exit exam) **closes Year 1** on those two curriculum deliverables. The runnable bar is OpenML
`adult` under the L020 fair protocol (public, regenerable — homework remains the Q4 audit/package/review
artifact). The fork classifier treats gaps inside ±0.002 ROC-AUC as ties; the L020 evidence of record
(ref XGB 0.9282, tuned LGBM 0.9296, stack 0.9297, OOF corr 0.997) predicts the modal honest outcome is
**TIE**, and the exit grades that as a pass when paired with a cold written account of the three biases and
an explicit STAND/REVISE on the L039 claim. Soft-selling Δ=+0.001 as a beat (M48) or treating a non-win as
exit failure (M49) are the failure modes the lesson exists to kill. Year 1 ends with a high, regenerable
flat-table bar and a written inductive-bias understanding — not with a fake RDL scalp on the ledger.

L041 (deep-tabular landscape & rtdl) opens Year 2 and starts hardening the *other* half of the
single-table bar — the neural one. Its lever for the thesis is entirely about the credibility of a future
comparison. Gorishniy 2021 shows that most pre-2021 "deep learning beats trees on tables" results were
unreliable because the field had no strong, simple, shared baseline and no shared tuning protocol, so
architectures were compared unfairly (the L038 HP-budget-parity failure, now diagnosed at subfield scale).
It fixes that with a well-tuned **ResNet** baseline that alone matches many "novel" models, and adds
**FT-Transformer** — whose Feature Tokenizer turns *every* feature (numeric included) into an attention
token and reads a [CLS] summary — as the strong universal DL model. Run fairly, the verdict is **no
universal winner**: a tuned GBDT still wins on a large share of datasets, the same conclusion Grinsztajn
reached from the inductive-bias side. For the mission this is pure BAR-raising: the neural single-table
opponent an RDL result must beat is now the strongest *fair* one (FT-Transformer via the rtdl reference
implementations), which forecloses the skeptic's move "a good tabular transformer would have won on a flat
table anyway." The one FOR thread is indirect — the tokenizer's per-entity embeddings and self-attention
are the exact machinery an RDL encoder is assembled from (L031/L032 → Years 3–5).

L042 (MLP & ResNet baselines — do these first) converts L041's map into a *trained* skill and, with it,
converts the neural half of the single-table bar from *named* to *trainable and fair*. The learner builds
the ResNet **from scratch** — the L028 residual block (promoted to `relkit.nets`) whose skip makes the
identity map free, so the degradation problem (He 2015) stops depth from hurting — and **validates it
against rtdl** (a reference is a *checker*, not a teacher; verified |Δ| ROC-AUC = 0.000), then tunes it,
plus a plain MLP, under a **shared protocol**: a shared frame (same split, metric, search budget,
validation selection) with only each model's search *space* differing. Two load-bearing ideas for the
thesis: the **baseline-first rule** (a tuned ResNet alone matches many published "novel" architectures, so
run the strong simple baselines *before* the fancy model; fairness is equal *budget*, not equal knobs —
L038) and **multi-dataset rigor** (no comparative conclusion from one table). Under that fairness the
verified credit_g result (L042 evidence of record — from-scratch models validated vs rtdl,
`labs/_verify_l042.py`) is a **tie within noise** — MLP 0.802 ≈ ResNet 0.790 ≈ GBDT 0.780. Read across
four small tables the nets rank ahead (mean ranks 1.25/1.75 vs 3.00, Friedman p=0.039), but that sample is
tiny and numeric-skewed — a *demonstration of the method*, not proof "nets beat trees"; the representative
**no universal winner** stays Grinsztajn 2022's ~45 datasets (L041/L024). This is pure BAR-raising: an
eventual RDL win over the single-table neural bar can no longer be waved away as beating an undertuned or
absent net, because the bar is now a *properly-trained* MLP/ResNet. The indirect FOR thread is that the
residual-MLP head trained here is a literal component of the RDL stack the later years assemble
(encoder → message passing → residual-MLP head).

L043 (TabNet — sequential attention) is the first time the bar L042 built is actually *used*, and it is
worth recording that the bar bit. TabNet is a genuinely interesting mechanism: **sparsemax** projects onto
the simplex so a mask can hold exact zeros — a real *selection*, not a weighting — and the **prior scale**
`P[i] = ∏_{j≤i}(γ − M[j])` remembers what earlier steps spent, which is what makes the attention
*sequential* rather than merely sparse (γ = 1 bans a fully-used feature outright). Built from scratch and
sparsemax-validated against `pytorch_tabnet` (max |Δ| = 2.4e-07), then held to the shared frame on four
small tables, it lands **behind** the tuned simple baselines it was designed to beat: mean ranks TabNet
**2.50** vs MLP **1.75** / ResNet **2.00** (GBDT 3.75), Friedman **p = 0.127** (`labs/_verify_l043.py`).
The disciplined reading matters more than the number: a large *p* on four datasets licenses only "cannot
distinguish on this sample", never "significantly worse" — but the burden of proof sits with the *new*
model, so the bar was not cleared. The paper's own Appendix A, where TabNet ties or trails
XGBoost/CatBoost, says the same thing more quietly. The interpretability claim gets the same treatment: on
the paper's own synthetic generators the aggregate mask recovers **global** relevance convincingly (Syn2,
76.8% of mass on the true X3–X6) but only **partially** recovers **instance-wise** relevance (Syn4, 15.6%
vs 97.9% of rows favouring their own group) — so attributions are evidence to *validate*, not explanation
to *trust* (M53). One honest loose end is preserved rather than tidied away: the from-scratch model
outscored the reference end-to-end on credit_g (0.748 vs 0.694), and both hypotheses tested (training
length, LR schedule) were refuted, so the discrepancy is recorded as **unexplained**. For the thesis this
is BAR-raising with an indirect FOR: *instance-wise* selection is the single-table shadow of the real
claim — different rows genuinely need different context — and TabNet shows how expensive that is to buy
by masking columns of an already-flattened table.

L044 (NODE — differentiable oblivious trees) is the sharpest single illustration so far of *why* the thesis
cares about structure at all, precisely because it is an honest **loss** on flat tables. NODE and CatBoost
are the **same tree shape** — an ensemble of oblivious (symmetric) trees, one shared (feature, threshold)
per level (L016) — so the experiment isolates one variable: *make the tree differentiable*. Three discrete
steps are softened — **entmax15** feature choice (α = 1.5, the middle of softmax → entmax15 → sparsemax,
real zeros with a smoother gradient than L043's sparsemax), the **entmoid** soft split, and **outer-product
routing** that sends a fraction of the row to all 2^d leaves as a distribution — so hundreds of trees per
layer train by backprop and stack **DenseNet-style**. Built from scratch and validated to machine precision
against the `entmax` package (entmax15 \|Δ\| = 5.6e-16, entmoid \|Δ\| = 3.3e-16, `relkit.node`), then held
to the shared frame on four small tables, NODE lands **last**: mean ranks NODE **3.50** vs CatBoost 2.50,
MLP 2.00, ResNet 2.00 (Friedman χ² = 3.6, **p = 0.308**), beats CatBoost on **1/4**, and trains **~70×
slower** (60.2 s vs 0.9 s on credit_g; `labs/_verify_l044.py`). The disciplined reading is the same as L043:
a large *p* on four datasets licenses only "cannot distinguish on this sample", but the burden sits with the
expensive new model, and it did not clear it *here* (this is a down-scaled demonstration; the paper's small
win over GBDT is at benchmark scale with thousands of trees + heavy tuning). What makes this **FOR** the
thesis rather than merely against NODE is the diagnosis of *what* differentiability buys: a GBDT's greedy
splits have **no gradient**, so it can never co-learn embeddings, stack hierarchically so later trees split
on earlier decisions, or sit inside an end-to-end multi-modal pipeline. On one flat table that capability is
pure liability (it loses and costs 70×); it earns its keep only when the tree must **compose with more
structure** — which is exactly the relational regime. The lineage matters: this is CatBoost's symmetric tree
(Y1 L016) made to plug into deep learning, and it is the first concrete demonstration that "keep the
structure / keep the gradient" is a bet you *lose* on isolated tables and only win when the surrounding
structure exists to connect to.

L045 (TabTransformer — contextual categorical embeddings + self-supervised pre-training) trains and
pre-trains the architecture L032 previewed forward-only, and lands the same **BAR + FOR** shape as the rest
of the Q4/Y2Q1 cascade. The mechanism is the L031 static **entity embedding** promoted to a **contextual**
one: a stack of Transformer self-attention blocks re-mixes each categorical column's vector with the other
columns *in the same row*, so the same category can mean different things in different rows. Built from
scratch and validated to machine precision against torch's own kernels (`scaled_dot_product_attention`
\|Δ\| = 6.7e-16, `nn.MultiheadAttention` \|Δ\| = 1.1e-16, `relkit.tabtransformer`), with the contextual
property confirmed on a real row — a column's vector moves **0.259** under a neighbour flip **with**
attention and exactly **0** at `n_layers=0`, the ablation that *is* the L031/L032 static-embedding model.
Held to the shared frame on three categorical-rich tables × three seeds (`labs/_verify_l045.py`): contextual
edges the context-free MLP on **2/3** — a **small, within-noise** gain (mean ranks TabTransformer **2.33** vs
context-free **2.67**) — but beats **CatBoost** on **0/3** (CatBoost mean rank **1.00**, Friedman p = 0.097).
The ceiling is the paper's own limitation, and it is the thesis-relevant point: **only categoricals go
through the attention; numeric features are LayerNorm'd and concatenated, never attending to anything** —
which is exactly what FT-Transformer (L046) removes by tokenising numerics too. The **self-supervised**
half is the one genuinely new lever: RTD pre-training (corrupt categorical tokens, detect the swaps —
detector ROC-AUC ≈ 0.82) learns from **unlabeled** rows, something a GBDT structurally cannot do, and the
detector can only succeed because a swap is visible **only in context**, so the pretext sharpens the very
contextualisation the architecture adds. But the payoff is honestly **small and fragile**: +0.008 AUC at 3%
labels (all seeds positive) shrinking to +0.001 at 10%, and it **collapses to negative** under a small
unlabeled pool or an aggressive fine-tune LR (catastrophic forgetting; the fix was a gentle FT-LR of 5e-4).
For the thesis this is **BAR** (another deep single-table architecture that ties the static embedding and
loses to trees — an RDL win cannot lean on "attention is powerful" applied *within a flattened row*) with a
double **FOR**: a contextual embedding is a weighted aggregate of related vectors — the exact operation a
GNN runs, here *within a row over columns*, that RDL runs *across rows over foreign-key neighbours* — and
self-supervision on abundant unlabeled rows foreshadows the relational foundation models of Year 5. The
same lesson recurs: adding attention inside one table adds no new **structural** information, so the
untapped value stays across the join.

L046 (FT-Transformer — the Feature Tokenizer + [CLS] readout) closes the Q1 classic-neural cascade by
removing the exact ceiling L045 named. The edit is surgical: make **every** feature a token, numerics
included. A numeric feature *j* becomes the **affine** token `T_j = b_j + x_j·W_j` — the scalar placed on a
learned per-column direction `W_j` (validated from scratch, `labs/_check_l046.py`: bump `x_j` by Δ and token
*j* moves by exactly `Δ·W_j`, no other token moves), and a learned **[CLS]** token is prepended so the
Transformer pools the whole row into the vector the head reads. The probe makes the fix measurable: a numeric
change moves FT-T's [CLS] readout by **L2 ≈ 0.438** on adult but moves TabTransformer's representation
**exactly 0.0** — numerics now attend. Held to the shared frame on four tables × three seeds
(`labs/_verify_l046.py`, attention reused from the L045 kernel matched to torch at \|Δ\| ≈ 1e-16): mean ranks
FT-T **2.50** / MLP **2.75** / TabTransformer **3.75** / CatBoost **1.00** (Friedman p = 0.026). FT-T beats
TabTransformer on **3/4** (all but the most-categorical credit_g, where numerics matter least) and is the
**best single neural model** — but a tuned CatBoost still wins **all four**. For the thesis this is again
**BAR** (the strongest classic single-table neural architecture, built honestly, is still a notch below a
tree on flat data — the paper's own "no universal winner", cited: under a shared protocol FT-T ~ties tuned
GBDTs) with a clean **FOR**: FT-Transformer perfects attention *within a flattened row over columns* and it
buys the best neural rank yet *without* overtaking the tree — because adding attention inside one table adds
no new **structural** information. The untapped signal is still across the join, and Q1 has now exhausted the
single-table neural repertoire (MLP/ResNet → TabNet → NODE → TabTransformer → FT-Transformer) that a
relational model must eventually beat *fairly*.

The genuinely *supporting* evidence (C1, C2) is still conceptual — flattening is demonstrably lossy and
leakage-prone, and manual feature synthesis hints structure is recoverable, but no result yet shows a
relational model *beating the fair bar by keeping structure*. That demonstration is now the **Year 2–4**
burden (neural tabular honesty → GNNs → RelBench). Standing honestly on a high, fully-instrumented baseline
is the point: it is what will make an eventual win credible.


### L047 — BAR: define which other rows a prediction may read (2026-09-05)
From-scratch supervised SAINT stage matches the pinned official forward and input gradients on transplanted weights (both maximum differences 0 in the checked tensors). A three-substitute/three-seed ablation yields mean ranks feature-only 1.667, feature+row 1.667, budgeted CatBoost 2.667; Friedman p=0.368, Nemenyi CD=1.914. No overall difference is established; neither equivalence nor superiority over tuned trees is supported. At fixed diabetes seed-0 weights, evaluation batches 64→1 change a prediction by up to 0.2673 and AUC 0.7920→0.7648. Companion availability is part of the inference contract. This provides a methodological bridge to relational neighborhood sampling, not evidence that random mini-batch links recover foreign-key semantics. Paper Bank/Table 2 full budgets and Table 3 pretraining are not reproduced. Evidence: labs/_verify_l047_results.json and labs/_check_l047_results.json.


### L048 — BAR: explicit within-row interactions deserve a fair baseline (2026-09-05)
DCNv2 makes multiplication an explicit operation while keeping each forward pass within one embedded row. A three-substitute/three-seed comparison yields mean ranks MLP 2.333, dense cross+MLP 3.667, factored cross+MLP 1.667, CatBoost 2.333; Friedman p=.284 and Nemenyi CD=2.708 detect no overall difference, not equivalence. The full MovieLens binary-task closer run uses 739,012 rows and three seeds: log loss .355545 ± .000692, AUROC .864981 ± .000413. It remains INCOMPARABLE to Table 6 (.3170 log loss) because of protocol and configuration gaps. This raises the within-row baseline and reproduction standard; it supplies no direct evidence that relational information improves predictions. Sources: `labs/_verify_l048_results.json`, `labs/_scaleup_l048_results.json`, `labs/_sources_l048.json`. Existing contrary evidence is retained.


### L049 — BAR: a model claim includes its protocol (2026-09-05)
ExcelFormer’s selected numeric path matches released forward/input gradients exactly, but a three-task fixed-budget experiment detects no overall rank difference (Friedman p=.607; CD=1.914), and the larger author-split Pima run remains INCOMPARABLE to the published five-run/tuned benchmark. A separate timestamped MovieLens pipeline reverses its local neural/tree ranking between random and chronological test populations. These are baseline/protocol findings, not evidence of relational-model superiority. Trompt’s comparability claim and subgroup exceptions must remain distinct from the curriculum’s GBDT-surpassing shorthand. [L049](lessons/0049-excelformer-trompt.html) · [measured evidence](labs/_verify_l049_results.json).

| 052 | BAR | TabR strengthens the single-table baseline by retrieving labeled training rows. Local fixed-budget XGBoost leads on three author tasks; mean ranks MLP 2.667 / TabR-S 2.333 / XGB 1.000, Friedman p=.097, CD=1.914. Three seeds condition on fixed rows. This is no direct relational-versus-flat evidence and is INCOMPARABLE to the paper benchmark. | [Lesson](lessons/0052-tabr-retrieval.html) |


### L053 — BAR: a strong default has a development history (2026-09-07)
Numeric RealMLP-TD-S source parity passes, but the capped three-task comparison yields mean ranks TD-S 3.000 / XGB-fixed 1.667 / six-candidate XGB-tuned 1.333; Friedman p=.09697, CD=1.91362. The tree recipes lead on these task means; no general superiority or equivalence follows. Larger California TD-S, 6000 train rows, width256/256epochs, gives RMSE .53024 ± .00493 across three seeds. All remain INCOMPARABLE to the RealMLP benchmark. This raises the fair single-table baseline and dataset-level tuning standard; it is not direct relational-versus-flat evidence. [Lesson](lessons/0053-realmlp-strong-defaults.html) · [evidence](labs/_verify_l053_results.json).


### L054 — BAR: the single-table DL bar is now parameter-efficient ensembling (2026-09-09)
TabM (Gorishniy 2024) is, per the paper, the best-performing tabular DL model over 46 datasets — one MLP that imitates a deep ensemble of k submodels sharing most weights (BatchEnsemble: member i acts as Wᵢ = W ⊙ (sᵢ rᵢᵀ)), competitive with GBDT and far more efficient than attention/retrieval models. So the neural half of the honest bar an RDL win must clear is now "a strong MLP wrapped in cheap ensembling," not a plain net. Independent numeric TabM-mini built from scratch; layer identity, no-op adapter init, and member diversity verified by CHECK. Capped three-task run (width 64, k=32, 3 seeds): the **diversity mechanism reproduces on all three** (collective error < mean-individual submodel), and MLP×32 beats a single MLP on all three — ensembling works. But no arm dominates: mean ranks MLP×32 = XGB-tuned = 2.0, TabM-mini 2.67, MLP 3.33; Friedman p=.53 → **INCOMPARABLE**. A California k-sweep shows the individual-vs-collective gap widening with k yet absolute error rising — averaging cannot rescue jointly-biased narrow submodels. This raises the DL baseline standard; it is not relational-versus-flat evidence, and TabM still consumes one flat design matrix, so it does not touch the cross-table structure the thesis bets on. Carries a question to L055: what happens to these rankings under domain-aware (temporal) splits? [Lesson](lessons/0054-tabm-parameter-efficient-ensembling.html) · [evidence](labs/_verify_l054_results.json).


### L057 — BAR: compare against a selected ensemble, accept failed gains (2026-09-09)
TabArena Figure6 motivates cross-family single-table baselines. Our three-task, three-seed XGB/TabM/TabICL OOF stack improves versus the OOF-selected single on diabetes/phoneme and selects one model on blood transfusion, but never beats the lowest individual test score. Those individual test winners are oracle choices, not deployable selectors. Mean ranks and Friedman p=.0907 on three tasks cannot establish universal gains. Different data/metric/recipes/TFM handling make this INCOMPARABLE to Figure6. No relational-versus-flat claim follows; the lesson raises the baseline selection and leakage standard. [Lesson](lessons/0057-cross-family-ensembling.html) · [evidence](labs/_verify_l057_results.json).

<!-- FOUNDATION-058-070:begin -->
### L058 — BAR: Read a benchmark without inheriting its bias

No, selection biases the evaluation population. The executed local experiment or frozen-result audit sharpens the single-table baseline or information contract; it is not relational-versus-flat evidence or full paper reproduction. [Lesson](lessons/0058-surveys-meta-benchmarks.html) · [measured evidence](labs/_verify_l058_results.json) · [scope](labs/l058-reproduction.md).

### L059 — BAR: When validation becomes training

No, validation selects lucky noise; test performance remains chance in expectation. The executed local experiment or frozen-result audit sharpens the single-table baseline or information contract; it is not relational-versus-flat evidence or full paper reproduction. [Lesson](lessons/0059-validation-set-overfitting.html) · [measured evidence](labs/_verify_l059_results.json) · [scope](labs/l059-reproduction.md).

### L060 — BAR: A baseline comparison you can defend

No, the underlying task is shared; keep regimes separate. The executed local experiment or frozen-result audit sharpens the single-table baseline or information contract; it is not relational-versus-flat evidence or full paper reproduction. [Lesson](lessons/0060-broad-model-comparison.html) · [measured evidence](labs/_verify_l060_results.json) · [scope](labs/l060-reproduction.md).

### L061 — BAR: Learn Bayesian prediction across tasks

No, a learned finite-range approximation need not preserve the analytic rule outside that range. The executed local experiment or frozen-result audit sharpens the single-table baseline or information contract; it is not relational-versus-flat evidence or full paper reproduction. [Lesson](lessons/0061-prior-data-fitted-networks.html) · [measured evidence](labs/_verify_l061_results.json) · [scope](labs/l061-reproduction.md).

### L062 — BAR: Trace the original TabPFN

No, only labeled context rows supply keys in this inductive path. The executed local experiment or frozen-result audit sharpens the single-table baseline or information contract; it is not relational-versus-flat evidence or full paper reproduction. [Lesson](lessons/0062-tabpfn-v1.html) · [measured evidence](labs/_verify_l062_results.json) · [scope](labs/l062-reproduction.md).

### L063 — BAR: What a synthetic causal prior encodes

No, the synthetic prior is an assumption over tasks, not an identification result. The executed local experiment or frozen-result audit sharpens the single-table baseline or information contract; it is not relational-versus-flat evidence or full paper reproduction. [Lesson](lessons/0063-synthetic-scm-prior.html) · [measured evidence](labs/_verify_l063_results.json) · [scope](labs/l063-reproduction.md).

### L064 — BAR: Trace TabPFN v2 across both table axes

No, it checks an operator boundary, not pretrained weights, data, tuning or scores. The executed local experiment or frozen-result audit sharpens the single-table baseline or information contract; it is not relational-versus-flat evidence or full paper reproduction. [Lesson](lessons/0064-tabpfn-v2.html) · [measured evidence](labs/_verify_l064_results.json) · [scope](labs/l064-reproduction.md).

### L065 — BAR: Extract embeddings without their own labels

No, their information roles differ because one can contain its own target. The executed local experiment or frozen-result audit sharpens the single-table baseline or information contract; it is not relational-versus-flat evidence or full paper reproduction. [Lesson](lessons/0065-tabpfn-query-embeddings.html) · [measured evidence](labs/_verify_l065_results.json) · [scope](labs/l065-reproduction.md).

### L066 — BAR: How TabICL builds and uses context

No, the final dataset-level context attention retains a quadratic component. The executed local experiment or frozen-result audit sharpens the single-table baseline or information contract; it is not relational-versus-flat evidence or full paper reproduction. [Lesson](lessons/0066-tabicl-column-row-attention.html) · [measured evidence](labs/_verify_l066_results.json) · [scope](labs/l066-reproduction.md).

### L067 — BAR: Retrieve locally, then test adaptation

No, retrieval changes inputs; fine-tuning requires measured parameter updates. The executed local experiment or frozen-result audit sharpens the single-table baseline or information contract; it is not relational-versus-flat evidence or full paper reproduction. [Lesson](lessons/0067-local-pfn-retrieval-finetuning.html) · [measured evidence](labs/_verify_l067_results.json) · [scope](labs/l067-reproduction.md).

### L068 — BAR: Teach a PFN about changing mechanisms

No, its label must also be available by prediction time. The executed local experiment or frozen-result audit sharpens the single-table baseline or information contract; it is not relational-versus-flat evidence or full paper reproduction. [Lesson](lessons/0068-pfns-under-temporal-shift.html) · [measured evidence](labs/_verify_l068_results.json) · [scope](labs/l068-reproduction.md).

### L069 — BAR: Find the broken prediction contract

No, dropping them changes the evaluated population and hides coverage failures. The executed local experiment or frozen-result audit sharpens the single-table baseline or information contract; it is not relational-versus-flat evidence or full paper reproduction. [Lesson](lessons/0069-tabpfn-open-environment-failures.html) · [measured evidence](labs/_verify_l069_results.json) · [scope](labs/l069-reproduction.md).

### L070 — BAR: Defend your foundation-model baseline

No, version identity is part of the model and the missing arm must stay visible. The executed local experiment or frozen-result audit sharpens the single-table baseline or information contract; it is not relational-versus-flat evidence or full paper reproduction. [Lesson](lessons/0070-foundation-model-checkpoint.html) · [measured evidence](labs/_verify_l070_results.json) · [scope](labs/l070-reproduction.md).
<!-- FOUNDATION-058-070:end -->


### L071 — BAR: unlabeled single-table structure is a baseline resource

VIME provides a testable way to learn a flat-table encoder from unlabeled rows. The local five-arm label-efficiency curve uses paired splits and counts validation labels. Its conclusions are restricted to sklearn digits and the declared training recipe; it supplies no relational-versus-flat result. [Lesson](lessons/0071-vime-masked-tabular-ssl.html) · [Evidence](labs/_verify_l071_results.json) · [Scope](labs/l071-reproduction.md).


### L072 · BAR · Views encode task assumptions

Three-dataset frozen-probe experiments show SCARF improving mean accuracy over its random encoder, while raw-feature logistic regression has higher mean accuracy on all three datasets. This is a local representation audit, not a relational-versus-flat comparison or original-paper reproduction. SubTab contrastive operator outputs and gradients match a pinned author release; complete training parity is unestablished. [Evidence](labs/_verify_l072_results.json) · [Contract](labs/l072-reproduction.md).

- **L073 · BAR:** label-budget gains require matched supervised controls, paired row/seed accounting and validation-label costs. The local SCARF study provides no direct evidence for relational superiority.

| L074 | BAR | CARTE provides schema-variable within-row encodings; local YAGO transfer is measured on three related wine tables with scratch controls. Pretrained fine-tuning trails scratch at all three dataset means; frozen pretraining beats random on one. This does not establish foreign-key reasoning or temporal validity. Full benchmark INCOMPARABLE. See labs/_verify_l074_results.json. |

| L075 | BAR | PyTorch Frame provides the typed row-encoding interface needed before relational message passing. Local five-type and real credit_g checks establish shape, fit scope and gradient behavior, not relational advantage or predictive quality. No benchmark run; random row vectors remain interface evidence. |

| L076 | BAR | Executed real Frame encoders → explicit eligible one-hop mean → binary head, with identity, fit-scope and gradient checks. Demonstrates the integration interface, not relational predictive advantage. Historical RelBench driver-dnf replay packaged; benchmark NOT_RUN due to historical runtime prerequisites. |

| L077 | FOR / BAR | Full paired-history construction reproduced on five seeds: specified flat input has exact .5 accuracy ceiling; adding eligible order delta permits1 with the same tabular stump. Supports an information-loss mechanism. Limits the broader claim: adequate manually derived relational features solve this task; neither GNN superiority nor real-world prevalence follows. |

| L078 | BAR | Full Cora GCN release-protocol port,100 initializations:81.401% mean versus published81.5%; implementation makes learned neighbor aggregation concrete. Single static transductive graph, modern-framework deviations and no tabular baseline comparison: neither temporal safety nor relational-database superiority follows. |

| L080 | BAR | Prepared reproducible four-family random/temporal comparison and an assessed information-ceiling argument. Two capped datasets cannot establish population superiority or relational benefit; no learner pass inferred. |

| L081 | BAR | Executable learned graph aggregation and invariant molecular readout, with honest QM9 reconstruction boundaries. Mechanism tests and smoke training do not establish relational advantage, historical paper parity, or learner mastery. |

| L082 | BAR | Full Cora fixed-split GCN port,100 initializations; measured81.401% versus81.5% target. Supports reproducible static graph aggregation, not temporal RDL advantage or cross-framework identity. |

| L085 | BAR | Full karate untrained setup reconstruction and separate Cora depth sweep expose information loss from repeated mixing. Degree scaling, components, optimization and feature magnitude constrain the diagnosis. Greater relational depth alone is not evidence of better predictions; no database benchmark superiority follows. |

| L087 | BAR | Full8-graph,10-split CN/AA/RA baseline reconstruction and separate GCN decoder/ranking lab expose how graph visibility and candidate selection define the task. Static edge recovery is not temporal recommendation evidence; historical numerical parity INCOMPARABLE, full SEAL training NOT_RUN. |

| L088 | BAR | GIN makes multiset counting and the 1-WL ceiling explicit. A six-cycle and two triangles remain indistinguishable under identical initial labels; empirical graph classification cannot establish universal relational expressiveness. MUTAG execution coverage and historical protocol gaps are reported separately. |

| L089 | BAR | Training feasibility depends on which relational messages fit inside each batch. PPI teaching runs quantify retained edges, hidden-state proxies and F1, but equal passes give different update counts. Full published-scale recipe is supplied; its result is not reproduced. |

- **L090 · BAR · 2026-09-19:** Full100-seed Cora GCN reconstruction reaches81.401% mean versus81.5% reported; full protocol audit and a separate inductive extension support implementation fluency. One citation graph is not evidence of relational-database superiority. Historical framework identity remains INCOMPARABLE.

- **L091 · BAR · 2026-09-20:** R-GCN preserves foreign-key roles through relation-specific transformations. Complete AIFB reconstruction and a separate basis-sharing extension test the implementation; a static identity-based RDF classifier is not evidence of temporal relational-database superiority. Full historical identity remains INCOMPARABLE.

- **L092 · BAR · 2026-09-20:** HAN makes multi-hop relation choice explicit through meta-path endpoint graphs and two attention levels. Full ACM release reconstruction tests the implementation and its KNN probe. Attention weights alone are not causal evidence, and one static bibliographic graph does not establish temporal relational-database superiority. Paper/code discrepancies remain explicit.

- **L094 · BAR · 2026-09-20:** Taxonomy does not rank model quality. The complete NN release matches all five node counts yet differs from printed edge statistics. Data identity and matched task protocols must precede claims that relational encoders outperform alternatives. No database-superiority or full-paper reproduction claim follows.

- **L095 · BAR · 2026-09-20:** Complete ML-100K five-fold course experiment gives walk NDCG@10 .327886 versus popularity .204553. This shows useful relational structure under one declared offline protocol, not GNN superiority, temporal forecasting success or general database superiority. Published release counts and partitions match; historical model-score parity is not claimed.

- **L098 · BAR · 2026-09-20:** Native full-neighbor batching matches independent dense seed outputs/gradients across 96 synthetic configurations; temporal disjoint queries preserve distinct histories. Nine toy fits establish an executable training mechanism, not relational-database superiority or RelBench reproduction.

- **L100 · BAR · 2026-09-20:** Native heterogeneous sampling passes typed-ID, seed-only loss and fixed-weight full-neighbor gradient audits; all24 matched ACM fits execute. HGT exceeds its uniform ablation by1.70pp under this protocol, while R-GCN collapses to the majority class. One graph/split and a failed optimization trajectory cannot rank families universally. Named AIFB replay is close; full HGT CS/full-paper reproduction remains unestablished.

- **BAR · L101 (2026-09-22):** Chronological targets do not protect against late-arriving or multi-hop future inputs. Nine constructed diagnostic fits expose this; 384 SQL-oracle graph cases pass. Ten published RelBench heuristic MAE cells match, but no GNN superiority or full-paper reproduction is established.

- **BAR · L102 (2026-09-22):** Complete ten-run TGN-attn Wikipedia replay reaches 98.512% all-event and 97.829% new-node batch-mean AP. Shared temporal state must be audited independently of weights: omitting pending messages changes predictions. This single-dataset sampled-candidate experiment does not establish relational-database superiority, full-catalog retrieval quality or full-paper reproduction.

- **BAR · L104 (2026-09-26):** A causal score comparison requires fixed prediction questions and a declared information boundary. Fresh TGAT checkpoint evaluation and illegal-history interventions test that boundary; event-only data cannot certify historical feature availability. Reused training and selected Wikipedia evidence do not establish full-paper reproduction or general relational-model superiority.

- **BAR · L103 (2026-09-26):** Complete selected TGAT Wikipedia release replay: 95.282% all-event / 93.779% new-node batch-mean AP. Temporal encoding needs an independent recursive information boundary. Release defects and population differences prevent historical equivalence; the three-seed matched course slice does not establish broad TGAT/TGN superiority or relational-database advantage.

- **BAR · L106 (2026-09-26):** A full Wikipedia EdgeBank replay tests the evaluation before crediting learned relational structure. Candidate distribution changes scores with the model fixed; original-code parity and selected target closeness do not demonstrate general relational-model superiority, operational forecasting quality or full-paper reproduction.

- **BAR · L107 (2026-09-26):** 3600 AP 0.9112, 86400 AP 0.8411, tgn AP 0.9680. These matched-candidate single-dataset system comparisons include architecture, delay and update-budget differences; they do not prove broad temporal or relational superiority. The released O fixed-window recurrence has no deterministic path from earlier graph contents to the final output, showing why a recurrent architecture label is insufficient evidence of learned history. Full selected SBM replay retains source/prose and runtime deviations.

- **BAR · L108 (2026-09-26):** Exact sampling optimization gives a 7.82× local CPU sampler speedup, while changing frozen TGAT fanout from 20 to 5 gives only 1.64× measured GPU inference throughput and loses 11.57 pp all-event AP. Context budgets and inference distributions can confound relational-model comparisons. Complete selected checkpoint evaluation does not establish fresh-training, production-scale, or full-paper equivalence.

- **BAR · L109 (2026-09-26):** All 8,712 F1 driver-position labels and ten selected heuristic cells reproduce, but the archive lacks ingestion/version histories. Thirty-three validation and 42 test query rows have no prior-year result; the released activity predicate is future-permissive. Exact benchmark parity does not certify a deployment population or historical feature availability.

- **L110 · BAR · 2026-09-26:** twenty fresh full-data temporal-GNN fits and a time-travel audit. Release Wikipedia AP 98.5123% / 97.8295% is numerically CLOSE/CLOSE; strict sampling alone cannot protect a memory path that crosses timestamp ties. Complete checkpoint consistency and known availability remain prerequisites for relational evidence. This is one interaction benchmark, not database superiority or full-paper parity. [Contract](labs/l110-reproduction.md).

- **BAR · L112 (2026-09-26):** Full selected ogbn-arxiv GCN reproduction: ten fresh 500-epoch fits, test 71.8480% ± 0.3119 pp versus published 71.74% ± 0.29 pp (CLOSE); all final node classes replay exactly through original GCN. This trains on the complete transductive feature graph with a year-based label split; it does not establish point-in-time deployment validity or relational superiority on database tasks. Historical identity and full-paper parity remain NOT_ESTABLISHED.

- **BAR · L113 (2026-09-26):** valid: 91.6919% ± 0.0908pp (sample seed SD), CLOSE; test: 78.3660% ± 0.2749pp (sample seed SD), OUTSIDE_TOLERANCE. Historical identity and whole-paper parity are not established. Scaling evidence must specify both sampler and aggregation, preserve split/selection rules, and account for training AND inference. This one transductive products benchmark does not establish database superiority or deployment-time validity. [Contract](labs/l113-reproduction.md).

- **BAR · L114 (2026-09-26):** On ogbn-arxiv, reused GCN test71.8480% versus freshly reproduced MLP55.8036%, but the validation-nominated homophily<.25 rule reverses the comparison on8585test nodes:17.2801% versus27.2231%,gap−9.9429pp. Both paper mean targets are CLOSE; this new retrospective slice finding is not a published target. Graph advantage is population-dependent; hidden-label diagnostics, correlated difficulty and differing baseline BN/initialization protocols preclude a causal or serving-policy claim. [Evidence](labs/l114-reproduction.md).

- **BAR · L116 (2026-09-26):** Full selected GCN reproduction reaches test 72.0200% ±0.3056 pp, CLOSE to OGB's 71.74% mean. A paired full-data missing-step intervention preserves nonzero gradients while freezing all parameters; repairing the update lowers loss from 4.050962 to 0.948045 in 80 epochs. This shows why training integrity and label/ID boundaries must precede claims of relational advantage. Neither one citation benchmark nor diagnostic success establishes database superiority or historical whole-paper parity. [Protocol and evidence](labs/l116-reproduction.md).

- **BAR + FOR · L117 (2026-09-27):** A complete selected RelBench RDL release replay on F1 driver-position yields test MAE4.13392±.15660 versus cited4.022. The graph/model pipeline is executable and its final predictions match original-model computation within numerical tolerance. This supports feasibility on this named task; it does not reproduce the whole benchmark, prove superiority over freshly tuned engineered features, or repair absent ingestion histories and non-train-only feature statistics. [Protocol](labs/l117-reproduction.md).

- **BAR + AGAINST overgeneralization · L118 (2026-09-27):** Cvitkovic supplies an early row-to-graph pipeline; the expanded paper reports mixed dataset outcomes and a competitive DFS+GBDT Home Credit baseline. Our source-algebra and full-fold identity checks pass; a real-data T4 pilot establishes a cost bottleneck, not predictive superiority. Full Home Credit five-fold replay remains NOT_RUN under the USD10 cap. Global released feature statistics and unverified pilot-vs-Neo4j graph identity remain explicit boundaries. [Protocol](labs/l118-reproduction.md).

- **BAR · L119 (2026-09-27):** A chosen flat summary can lose information, while a richer engineered feature can restore it; ordinary message passing also fails on equal-feature cycle/triangle examples. Five fresh complete RelBench v1 F1 RDL fits achieved test MAE4.01457±.12673 (paper4.022; descriptive CLOSE). This establishes selected release execution, not a fresh matched tabular win or whole-paper parity. Preprocessing is test-censored-database fitted; ingestion histories remain unavailable. L118 Home Credit stays NOT_RUN. [Protocol](labs/l119-reproduction.md) · [Evidence](labs/evidence/l119/summary.json).

- **BAR + AGAINST overgeneralization · L121 (2026-09-27):** Relational learning predates GNNs. The original grouping example gives68/100 despite identical leaf count/sum/max, and an explicit nested engineered feature repairs the collision. Fresh Neo4j3.5.4 sample audit matches32 applicant graphs after correcting123 empty/null cells; encoded tensors are unchanged. Caching reduces preparation cost but the five-fold Home Credit projection remainsUSD15.70 before overhead, aboveUSD10cap, so full predictive reproduction is NOT_RUN. No new evidence of superiority over DFS is claimed. [Protocol](labs/l121-reproduction.md).


### L122 · Construction evidence versus predictive evidence (2026-09-27)

A complete row/key audit on F1 establishes exact topology for the released database: 74,063 nodes and 338,842 directed edges, independently checked with SQL and original code. Three student functions rebuild the full key-only topology. Five new selected released-protocol RDL runs yield test MAE4.05511±.15171 versus published4.022 (descriptive CLOSE). These results support the ability to construct and execute a reproducible RDL pipeline; they do not establish superiority over newly run feature-engineering baselines, whole-paper parity, real ingestion-time validity, or learner mastery. See labs/l122-reproduction.md.


### L123 · Temporal access is an explicit information contract (2026-09-27)

BAR: complete F1 temporal graph retained; nine neighborhoods match SQL/PyG and every training/evaluation batch passes root-time, edge-identity and query-isolation checks. Five fresh released-protocol RDL fits yield testMAE4.06491±.06990 versus published4.022, descriptive CLOSE. This supports reproducible execution, not universal point-in-time correctness: ingestion/mutable histories are absent and release preprocessing uses the test-censored database. No newly run baseline comparison or whole-paper claim. [Protocol](labs/l123-reproduction.md).

- **L126 · BAR (2026-09-27):** Benchmark claims require task, split, metric and archive identity. Beta source semantics pass independent checks; all 74,063 modern F1 rows and 760 predictions are audited. Historical beta archives remain unavailable; no beta performance reproduction or RDL-superiority claim follows.

- **L128 · BAR · 2026-09-27:** Five fresh complete driver-dnf RDL runs with reconstructed historical labels yield test AUROC71.6708±1.1760pp versus paper72.62±.27; descriptive CLOSE. All6340 predictions and605190 query occurrences audited. Changed archive/label polarity shows why task identity must be verified; historical archive/order and whole-paper identity remain unestablished. No new matched manual-FE evidence; recommendation exercises are COURSE_ONLY.


### L130 · Q1 checkpoint · 2026-09-27

**BAR / AGAINST a universal superiority claim:** five fresh full F1 RDL runs give validation3.164116±.038287/test4.070921±.075000MAE, CLOSE to the selected Table7 means under predeclared0.2 tolerance. Every final prediction and sampled query audited. Reused L129 tuned manual FE scores3.948917±.070469testMAE on identical query keys; FE minus RDL−.122004. This task-specific descriptive advantage does not compare equal search/preprocessing budgets or reproduce human effort. Full selected released pipeline complete; historical/whole-paper identity NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE. Protocol: labs/l130-reproduction.md.

- **L136 · BAR · 2026-09-27:** Exact reconstruction of27 submitted regression scores (2480739 predictions) establishes evaluation reproducibility for three entries, not controlled model superiority. Temporal input regimes, search budgets and normalization provenance remain separate audit dimensions. Five fresh full-data historical RDL fits yield test4.097695±.099605MAE, CLOSE to the selected paper target under the inherited.2 descriptive tolerance. This selected training evidence cannot identify the current GNN entry or recreate Kapso's search. Protocol: labs/l136-reproduction.md.

- **L137 · BAR · 2026-09-27:** Fresh full selected GNN/FE pipelines yield testMAE4.089695/3.948917, gap+.140779 with conditional driver-cluster95%[-.110617,.402166]. Validation-selected high-history gap attenuates on test; observed-recent-feature support vanishes. This constrains claims about stable weaknesses and architecture-only explanations, without establishing equivalence or broad FE superiority. Both use a frozen snapshot; transformations and tuning differ. [Lesson](lessons/0137-error-analysis-reg.html) · [Protocol](labs/l137-reproduction.md).

- **L138 · BAR · 2026-09-27:** All 5,470,060 Amazon user-churn labels reconstructed; fixed recency baseline test AUROC 0.582276. Full five-seed released-pipeline evidence is in [the execution ledger](labs/evidence/l138/reproduction.json). The released training table has 24,172 fewer rows than the paper, so exact historical reproduction remains unestablished. Review absence does not establish purchasing churn or intervention effects. [Lesson](lessons/0138-ecommerce-amazon.html).


### L141 · BAR — Atomic routes are implementable; historical training remains gapped

The full RelGNN F1 architecture has exact CPU operator parity and matching original-model predictions on all 7,554 primary held-out rows. Independent label reconstruction covers all 8,712 task queries. A checkpoint-compatible replay scores 3.737381 test MAE, but requires an inferred numerical type for qualifying.position. Five fresh documented reconstruction fits score 4.259311 ± .267820, outside the frozen ±.20 band around the published3.798. The historical training recipe is missing, and matching nonfinite encoder gradients limit optimization-health claims. These measurements support an executable mechanism, not a demonstrated causal advantage over the previous model or a full historical reproduction. [Evidence and deviations](labs/l141-reproduction.md). Learner defense pending.


### L144 · BAR — Contextual ranking is executable; the claimed benefit remains unreproduced

The released ContextGNN local replacement operator matches original outputs and gradients; full-graph timing pilots match original forward predictions within declared tolerance, and all733741 site-sponsor-run labels were independently reconstructed. The USD10 cap stops the complete tuning/repeat protocol: a pilot-throughput scenario projects USD28.75 even for one epoch per fit. Partial validation scores cannot establish the published ContextGNN advantage. Stale evaluation embeddings and nonfinite encoder gradients also limit source-matching evidence. [Lesson](lessons/0144-contextgnn.html) · [Evidence and deviations](labs/l144-reproduction.md). Learner defense pending.

### L145 · BAR — RelGT is executable; source agreement does not establish temporal validity

The complete source-adapted model matches original CPU fixture outputs and149parameter-gradient tensors; a real F1 batch matches within1.92e-6. All8712labels are independently reconstructed. Yet released cache semantics expose569502training and21368validation future-token occurrences. This is evidence about the pinned released pipeline, not proof that historical paper runs used the same cache. One partial pilot cannot reproduce the paper's3.9170MAE or establish a GNN advantage. Three shallow full configurations projectUSD26.80 before overhead, exceeding the USD10 cap. Full selected reproduction INCOMPLETE. [Lesson](lessons/0145-relational-graph-transformer.html) · [Evidence](labs/l145-reproduction.md). Learner defense pending.

### L146 · BAR — A comparison requires an information and selection contract

A corrected small-model F1 comparison gives the GNN lower test MAE in all three seed pairs while reduced RelGT has lower validation MAE in every pair. Test means4.293632versus4.617536,pairedGNN−RelGT−.323904±.253784(sample seed SD). Same legal contexts and full task rows make the conditional comparison auditable; unequal encodings/readout/state/compute prevent isolating attention or ranking paradigms generally. The printed RelGT table also leaves cross-configuration selection unresolved. Source full reproduction remainsINCOMPLETE. [Lesson](lessons/0146-gnn-vs-graph-transformer.html) · [Evidence and limits](labs/l146-reproduction.md). Learner defense pending.

### L147 · BAR — A research agenda needs falsifiable questions and appropriate holdouts

Freshly rescoring all7,554saved L146 predictions reproduces the validation/test ordering reversal. This strengthens traceability of that conditional result, not the number of independent experiments supporting the thesis. The survey's foundation-model agenda requires database-level evaluation that the F1 seed pairs do not provide. An explicit research map makes the required test, held-fixed quantities, falsifier and planning judgments inspectable. No new training; full RelGT reproduction still INCOMPLETE. [Lesson](lessons/0147-next-generation-architectures.html) · [Protocol](labs/l147-reproduction.md). Learner defense pending.


## L148 · Ablation discipline — BAR / AGAINST an unconditional propagation claim

Fresh full-data F1 experiment, five seeds per arm, ten epochs. Full RDL validation3.2245±.0531/test4.1931±.2428MAE, both descriptively CLOSE to the selected paper means under frozen±.2. No-message arm worsens validation by+.3990 but improves test by−.1538; restricted history improves validation by−.2940 but worsens test by+.7708. Test encoder–message interaction+.0554±.2850MAE has mixed seed signs. These conditional exploratory effects challenge an unconditional more-history/more-propagation story; they do not identify a single cause or establish cross-database effects. Post-sampling history removal leaves empty sampling slots. The initial real batch shows512nonfinite gradient entries in every arm; released numerical behavior preserved. Independent reconstruction covers8712labels and keyed rescoring31475primary predictions. See `labs/l148-reproduction.md`. Whole paper NOT_RUN; historical identity NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE.

## L149 · AGAINST an unconditional RDL advantage / BAR for causal explanations

Fresh complete F1 basic RDL versus released manual-FE pipelines: five10epoch GNN fits and five10trial LightGBM searches plus refits. TestMAE4.123071 vs3.948917; descriptive driver-cluster loss-penalty interval[−.085850,.449621] crosses zero. High-history is the validation-nominated weakness; its test interval[−.397691,.638169] also crosses zero. Low history does not explain the largest observed validation gap. Published Figure3 catalog identifies driver-top3 and item-sales as largest negative point gaps within their respective metric panels; regression uses boosted RDL, not this basic-GNN replay. Plot extraction is not original scalar/significance recovery; user-votes/post-votes identity unresolved. Evidence is conditional and exploratory; historical identity NOT_ESTABLISHED, wholepaper NOT_RUN, learner PENDING_WRITTEN_DEFENSE. [Lesson](lessons/0149-weakest-relbench-tasks.html) · [Protocol](labs/l149-reproduction.md).

## L150 · A completed reproduction can fail its performance goal

Fresh full RelGNN F1 checkpoint: reference test4.314567±0.371880, selected lr0.003 test4.079708±0.099947MAE; both outside descriptive±0.20 around published3.798. The selected configuration won three-candidate seed100validation search before five fresh seeds10–14. Reference and selected seed sets differ, so this is not a paired causal learning-rate estimate. Released replay3.737381does not repair fresh-training failure. Current nMAE normalization and protocol remain unreconciled; no current SOTA/near-SOTA claim. Gradient source agreement retains nonfinite entries, and earlier test exposure limits confirmation. A report can explain all of this well while the performance objective remains unmet. [Evidence](labs/_verify_l150_results.json) · [Report](labs/l150-report-template.md).


## L151 · A portable classification result with a bounded claim

Prepared L151: fresh full graph and13full20-epoch fits (5reference,3search,5selected); frozen rate0.0001. Reference test69.258775±0.748950AUROC points, CLOSE; selected test68.457012±1.350223. Same chosen rate as reference: no tuning-gain claim. All13779labels and20730predictions independently checked; 3388770query occurrences, zero future-timestamp violations. Historical identity/feature-arrival legality NOT_ESTABLISHED; wholepaper/freshFE NOT_RUN; learner PENDING_WRITTEN_DEFENSE. The search returned the reference learning rate; separate seed sets cannot establish a tuning improvement. Published raw-entity LightGBM70.09±1.41is cited counter-evidence, not fresh manual-FE evidence. This entry supports reproducible execution, while preserving the missing comparator and historical availability burden. [Protocol](labs/l151-reproduction.md).

- **L152 · BAR · 2026-10-01:** Five fresh full-data ten-epoch fits: validation 3.180178 ± 0.049613, test 3.970840 ± 0.162612 MAE; descriptive CLOSE against Table7 test4.022±.20. All6295predictions independently rescored, all8712labels reconstructed against released SQL/archive,403895query occurrences audited. First backward retains640nonfinite gradient entries per seed. Median diagnostics use validation-fixed bins with counts/ties; no correction or test-led retuning. Historical identity/feature-arrival legality NOT_ESTABLISHED; whole paper/freshFE NOT_RUN; learner PENDING_WRITTEN_DEFENSE. Seed0 bins2/3 have empirical median violations.207/.204 with116/54queries; a close aggregate MAE is not a calibration guarantee or evidence of beating fresh manual FE.

## L153 · A ranking entry can expose an execution limit

The source-pinned GraphSAGE/site-sponsor-run pipeline reconstructs733741labels and independently rescores37003pilot validation rankings, but the five-seed20epoch procedure projectsUSD41.23compute beforeoverhead against aUSD10lesson cap. Full reproductionINCOMPLETE; testNOT_RUN. This is evidence about protocol implementation and bounded feasibility, not recommendation superiority or a third completed baseline. The source also retains384nonfinite gradient entries and356observed positive-negative collisions in8388608shared training comparisons; cutoff-safe sampling does not prove complete historical availability or healthy optimization. [Protocol](labs/l153-reproduction.md) · [Evidence](labs/_verify_l153_results.json).

## L155 · BAR · quality and effort require different observations

Fresh complete F1 FE/GNN comparison: FE test3.948917±.070469 versus basicRDL4.013141±.208250MAE. Signed RDL benefit−.064225, driver-cluster conditional95%[−.312329,+.172961]. Point estimate favors FE; no decisive superiority or equivalence claim. All12590held-out predictions independently rescored. One matched task does not establish a portfolio advantage. Human effortNOT_OBSERVED; original human trialNOT_RUN; machine runtime is not human effort. Figure3 boosted regression not reproduced. Source first-backward640nonfinite entries per seed and feature-arrival uncertainty retained. [Protocol](labs/l155-reproduction.md).

| L156 | BAR | All8,712F1labels,443,552SQLvalues and12,590held-out predictions audited. Released dated preprocessing violates a declared2005fit horizon while matching source; corrected test MAE4.200381vs4.128265. Static/schedule availability remains NOT_ESTABLISHED; no leak-free or superiority claim. | labs/evidence/l156/report.md |

## L157 · BAR · reproducibility is part of a contribution's value

Fresh standalone F1 runs reproduce the released selected protocol and a fixed-2005 intervention: test MAE4.015191±.150212 versus4.073480±.250688, descriptive difference+.058289. All8712labels,443552SQLvalues and12590held-out predictions independently checked. The public claim must remain scoped: computational reproduction does not establish historical arrival legality, modern-upstream defect, all-paper reproduction or superiority. Source numeric-gradient limitations remain. Local package and draft are reviewable; public contribution PENDING_PUBLICATION. [Lesson](lessons/0157-open-source-contribution.html) · [Package](labs/releases/l157-f1-audit.zip).

## L158 · BAR · Synthesis does not multiply evidence
The CPU replay rescores98,918 saved rows with45 selection checks across220 frozen inputs. Two completed tasks on two databases and one matched FE task support computational traceability, not broad superiority. L155 benefit−0.064225MAE has conditional driver95%[−0.312329,+0.172961]; human effort remains NOT_OBSERVED. Repeated F1 runs/reports add no new database. Temporal policy and optimization limitations persist; graph-construction157b is absent. [Report](labs/evidence/l158/report.md). Author preparation; learner PENDING_WRITTEN_DEFENSE.

| 2026-10-01 · L159 | BAR | Masked reconstruction provides a plausible reusable-representation objective, but neither the paper's initial single-table results nor our synthetic binary mechanism establishes unseen-database transfer. Published wikiTables target NOT_RUN: exact source/protocol artifacts not located. USD0 cloud; learner defense pending. | [Protocol](labs/l159-reproduction.md) |

## L160 · BAR · A reproducible packet can still be an incomplete portfolio
The complete selected CPU replay checks98,918 saved prediction rows and45 validation decisions. Three declared tasks contain two complete experiments and one matched FE comparison; human effort and temporal sign-off remain missing. The Year4 exit is INCOMPLETE despite replay PASS. A negative result can satisfy an experiment requirement; a missing experiment cannot. [Report](labs/evidence/l160/report.md). Learner PENDING_WRITTEN_DEFENSE.

## L161 · BAR · Reuse needs a scoped transfer test
Broad pretraining and multi-task adaptation define an ambition worth testing, but frozen weights, many parameters or a clean design checklist do not establish relational advantage. The complete15-case synthetic metadata audit supplies no empirical transfer result. Require canonical database provenance, legal target context, matched baselines and an inspected held-out comparison. [Lesson](lessons/0161-what-is-a-foundation-model.html) · [Protocol](labs/l161-reproduction.md). TrainingNOT_RUN; transferNOT_ESTABLISHED; learnerPENDING_WRITTEN_DEFENSE.

## L162 · BAR · Accessible context is not established transfer
Row-wise language encoding plus graph paths supplies a plausible mechanism for combining semantics and relational context. The complete synthetic 16 graph + 8 token + 64 declaration audit verifies information access and arithmetic, not model performance. The historical single-table reconstruction results cannot establish transfer to unseen multi-table databases; exact historical reproduction remains source-gated. [Lesson](lessons/0162-the-relational-fm-vision.html) · [Protocol](labs/l162-reproduction.md). Learner PENDING_WRITTEN_DEFENSE.

## L163 · measured row encoder boundary

Complete synthetic numeric/additive comparison: typed baseline MAE1.714±0.151 versus frozen pooled BART14.265±1.285 across three overlapping splits. BART schema perturbations use fixed baseline heads; renamed MAE705.122. This task is deliberately favourable to numeric/one-hot features. No general model ranking, natural-language semantic gain or relational transfer is established. Historical Vogel Table1 NOT_RUN; fidelity NOT_ESTABLISHED. [Protocol](labs/l163-reproduction.md).

| 2026-10-01 | L164 Griffin | BAR | Visible model matches released outputs/gradients; a real512-wide checkpoint and complete F1input audit support implementation readiness. The20fit Others-2→F1 transfer comparison remains INCOMPLETE_BUDGET_GATE after a timed probe; no local transfer-gain claim. General index-based few-shot temporal weakness retained; F1static-root finding is narrower. | `labs/l164-reproduction.md`; `labs/evidence/l164/report.md` |

## Lesson 165 · proprietary relational ICL comparator

KumoRFM v1 provides an author-reported comparator, not a performance ceiling or independently reproduced result. Selected Table2 rel-f1/driver-dnf in-context82.41AUROC remains NOT_RUN, historical fidelity NOT_ESTABLISHED. Original report bytes are pinned; modern service identity cannot substitute for v1 weights/data/context policy. The complete local144context/96graph/24scoring audit and two query-label interventions establish only explicit synthetic contracts. No new evidence of relational superiority, transfer or practical effort reduction; USD0 cloud/API. [Protocol and gaps](labs/l165-reproduction.md). Learner PENDING_WRITTEN_DEFENSE.

## Lesson 166 · synthetic relational prior evidence

The complete Table9 driver-dnf512-context checkpoint replay matches the rounded paper means for RDBPFN (.721938), its single-table initialization (.663990), and TabICLv1.1 (.717568). All three receive identical DFS features and paired support keys across ten seeds. The relational checkpoint gains .057948 over initialization but also received additional training; this is not a matched-compute causal isolation of the prior. Its .004369 mean gain over TabICL has paired SD .019808 on this one task. Do not extrapolate to databases or effort savings. Released label orientation differs from current raw SQL. Fresh pretraining/wholepaper NOT_RUN; historical identity NOT_ESTABLISHED. [Evidence and limits](labs/l166-reproduction.md).

## Lesson 168 · second-database transfer evidence

Fresh trial inference adds a second selected database to the reused F1 result. RDBPFN−TabICL mean AUROC gains are .005735353 (trial) and .004369297 (F1), with 6/10 positive support draws on each. Equal-database gain .005052325 describes these two tasks; it does not estimate broad database superiority. Trial's relational-versus-single-table checkpoint gain is only .002454809 and the checkpoints have different training histories. Matching preparation/supports improves attribution, but does not isolate relational pretraining at equal compute. Reported synthetic predictor training coexists with real-schema training of LayerDAG; exact target-schema exclusion and full checkpoint lineage remain unestablished. [Lesson](lessons/0168-cross-database-generalization.html) · [Protocol and evidence](labs/l168-reproduction.md).

## Lesson 169 · context scaling is conditional evidence

The selected two-task sweep supplies all five published context sizes and ten seeds for RDB-PFN, its single-table ablation and TabICLv1.1. All 300 evaluations are accounted for (240 fresh,60 reused); every mean is within the predeclared .02AUROC descriptive distance from the paper. RDB-PFN gains from64→1024 on both tasks, but intermediate reversals and comparator curves reject an unqualified “more context always helps” claim. The released sampler is not nested across sizes. Model/data/schema pretraining axes were not varied, so no pretraining scaling law follows. TabICL exact-repeatability diagnostics show small numerical differences, explicitly retained. [Lesson](lessons/0169-scaling-laws-open-questions.html) · [Evidence](labs/evidence/l169/report.md) · [Protocol](labs/l169-reproduction.md). Historical lineage/availability NOT_ESTABLISHED; fresh pretraining/whole paper NOT_RUN; learner PENDING_WRITTEN_DEFENSE.

- **L170 · BAR · FM design checkpoint:** Complete CPU replay of all300 selected evaluations /229,050 probabilities supports traceable comparisons, not a matched three-paradigm victory. RDB-PFN−TabICL signs change with task/context; two selected tasks and support-draw SD cannot establish general superiority. Griffin budget stop, TabICL exact-repeatability failure, RDBLearn pipeline NOT_RUN and DFS/temporal/lineage gaps remain. Written design and learner mastery PENDING_WRITTEN_DEFENSE. [Protocol](labs/l170-reproduction.md).

- **L171 · BAR · corpus evidence:** The seven-source inventory and full F1 archive audit make inclusion decisions inspectable:97,606rows,13FKcolumns,227,716non-null references with no dangling keys. This establishes snapshot integrity, not unseen-source transfer or historical availability. Three tables/1,145rows have no time column; other six row audits and pretraining NOT_RUN. Declared-lineage holdouts cannot certify unknown ancestry. [Protocol](labs/l171-reproduction.md).

### L172 — typed input contracts, 2026-10-02

Full F1 tokenization audit covers97,606rows /67columns /866,746cells and preserves227,716foreign-key references. Independently reconstructed scalar payloads; tested fit isolation, column reorder and masking. This supports deterministic input engineering under declared policies. It provides no learned-transfer or model-performance evidence. Untimed availability unresolved; fresh pretraining/wholepaper NOT_RUN. Local categories are not shared semantic coordinates; a subsequent model needs an explicit embedding, objective and query-access contract. Learner PENDING_WRITTEN_DEFENSE.

## Lesson 173 · objective design, not transfer evidence

The six-fit F1 course experiment trains a shared encoder across21masked-cell tasks. Both loss policies use the same inputs, initializations and batches; all715,560selected-checkpoint test predictions are independently rescored. Equal-task weighting improves the mean test macro loss in this small comparison, with one of three seeds reversing direction. Every trained run has higher test macro loss than the constant baseline; this schedule does not establish aggregate predictive benefit. This does not establish broad superiority, forecasting performance, cross-database generalization, RT parity or KumoRFM parity. Autocomplete can exploit same-row outcome proxies. [Protocol](labs/l173-reproduction.md) · [Per-task evidence](labs/evidence/l173/report.md). Whole-paper reproduction NOT_RUN; transfer NOT_ESTABLISHED; learner PENDING_WRITTEN_DEFENSE.

## Lesson 174 · adaptation is not evidence that pretraining helped

The complete fixed-schedule F1 comparison has twelve fresh fits across freeze/full/adapter/scratch and three paired seeds. Scratch has lower test macro loss in every seed than every pretrained adaptation arm; its mean is 1.158085 versus 1.247579 for full fine-tuning and 1.175455 for the constant baseline. Adapter tuning improves over leaving inherited weights unchanged, but does not beat the constant control. This distinguishes adaptation gain from a pretraining benefit. These are exposed same-task autocomplete data with fixed shared preprocessing and untuned per-arm hyperparameters, not a foundation-model transfer result. [Protocol](labs/l174-reproduction.md) · [All-task evidence](labs/evidence/l174/runs/report.md). Whole-paper reproduction NOT_RUN; learner PENDING_WRITTEN_DEFENSE.

### L175: zero-shot claims need an input-access audit

Complete RT-v1 F1 sampler audit:2,106contexts,385future-dated schedule cells in77contexts. Strict event-time filter fails; historical schedule availability is unknown, so this does not prove future-outcome leakage. Six approved checkpoint evaluations remain NOT_RUN; there is no new zero-shot AUROC or evidence of relational-FM superiority. Target validation and context labels are separate from gradient training. [Contract](labs/l175-reproduction.md).

| 2026-10-02 | L181 | BAR | Complete F1 autocomplete aggregation baselines reproduce all40 rounded paper MAE/R² entries over68,925 predictions. This establishes task/baseline fidelity, not relational-model superiority. Fresh GNN stopped at real missing-value numerical encoder gradients under current CPU dependencies; no repair or paid fit. RelGT-AC validation and mask-policy comparisons require matched-information/source audits. |

### L185 · Decision claim boundary · 2026-10-02

**BAR:** a complete five-seed synthetic counterexample gives badge-only AUROC0.898645±0.009651 but exactly zero badge intervention effect. Observed action association0.730164 becomes0.048060 after demand adjustment; simulator action ATE0.05. Legal joins and held-out ranking do not establish a deployable causal strategy. This illustrates a logical boundary, not evidence against RDL predictive value or real-world policy efficacy. [Evidence](labs/evidence/l185/report.json). Learner PENDING_WRITTEN_DEFENSE.

| L184 GelGT | BAR | Original source loses cutoff identity in context caching; executable temporal counterexample. All 8,712 F1 labels reconstructed and attention arithmetic checked, but selected reproduction INCOMPLETE_SOURCE_TEMPORAL_GATE, fresh training NOT_RUN. No evidence for downstream gain from this lesson. [Protocol](labs/l184-reproduction.md) |

- **L182 · FOR (bounded) / BAR:** Fresh full selected RDB-PFN Table 9 F1 comparison completed, 30 evaluations and 21,060 predictions; all three means match rounded paper targets. Relational PFN exceeds its single-table checkpoint in all ten support draws; its TabICL difference is mixed (6 positive / 4 negative). Same test population as L166, not independent dataset replication. Composite-message specialization and derivative checks pass, but a prior×composite-encoder hybrid and four-arm interaction remain NOT_RUN. Released label orientation, DFS provenance, historical availability, prior training-health stops and learner defense remain explicit limits. [Protocol](labs/l182-reproduction.md).


### L187 · Useful relationships expand privacy obligations · 2026-10-02

**BAR:** complete F1 audit covers 857 drivers, 70,876 declared owned records and 175,933 incident edges. The largest contribution has 1,096 records; a profile-node deletion leaves 1,095 child records. Shared constructor aggregates retain 24,501 driver/aggregate links, which are not proven attacks. Complete 270-release public simulation demonstrates clipping/noise tradeoffs; it establishes neither production DP nor GNN privacy or superiority. The ideal bounded-histogram proof is limited to the declared owner adjacency and fixed public domain. [Evidence](labs/evidence/l187/report.json). Learner PENDING_WRITTEN_DEFENSE.


### L186 · Predictive evidence needs a serving contract · 2026-10-02

**BAR:** complete81-cell course simulation/810000responses demonstrates distinct latency and source-age constraints under explicitly hypothetical service times. At100requests/s the15ms single worker overloads; a1ms precomputed lookup can still return old dependency snapshots. This is no empirical hardware/model speed ranking. Separate300original batch-receipt replay confirms enclosing timer accounting but cannot establish request p99. No real serving benchmark, fresh inference or fallback-quality evaluation. [Evidence](labs/evidence/l186/report.json). Learner PENDING_WRITTEN_DEFENSE.

| 2026-10-02 | L183 | BAR | All7554saved L146 predictions independently replayed: reduced RelGT wins validation, course GNN wins test in allthreepairs. Neither arm has pretraining; no transfer interaction follows. Original RelGT temporal/budget and Griffin budget gates retained. Four-arm hybrid probe is a falsifiable unrun proposal; broad novelty and transfer superiority NOT_ESTABLISHED. |

### L189 · Ranked research decisions, not validated contributions

The approved source/report audit and complete 27-setting rubric replay rank temporal comparison sensitivity, composite structure in an ICL predictor, and temporal-objective transfer under database holdout. The ranking uses authored impact/feasibility judgments; it does not measure scientific value. All proposed full experiments remain NOT_RUN and their complete costs unverified. Selected current primary texts narrow the gaps, especially temporal pretraining §4.4, but novelty and complete literature coverage remain NOT_ESTABLISHED. Source reports are authenticated without rerunning their predictions. The shortlist is input to L190’s research-gap document; it does not complete that checkpoint or establish learner mastery. See `labs/evidence/l189/shortlist.md` and `labs/l189-reproduction.md`.

- **BAR · L192 (2026-10-02):** an open implementation is not automatically an admissible comparison. Original RDBLearn preprocessing fails a synthetic support/query representation invariant; clean original-source execution and100independent generated cases confirm.13779task rows authenticate and their annual window-end schedule passes, but full model search remains INCOMPLETE_SOURCE_PREPROCESSING_GATE. No model score, benchmark harm, historical-paper failure or repaired-run superiority inferred.USD0cloud/API; learnerPENDING_WRITTEN_DEFENSE.

### L193 · full-task coverage before model claims (2026-10-02)

The approved RDBLearn full21task reproduction stopped at a freshly verified shared preprocessing invariant: new query categories can renumber fitted support codes. No benchmark model inference ran; this is not evidence that RDBLearn loses on any task. All567validation and63selected-test slots remain visible and NOT_RUN. The lesson can substantiate source behavior, published table arithmetic and missingness-safe reporting. It cannot substantiate fresh accuracy, database transfer, regression aggregate parity or whole-paper historical reproduction. A repaired pipeline is a distinct experiment. See [L193 ledger](labs/l193-reproduction.md).

- **BAR · L194:** complete 21-task report preserves0/21 fresh results. Published RDBLearn versus AutoGluon+DFS signs 17/3/1 are descriptive; both pipelines use relational features, so this does not identify the benefit of relational structure. Original-source preprocessing failure remains a reproduction barrier, not evidence of benchmark inferiority. [Report](labs/evidence/l194/report.md).

### L197 evidence-ledger addition — 2026-10-02

**BAR / AGAINST broad claims:** Complete selected evidence reconstruction/replay reuses L191/L195 and does not add independent model replication. The strongest open-code comparator depends on the declared pool; L149's conditional interval crosses zero; L182 is one task with support-draw variation; L194 retains all 21 null fresh scores. Preserve C1–C4 separately. No causal architecture attribution or economic undervaluation established. [Landscape essay](lessons/0197-year-5-essay.html) gives an author defense and explicit revision conditions, not learner mastery.

- **2026-10-02 · L198 · BAR:** Complete proposal/evidence audit reconstructs all27 original priority settings and full L197 evidence (491 published numeric cells,33,650 saved predictions,21-task missing-result inventory). Three ranked candidates now expose full matrices, useful-effect criteria, controls and execution gaps. This narrows testable research rather than supplying new support for economic undervaluation. Default temporal/composite/transfer scores18.67/15/10 are authored judgments; temporal impact3 reverses the first two. All future models NOT_RUN, novelty NOT_ESTABLISHED; RDBLearn reproduction INCOMPLETE_SOURCE_PREPROCESSING_GATE. Reused evidence is not independent replication. Learner PENDING_WRITTEN_DEFENSE.

- L200 · FOR/BAR: fresh full selected RDB-PFN checkpoint run,30evaluations/21,060predictions; meanAUROC .721938 vs TabICL .717568,positive6/10paired draws. Supports selected numerical result,not broad superiority,independent replication or historical availability. RDBLearn source gate and learner proposal/defense remain unresolved.
