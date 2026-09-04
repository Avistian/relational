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

## Current verdict (updated 2026-07-29, after L040 / Year 1 exit exam)

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
