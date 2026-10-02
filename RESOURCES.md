# Relational Deep Learning Resources

Organized by curriculum year. Full sequencing in [CURRICULUM.md](./CURRICULUM.md).

## Knowledge — Year 1 (Tabular foundations)

- [XGBoost — Chen & Guestrin, KDD 2016](https://arxiv.org/abs/1603.02754)
  Canonical gradient boosting. Use for: the baseline every RDL result must beat fairly.
- [LightGBM — Ke et al., NeurIPS 2017](https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
  Histogram + leaf-wise growth + GOSS (§3) + EFB (§4). **No arXiv** — use the NeurIPS PDF; the official [LightGBM "Features" docs](https://lightgbm.readthedocs.io/en/latest/Features.html) are the best reference for leaf-wise growth and histogram subtraction. Use for: Lesson 015 primary reading; the fast/scalable GBDT baseline. The famous "20× faster" is vs *conventional* pre-histogram GBDT, not modern XGBoost-`hist`.
- [CatBoost — Prokhorenkova et al., NeurIPS 2018](https://arxiv.org/abs/1706.09516)
  Ordered boosting + categoricals. Use for: high-cardinality categoricals without target leakage.
- [Random Search for Hyper-Parameter Optimization — Bergstra & Bengio, JMLR 2012](https://www.jmlr.org/papers/v13/bergstra12a.html)
  Low effective dimensionality: only a few hyperparameters matter, so at an equal budget random search covers the important axes far better than a grid (Fig 1). Use for: Lesson 017 primary reading; the fair-budget tuning baseline. Reference implementation: sklearn [`RandomizedSearchCV`](https://scikit-learn.org/stable/modules/grid_search.html) + `HalvingRandomSearchCV` (successive halving).
- [Stacked Generalization — Wolpert, Neural Networks 1992](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231)
  A level-1 generalizer trained on the level-0 models' predictions of *held-out* data (out-of-fold). Use for: Lesson 018 primary reading (skim §1–3); the leak-free stacked ensemble that is the real single-table baseline the RDL thesis must beat. Reference implementation: sklearn [`StackingClassifier` / `VotingClassifier`](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization).
- [Random Forests — Breiman, Machine Learning 2001](https://doi.org/10.1023/A:1010933404324)
  Bagging + per-split feature subsampling to decorrelate trees; OOB error. Use for: Lesson 012 primary reading; the variance-reduction baseline (contrast with boosting's bias reduction).
- [Greedy Function Approximation: A Gradient Boosting Machine — Friedman, Annals of Statistics 2001](https://doi.org/10.1214/aos/1013203451)
  Boosting as stagewise additive modelling / gradient descent in function space; fit the negative gradient (residual for squared error), shrinkage as regularization. Use for: Lesson 013 primary reading; the bias-reduction mechanism behind XGBoost/LightGBM.
- [Why trees beat DL on tabular — Grinsztajn et al., NeurIPS 2022](https://arxiv.org/abs/2207.08815)
  Three inductive biases (irregular targets, uninformative features, rotation non-invariance). Use for: Lesson 010 Q1-checkpoint primary reading; **Lesson 039 Year 1 synthesis primary reading** (re-read full paper with the essay skeleton in hand — each section earns a paragraph and forces a boundary); **Lesson 040 Year 1 exit exam primary reading** (exit re-read of abstract + §5 — regenerable XGB bar + written biases; TIE/EXPLAIN are full passes). Explains why a competently-built GBDT is the baseline the thesis must beat — and why that result is conditional, not universal.
- [Designing Machine Learning Systems — Huyen](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)
  Evaluation, leakage, deployment. Use for: year-1 evaluation lectures.
- [A Few Useful Things to Know about Machine Learning — Domingos, CACM 2012 (PDF)](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)
  Twelve folk-truths of applied ML. Use for: Lesson 033 primary reading — the sections *"Feature engineering is the key,"* *"More data beats a cleverer algorithm,"* and *"Overfitting has many faces,"* which together give the budget-allocation / when-to-stop rule (stop when the marginal held-out gain sinks into the CV noise band). The thesis reframe: manual single-table FE returns diminish fast — the value moved *across the join*, where RDL learns the relational aggregates DFS builds by hand.
- [Fact Tables and Dimension Tables — Kimball Group (2003)](https://www.kimballgroup.com/2003/01/fact-tables-and-dimension-tables/) · [Dimensional Modeling Techniques (summary PDF)](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/)
  Ralph Kimball & Margy Ross' authoritative, free summaries of the star schema: fact/event vs dimension tables, PK/FK links, grain. Use for: Lesson 034 primary reading — the vocabulary for reading any relational schema before you flatten it (fact / dimension / grain / PK / FK). Fuller reference: *The Data Warehouse Toolkit*, 3rd ed. (Kimball & Ross, Wiley 2013), Ch. 2. The thesis context: this is the single-table paradigm — the hand-written join + point-in-time aggregation whose cost L035 (Fey 2024 §2) quantifies and RDL replaces by learning over the PK/FK graph.
- [Improving Reproducibility in Machine Learning Research — Pineau et al., JMLR 22(164) 2021](https://jmlr.org/papers/v22/20-303.html)
  The NeurIPS 2019 Reproducibility Program, reported by the people who ran it: a mandatory **ML Reproducibility Checklist** at submission, a voluntary code-submission policy (uptake rose ~50 % → 75 % of accepted papers; the authors are explicit that the correlation with higher reviewer scores is *not* causal), and the community Reproducibility Challenge. Use for: **Lesson 037 primary reading** — §5 and the checklist in Appendix Fig. 8. Read the checklist items as *things a run manifest emits*, not things you promise: computing infrastructure, exact number of evaluation runs, the specific measure or statistic (the estimator, not just the metric name), central tendency **and** variation, and every hyper-parameter with how it was chosen. Companions: [Raff, *A Step Toward Quantifying Independently Reproducible ML Research*](https://arxiv.org/abs/1909.06674) (NeurIPS 2019) — 255 papers hand-implemented without the authors' code, **63.5 %** reproduced, the base rate for the hardest rung; and [Sculley et al., *Hidden Technical Debt in ML Systems*](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) (NeurIPS 2015) §6, two pages on **configuration debt** and the origin of the one-diffable-`baseline.yaml` argument. Terminology: [ACM Artifact Review and Badging v1.1](https://www.acm.org/publications/policies/artifact-review-and-badging-current) — which **swapped** *reproducible* and *replicable* in 2020, so name the artifacts rather than the noun.
- [Avoiding common machine learning pitfalls — Lones, Patterns 5(10):101046, 2024](https://doi.org/10.1016/j.patter.2024.101046) · [continuously-updated preprint (arXiv:2108.02497)](https://arxiv.org/abs/2108.02497)
  A "dos and don'ts" tour of the whole ML lifecycle in five stages; the middle three *are* the peer-review axes. Use for: **Lesson 038 primary reading** — read "How to robustly evaluate models" (leakage, **sequential overfitting**/over-hyping, choose metrics carefully, evaluate multiple times), "How to compare models fairly" (do not assume a bigger number means a better model, **meaningful baselines** freshly implemented and **equally tuned** = HP budget parity, statistical tests, **correct for multiple comparisons**), and "How to report your results" (be transparent, report performance multiple ways, significance **vs** effect size, use an ML checklist). Map each "do not" onto a course finding: the 0.0032-nat winner's curse (L036), the corrected test (L023), the ECE estimator ambiguity (L037). Companions (checklists the deliverable descends from): [Kapoor & Narayanan, *Leakage and the reproducibility crisis in ML-based science*, Patterns 2023](https://doi.org/10.1016/j.patter.2023.100804) (the 8-type taxonomy + model info sheet, already used in L022) and [REFORMS — Kapoor, Cantrell, … Lones, Malik et al., Science Advances 2024](https://doi.org/10.1126/sciadv.adk3452) (consensus reporting checklist for ML-based science).
- [On Over-fitting in Model Selection… — Cawley & Talbot, JMLR 2010](https://jmlr.org/papers/v11/cawley10a.html)
  Selection bias from tuning + scoring on the same data; argues for nested ("double") CV. Use for: Lesson 004 primary reading and the citation behind every honest tuned baseline.
- [scikit-learn — Cross-validation iterators for grouped data (§3.1.2.4)](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)
  `GroupKFold` / `StratifiedGroupKFold` API and rationale. Use for: Lesson 004 grouped splits; the non-i.i.d. fix that generalizes to temporal splits (Y4).
- [Flexible Imputation of Missing Data (2nd ed.) — Stef van Buuren](https://stefvanbuuren.name/fimd/) (free online)
  §1.2 + §2.2.4 define MCAR/MAR/MNAR by what the missingness probability depends on (nothing / observed / unobserved), with the weighing-scale examples. Use for: Lesson 006 primary reading and the canonical citation for "I named my missingness mechanism."
- [Inference and Missing Data — Rubin, Biometrika 1976](https://doi.org/10.1093/biomet/63.3.581)
  Origin of the MCAR/MAR/MNAR taxonomy and the "ignorability" idea. Use for: the foundational citation behind every missing-data method.
- [scikit-learn — Imputation of missing values (§6.4)](https://scikit-learn.org/stable/modules/impute.html)
  `SimpleImputer`, `add_indicator` / `MissingIndicator`, and `IterativeImputer` (behind `from sklearn.experimental import enable_iterative_imputer`). Use for: Lesson 006 lab; imputers refit inside the Pipeline from Lesson 005.
- [Learning from Imbalanced Data — He & Garcia, IEEE TKDE 2009](https://doi.org/10.1109/TKDE.2008.239)
  The canonical survey of the imbalanced-learning problem: why accuracy misleads, the sampling/cost-sensitive/threshold families, and assessment metrics. Use for: Lesson 007 primary reading; the citation behind "accuracy is the wrong metric here."
- [SMOTE — Chawla et al., JAIR 2002](https://www.jair.org/index.php/jair/article/view/10302)
  Synthetic Minority Over-sampling. Use for: the most-cited resampling method; the thing you must keep inside the CV fold.
- [imbalanced-learn — Common pitfalls and recommended practices](https://imbalanced-learn.org/stable/common_pitfalls.html)
  Why resampling the whole dataset before CV leaks and tests on an unrealistic distribution; the `imblearn.pipeline` fix that resamples train folds only. Use for: Lesson 007 lab; the leak-free pattern that extends Lesson 005's Pipeline discipline to steps that also rewrite `y`.
- [The Precision-Recall Plot Is More Informative than the ROC Plot… — Saito & Rehmsmeier, PLOS ONE 2015](https://doi.org/10.1371/journal.pone.0118432)
  Why ROC is deceptively optimistic under imbalance (its FPR denominator is true-negative-rich) while PR reflects deployment performance. Use for: Lesson 008 primary reading; the citation for "report PR/MAP on rare-positive tasks."
- [The Relationship Between Precision-Recall and ROC Curves — Davis & Goadrich, ICML 2006](https://doi.org/10.1145/1143844.1143874)
  A curve dominates in ROC iff it dominates in PR; AUROC and AUPRC can disagree on rankings. Use for: the formal relationship behind Lesson 008's curve choice.
- [Predicting Good Probabilities With Supervised Learning — Niculescu-Mizil & Caruana, ICML 2005](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)
  Reliability diagrams; max-margin methods (boosted trees, SVMs) give sigmoid-distorted scores; Platt scaling & isotonic regression fix them. Use for: Lesson 008 calibration section; the canonical calibration reference.
- [scikit-learn — Probability calibration (§1.16)](https://scikit-learn.org/stable/modules/calibration.html)
  `CalibratedClassifierCV` (sigmoid/isotonic), `calibration_curve`, Brier score. Use for: Lesson 008 lab; the API for checking and fixing calibration without leakage.
- [Deep Feature Synthesis: Towards Automating Data Science Endeavors — Kanter & Veeramachaneni, DSAA 2015](https://www.maxkanter.com/papers/DSAA_DSM_2015.pdf)
  The algorithm that automatically generates features for relational data by following foreign keys to a base table and stacking aggregation/transform primitives ("depth"); beat 615/906 human teams. Use for: Lesson 009 primary reading; the direct conceptual bridge from manual relational FE → DFS → RDL.
- [An Empirical Analysis of Feature Engineering for Predictive Modeling — Heaton, IEEE SoutheastCon 2016](https://arxiv.org/abs/1701.07852)
  Tests which engineered features NN/RF/GBDT/SVM can synthesize alone: simple single-column transforms yes, but **none** could learn a ratio of differences. Use for: Lesson 009; the citation for "engineer the features the model can't learn itself."
- [Featuretools — Deep Feature Synthesis documentation](https://docs.featuretools.com/en/stable/getting_started/afe.html)
  The open-source implementation of DFS (EntitySets, primitives, `max_depth`). Use for: a concrete tool to compare a hand-engineered baseline against later in the thesis.
- [scikit-learn — Preprocessing data (§6.3) incl. TargetEncoder](https://scikit-learn.org/stable/modules/preprocessing.html)
  Stateless vs fit-bearing transforms; `TargetEncoder` with built-in cross-fitting. Use for: Lesson 009 lab; keeping target encoding leak-free inside the Pipeline.
- [scikit-learn — Pipelines & composite estimators (§6.1)](https://scikit-learn.org/stable/modules/compose.html)
  `Pipeline`, `ColumnTransformer`, `FunctionTransformer` to route columns and bundle the whole baseline into one fit-able object. Use for: Lesson 010 capstone; the API for the reproducible baseline harness.
- [scikit-learn — Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
  Data leakage, inconsistent preprocessing, controlling randomness with `random_state`. Use for: Lesson 010; the reproducibility + no-leak checklist behind the Q1 checkpoint.

_Optional / extension (◆):_
- [Statistical Analysis with Missing Data (3rd ed.) — Little & Rubin, 2019](https://www.wiley.com/en-us/Statistical+Analysis+with+Missing+Data%2C+3rd+Edition-p-9780470526798)
  The graduate reference. Use for: likelihood/ignorability theory once the taxonomy is second nature.

_Optional / extension (◆ — read after the year exit criterion):_
- [SHAP — Lundberg & Lee, NeurIPS 2017](https://arxiv.org/abs/1705.07874)
  Model-agnostic feature attribution with uniqueness guarantees. Use for: explaining *why* a baseline predicts what it does; the interpretability tool reused on every later model.
- [Interpretable Machine Learning (3rd ed., 2025) — Christoph Molnar](https://christophm.github.io/interpretable-ml-book/)
  Free book. Use for: PDP/ICE, SHAP, and "goals of interpretability" when auditing tree/MLP baselines.

_Your own work (Tier-A artifact, not a published resource):_
- **`~/Projects/homework` — the ReAction L&D response-prediction submission** (`report.md` Tasks 1–4, `src/features.py`, `src/modeling.py`, `notebooks/01–04`, `artifacts/cv_folds_*.csv` + OOF `.npz`)
  119,498 situations joined from persons/situations/responses, 5,587 labelled (4.68 %), 5 classes, 4,851 persons of whom 675 repeat; person-grouped `StratifiedGroupKFold(5)`, LightGBM + isotonic calibration, log-loss/ECE ship-gate. **Primary reading for Lesson 036** — read as a reviewer beside `src/modeling.py`, not as its author. Use for: the Tier-A dataset and the pipeline under audit for the rest of Q4 (L037 packages the fixed version). Audit harness in this workspace: `labs/_audit_l036.py`, `labs/_selection_l036.py`.

## Knowledge — Year 2 (Advanced tabular)

- [DCN V2 — Wang et al., WWW 2021 / arXiv v2](https://arxiv.org/html/2008.13535v2)
  Lesson 048 primary reading: §3.2 Eq.1 (anchored matrix cross), Fig.1 (parallel/stacked), §3.5 Eqs.2–4 (linear factors versus nonlinear mixtures), §7.1 and Table 6 (protocol and results). The L+1 degree bound describes the basic cross stack in embedding coordinates, not nonlinear mixtures or the complete classifier. MovieLens is a filtered binary-target task despite the prose calling it regression; the exact six-field mapping and winning MovieLens hyperparameters remain gaps in our reproduction.
- [TensorFlow Recommenders Cross layer, release 0.7.3](https://github.com/tensorflow/recommenders/blob/v0.7.3/tensorflow_recommenders/layers/feature_interaction/dcn.py)
  L048 implementation validation source. Audit identity preactivation, zero diagonal scale, bias placement and factor orientation. The literal call method is checked through torch-backed Dense adapters; that does not validate TensorFlow kernels or training. Companion [official tutorial](https://www.tensorflow.org/recommenders/examples/dcn) uses raw MovieLens ratings and RMSE, so its scores cannot reproduce the paper's binary log-loss table.
- [MovieLens 1M — GroupLens](https://grouplens.org/datasets/movielens/1m/)
  L048 actual paper-dataset track: archive checksum, raw ratings/users schema, filtered 1/2 versus 4/5 task, six single-valued fields, random 80/10/10 split. Full-data closer attempt is measured and INCOMPARABLE; retain the exact protocol and fitted vocabularies.

**Architectures**
- [TabNet — Arik & Pfister, 2019](https://arxiv.org/abs/1908.07442)
  Sequential attention on features. Use for: interpretable neural tabular.
- [NODE — Popov et al., 2019](https://arxiv.org/abs/1909.06312)
  Differentiable oblivious trees. Use for: piecewise / irregular-function modeling.
- [TabTransformer — Huang et al., 2020](https://arxiv.org/abs/2012.06678)
  Contextual embeddings for columns. Use for: bridge to transformers on tables.
- [FT-Transformer / Revisiting DL — Gorishniy et al., NeurIPS 2021](https://arxiv.org/abs/2106.11959) ★
  rtdl baselines. Use for: **Lesson 041 primary reading** (Year 2 opener — abstract + intro for the baseline problem, §3 for the models and the Feature Tokenizer, §5 for the *no universal winner* verdict); FT-Transformer reproduction (L046); strongest *classic* neural single-table model. The methodological point that opens the year: a strong simple baseline (tuned ResNet) + a shared tuning protocol are the precondition for believing any "DL beats GBDT" claim — the L038 discipline applied to a whole subfield. Reference implementations: [`rtdl`](https://github.com/yandex-research/rtdl) / the `rtdl_revisiting_models` PyPI package. Companion survey: [Borisov et al. 2021, *Deep Neural Networks and Tabular Data*](https://arxiv.org/abs/2110.01889).
- [SAINT — Somepalli et al., 2021](https://arxiv.org/abs/2106.01342)
  Row + column attention. Use for: inter-sample attention idea.
- [ExcelFormer — Chen et al., 2023](https://arxiv.org/abs/2301.02819) · [Trompt — Chen et al., 2023](https://arxiv.org/abs/2305.18446)
  GBDT-surpassing claims. Use for: reading architecture claims *critically* against protocol.

**Modern DL & honest baselines (read critically)**
- [TabR — Gorishniy et al., 2023](https://arxiv.org/abs/2307.14338)
  Retrieval-augmented DL with kNN-attention. Use for: first DL model to beat GBDT on the GBDT-friendly benchmark.
- [RealMLP / "Better by Default" — Holzmüller et al., NeurIPS 2024](https://arxiv.org/abs/2407.04491)
  Strong pre-tuned MLP + GBDT defaults. Use for: the reality check — a good MLP needs no heavy tuning.
- [TabM — Gorishniy et al., ICLR 2025](https://arxiv.org/abs/2410.24210)
  Parameter-efficient MLP ensembling. Use for: **the current best/most practical DL baseline** — beats attention/retrieval models.
- [TabReD — Rubachev et al., 2024](https://arxiv.org/abs/2406.19380)
  Industry-grade datasets with temporal splits. Use for: proving why *time-based* evaluation flips model rankings.
- [Understanding temporal-shift limits — Cai & Ye, ICML 2025](https://arxiv.org/abs/2502.20260)
  Why DL fails under temporal drift; val-split protocol + Fourier temporal embedding. Use for: pairing with TabReD in Y2 Q2.
- [TabArena — Erickson et al., NeurIPS 2025](https://arxiv.org/abs/2506.16791)
  Living, maintained tabular leaderboard. Use for: ground truth on what actually wins, with ensembling.
- [BeyondArena — Purucker et al., 2026](https://arxiv.org/abs/2606.30410)
  Holistic benchmark: IID vs temporal vs grouped; TFMs win tiny/medium IID, trees/DL win non-IID/large/high-dim. Use for: failure-mode literacy beyond TabArena.
- [TALENT — Ye et al., 2024](https://arxiv.org/abs/2407.00956)
  300+ dataset meta-analysis; TALENT-tiny (45) for fast eval. Use for: meta-features predicting which family wins.
- [DL & Tabular Data: A Survey — Borisov et al., 2021](https://arxiv.org/abs/2110.01889) · [Survey on Deep Tabular Learning, 2024](https://arxiv.org/abs/2410.12034)
  Use for: mapping architecture families.

**Tabular foundation models / in-context learning**
- [PFNs — Müller et al., 2022](https://arxiv.org/abs/2112.10510)
  Prior-Data Fitted Networks; transformers approximating Bayesian inference. Use for: the ICL foundation.
- [TabPFN v1 — Hollmann et al., 2022](https://arxiv.org/abs/2207.01848)
  ICL on ≤1K rows. Use for: first tabular foundation model.
- [TabPFN v2 — Hollmann et al., Nature 2025](https://www.nature.com/articles/s41586-024-08328-6)
  Up to ~10K rows, beats 4h-tuned GBDT in seconds. Use for: the headline tabular-FM result.
- [A Closer Look at TabPFN v2 — Ye et al., 2025](https://arxiv.org/abs/2502.17361)
  How it handles heterogeneity; feature-extractor mode. Use for: understanding *why* it works.
- [TabICL — Qu et al., ICML 2025](https://arxiv.org/abs/2502.05564)
  Scales ICL to 500K rows via column-then-row attention. Use for: the scalability frontier of tabular ICL.
- [TabICLv2 — Qu et al., 2026](https://arxiv.org/abs/2602.11139)
  Open SOTA tabular ICL; QASSMax + Muon optimizer. Use for: current best open tabular FM baseline.
- [TabH2O — Qu et al., 2026](https://arxiv.org/abs/2605.18383)
  Unified classification+regression tabular FM in one forward pass. Use for: single-model cls/reg coverage.
- [TabPFN-2.5 — Grinsztajn et al., 2025](https://arxiv.org/abs/2511.08667)
  RealTabPFN-2.5; strong on small/medium datasets. Use for: tuned FM baseline on TabArena.
- [TabPFN-3 — Grinsztajn et al., 2026](https://arxiv.org/abs/2605.13986)
  Enterprise-scale tabular FM (up to ~1M rows). Use for: production-scale tabular ICL frontier.
- [LoCalPFN — Thomas et al., 2024](https://arxiv.org/abs/2406.05207) · [Drift-Resilient TabPFN — Helli et al., NeurIPS 2024](https://arxiv.org/abs/2411.10634)
  Retrieval+fine-tuning; distribution shift. Use for: making PFNs practical.
- [Realistic eval of TabPFN v2 in open environments — Cheng et al., 2025](https://arxiv.org/abs/2505.16226)
  **Critical counterweight:** where tabular FMs fail (covariate shift, scale, imbalance) and trees still win.
- [Operational TTF — Klein & Hoffart, 2026](https://arxiv.org/abs/2606.29091)
  Values-only TFMs hit Bayes bound; rule-derived audits needed. Use for: enterprise deployment skepticism.
- [PyTorch Frame — Hu et al., 2024](https://arxiv.org/abs/2404.00776)
  Deep tabular row encoder. Use for: RDL stack (encoder → GNN); read in Y2, apply heavily in Y4. (**Correct ID:** not `2402.05964`.)

_Optional / extension (◆):_
- [CARTE — Kim, Grinsztajn & Varoquaux, 2024](https://arxiv.org/abs/2402.16785)
  String-aware graph-attention pretraining; no schema/entity matching needed. Use for: schema-agnostic transfer across tables and an early glimpse of the relational graph view.
- [Interpretable ML for TabPFN — Rundel et al., 2024](https://arxiv.org/abs/2403.10923)
  SHAP/LOCO adapted to in-context models (avoids retraining). Use for: interpreting foundation-model predictions; pairs with Molnar's "interpretability tax".
- [TabLLM — Hegselmann et al., 2022](https://arxiv.org/abs/2210.10723)
  LLM serialization of rows. Use for: understanding the boundary the thesis rejects — text-only ≠ relational structure (see [MISSION.md](./MISSION.md) out-of-scope).
- [Tabular Foundation Models (in-progress book) — Christoph Molnar](https://tabularfoundationmodels.com) · [Mindful Modeler newsletter](https://mindfulmodeler.substack.com/)
  Practitioner's tour of TabPFN/TabICL: usage, architecture, interpretability cost, failure modes. Use for: tracking the tabular-FM frontier between curriculum updates.

## Knowledge — Year 3 (Graph ML)

- [Neural Message Passing — Gilmer et al., ICML 2017](https://arxiv.org/abs/1704.01212)
  MPNN framework. Use for: unified view of GNN layers.
- [GCN — Kipf & Welling, ICLR 2017](https://arxiv.org/abs/1609.02907)
  Spectral motivation, simple propagation. Use for: first GNN implementation.
- [GraphSAGE — Hamilton et al., NeurIPS 2017](https://arxiv.org/abs/1706.02216)
  Inductive neighbor sampling. Use for: scaling GNN training.
- [GAT — Veličković et al., ICLR 2018](https://arxiv.org/abs/1710.10903)
  Attention over neighbors. Use for: adaptive aggregation.
- [R-GCN — Schlichtkrull et al., ESWC 2018](https://arxiv.org/abs/1703.06103)
  Relation-specific convolutions. Use for: heterogeneous graphs; direct precursor to REG edges.
- [HGT — Hu et al., WWW 2020](https://arxiv.org/abs/2003.01332)
  Heterogeneous graph transformer. Use for: multi-type message passing at scale.
- [TGN — Rossi et al., ICML 2020](https://arxiv.org/abs/2006.10637)
  Temporal graph networks. Use for: time-respecting relational data before RDL.

_Optional / extension (◆) — GNN pathologies & graph transformers:_
- [Deeper Insights / over-smoothing — Li et al., 2018](https://arxiv.org/abs/1801.07606)
  Formalizes over-smoothing (lecture 085). Use for: why stacking GNN layers stops helping.
- [DropEdge — Rong et al., 2019](https://arxiv.org/abs/1907.10903)
  Random edge dropping. Use for: a cheap remedy for over-smoothing/over-fitting in deep GCNs.
- [PNA — Corso et al., 2020](https://arxiv.org/abs/2004.05718)
  Multiple aggregators + degree-scalers. Use for: why a single aggregator throws away neighborhood information.
- [Over-squashing bottleneck — Alon & Yahav, 2021](https://arxiv.org/abs/2006.05205)
  Long-range signal collapse. Use for: predicting where deep REG message passing loses distant-table signal.
- [Over-squashing via curvature — Topping et al., 2021](https://arxiv.org/abs/2111.14522)
  Ricci-curvature diagnosis + rewiring. Use for: a geometric account of graph bottlenecks.
- [Graphormer — Ying et al., 2021](https://arxiv.org/abs/2106.05234)
  Structural encodings for transformers on graphs. Use for: precursor to relational graph transformers.
- [GraphGPS — Rampášek et al., 2022](https://arxiv.org/abs/2205.12454)
  MPNN + global attention recipe. Use for: the direct conceptual parent of RelGT (Y4).

## Knowledge — Year 4 (Relational DL)

- [Supervised Learning on Relational Databases with GNNs — Cvitkovic, 2019](https://arxiv.org/abs/2002.02046)
  Early relational DL line. Use for: historical context.
- [Position: Relational Deep Learning — Fey et al., ICML 2024](https://proceedings.mlr.press/v235/fey24a.html)
  REG blueprint. Use for: north-star thesis and vocabulary.
- [Relational Deep Learning — Fey et al., 2024 (arXiv:2312.04615)](https://arxiv.org/abs/2312.04615)
  The foundational RDL paper (rows = nodes, PK/FK = edges; RelBench). Use for: **Lesson 035 primary reading — §1–2 only** (★ preview): the five issues with manually joining+aggregating tables into one training table — (1) manual/slow, (2) arbitrary/suboptimal choices, (3) only a tiny fraction of features explored, (4) **forcing data into a single table aggregates into lower-granularity features, thus losing valuable fine-grain signal** (the *aggregation collision* — the load-bearing claim), (5) drift makes hand features obsolete — and the definition of the *relational entity graph*. The Gabor-filter → raw-pixel CV analogy for "learn the features instead of authoring them." Message passing / temporal sampling / RelBench results are Y3–Y4, not now.
- [RelBench v1 — Robinson et al., NeurIPS 2024](https://arxiv.org/abs/2407.20060)
  7 databases, FE user study. Use for: hands-on experiments; manual FE comparison.
- [ContextGNN — Yuan et al., 2024](https://arxiv.org/abs/2411.19513)
  Pair-wise + two-tower fusion for recommendation/link prediction. Use for: RDL applied to recsys.
- [RelGNN — Chen et al., ICML 2025](https://arxiv.org/abs/2502.06784)
  Composite message passing + atomic routes; SOTA on most RelBench tasks. Use for: current SOTA GNN architecture.
- [Relational Graph Transformer (RelGT) — Dwivedi et al., ICLR 2026](https://arxiv.org/abs/2505.10960)
  Multi-element tokenization + local/global attention. Use for: graph-transformer paradigm on REG.
- [Desired graph for RDL — Cheng & Luo, ICML 2026](https://arxiv.org/abs/2606.08491)
  Schema graphs need filtering + injection. Use for: when raw FK graphs fail.
- [Universal Row Encoder — Peleška & Šír, ECML PKDD 2026](https://arxiv.org/abs/2606.21434)
  Modular row encoder decoupled from GNN message passing. Use for: cross-database encoder pretrain.
- [RDL survey — arXiv 2025](https://arxiv.org/abs/2506.16654)
  Challenges + next-gen architectures. Use for: year-4/5 frontier mapping.

_Optional / extension (◆):_
- [DBFormer — Peleška & Šír, 2024](https://arxiv.org/abs/2412.05218)
  SQL-native relational transformer; contrast to graph-native RDL.
- [ReDeLEx — 2025](https://arxiv.org/abs/2506.22199)
  70+ CTU databases; classical vs RDL benchmarking framework.
- [4DBInfer — Wang et al., NeurIPS 2024](https://arxiv.org/abs/2404.18209)
  Graph-centric RDB benchmarking toolbox ([code](https://github.com/awslabs/multi-table-benchmark)). Use for: comparing table→graph construction strategies and subsampling against RelBench, to test whether your conclusions are benchmark-dependent.

## Knowledge — Year 5 (Foundation relational models)

- [Towards Foundation Models for Relational DBs — Vogel et al., 2023](https://arxiv.org/abs/2305.15321)
  LM + GNN pre-training vision. Use for: foundation-model thesis and open problems.
- [Griffin — Wang et al., ICML 2025](https://arxiv.org/abs/2505.05568)
  Graph-centric RDB foundation model; unified encoder/decoder, cross-attention, pretrained on 150M+ nodes. Use for: the first serious open RDB FM.
- [RDB-PFN — Wang et al., 2026](https://arxiv.org/abs/2603.03805)
  First relational FM trained *purely on synthetic data* (Relational Prior Generator + PFN). Use for: open, reproducible relational ICL.
- [RDBLearn toolkit — Zhang et al., 2026](https://arxiv.org/abs/2602.18495); [companion encoder analysis — Xu et al.](https://arxiv.org/abs/2602.13697)
  DFS featurize + off-the-shelf TabICL/TabPFN; no RDB FM training. Use for: **training-free baseline** that sometimes beats supervised RDL.
- [Relational Transformer (RT) — Ranjan et al., ICLR 2026](https://arxiv.org/abs/2510.06377)
  Cell-level tokenization + relational attention; zero-shot on unseen schemas. Use for: schema-agnostic relational FM baseline.
- [OpenRFM — Chen et al., 2026](https://arxiv.org/abs/2606.04320)
  Open relational ICL; dual-stage architecture + homophily-aware pretrain; ~30% over RT. Use for: best open reproducible relational FM.
- [KumoRFM-2 — Fey et al., 2026](https://arxiv.org/abs/2604.12596)
  Current RelBench SOTA; first few-shot FM to beat supervised RelGNN. Use for: SOTA tracking (proprietary weights).
- [GelGT (Gaussian Relational Graph Transformer) — 2026](https://arxiv.org/abs/2605.15575)
  Long-range dependency fixes for relational graph transformers. Use for: tracking architecture frontier.
- [RelBench v2 — arXiv 2602.12606](https://arxiv.org/abs/2602.12606) · [RelGT-AC — arXiv 2606.03040](https://arxiv.org/abs/2606.03040)
  Expanded benchmark + autocomplete tasks. Use for: year-5+ experiments.
- KumoRFM v1 (2025) — no arXiv; [PDF](https://kumo.ai/research/kumo_relational_foundation_model.pdf). Track via KumoRFM-2 for reproducible numbers.

## Knowledge — Year 5 → 6 research bridge (September 2026)

The [B01–B24 lecture specifications and primary readings](./plan/year-5-6-bridge.md) provide 16 core units and eight electives. The [influence audit](./plan/research-influence-audit-2026-09.md) distinguishes lineage and benchmark uptake from newly reported performance.

New required families: [numerical embeddings](https://arxiv.org/abs/2203.05556), [TabDPT](https://arxiv.org/html/2410.18164v3), [Mitra](https://arxiv.org/html/2510.21204v1), [ConTextTab](https://arxiv.org/html/2506.10707v1), [TabSTAR](https://arxiv.org/html/2505.18125v2), dedicated [Relational Transformer](https://arxiv.org/html/2510.06377v1) and [PluRel](https://arxiv.org/html/2602.04029v1). Existing TabM, TabPFN, TabICL, RelGNN/RelGT and relational FM families receive comparative updates. Recent diagnostics and speculative mechanisms have explicit scope limits and elective placements.

Research checked **2026-09-26** using primary papers, method/evaluation sections for major additions, official repositories and benchmark infrastructure. RT-J's full paper was inaccessible; its author page supports only the scoped update. Optional papers retained from the initial abstract-level pass still require method/code audits before lesson authoring. No experiment or learner completion is claimed.

## Wisdom (Communities)

- [RelBench mailing list](https://groups.google.com/forum/#!forum/relbench/join)
  Benchmark authors and practitioners. Use for: reproduction questions, leaderboard norms.
- [PyTorch Geometric Discussions](https://github.com/pyg-team/pytorch_geometric/discussions)
  GNN implementation. Use for: PyG + RelBench integration.
- [RelBench GitHub Issues](https://github.com/snap-stanford/relbench/issues)
  Bug reports and baseline discrepancies. Use for: when numbers don't match papers.
- [r/MachineLearning](https://reddit.com/r/MachineLearning)
  New paper alerts. Use for: spotting RDL papers; filter for signal.
- [Mindful Modeler — Christoph Molnar (Substack)](https://mindfulmodeler.substack.com/)
  High-signal newsletter on interpretability and tabular foundation models, written with statistical mindfulness. Use for: a practitioner's read on TabPFN/TabICL failure modes and the "interpretability tax" of in-context models.

## Gaps

- No canonical textbook for relational foundation models — curriculum is paper-driven.
- Proprietary FM weights (KumoRFM v1/v2) limit full reproduction; plan ablations on open components (Griffin, RDB-PFN, OpenRFM, RDBLearn).
- Lesson HTML published through 015 (LightGBM); agent produces lessons as you progress through CURRICULUM.md rows.
- Fast-moving frontier: re-run an arXiv search each quarter (sort by `submitted`) and add only papers that set SOTA, expose a failure mode, or are a baseline to beat. Full verified ID index + exhaustive registry in [CURRICULUM.md](./CURRICULUM.md#exhaustive-paper-registry-july-2026).
- Resources are tagged **core** (the default entries) vs **◆ optional / extension** (read only after the year's core papers and lab are done). Optional papers are ~2 h skims, never a reason to skip reproduction or exit exams. Full optional index: [CURRICULUM.md → Optional / extension reading](./CURRICULUM.md).
- **July 2026 merge:** PyTorch Frame ID corrected to `2404.00776`; TALENT to `2407.00956`. See [Research merge status](./CURRICULUM.md#research-merge-status-july-2026-deep-research-pass) for what's already solved vs newly added.


### Lesson 047 — SAINT source audit (2026-09-05)
- **Primary:** Somepalli, Goldblum, Schwarzschild, Bruss & Goldstein (2021), [SAINT, arXiv:2106.01342v1](https://arxiv.org/abs/2106.01342v1). §3/Algorithm 1 for row packing, §4/Eqs.3–5 for contrastive and denoising pretraining; Tables 1–2 and Appendix C for supervised protocol. Bank target: 0.9330 AUROC, five-trial mean; Table 6 SE 0.0009.
- **Executable companion:** [official repository at e288e84](https://github.com/somepago/saint/tree/e288e84c77a54cfd2ffb55a53678fb7cbbb16630). Inspect models/model.py, models/pretrainmodel.py, train.py, data_openml.py. SHA-256 audit in labs/_sources_l047.json.
- **Why both:** normalization placement, numeric embedding, split proportions, heads, attention dropout, and binary checkpoint selection differ across descriptions. The lesson makes each fidelity choice explicit; reference-stage parity does not establish full training parity.
- **Course use:** [L047](lessons/0047-saint.html), [reproduction card](reference/saint-reproduction.html). Supervised model + contrastive key-parts exercise; full semi-supervised benchmark remains outside the executed claim.

## L049 — pinned primary sources and released data (2026-09-05)

- [ExcelFormer arXiv v5 / KDD 2024](https://arxiv.org/html/2301.02819v5): §3.1 Eq.1–4 for mask and initialization; §4 Eq.7–8 for Feat-Mix; §6.1 and Appendix G for protocol and exact dataset rows. The older curriculum's 2023 date refers to initial submission.
- [Released ExcelFormer code, revision 17f7052](https://github.com/WhatAShot/ExcelFormer/blob/17f70526390e70390bb8c8ec3850697eb730f9cd/bin/excel_former.py): numeric/pre-normalization path validated on copied weights. Record positional tie handling, initialization differences and unvalidated training augmentation separately.
- [Authors' prepared dataset release](https://huggingface.co/datasets/jyansir/excelformer): L049 uses original split files for one Pima variant, Breast Cancer Dataset and Swiss banknote. File hashes and exact names are in `labs/relkit/claim_data.py`; upstream fit-scope provenance is not independently reconstructed.
- [Trompt v2 / ICML 2023](https://arxiv.org/html/2305.18446v2): §3 Eq.1–9, §4.2 subgroup limits, Appendix A.4 / F search accounting. Full benchmark unreplicated; key parts only in the lab.
- [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/): real timestamps for a separate transfer probe using the existing checked archive. This is neither paper's reported protocol; snapshot feature availability remains a limitation.


## Lesson 050 — fair comparison checkpoint (2026-09-06)

- Gorishniy et al., *Revisiting Deep Learning Models for Tabular Data*, arXiv 2106.11959v5: https://arxiv.org/html/2106.11959v5 — §4.5 separates ensemble comparison from Table 2 individual models; Appendix D.2 examines time budgets; Appendix E defines FT-T. Checked against the primary text for L050.
- Authors' reference: https://github.com/yandex-research/rtdl-revisiting-models — installed `rtdl_revisiting_models==0.0.2` used only for copied-weight numeric evaluation parity; source SHA saved in L050 results.
- Demšar, *Statistical Comparisons of Classifiers over Multiple Data Sets* (JMLR 2006): https://www.jmlr.org/papers/v7/demsar06a.html — dataset-level ranks and Friedman/Nemenyi. With three tasks the lesson treats inference as low-power and adds the exact sign test for the primary pair.
- XGBoost Python documentation: https://xgboost.readthedocs.io/en/stable/python/python_intro.html#early-stopping — validation monitoring and best-iteration behavior; local measured version 3.3.0.


## Lesson 051 — intervention audit (2026-09-06)

- [Grinsztajn et al., v1 §5 and A.4](https://arxiv.org/html/2207.08815v1): source of the three-bias investigation. Use alongside code, not as an unrestricted model ranking.
- [Pinned released transformations](https://github.com/LeoGrin/tabular-benchmark/blob/9d54cf53d9fd3159e367e70a00005f4fcbf2c79d/src/data_transforms.py): verify strict hard-label threshold, common rotation and training-moment-matched noise; our noise variant follows the paper caption instead.
- [OpenML suite 337](https://www.openml.org/search?type=benchmark&study_type=task&id=337): authors' January 2023 numeric classification release; L051 uses three exact dataset IDs. This release is newer than the paper v1.
- [Rahaman et al. (2019)](https://proceedings.mlr.press/v97/rahaman19a.html): distinguishes expressivity from frequency-dependent learning behavior.
- [Demšar (2006)](https://www.jmlr.org/papers/v7/demsar06a.html): dataset-level comparison; only three tasks limits inference here.

## Lesson 052 — TabR source audit

- [TabR v2 — Gorishniy et al. (2023)](https://arxiv.org/abs/2307.14338v2). Read §3.2 Eq. 5, §4 TabR-S and Appendix B for learned distances, corrected label values and deployment eligibility. Distinguish single-model, ensemble and 43-task claims.
- [Pinned released implementation](https://github.com/yandex-research/tabular-dl-tabr/tree/17baa9082506f8e7a0f8d11bb1e08212926a1507). Validation source and California selected hyperparameters; local audit records neural parity but not full-training parity.
- [Authors’ data archive](https://huggingface.co/datasets/puhsu/tabular-benchmarks). Source of original train/val/test arrays; L052 records byte ranges and hashes for three numeric tasks.


## Lesson 053 — RealMLP and strong defaults (2026-09-07)

- **Core paper:** Holzmüller, Grinsztajn & Steinwart, [Better by Default](https://arxiv.org/html/2407.04491v3). Read §2 for dataset-level meta-train/meta-test, §3 for the recipe, Table A.1 and Appendix A.2 for TD-S. Verified live during authoring.
- **Primary code:** [Standalone TD-S](https://github.com/dholzmueller/realmlp-td-s_standalone/tree/a8c73f75dbeae4ab1ba5bc92372c444ffa0f766e), pinned source and MIT license under labs/sources/l053. Independent numeric forward and gradient parity checked.
- **Full benchmark and toolkit:** [pytabkit](https://github.com/dholzmueller/pytabkit), inspected revision c126ea51187c5080b91f28d352481dbd3b2194b0. Its README links the [benchmark archive](https://doi.org/10.18419/darus-4555); exact benchmark reconstruction remains unrun.

## Lesson 055 source audit — 2026-09-09

- [TabReD v4, §5.4 / Figure 2](https://arxiv.org/html/2406.19380v4#S5.SS4): primary reading for temporal/random split comparison. Match row counts and distinguish protocol sensitivity from isolated drift causality.
- [Official source at b5ef15b](https://github.com/yandex-research/tabred/tree/b5ef15b3749f30da7a1eb8fba21a5b54d706bf32): registry checksums, released paired split indices and Sberbank target transformation. Public July 2026 release counts differ from paper Table 2; full paper identity is not established.


## Lesson 056 — TabArena methodology and result audit

- [TabArena v1, Erickson et al.](https://arxiv.org/html/2506.16791v1): §§2.1–2.3 define method, data and evaluation protocols; §3.1 interprets default/tuned/ensembled rankings. Initial IID scope is crucial after L055.
- [Pinned official frozen-paper example](https://github.com/autogluon/tabarena/blob/e7cc6b049f9be11a6df29eb2560d9ccb2399d95c/examples/reproducibility/run_generate_main_leaderboard_neurips2025.py): full-pool reconstruction route, not run locally. Its collection includes camera-ready replacements; reconcile the exact paper version.
- [Pinned evaluator](https://github.com/autogluon/tabarena/blob/e7cc6b049f9be11a6df29eb2560d9ccb2399d95c/packages/bencheval/src/bencheval/elo_utils.py): equal-dataset rank-to-win primitive validates the student implementation; current solver details are not silently attributed to v1.
- [Demšar 2006](https://www.jmlr.org/papers/v7/demsar06a.html): independent-dataset blocks, Friedman and Nemenyi; L056 supplementary statistics retain this dataset unit.
- Artifact URLs, exact hashes and scope: `labs/_sources_l056.json`; four public score files include all 51 datasets. This is released-score reanalysis, not fresh model training.


## Lesson 057 — Cross-family OOF ensembles

- [TabArena v1 §3.2, Figure6](https://arxiv.org/html/2506.16791v1#S3.SS2): cross-model ensemble evidence, member weights, and limits of individual rankings. Distinguish the paper result from our small OOF stack.
- [Caruana et al. 2004](https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf): forward selection with replacement and validation-based ensemble construction; local scope omits library bagging/sorted initialization.
- [Pinned AutoGluon selector](https://github.com/autogluon/autogluon/blob/946b65c683e975df34456bbe1f68366c8f4570be/core/src/autogluon/core/models/greedy_ensemble/ensemble_selection.py): executable source validation with a local metric adapter; rounding and tie policies differ.
- [TabICL0.1.4](https://pypi.org/project/tabicl/0.1.4/): provided foundation-model inference baseline. Original v1.1 checkpoint, wheel and revision identities in labs/_sources_l057.json; one-view local reduction is not its full default.

<!-- FOUNDATION-058-070:begin -->
## Lessons 058–070 · primary reading and exact evidence contracts
- **L058:** [TALENT Sections 4–5 and 8](https://arxiv.org/html/2407.00956v3). [Local scope and regeneration](labs/l058-reproduction.md).
- **L059:** [Cawley and Talbot 2010](https://jmlr.org/papers/v11/cawley10a.html). [Local scope and regeneration](labs/l059-reproduction.md).
- **L060:** [TabArena and TabReD protocols](https://arxiv.org/html/2506.16791v1). [Local scope and regeneration](labs/l060-reproduction.md).
- **L061:** [Transformers Can Do Bayesian Inference](https://arxiv.org/html/2112.10510v7). [Local scope and regeneration](labs/l061-reproduction.md).
- **L062:** [TabPFN v1](https://arxiv.org/html/2207.01848v6). [Local scope and regeneration](labs/l062-reproduction.md).
- **L063:** [TabPFN v1 prior](https://arxiv.org/html/2207.01848v6). [Local scope and regeneration](labs/l063-reproduction.md).
- **L064:** [Nature TabPFN v2](https://www.nature.com/articles/s41586-024-08328-6). [Local scope and regeneration](labs/l064-reproduction.md).
- **L065:** [A Closer Look at TabPFN v2, Section 6](https://arxiv.org/html/2502.17361v1). [Local scope and regeneration](labs/l065-reproduction.md).
- **L066:** [TabICL, Sections 3–4](https://arxiv.org/html/2502.05564v1). [Local scope and regeneration](labs/l066-reproduction.md).
- **L067:** [Retrieval & Fine-Tuning for In-Context Tabular Models](https://arxiv.org/html/2406.05207v1). [Local scope and regeneration](labs/l067-reproduction.md).
- **L068:** [Drift-Resilient TabPFN](https://arxiv.org/html/2411.10634v1). [Local scope and regeneration](labs/l068-reproduction.md).
- **L069:** [Realistic Evaluation of TabPFN v2 in Open Environments](https://arxiv.org/html/2505.16226v1). [Local scope and regeneration](labs/l069-reproduction.md).
- **L070:** [TabPFN-3 technical report and historical controls](https://arxiv.org/abs/2605.13986). [Local scope and regeneration](labs/l070-reproduction.md).
<!-- FOUNDATION-058-070:end -->


## Lesson 071 · VIME source bundle

- Yoon et al., NeurIPS 2020, [VIME paper](https://proceedings.neurips.cc/paper/2020/file/7d97667a3e056acab9aaf653807b4a03-Paper.pdf), §4, Eqs. 3–10: corruption, dual pretext tasks and downstream consistency.
- [Supplement](https://proceedings.neurips.cc/paper/2020/file/7d97667a3e056acab9aaf653807b4a03-Supplemental.pdf), §§2, 5–6: data splits and model selection.
- [Author code, pinned commit](https://github.com/jsyoon0823/VIME/tree/996c58cf4c570061b30c38ecf2a754a9af85aafd): collision-aware targets, fixed corruption, frozen encoder and logit variance; checked in labs/_sources_l071.json.


## Lesson 072 · contrastive tabular views

- Bahri et al., SCARF (ICLR 2022; 2021 preprint), Algorithm 1 and Figure 1: https://arxiv.org/html/2106.15147v2
- Ucar et al., SubTab (NeurIPS 2021), sections 2.1–2.3: https://arxiv.org/html/2110.04361v1
- Author SubTab implementation, pinned aa3ab1b97231fc37229ef3e55d98ae13bdbfb4fc: https://github.com/AstraZeneca/SubTab/tree/aa3ab1b97231fc37229ef3e55d98ae13bdbfb4fc
- Local source/operator and protocol audit: labs/_sources_l072.json and labs/l072-reproduction.md.

## Lesson 073 · SSL evaluation
- Wang et al., [A Survey on Self-Supervised Learning for Non-Sequential Tabular Data](https://arxiv.org/html/2402.01204v3), taxonomy and transfer context.
- Bahri et al., [SCARF](https://arxiv.org/html/2106.15147v2), Figure 1 and training/evaluation sections; pretraining then fine-tuning.
- Oliver et al., [Realistic Evaluation of Semi-Supervised Learning Algorithms](https://arxiv.org/html/1804.09170v2), §4.6 and §5; image-domain evidence used to motivate validation-label accounting, not tabular scores.

## L074 CARTE

- [CARTE v2](https://arxiv.org/html/2402.16785v2), Figures 1–3 and Sections 3.1–3.3: graph representation and cross-schema transfer.
- [Pinned release](https://github.com/soda-inria/carte/tree/f54690da4cddbedd1e1a9113a312f85783d2c125): implementation/checkpoint/example-table provenance.
- [FastText English crawl vectors](https://fasttext.cc/docs/en/crawl-vectors.html): real language inputs and licensing.

## L075 PyTorch Frame

- [Hu et al., v2, Figure1 and §3](https://arxiv.org/html/2404.00776v2): materialization, encoding, column interactions and decoding.
- [Pinned release0.3.0](https://github.com/pyg-team/pytorch-frame/tree/d998aae368db6a4e36139ccc56bd54579a70874b): executable enum, converter, statistics and encoders; local files byte-matched.
- [Official heterogeneous-type tutorial](https://pytorch-frame.readthedocs.io/en/latest/handling_advanced_stypes/handle_heterogeneous_stypes.html): conceptual configuration guide; latest docs can differ from the pin.

## L076 relational stack

- [PyTorch Frame v2, Figure1 and §3](https://arxiv.org/html/2404.00776v2): typed table-to-row interface.
- [RelBench v1, Table6 and AppendixB](https://arxiv.org/html/2407.20060v1): driver-dnf target and training protocol.
- [Publication-day source](https://github.com/snap-stanford/relbench/tree/5894184f3d1b2432feb9a208a8aaf18b106fbdf4): original model, neural layers and gnn_node trainer.

## L077 single-table ceiling

- [Fey et al., ICML2024 position paper](https://proceedings.mlr.press/v235/fey24a.html): primary motivation for graph learning over databases.
- [Zaheer et al., Deep Sets](https://arxiv.org/abs/1703.06114): permutation-invariant set functions; supporting reading, no paper experiment claimed.
- [RelBench v1](https://arxiv.org/html/2407.20060v1): real-data evaluation and feature-engineering comparison, distinct from the authored collision construction.

## L078 message passing

- [Gilmer et al., ICML2017, §2](https://proceedings.mlr.press/v70/gilmer17a.html): message, sum, update and graph readout.
- [Kipf & Welling, ICLR2017](https://arxiv.org/html/1609.02907v4): Eq2/9 and Table2 Cora fixed-split experiment.
- [Pinned GCN release](https://github.com/tkipf/gcn/tree/39a4089fe72ad9f055ed6fdb9746abdcfebc4d81): preprocessing, architecture, loss, initialization and exact stopping code; paper/code stopping mismatch documented.

## L079 decision guide

Primary historical sources for mechanism-based shortlists: [trees](https://arxiv.org/abs/2207.08815), [FT-Transformer](https://arxiv.org/abs/2106.11959), [RealMLP](https://arxiv.org/abs/2407.04491), [TabM v3](https://arxiv.org/abs/2410.24210v3), [TabPFN v2](https://www.nature.com/articles/s41586-024-08328-6), [TabICL 2025 v2](https://arxiv.org/abs/2502.05564v2). This is a course synthesis, not a current leaderboard. Local quantitative evidence uses corrected L060 v2 predictions only; source/hash inventory: labs/_sources_l079.json.

## L080 exit exam

Historical synthesis of Grinsztajn three biases, FT-Transformer, TabM, Nature TabPFN v2, TabICL 2025 and TabReD. Pinned sources: `labs/_sources_l080.json`. FT uses the released architecture (including ReGLU and first-layer normalization exception). Local cross-paper comparison has no matching published target; original benchmarks remain NOT_RUN.

## L081 · Gilmer MPNN framework

- [Primary paper](https://proceedings.mlr.press/v70/gilmer17a.html): framework §2, features §6, training §7.
- [Supplement](https://proceedings.mlr.press/v70/gilmer17a/gilmer17a-supp.pdf): GCN mapping §1.1; sparse GG-NN Table3 and chemical accuracy Table1.
- [Official source](https://github.com/brain-research/mpnn/tree/4a1f0ddea3cd7de5eebc96e509da2161624aaacd): model-only release with missing reader/trainer; pin and hashes in labs/_sources_l081.json.

## L082 · GCN

- [Kipf & Welling 2017](https://arxiv.org/html/1609.02907v4): spectral motivation, Eq2 propagation, Table2 Cora target.
- [Pinned TensorFlow release](https://github.com/tkipf/gcn/tree/39a4089fe72ad9f055ed6fdb9746abdcfebc4d81): actual stopping rule, regularization, dropout and Planetoid data. Source/data provenance shared with L078 and verified again for L082.

## L083 · GraphSAGE

- [Hamilton, Ying & Leskovec 2017](https://arxiv.org/html/1706.02216v4): Algorithms1–2, Table1 PPI supervised mean .598, AppendixC hyperparameters.
- [Pinned official implementation](https://github.com/williamleif/GraphSAGE/tree/a0fdef95dca7b456dab01cb35034717c8b6dd017): inspect mean concat, reversed sampling, final normalization, fixed adjacency and training access.
- [Full PPI data](https://snap.stanford.edu/graphsage/ppi.zip): archive SHA-256 in labs/_sources_l083.json; not the tiny example_data subset.

## L084 · GAT

- [Veličković et al.2018](https://arxiv.org/html/1710.10903v3): attention equations, Table2 Cora target and100-run evaluation.
- [Pinned official implementation](https://github.com/PetarV-/GAT/tree/5af87e7fce2b90ae1cbd621cd58059036a3c7436): inspect head biases/dropout, parameter regularization and checkpoint decisions.
- [Lesson reproduction contract](labs/l084-reproduction.md): full protocol audit, local run evidence and historical gaps.

## L085 · Over-smoothing

- Li, Han & Wu (AAAI2018), [Deeper Insights](https://arxiv.org/abs/1801.07606), §3 and Figure2. Primary source for fixed-propagation analysis and untrained karate experiment.
- [Authors’ source, pinned AAAI-18 branch](https://github.com/liqimai/gcn/tree/3b30a2d35ca2b144bf0f36337f233d407a7e2dd6): hidden ReLU, final identity, Glorot initialization. Figure2 seed/driver not identified.

## L086 · PyG fundamentals

- [Fey & Lenssen2019](https://arxiv.org/abs/1903.02428): primary framework paper; Table1 fixed-Cora target and100-run protocol.
- [Archived PyG1.2.0](https://github.com/pyg-team/pytorch_geometric/tree/d5aff37604c8e3247f5e807f2ba0ec6eeb4c661b/benchmark/citation): audited release proxy, model and selection details differ from original Kipf trainer.
- [MessagePassing](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MessagePassing.html) and [NeighborLoader](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/neighbor_loader.html): source/destination lifting, seed ordering and ID contracts. Executed against PyG2.8.0.post1.

## L087 · Link prediction

- [Zhang & Chen2018](https://arxiv.org/abs/1802.09691): primary reading, enclosing subgraphs/DRNL; Table1 baseline target.
- [Pinned author source](https://github.com/muhanzhang/SEAL/tree/ca1f019a15fb0c21796042165b4e6bee73981dd3): data, split/sampling, heuristic scores, metric and labeling implementation.
- [OGB task contracts](https://ogb.stanford.edu/docs/linkprop/) and [evaluator](https://github.com/snap-stanford/ogb/blob/master/ogb/linkproppred/evaluate.py): candidate policies and average tied ranks.

## L088 · Graph classification

- [Xu et al., ICLR2019](https://arxiv.org/abs/1810.00826v3): §4 GIN and graph readout, §7 Table1 MUTAG target, expressiveness bound and scope.
- [Pinned powerful-gnns release](https://github.com/weihua916/powerful-gnns/tree/9a2ce8ac3e99278307093a464a95caf0fb04b602): model, data, training recipe and clarified common-epoch cross-validation.
- [PyTorch1.0 scheduler](https://github.com/pytorch/pytorch/blob/v1.0.0/torch/optim/lr_scheduler.py): historical StepLR initialization for faithful decay timing.

## L089 · Cluster-GCN

- [Chiang et al., KDD2019](https://arxiv.org/abs/1905.07953v2): §§3.1–3.3,Algorithm1 and Table10; batching, diagonal enhancement and PPI result target.
- [2019 release](https://github.com/google-research/google-research/tree/89c16e403d42015c3133634788ed0b7965f56395/cluster_gcn): PPI shell recipe, precomputed first layer, loss, partition and inference details.
- [Stanford GraphSAGE datasets](https://snap.stanford.edu/graphsage/): original PPI archive, fixed split, features and multi-label targets.

## L090 · GNN checkpoint

- Kipf & Welling, [GCN](https://arxiv.org/html/1609.02907v4), §§3,5.2, Table2; primary benchmark source.
- [Pinned GCN release](https://github.com/tkipf/gcn/tree/39a4089fe72ad9f055ed6fdb9746abdcfebc4d81): implementation resolves normalization, objective and stopping.
- Hamilton et al., [GraphSAGE](https://arxiv.org/html/1706.02216v4), Algorithm1 and §3.1: inductive sample/aggregate principle; Cora extension does not reproduce its tables.

## Lessons 071–090 · architecture reading route · 2026-09-19

The [model and learning map](reference/0071-0090-model-map.html) connects self-supervision, schema transfer, row encoding, graph operators, pair/graph prediction and sampling. Each revised lesson pairs a primary-paper reading prompt with an original arithmetic trace and an architecture diagram. Read the cited method section before interpreting benchmark results; the existing reproduction contracts remain the authority for execution coverage and deviations.

## Lesson 091 · R-GCN

- Schlichtkrull et al., [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103): equations2–4, entity classification and Table2 AIFB.
- [Author classification implementation, pinned revision](https://github.com/tkipf/relational-gcn/tree/4bec1341dd46b72bf482f7ed26c2dca4533577f6): exact AIFB settings, sparse support construction and testing mode.
- [Keras1.2.1 optimizer](https://github.com/keras-team/keras/blob/1.2.1/keras/optimizers.py): historical Adam equation.

## Lesson 092 — HAN / meta-paths

- [Wang et al. HAN, arXiv1903.07293v2](https://arxiv.org/html/1903.07293v2): §§3–4, Fig2, §5.3–5.4/Table3.
- [Pinned authors release](https://github.com/Jhy1993/HAN/tree/71bac29a07fb8fab908d50a806a7bc38aa6c6611): `SimpleAttLayer`, `HeteGAT_multi`, `ex_acm3025.py`, `jhyexp.py`. Paper-global versus release per-node attention is a documented discrepancy.
- [DGL ACM3025](https://data.dgl.ai/dataset/ACM3025.pkl): hash-verified mirror; original MAT byte identity unestablished.

## L093 · Heterogeneous Graph Transformer

- [Hu et al., WWW2020](https://arxiv.org/html/2003.01332v1): Figure2,§§3–4 and Table2. Typed attention, relative time, sampling and named CS Paper-Field L2 target.
- [Audited modern OAG release](https://github.com/acbull/pyHGT/tree/85eaccd482bc1d1af56c2de297b6e3a88b96d5cd/OAG): executable source operator and sampler oracle.
- [Publication-era source](https://github.com/acbull/pyHGT/tree/fd4a244db8efc72410537f3effec3b0c432892f7): historical comparison; differs from modern architecture.
- [Authors' OAG files](https://drive.google.com/drive/folders/1a85skqsMBwnJ151QpurLFSa9o2ymc_rq): NN and CS snapshots acquired and hashed; paper-era identity unestablished.

## L094 · HIN taxonomy and evidence accounting

- [Dong, Hu, Wang, Sun and Tang (2020), Heterogeneous Network Representation Learning](https://web.cs.ucla.edu/~yzsun/papers/2020_IJCAI_HIN_Survey.pdf): §1 schema, §3 representations, §4 and Table1. Corrects the roadmap citation placeholder.
- [metapath2vec author release](https://ericdongyx.github.io/metapath2vec/m2v.html): typed walks and lookup embeddings as a contrast to GNN encoders.
- [Pinned OAG source](https://github.com/acbull/pyHGT/tree/85eaccd482bc1d1af56c2de297b6e3a88b96d5cd/OAG): relation storage and released graph provenance; historical snapshot equality remains unestablished.

## L095 · Bipartite recommendation contract

- [GroupLens ML-100K release](https://grouplens.org/datasets/movielens/100k/) and [README](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt): complete counts and official five partitions. Data acknowledgment: Harper & Konstan (2015), DOI 10.1145/2827872.
- [PyG heterogeneous tutorial](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html): typed stores and bipartite identity.
- [PyG 2.8.0 RandomLinkSplit](https://pytorch-geometric.readthedocs.io/en/2.8.0/generated/torch_geometric.transforms.RandomLinkSplit.html): paired reverse relations; is_undirected alone does not cover bipartite types.
- [Local full-run protocol](labs/l095-reproduction.md): complete release audit plus explicitly course-defined walk and ranking experiment.

## L096 · SQL foreign keys to typed graphs

- [SQLite foreign keys](https://www.sqlite.org/foreignkeys.html): primary reading, NULL/reference integrity and explicit enforcement.
- [PostgreSQL 18 constraints](https://www.postgresql.org/docs/18/ddl-constraints.html): primary/composite keys and FK constraints.
- [SQLite SELECT](https://www.sqlite.org/lang_select.html): join rows, aggregation and outer joins.
- [PyG 2.6.1 HeteroData](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.data.HeteroData.html): typed stores and edge indices.
- [Fey et al. 2023](https://arxiv.org/abs/2312.04615): forward reading for relational graph construction, not a reproduced model target.
- [Full course contract](labs/l096-reproduction.md): archived source hashes and all executed coverage.

## Lesson 097 · Negative sampling

- [Rendle et al., BPR](https://arxiv.org/abs/1205.2618), §4.1–4.3 and §5.1: pairwise preference objective and factorized scoring; historical experiment is not replayed by the course ablation.
- [Krichene & Rendle, On Sampled Metrics for Item Recommendation (extended abstract)](https://www.ijcai.org/proceedings/2021/0651.pdf): sampled evaluations need not preserve model comparisons.
- [PyG 2.6.1 negative-sampling implementation](https://raw.githubusercontent.com/pyg-team/pytorch_geometric/2.6.1/torch_geometric/utils/_negative_sampling.py): bipartite support and provided-edge exclusion.
- [GroupLens ML-100K README](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt): complete release and official five-fold split contract; byte identities in `labs/_sources_l097.json`.

## Lesson 098 · Heterogeneous mini-batching

- [PyG NeighborLoader API](https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/loader/neighbor_loader.html): typed fanout, seed prefix, identity and time contract.
- [Pinned RelBench entity example](https://github.com/stanford-star/relbench/blob/584a03d518b2b655580ea8e1cfbbb26bec0a2841/examples/gnn_entity.py): real framework integration; not a benchmark replay.
- [Full course protocol](labs/l098-reproduction.md), archived source hashes and complete executed coverage.

## Lesson 099 · Controlled architecture comparison

- [R-GCN, Schlichtkrull et al.](https://arxiv.org/abs/1703.06103v4): Eq2–3, Table2 AIFB; named ten-run port replay.
- [HGT, Hu et al.](https://arxiv.org/abs/2003.01332v1): §3 operators, Table2 CS; full-setting track remains unrun.
- [Pinned ACM raw loader](https://github.com/dmlc/dgl/blob/3d16000b4170fa741ed9e9667f22ba84d3493026/examples/pytorch/han/utils.py): conference mapping and typed incidence; course split/trainer explicitly differ.
- [L099 protocol and evidence](labs/l099-reproduction.md):24 complete course fits, paired attention intervention, independent checkpoint/metric audit and separate published targets.

## Lesson 100 · Integrated heterogeneous checkpoint

- [R-GCN, Eq2 and Table2](https://arxiv.org/abs/1703.06103v4): relation-wise message normalization; separate full AIFB release-protocol replay.
- [HGT, §3 and Table2](https://arxiv.org/abs/2003.01332v1): typed attention, temporal encoding and HGSampling; source-era differences retained.
- [Native NeighborLoader](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.NeighborLoader): typed seed prefix, local/global IDs and directional neighborhoods. Executed wrapper/source hashes archived.
- [L100 full reproduction ledger](labs/l100-reproduction.md): frozen24-fit course protocol, complete AIFB replay and unrun CS full-setting track.

## Lessons 091–100: whole-computation reading route

The [sequence map](reference/0091-0100-model-map.html) connects these sources to the expanded lesson walkthroughs. Paper equations, released implementations, course experiments and historical reproduction claims remain distinct.

- [R-GCN](https://arxiv.org/html/1703.06103v4), §2.1 Eq. 2 → §2.2 Eqs. 3–4 → §3: relation routing, sharing and the classification handoff. L091 derives a two-layer path and contrasts mean aggregation with count preservation.
- [HAN](https://arxiv.org/html/1903.07293v2), §3–4 and Figure 2, alongside [pinned SimpleAttLayer](https://github.com/Jhy1993/HAN/blob/71bac29a07fb8fab908d50a806a7bc38aa6c6611/utils/layers.py): neighborhood, semantic and class softmax axes; paper-global versus released node-wise semantic fusion.
- [HGT](https://arxiv.org/html/2003.01332v1), §3–4, alongside [pinned convolution](https://github.com/acbull/pyHGT/blob/85eaccd482bc1d1af56c2de297b6e3a88b96d5cd/OAG/pyHGT/conv.py): score/value branches, type/relation sharing, residual update, temporal inputs and sampling.
- [metapath2vec author project](https://ericdongyx.github.io/metapath2vec/m2v.html), sections C–F: constrained walk generation → embedding objective → stored vectors → downstream labels. L094 adds a conceptual architecture map; no new training or paper reproduction is claimed.
- [MovieLens 100K README](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt): released row and split contracts; the three-hop scorer and candidate policy in L095 are course choices.
- [Relational Deep Learning](https://arxiv.org/abs/2312.04615) and [SQLite foreign keys](https://www.sqlite.org/foreignkeys.html): database motivation and referential semantics for L096; the SQL/graph audit is a construction exercise, not a model benchmark.
- [BPR](https://arxiv.org/abs/1205.2618), §4–5.1: pairwise ranking and factorization. L097 derives gradients and distinguishes sampling distributions from evaluation candidates; the course optimizer is not the historical LearnBPR protocol.
- [PyG 2.6.1 NeighborLoader source](https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/loader/neighbor_loader.html): typed fanouts, local IDs, seed prefix and query timing for L098/L100. The executed native runtime remains separately pinned in each lesson's source manifest.

## L101 — temporal query boundaries

- Fey et al. (ICML 2024), [position paper](https://proceedings.mlr.press/v235/fey24a.html): sections 3.2–3.3, Appendix A–B, temporal neighborhoods. Appendix A filters neighbor timestamps; Algorithm 1 prints a receiver timestamp. L101 follows the former.
- Robinson et al., [RelBench v1 Table 4](https://arxiv.org/html/2407.20060v1#S5.T4): complete five-heuristic driver-position slice replayed.
- [Pinned v1.1.0 baseline source](https://github.com/stanford-star/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/examples/baseline_node.py): validation train-only, test train+validation; entity cold starts map to zero.

## L102 · Temporal Graph Networks · 2026-09-22

- Rossi et al. (2020), [Temporal Graph Networks, v3](https://arxiv.org/html/2006.10637v3): §§3.1–3.2, Figure 2, Appendix A.2 and Wikipedia TGN-attn Table 2. Primary reading for memory and delayed training.
- [Publication-era source](https://github.com/twitter-research/tgn/tree/e38cdf85998c6ca077167610dc4e769a688efa95): pinned code, exact negative sampling, evaluator and checkpoint behavior.
- [JODIE Wikipedia data](https://snap.stanford.edu/jodie/): complete raw events; SHA-256 pinned in the L102 provenance manifest.
- [PyTorch GRUCell](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRUCell.html): gate convention used by the visible implementation.

## Lesson 103 — TGAT (2026-09-26)

- Xu et al. (ICLR 2020), [Inductive Representation Learning on Temporal Graphs](https://arxiv.org/html/2002.07962v1): §§3.1–3.4 functional time encoding and recursive attention; Tables 1/2 Wikipedia AP targets; Appendix A.5 implementation details.
- [Publication-era implementation](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs/tree/9293d10d1943c4bd4a186337cf38ba98e4c8bb99): original model/sampler/trainer, locally fetched by hash; no upstream license supplied.
- [SNAP/JODIE Wikipedia](https://snap.stanford.edu/jodie/): raw interaction stream shared with L102. Exact data and protocol identities are recorded in `labs/l103-reproduction.md`.

## L104 — temporal leakage audit

- [Kapoor & Narayanan, 2022 preprint](https://arxiv.org/abs/2207.07048): taxonomy L2/L3.1 and model info sheets. [Patterns 2023 article](https://doi.org/10.1016/j.patter.2023.100804). L104 operationalizes the audit; no civil-war reproduction claim.
- [Fey et al., ICML 2024](https://proceedings.mlr.press/v235/fey24a.html): §2.2 target windows, §3.3 time-consistent graphs, Appendix A temporal neighborhoods. Snapshot ≤ convention is distinguished from TGAT's strict before-event convention.
- [Xu et al., ICLR 2020](https://arxiv.org/html/2002.07962v1): §3.2 recursion and Wikipedia Tables 1/2. L104 reuses L103-trained checkpoints for fresh inference and a separate leakage intervention; source/data identities are frozen.

## Lesson 105 · continuous time / event streams versus snapshots

- [TGN v3 §2: Dynamic Graphs](https://arxiv.org/html/2006.10637v3#S2): primary definitions; timed interactions versus graph snapshots.
- [JODIE authors’ Wikipedia data](https://snap.stanford.edu/jodie/) and [format](https://github.com/claws-lab/jodie#dataset-format): complete hashed release; cite Kumar, Zhang & Leskovec, KDD 2019.
- [Pinned TGN preprocessing](https://github.com/twitter-research/tgn/blob/e38cdf85998c6ca077167610dc4e769a688efa95/utils/preprocess_data.py): executed typed-ID bijection check only; no new model-score parity claim.
- [Lesson](lessons/0105-continuous-time.html), [student lab](labs/0105-continuous-time.ipynb), [reference](reference/event-stream-snapshots.html), [reproduction contract](labs/l105-reproduction.md). Complete three-width course audit, distinct from full-paper reproduction.


## Lesson 106 · temporal link evaluation

- Poursafaei, Huang, Pelrine & Rabbany (NeurIPS 2022), [Towards Better Evaluation for Dynamic Link Prediction](https://arxiv.org/html/2207.10128v2): §§4–6, Appendix B Wikipedia targets. Read with the [publication-era released implementation](https://github.com/fpour/DGB/tree/7793e9449f5321c7e39b24c0585e3c3de7cf9f5e/EdgeBank/link_pred).
- [JODIE data](https://snap.stanford.edu/jodie/): complete authenticated Wikipedia event stream. Historical table-generation bytes remain unestablished.


## Lesson 107 · Snapshot methods

- Pareja et al. (AAAI 2020), [EvolveGCN](https://arxiv.org/html/1902.10191v3): §3 state evolution and §4/Table 2 SBM experiment.
- [Publication-era IBM source](https://github.com/IBM/EvolveGCN/tree/3f4996ac2a742a69fe6ce6e378b6317518bd99bf): H/O, released configurations, full SBM data, original sampler and metric definitions. Source O uses GRU-style recurrence, not paper LSTM.
- [TGN §3](https://arxiv.org/html/2006.10637v3#S3) and [JODIE data](https://snap.stanford.edu/jodie/): event-model baseline and shared Wikipedia raw input.


## Lesson 108 — temporal sampling

- Xu et al., *Inductive Representation Learning on Temporal Graphs*, ICLR 2020: [§3.4 and Appendix A.6](https://arxiv.org/html/2002.07962v1), [pinned released sampler](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs/blob/9293d10d1943c4bd4a186337cf38ba98e4c8bb99/graph.py). Used for recursive time cutoffs and uniform fanout; new strict-past/window sampler is a course extension.
- [L108 reproduction contract](labs/l108-reproduction.md): fixed checkpoint cohort, full evaluation, source quirks, paired questions and timing boundaries.


## Lesson 109 — database timestamp contracts

- [RelBench v1 §2, §5.2/Table 4](https://arxiv.org/html/2407.20060v1): full selected driver-position heuristic slice.
- [Pinned RelBench task SQL](https://github.com/snap-stanford/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/relbench/tasks/f1.py) and [dataset clock construction](https://github.com/snap-stanford/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/relbench/datasets/f1.py).
- [Fowler, Bitemporal History](https://martinfowler.com/articles/bitemporal-history.html): separate effective and record histories.
- [SQL Server temporal tables](https://learn.microsoft.com/en-us/sql/relational-databases/tables/temporal/overview?view=sql-server-ver17): system-managed history and its time semantics.
- [L109 reproduction contract](labs/l109-reproduction.md): data/source hashes, complete labels/scores and explicit scope boundaries.


## Lesson 110 — temporal GNN checkpoint

- [Rossi et al., TGN v3](https://arxiv.org/abs/2006.10637v3): §3 memory/message/embedding computation; Table 2 Wikipedia TGN-attn targets.
- [Pinned original trainer](https://github.com/twitter-research/tgn/blob/e38cdf85998c6ca077167610dc4e769a688efa95/train_self_supervised.py): checkpoint selection, snapshot branch resets and early stopping.
- [Original memory object](https://github.com/twitter-research/tgn/blob/e38cdf85998c6ca077167610dc4e769a688efa95/modules/memory.py): queued messages and their relationship to state_dict.
- [L110 reproduction contract](labs/l110-reproduction.md): fresh paired full-data runs, strict-time audit, source/data identities, cost ceiling and explicit paper/course boundaries.


### Lesson 112 · OGB GCN reproduction

- [Hu et al., Open Graph Benchmark, v6 §4.3/Table 6](https://arxiv.org/html/2005.00687v6#S4.SS3): selected ogbn-arxiv GCN target and task framing.
- [Pinned OGB implementation](https://github.com/snap-stanford/ogb/tree/61e9784ca76edeaa6e259ba0f836099608ff0586/examples/nodeproppred/arxiv): full model, schedule and logger selection.
- [Dataset protocol](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) and [leaderboard](https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv).
- Local [protocol/evidence](labs/l112-reproduction.md) and [reference](reference/ogb-reproduction.html).

## Lesson 113 · Scaling OGB (2026-09-26)

- Primary: Hu et al., [OGB v6 §4.1/Table4](https://arxiv.org/html/2005.00687v6#S4.SS1). The products ClusterGCN entry uses GraphSAGE aggregation; preserve the sales-ranking split and ten-run report.
- Mechanism: Chiang et al., [Cluster-GCN](https://arxiv.org/abs/1905.07953); Hamilton et al., [GraphSAGE](https://arxiv.org/abs/1706.02216). Sampling and aggregation are separate choices; the original Cluster-GCN Amazon split differs from OGB's split.
- Executable source: [OGB commit cf066f9](https://github.com/snap-stanford/ogb/blob/cf066f93311ab3099cad84d71085d1b0375dcc2e/examples/nodeproppred/products/cluster_gcn.py), source/license retained under labs/sources/l113. [PyG2.6.1 cluster implementation](https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/loader/cluster.html) supplies METIS and induced-batch infrastructure.
- Local package: [protocol](labs/l113-reproduction.md), [lesson](lessons/0113-scaling-ogb.html), [reference](reference/scaling-ogb.html). Original partition/seeds are unavailable; historical identity is not established.

## Lesson 114 · OGB error analysis (2026-09-26)

- Primary task and aggregate targets: Hu et al., [OGB v6 §4.3 / Table6](https://arxiv.org/html/2005.00687v6#S4.SS3). Selected MLP reproduction; the slice study is new course analysis.
- [Pinned MLP source](https://github.com/snap-stanford/ogb/blob/61e9784ca76edeaa6e259ba0f836099608ff0586/examples/nodeproppred/arxiv/mlp.py): three-layer network, train-row-only BN, full500epoch schedule, ten runs. Original seed list is unavailable.
- [Official arxiv data contract](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv): features, categories and time split. This does not imply a historically censored graph.
- [Protocol and evidence](labs/l114-reproduction.md) · [Error-analysis reference](reference/ogb-error-analysis.html). Complete graph/metric reconstruction distinguishes training variation from independent-node inference.


## L115 · Graph ML design patterns and modular GCN reproduction

- [Battaglia et al., 2018, §4.3 and Figure6b](https://arxiv.org/html/1806.01261v3#S4.SS3): composable graph blocks and encode-process-decode. Primary conceptual source; our modular GCN is a course decomposition, not a reproduction of the paper's demos.
- [Gilmer et al., ICML2017, §2](https://proceedings.mlr.press/v70/gilmer17a.html): message-passing and graph-level readout responsibilities.
- [Hu et al., OGB v6, §4.3/Table6](https://arxiv.org/html/2005.00687v6#S4.SS3): named GCN ogbn-arxiv experiment, mean validation73.00%,test71.74%.
- [Pinned OGB source](https://github.com/snap-stanford/ogb/blob/61e9784ca76edeaa6e259ba0f836099608ff0586/examples/nodeproppred/arxiv/gnn.py): all three GCN layers, full-node BN, train-label loss and released defaults. MIT snapshot in labs/sources/l115.
- [L115 protocol/evidence](labs/l115-reproduction.md): ten fresh fits, independent metrics and complete original-model replay. CLOSE is numerical proximity, not historical/whole-paper identity.

## Lesson 116 — diagnose GNN training with falsifiable probes

- [Pinned OGB GCN training loop](https://github.com/snap-stanford/ogb/blob/61e9784ca76edeaa6e259ba0f836099608ff0586/examples/nodeproppred/arxiv/gnn.py): primary line-by-line reading; train-label loss, backward, optimizer update and validation selection. Fresh source-byte check in labs/_upstream_l116_results.json.
- [OGB v6 §4.3/Table 6](https://arxiv.org/html/2005.00687v6#S4.SS3): full selected ogbn-arxiv GCN target; ten fresh fits in the L116 protocol.
- [Li, Han and Wu, AAAI 2018](https://arxiv.org/abs/1801.07606): Laplacian smoothing interpretation and limits; used for mechanism teaching, not a claim to reproduce their co-/self-training results.
- [PyG 2.6.1 neighbor-sampling tutorial](https://pytorch-geometric.readthedocs.io/en/2.6.1/tutorial/neighbor_loader.html): n_id mapping and seed-only loss; independently checked against a real NeighborLoader batch.
- [PyTorch autograd and evaluation modes](https://docs.pytorch.org/docs/stable/notes/autograd.html): gradient recording and module evaluation mode are distinct.
- [L116 protocol](labs/l116-reproduction.md): pinned source/data, diagnostic interventions, benchmark results, budget and exact commands.

## Lesson 117 — RDL bridge and selected full-data reproduction

- [Fey et al., ICML 2024 final paper](https://proceedings.mlr.press/v235/fey24a.html): complete reading; §§2–3 define query tables, schema/REG/computation graphs and temporal message passing; §4 research agenda; §5 history; §6/Appendices C–D beta benchmark. Algorithm1's printed receiver-time filter is distinguished from AppendixA's neighbor-time definition.
- [Robinson et al., RelBench v1](https://arxiv.org/html/2407.20060v1): §3 implementation; Table7 selected F1 regression result; Table9 protocol. Five fresh complete release runs: validation3.18180±.04348, test4.13392±.15660 MAE; both descriptively CLOSE. This is a companion experiment, not the Fey beta table or whole paper.
- [Pinned released trainer](https://github.com/snap-stanford/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/examples/gnn_node.py): actual [128,64] fanout, query labels, mean L1, train-percentile clipping and stochastic validation selection. Full sources, licenses, GloVe revision and primitive source files are in labs/sources/l117.
- [L117 exact commands and deviations](labs/l117-reproduction.md): full released-data experiment, explicit non-train-only preprocessing and missing ingestion-history boundary; independent SQL/key/query/score audit and original-model replay.


## Lesson 118 — Cvitkovic relational GNN

- [2019 workshop paper](https://rlgm.github.io/papers/55.pdf) and [expanded 2020 arXiv v1](https://arxiv.org/abs/2002.02046v1): version distinction, Table1, Algorithm1, Table4 and AppendixC.
- [Author code at 57195cc](https://github.com/mwcvitkovic/Supervised-Learning-on-Relational-Databases-with-GNNs/tree/57195ccab62d23dcbcac1a317f8a9811a9fd6cb5): actual Home Credit query, encoders, GCN, graph pooling, split and training schedule.
- [DGL v0.3.1](https://github.com/dmlc/dgl/tree/v0.3.1/python/dgl/nn/pytorch): GraphConv and GlobalAttentionPooling original Python source; checked with an independent dense adapter, not the historical binary.
- [Home Credit competition](https://www.kaggle.com/competitions/home-credit-default-risk/data): authenticated raw download, locally retained.
- [Protocol and cost boundary](labs/l118-reproduction.md): source checks and real-data pilot completed; full five-fold experiment NOT_RUN.

## Lesson 119 · Year 3 synthesis (2026-09-27)

- [Fey et al., ICML2024 relational deep learning blueprint](https://proceedings.mlr.press/v235/fey24a.html): primary synthesis reading.
- [Xu et al., How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826): aggregation expressiveness and local-message-passing limits; the lesson's cycle/triangles and order examples are course constructions.
- [RelBench v1 Table7 and AppendixB](https://arxiv.org/html/2407.20060v1): complete selected F1 RDL replay; five NEW runs, test4.01457±.12673 MAE, historical identity and whole-paper parity NOT_ESTABLISHED.
- [Pinned RelBench release](https://github.com/snap-stanford/relbench/tree/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639): reused L117 model/trainer, separate L119 evidence and budget.

## Lesson 120 · Year 3 exit exam

- Fey et al. ICML2024 final paper, https://proceedings.mlr.press/v235/fey24a.html — complete reading/claim ledger, including appendices.
- RelBench v1, https://arxiv.org/html/2407.20060v1 — selected Table7 F1 RDL and AppendixB protocol; five fresh runs.
- PyG batching, https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html — disjoint batches and type-specific offsets.

### Lesson 121 · history and representation choices

- [Muggleton1991, Inductive Logic Programming](https://www.doc.ic.ac.uk/~shm/Papers/ilp.pdf), §§1–2: distinguish learning relational rules from executing supplied ones.
- [Lavrač and Flach2001, An extended transformation approach to ILP](https://research-information.bris.ac.uk/en/publications/an-extended-transformation-approach-to-inductive-logic-programmin/): authors' institutional abstract grounds propositionalization as relational-to-attribute transformation.
- [Kanter and Veeramachaneni2015, Deep Feature Synthesis](https://www.jmaxkanter.com/papers/DSAA_DSM_2015.pdf), §II/Algorithm1: composed feature primitives. [Official Featuretools guide](https://featuretools.alteryx.com/en/stable/getting_started/afe.html) supplies nested aggregation examples. A course feature recipe is not full DFS search reproduction.
- [Lam et al.2018, Neural Feature Learning From Relational Database](https://arxiv.org/abs/1801.05372): earlier neural relational work, included as a historical branch rather than reproduced here.
- Cvitkovic [workshop2019](https://rlgm.github.io/papers/55.pdf) versus [expanded2020](https://arxiv.org/html/2002.02046v1): use the latter's Table4 for Home Credit GCN .780±.004. Selected full experiment remains NOT_RUN under the compute cap; local extraction/timing evidence is separate.


## Lesson 122 · construct and verify a REG

- [Fey et al., ICML2024, §3.1–3.2/Figure4](https://proceedings.mlr.press/v235/fey24a.html): schema graph versus entity graph versus query computation; row-to-node representation.
- [Pinned RelBench graph constructor](https://github.com/snap-stanford/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/relbench/modeling/graph.py): consecutive-key requirement, FK-column relation names, reverse edges, key removal and constant feature fallback.
- [PyG2.6.1 HeteroData](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.data.HeteroData.html): typed stores, explicit node counts and edge-index coordinates.
- [RelBench v1 Table7/AppendixB](https://arxiv.org/html/2407.20060v1): separate full selected F1 RDL replay, five fresh runs. Protocol and actual evidence: [L122](labs/l122-reproduction.md).


## Lesson 123 · Temporal heterogeneous graphs (2026-09-27)

- Fey et al., ICML2024, Sections3–4: https://proceedings.mlr.press/v235/fey24a.html — primary blueprint for temporal REG neighborhoods.
- Robinson et al., RelBench v1 Sections2–3 / Table7 / AppendixB: https://arxiv.org/html/2407.20060v1 — selected F1 driver-position RDL target and release deviations.
- RelBench pinned constructor/loader: https://github.com/snap-stanford/relbench/tree/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639 — node timestamps, reverse stores, input_time and released fanouts.
- PyG2.6.1 neighbor sampler: https://github.com/pyg-team/pytorch_geometric/blob/2.6.1/torch_geometric/sampler/neighbor_sampler.py — disjoint temporal components.
- pyg-lib0.4.0 CPU neighbor kernel: https://github.com/pyg-team/pyg-lib/blob/0.4.0/pyg_lib/csrc/sampler/cpu/neighbor_kernel.cpp — inclusive timestamp upper bound and root query identity.
- Executed protocol: labs/l123-reproduction.md. Availability-time fixture is an explicit course extension; F1 ingestion histories are unavailable.

## Lesson 124 · Entity vs task table

- Fey et al. (2024), RDL blueprint: https://proceedings.mlr.press/v235/fey24a.html — reusable REG versus prediction task.
- Robinson et al., RelBench v1: https://arxiv.org/html/2407.20060v1 — named F1 Table7 target; not the Fey beta benchmark.
- RelBench commit9aa346267c2e1c560bd92da07d6f4ad1ca2f0639, task SQL and BaseTask split grid: labs/sources/l124/; hashes in labs/_sources_l124.json. The released one-year cohort condition has no upper bound; independent full-table reconstruction and past-only intervention are documented separately.

## Lesson 125 · PyTorch Frame and the row-to-graph contract · 2026-09-27

- Hu et al., [PyTorch Frame](https://arxiv.org/html/2404.00776v2), §§3–4: materialization, semantic-type encoding, column interaction and decoding. §5.3/Table2 names the historical rel-stackex-engage target (ROC-AUC .854).
- [Frame 0.3.0 source](https://github.com/pyg-team/pytorch-frame/tree/d998aae368db6a4e36139ccc56bd54579a70874b): pinned operator/model reference, not historical runtime identity. Local source manifest: `labs/_sources_l125.json`.
- Contemporaneous RelBench source preserved in `labs/sources/l125/relbench/`; original archive hashes and six HTTP404 probes in `labs/_paper_audit_l125_results.json`. No matching original data recovered; exact Table2 protocol unestablished; full target NOT_RUN.
- [Lesson](lessons/0125-pytorch-frame-deep-dive.html), [reference](reference/pytorch-frame-deep-dive.html), [contract](labs/l125-reproduction.md). Complete F1 feature path is separate mechanism evidence.

## Lesson 126 — RelBench beta package and historical contract (2026-09-27)

- [Fey et al., arXiv:2312.04615v1](https://arxiv.org/html/2312.04615v1), §4/Figure5: two databases, four tasks, standardized loading/splits/evaluation. Read the complete paper with the lesson reading ledger. No numerical results table in this version.
- [Beta source0433616e](https://github.com/stanford-star/relbench/tree/0433616ee94003fb15a4ac4d633e499d0f129077): package0.1.1, pre-publication snapshot. Task horizon730days; AP; separate masked/full test tables. Source files and license preserved.
- [Average precision definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html): recall-increment weighting without trapezoidal interpolation.
- [Protocol](labs/l126-reproduction.md): original archives missing; synthetic source checks and complete modernF1 API tour are separate evidence.

## Lesson 127 · RelBench v1 baseline and experiment audit

- Robinson et al., [RelBench v1](https://arxiv.org/html/2407.20060v1), §2–3, Table1, Table7 and AppendixB: seven-database overview and complete selected F1 RDL replay. Five fresh ten-epoch runs, test3.967165±.093906MAE; whole-paper/historical identity NOT_ESTABLISHED.
- [Pinned released implementation](https://github.com/snap-stanford/relbench/tree/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639): source/AST/gradient/update parity, full query scoring and temporal batch checks. [L127 protocol](labs/l127-reproduction.md) records fanout/preprocessing differences and exact commands.
- [Modal pricing](https://modal.com/pricing), checked2026-09-27: USD10 aggregate plan; bounded pilot plus five fits, no automatic retries. Worker time estimates exclude unitemized billing.


## Lesson 128 · Task contracts and label provenance
- Robinson et al., RelBench v1 https://arxiv.org/html/2407.20060v1 — §5, Tables2/6/8/13; classification/regression/recommendation contracts and five-run target.
- Pinned RelBench API and examples, commit9aa346267c2e1c560bd92da07d6f4ad1ca2f0639: labs/sources/l128/manifest.json. Includes task enum, entity/recommendation evaluators, metrics and two-tower BPR training.
- Upstream label-flip commit https://github.com/snap-stanford/relbench/commit/c348273a8e66 and pre-flip parent e416c3d208cb8a6d8bd31a68c46ca9090ecb334a. Every recovered historical key/label matches original SQL; archive bytes/order unestablished.

## Lesson129 · Manual feature engineering and the expert study

- [RelBench v1§6 and AppendixC](https://arxiv.org/html/2407.20060v1#S6): expert protocol, marginal human work, exclusions and normalized Figure3 comparison. Do not substitute Table7 raw-entity LightGBM.
- [Released manualFE repository](https://github.com/snap-stanford/relbench-user-study/tree/445bb7a3b1230f49f8e5890ae81754d3e365680f): complete F1 SQL, schemas, trainer and worked notebook.
- [PyTorch Frame0.2.2 LightGBM](https://github.com/pyg-team/pytorch-frame/blob/56f687ddf4bf1c4a7d7b72ab0ef4117493256199/torch_frame/gbdt/tuned_lightgbm.py): input conversion, eight tuning dimensions, L1 objective, early stopping and refit. Local source-parity pilot matches exactly.
- Local protocol: labs/l129-reproduction.md; full evidence: labs/evidence/l129/summary.json. Historical staging404; pinned v1 archives substituted with independent full SQL/label audit.


## Lesson 130 · Q1 full RDL checkpoint

- [Fey et al., RDL blueprint](https://proceedings.mlr.press/v235/fey24a.html): table rows, foreign-key graph, temporal sampling and learned prediction chain.
- [Robinson et al., RelBench v1](https://arxiv.org/html/2407.20060v1), §3/Table7/AppendixB: full selected F1 driver-position basic-RDL target; five fresh ten-epoch runs, validation3.164116±.038287/test4.070921±.075000MAE, descriptive CLOSE under0.2 tolerance.
- [Pinned source and protocol](labs/l130-reproduction.md), [fresh evidence](labs/evidence/l130/summary.json), and [reused manual-FE comparison](labs/evidence/l130/comparison.json). Complete selected released-protocol replay does not establish historical or whole-paper identity.

## Lesson 131 — measured GNN + tabular stack

- [Fey et al.2024, RDL blueprint](https://proceedings.mlr.press/v235/fey24a.html), §5: row representation, temporal heterogeneous message passing and task prediction.
- [Robinson et al.2024, RelBench v1](https://arxiv.org/html/2407.20060v1), §3,Table7,AppendixB: selected full F1 driver-position reproduction.
- [Pinned released model](https://github.com/snap-stanford/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/examples/model.py) and [typed encoders/GNN](https://github.com/snap-stanford/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/relbench/modeling/nn.py): exact instrumented computation; historical training identity unestablished.
- Local [protocol](labs/l131-reproduction.md), [trace audit](labs/_trace_audit_l131_results.json) and [missing-value diagnostic](labs/_missing_gradient_l131_results.json). Matched nonfinite gradients are not hidden by a parity claim.


## Lesson132 · Identity-aware message passing

- [You et al., Identity-aware Graph Neural Networks](https://proceedings.aaai.org/index.php/AAAI/article/view/17283): root-aware message functions and expressiveness.
- [RelBench v1, Table8 and AppendixB](https://arxiv.org/html/2407.20060v1): condition-sponsor-run comparison and protocol.
- [Pinned released models](https://github.com/snap-stanford/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/examples/model.py): distinguish additive root marking from original ID-GNN.
- [Local protocol and evidence boundaries](labs/l132-reproduction.md): two feasibility pilots completed; full selected reproduction INCOMPLETE under the USD10 cap.

## Lesson133 · Heterogeneous convolution on REG

- [Pinned RelBench HeteroGraphSAGE](https://github.com/snap-stanford/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/relbench/modeling/nn.py): separate per-relation SAGE modules, outer sum and node LayerNorm.
- [PyG2.6.1 HeteroConv](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.conv.HeteroConv.html) and [SAGE source](https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/nn/conv/sage_conv.html): bipartite indices, relation grouping, root transform and bias.
- [RelBench v1 paper](https://arxiv.org/html/2407.20060v1), implementation section and Table7: named full-data F1 driver-position target. Protocol/deviations and fresh evidence in labs/l133-reproduction.md.

## Lesson134 · Training at scale

- RelBench v1: https://arxiv.org/html/2407.20060v1 — selected Table7 F1 reproduction plus a separately scoped full-topology systems workload.
- Released temporal loader: https://github.com/snap-stanford/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/examples/gnn_node.py — uniform sampling, decreasing fanouts and query transforms.
- PyG2.6.1 sampler: https://github.com/pyg-team/pytorch_geometric/blob/2.6.1/torch_geometric/sampler/neighbor_sampler.py — temporal disjoint sampling, local/global coordinates and query ownership.
- CUDA timing semantics: https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution — synchronization is required for meaningful device wall time.
- Exact scope/source/deviations: labs/l134-reproduction.md; pinned scale sources labs/sources/l134/scale/manifest.json.

## Lesson 135 — Tuning on the REG

- [RelBench v1 Appendix B.2 and Table 7](https://arxiv.org/html/2407.20060v1#A2): default hyperparameters and five-run selected baseline targets; the course search is a separate extension.
- [Pinned released node trainer](https://github.com/snap-stanford/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/examples/gnn_node.py): actual fanout schedule, clipping and checkpoint selection.
- [Modal resource prices](https://modal.com/pricing), checked 2026-09-27: aggregate budget calculation including CPU, memory, checks and retry reserves.

## Lesson 136 — Leaderboard literacy

- [Pinned RelBench submission implementation](https://github.com/stanford-star/relbench/blob/584a03d518b2b655580ea8e1cfbbb26bec0a2841/relbench/submit.py): exact query-key, finite-value, complete-board and aggregation contract.
- [Pinned metric source](https://github.com/stanford-star/relbench/blob/584a03d518b2b655580ea8e1cfbbb26bec0a2841/relbench/metrics.py): NMAE uses the resolved train-target scale. Hosted constants audited against pinned raw train tables.
- [Kapso submission393](https://github.com/stanford-star/relbench/issues/393), [GNN380](https://github.com/stanford-star/relbench/issues/380), [RT-PluRel397](https://github.com/stanford-star/relbench/issues/397): exact attached predictions, distinct from their generating training procedures.
- [RelBench v1 Table7](https://arxiv.org/html/2407.20060v1): named historical full-data five-seed RDL target; raw MAE distinct from the current normalized leaderboard.
- Frozen URLs and SHA256 values, including Kapso's temporal-regime document: `labs/_sources_l136.json`. Exact execution, provenance discrepancies and boundaries: `labs/l136-reproduction.md`.


## Lesson 137 — Error analysis on REG

- [RelBench v1, Table7 and expert-feature study](https://arxiv.org/html/2407.20060v1): selected basic RDL target plus distinct manual-FE pipeline provenance.
- [Released user-study SQL](https://github.com/snap-stanford/relbench-user-study/blob/445bb7a3b1230f49f8e5890ae81754d3e365680f/f1/driver-position/feats.sql): strict-past standings, recent global-race slots and explicit schedule availability caveats.
- [Pinned GNN implementation](https://github.com/snap-stanford/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/examples/gnn_node.py). The lesson's slice protocol is an original course extension; source hashes and executed boundaries live in `labs/l137-reproduction.md`.


## Lesson138 — Amazon review churn

- [RelBench v1 §4/5.1/Table6](https://arxiv.org/html/2407.20060v1): named basic RDL user-churn target.
- [Pinned task SQL](https://github.com/snap-stanford/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/relbench/tasks/amazon.py): eligibility, horizon and exact endpoints.
- [Pinned dataset construction](https://github.com/snap-stanford/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/relbench/datasets/amazon.py): Books reviews, table schema and source limitations.

## Lesson 139 — Clinical-trial pipeline transfer

- [RelBench v1 Table6 and AppendixB.2](https://arxiv.org/html/2407.20060v1): study-outcome target and the trial-specific mean aggregation, learning rate, fanout and epoch exception.
- [Pinned clinical task SQL](https://github.com/stanford-star/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/relbench/tasks/trial.py): observed-query inclusion, numeric p-values, modifier behavior and365-day windows.
- [Pinned clinical dataset constructor](https://github.com/stanford-star/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/relbench/datasets/trial.py): retrospective completed-study cohort and timestamps inferred from start/completion dates. These do not establish recorded historical arrival times.
- Source hashes, independent audit and exact commands: `labs/sources/l139/manifest.json`, `labs/l139-reproduction.md`.

- L140 checkpoint: [RelBench v1 Table 6 and Appendix B.2](https://arxiv.org/html/2407.20060v1#A2.SS1), [pinned source trainer](https://github.com/snap-stanford/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/examples/gnn_node.py), and [Modal resource pricing](https://modal.com/pricing) checked2026-09-30. Use for two-task released-protocol reproduction, validation selection, seed variance and limits of historical claims.

## Lesson 141 · RelGNN atomic routes

- [ICML2025 RelGNN v2](https://arxiv.org/html/2502.06784v2): §§4.1–4.3 and Table2.
- [Pinned implementation](https://github.com/snap-stanford/RelGNN/tree/cffdb8b54627e92c7dd112c1243dde739c90d35b): route derivation, composite attention, inference configuration. Main entry point is checkpoint evaluation, not training.
- [Pinned checkpoint release](https://huggingface.co/tianlangchen/RelGNN/tree/321e6f6e7af5d7546b637f147783fc28ab5d4a7a): F1 driver-position weights; feature-type compatibility reconstruction documented in [protocol](labs/l141-reproduction.md).


## Lesson 142 — many-to-many edge pathology
- Chen et al., RelGNN v2 §§3.1–4.3 and Table2: https://arxiv.org/html/2502.06784v2 — bridge return paths, hub role mixing and composite routes; architectural motivation is not a universal information-loss theorem.
- Released code pinned cffdb8b54627e92c7dd112c1243dde739c90d35b: https://github.com/snap-stanford/RelGNN/tree/cffdb8b54627e92c7dd112c1243dde739c90d35b — historical training recipe incomplete; L142 ordinary edge-attention comparator is a course experiment.
- [Lesson](lessons/0142-many-to-many-edge-pathology.html) · [Full protocol](labs/l142-reproduction.md). Five full fits per arm, shared freshly materialized archives, independent keyed scoring and separate notebook-validation fits.

## Lesson 143 — RelGNN reproduction

- [RelGNN §4/§5.2/Table2](https://arxiv.org/html/2502.06784v2): composite computation and selected full F1 target3.798MAE.
- [Pinned released evaluation code](https://github.com/snap-stanford/RelGNN/tree/cffdb8b54627e92c7dd112c1243dde739c90d35b): architecture and checkpoint evaluation; historical training recipe incomplete.
- [Pinned checkpoint](https://huggingface.co/tianlangchen/RelGNN/tree/321e6f6e7af5d7546b637f147783fc28ab5d4a7a): qualifying numerical-buffer evidence for the documented compatibility reconstruction.
- [Protocol](labs/l143-reproduction.md): fresh full-task five-seed training, separate checkpoint replay, exact label/key audits and aggregate cost. Close scores do not establish historical identity.

## Lesson 144 · ContextGNN
- [Yuan et al., 2024, ContextGNN v1](https://arxiv.org/html/2411.19513v1): §§3–5; selected Table2 site-sponsor-run MAP targets.
- [Pinned authors implementation](https://github.com/kumo-ai/ContextGNN/tree/ca4a96985b7ef73c36a40da32e70710ad9b59a1e): local replacement, distinct branch offsets, full-catalog training and search loop. Release differences and evaluation-cache behavior are recorded in labs/l144-reproduction.md.

## Lesson 145 · RelGT
- [RelGT v1](https://arxiv.org/html/2505.10960v1): five-element tokens, local/global attention, selected F1 Table1.
- [Pinned source](https://github.com/snap-stanford/relgt/tree/19e423ca3e7cac761130aba790857f2dc3a46ef7): full nine-config search and confirmed cache/evaluation defects; see labs/l145-reproduction.md.

## Lesson 146 · GNN versus graph transformer

- [RelGT v1 §4 and Table6](https://arxiv.org/html/2505.10960v1#A4), checked2026-09-30: compare displayed validation/test rows without assuming historical selection intent.
- [Pinned RelGT release](https://github.com/snap-stanford/relgt/tree/19e423ca3e7cac761130aba790857f2dc3a46ef7): distinguish checkpoint selection in`main_node_ddp.py`from sweep launching. Source changes and reused-file hashes: [ledger](labs/evidence/l146/sources.json).
- [Reproduction and comparison contract](labs/l146-reproduction.md): full-data corrected course experiment, retained complete source schedules, cost cutoff and independent keyed evaluation. [Reference sheet](reference/gnn-vs-graph-transformer.html).

## Lesson 147 · next-generation architecture survey

- [Dwivedi et al., RDL: Challenges, Foundations and Next-Generation Architectures, v1](https://arxiv.org/html/2506.16654v1), 19June2025; checked2026-09-30. Read §§2.3–2.4,4.1,5. Scope: historical synthesis, not a current frontier census or a new experimental table.
- [L147 source/claim ledger](labs/evidence/l147/sources.json) pins fetched source bytes and separates survey proposals, local reanalysis and author planning judgments.
- [Executable research map](labs/evidence/l147/questions.json) and [reproduction boundary](labs/l147-reproduction.md). Six L146 fits reused; all7,554held-out predictions freshly rescored, no training/cloud cost.


## Lesson 148 · Ablation discipline (2026-09-30)

- RelBench v1, §3, Table 7 and Appendix B.3: https://arxiv.org/html/2407.20060v1 . Primary RDL pipeline, regression target and published ablation context. L148 freshly replays the F1 RDL baseline; its four controlled interventions are exploratory course extensions.
- Pinned RelBench release: https://github.com/snap-stanford/relbench/tree/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639 . Current byte audit in `labs/evidence/l148/sources.json`; exact protocol and limitations in `labs/l148-reproduction.md`.

## Lesson 149 · weakest RelBench tasks
- [RelBench v1 Figure3, §6 and AppendixC.2](https://arxiv.org/html/2407.20060v1#S6): published 15-task user study; boosted regression head must be separated from Table7 basic RDL. Figure SVG freshly checksum-verified2026-09-30; reconstructed means are PLOT_DERIVED, with unresolved user-votes/post-votes label mismatch.
- [Pinned FE release](https://github.com/snap-stanford/relbench-user-study/tree/445bb7a3b1230f49f8e5890ae81754d3e365680f) and [pinned RDL release](https://github.com/snap-stanford/relbench/tree/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639): complete fresh F1 selected-pipeline replay and separate exploratory slice profile. Ledger: labs/l149-reproduction.md.

## Lesson150 checkpoint sources · 2026-09-30

- [RelGNN v2 §5.2/Table2](https://arxiv.org/html/2502.06784v2#S5.T2): selected historical F1 target3.798 rawMAE. [Pinned release](https://github.com/snap-stanford/RelGNN/tree/cffdb8b54627e92c7dd112c1243dde739c90d35b); fresh source/hash audit in labs/evidence/l150/sources.json.
- [Current RelBench leaderboard](https://star-project.stanford.edu/relbench/leaderboard/) and [RelArena protocol](https://star-project.stanford.edu/relarena/): metric normalization and data-state/tuning/refit differences block a direct historical-current score comparison. Live JS/data snapshots retained; do not interpret the static empty fallback as no submissions.
- [Checkpoint lesson](lessons/0150-q3-reproduction-checkpoint.html), [protocol](labs/l150-reproduction.md), [report scaffold](labs/l150-report-template.md).


## Lesson 151 · Classification portfolio

- [RelBench v1 §5.1, Table6 and AppendixB.2](https://arxiv.org/html/2407.20060v1#A2): named RDL rel-trial/study-outcome experiment and trial-specific hyperparameters.
- [Pinned RelBench release](https://github.com/snap-stanford/relbench/tree/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639): exact model, graph and task provenance in labs/sources/l151; full protocol in labs/l151-reproduction.md.
- [Modal pricing](https://modal.com/pricing), verified2026-10-01: aggregate USD10 cap; failed audit packaging reservation retained.


## Lesson 152 · Regression portfolio

- [Robinson et al., RelBench v1 Table7 and Appendix B.2](https://arxiv.org/html/2407.20060v1#A2.T7): selected basic RDL driver-position regression experiment, full five-seed replay; see labs/l152-reproduction.md.
- [Pinned released F1 task SQL](https://github.com/snap-stanford/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/relbench/tasks/f1.py): target window, cohort and entity-cutoff identity. Full timestamp grid compared with independent pandas reconstruction and archive.
- [Gneiting and Resin, Regression Diagnostics meets Forecast Evaluation](https://arxiv.org/abs/2108.03210): calibration must match the elicited target functional. L152's simple fixed-bin median diagnostic is a course extension, not a reproduction of their estimation methods.
- [Modal pricing](https://modal.com/pricing), checked2026-10-01: T4+2physicalCPU+16GiB=.00022572USD/sec; aggregate USD10cap and immutable reservations.

### Lesson 153 · Ranking portfolio protocol

- [RelBench v1, Table 8 and Appendix B.2](https://arxiv.org/html/2407.20060v1#A2.T8). Primary target: GraphSAGE on rel-trial/site-sponsor-run, five-run MAP@10. Use with the pinned [trainer and loader](labs/sources/l153/manifest.json): full-catalog evaluation, BPR shared negatives, timestamp batching and tied validation selection. L153's measured cost gate leaves full reproduction INCOMPLETE; the pilot is separate evidence.


### Lesson154 · Portfolio synthesis sources

- [RelBench v1, Tables6–8 and Section6](https://arxiv.org/html/2407.20060v1#A2.SS1): published baseline identity, metric units and manual-FE study distinction. Retrieved/checked2026-10-01; frozen table excerpts and extracted context at `labs/evidence/l154/`.
- [L151–153 provenance and replay contract](labs/l154-reproduction.md): current author evidence, frozen inputs, local comparison gaps and budget-stopped recommendation lane. These do not establish historical identity or learner mastery.

## Lesson155 · human effort and matched FE comparison

Primary: [RelBench v1 Section6 and AppendixC](https://arxiv.org/html/2407.20060v1#S6); [released user study](https://github.com/snap-stanford/relbench-user-study/tree/445bb7a3b1230f49f8e5890ae81754d3e365680f). Marginal human work excludes reusable infrastructure. Full F1 computational replay is separate from original human-study replication. See labs/l155-reproduction.md.

## Lesson156 · temporal audit sources

- [RelBench v1 Section2 and Table7](https://arxiv.org/html/2407.20060v1): temporal split intent and selected basic-GNN target.
- [Pinned graph feature materialization](https://github.com/snap-stanford/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/relbench/modeling/graph.py): processor fitting population versus neighbor timestamps.
- [Pinned user-study F1 SQL](https://github.com/snap-stanford/relbench-user-study/blob/445bb7a3b1230f49f8e5890ae81754d3e365680f/f1/driver-position/feats.sql): strict historical joins,upcoming schedules and cohort-conditioned maxima; verified upstream bytes in labs/_sources_l156.json.

## Lesson157 · Reproducibility contribution

- [RelBench contribution guide](https://github.com/stanford-star/relbench/blob/main/CONTRIBUTING.md), checked2026-10-01: current development workflow and small synthetic regression tests. Mutable contributor mechanics are distinct from the pinned historical experiment.
- [Pinned RelBench implementation](https://github.com/stanford-star/relbench/tree/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639), MIT; original experiment reference.
- [User-study SQL](https://github.com/snap-stanford/relbench-user-study/blob/445bb7a3b1230f49f8e5890ae81754d3e365680f/f1/driver-position/feats.sql): hash-checked download, omitted from the release archive; root license not found at that commit.
- [Modal pricing](https://modal.com/pricing), checked2026-10-01: T4.000164/s,physicalCPU.0000131/core/s,memory.00000222/GiB/s; used for aggregate reservations, not represented as an invoice.

## Lesson158 · Evidence-bounded synthesis
- [RelBench v1 Section6 and AppendixC](https://arxiv.org/html/2407.20060v1#S6): primary reading for the predictive-quality/human-effort distinction. Local basic-GNN replay is not its boosted regression or human-work study.
- [L158 reproducible evidence report](labs/evidence/l158/report.md): frozen portfolio, matched comparison and temporal-policy evidence; inherited versus fresh checks explicitly separated.

## Lesson 159 · Foundation-model preview

- [Vogel, Hilprecht and Binnig (2023), pinned v1](https://arxiv.org/html/2305.15321v1): row LM plus GCN, masked reconstruction and initial single-table experiments. Corrects prior author attribution.
- [L159 source audit](labs/sources/l159/manifest.json) and [protocol/deviations](labs/l159-reproduction.md): selected wikiTables Table1 target NOT_RUN; exact release/subset/configuration not located.

## Lesson 160 · Year 4 exit evidence
- [RelBench v1 Section6 and AppendixC](https://arxiv.org/html/2407.20060v1#S6): distinguish published expert work from replaying the released pipeline.
- [L160 frozen report](labs/evidence/l160/report.md) and [reproduction/remediation contract](labs/l160-reproduction.md): full selected saved-evidence replay with unmet experimental gates.

## Lesson 161 · Foundation-model definition and scope (2026-10-01)

- **Primary:** Bommasani et al. (2021), [On the Opportunities and Risks of Foundation Models, v1](https://arxiv.org/abs/2108.07258v1). Read the introduction and adaptation discussion. [CRFM overview](https://crfm.stanford.edu/report.html) provides navigation. Broad pretraining and task adaptation ground the scope note; emergence/homogenization motivate evidence and failure analysis. This synthesis report is not represented as a selected relational training benchmark.
- **Prepared practice:** [Lesson 161](lessons/0161-what-is-a-foundation-model.html), [protocol](labs/l161-reproduction.md). Synthetic design audit, no trained model or numerical paper-result claim.
- **Year 5 boundary source pass:** [record](docs/plans/2026-10-01-year-5-source-check.md). Existing bridge already covers Context Window Failures and RelArena/TabPFN-Rel. The newly accessible [TabFM report](https://arxiv.org/abs/2609.37959) is a candidate method/code audit, not a new performance conclusion or required L161 experiment.

## Lesson 162 · Relational FM vision
Vogel, Hilprecht and Binnig (2023), [Towards Foundation Models for Relational Databases, arXiv:2305.15321v1](https://arxiv.org/html/2305.15321v1), §§2–4 and Table1. Primary reading for row-wise LM encoding, graph context, staged training and scaling obstacles. Initial single-table reconstruction is distinct from held-out multi-table transfer. [Pinned source ledger](labs/sources/l162/source-ledger.json) records bytes, search scope and unresolved protocol details. The complete local context/scale audit is synthetic; historical wikiTables training remains NOT_RUN.

## Lesson163 · row encoders

- Vogel, Hilprecht and Binnig, [2023 vision paper §§2–4](https://arxiv.org/html/2305.15321v1): primary conceptual/historical target. Exact artifacts and training protocol remain unresolved.
- Meta [BART-base model card](https://huggingface.co/facebook/bart-base) and [Transformers4.57.1 BART API](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/bart): real encoder/tokenizer used at immutable revision aadd2ab0ae0c8268c7c9693540e9904811f36177; hashes in labs/evidence/l163/encoding-receipt.json. Mean pooling and ridge are course choices.
- [L163 contract](labs/l163-reproduction.md): complete synthetic comparison and separately unrun historical target; no general encoder superiority claim.

## Lesson164 · Griffin · pinned2026-10-01

- Wang et al., [Griffin: Towards a Graph-Centric Relational Database Foundation Model](https://arxiv.org/html/2505.05568v1), §§3–4,Table12. Shared cell/task representation,graph-centric adaptation and low-data transfer.
- [Official code at b9d0e1f](https://github.com/yanxwb/Griffin/tree/b9d0e1fa8d89dfb1cd8bd5976b71de8a3b515427),Apache2.0; unmodified archive/license in `labs/sources/l164/upstream`.
- [Processed data](https://huggingface.co/datasets/yamboo/Griffin_datasets_joint_v65/tree/e0c54ceada75317b06f11f8dcda7aa8304fbb593) and [released checkpoints](https://huggingface.co/yamboo/Griffin_models/tree/bd8c5be5130f34e7faa31099d0bd81d95d0aa995). Artifact hashes and source deviations in `labs/l164-reproduction.md`; published scores are not local results.

### Lesson 165 · KumoRFM v1 · 2026-10-01

- Fey, Kocijan, Lopez, Lenssen and Leskovec (2025), [original archived technical report](https://web.archive.org/web/20260702201348id_/https://kumo.ai/research/kumo_relational_foundation_model.pdf): Eq.1, §§2.2–2.3 and Table2. [Pinned local PDF](labs/sources/l165/paper.pdf), SHA256 `805febc5e9af8a8e9d34c6123fa37af4d9c2f29ea2d1d04be0e87d63e249cfa2`.
- Historical selected target: rel-f1/driver-dnf, in-context AUROC82.41. Weights, exact data identity and complete context/evaluation recipe unresolved; NOT_RUN/fidelity NOT_ESTABLISHED. [Full protocol](labs/l165-reproduction.md).
- Current NVIDIA [research overview](https://docs.nvidia.com/sdgm/research/kumorfm-paper) is a source-drift observation: its paper link points to KumoRFM-2. The original Kumo report URL redirects to current documentation; a live client cannot establish v1 historical identity. [Retrieval ledger](labs/sources/l165/source-ledger.json).
- Local teaching evidence: complete deterministic context, graph and keyed-scoring audit. No inference or training; USD0 cloud/API. Author preparation is not learner mastery.

### Lesson 166 · RDB-PFN · 2026-10-01

- Wang, You, Shi and Zhang, [Relational In-Context Learning via Synthetic Pre-training with Structural Prior, v5](https://arxiv.org/html/2603.03805v5), §§4–6, Appendix A.3/C.2/F; Table9 selected512-context driver-dnf comparison.
- [Official code at a953782](https://github.com/MuLabPKU/RDBPFN/tree/a95378225478daa262b85f180d482da7516b0af6), [processed data](https://huggingface.co/datasets/yamboo/RDB_PFN/tree/d6a88c0a8cce79607cfc0fca0dcba78ba262ffad). Exact checkpoint/data/library hashes and deviations in [contract](labs/l166-reproduction.md).
- Fresh complete30-run released-checkpoint replay matches all three paper means at four decimals. One released prior draw completed. Fresh pretraining and wholepaper NOT_RUN; historical identity NOT_ESTABLISHED. Author preparation, not learner mastery.


## Lesson 167 — tabular to relational FM transfer

- Hollmann et al., [TabPFN v2 / Nature2025](https://www.nature.com/articles/s41586-024-08328-6), architecture and context caching. Conceptual comparison; no TabPFN benchmark arm added.
- Qu et al., [TabICL / ICML2025](https://proceedings.mlr.press/v267/qu25d.html), row representations followed by dataset ICL. The audited release is L166's pinned TabICLv1.1,32estimators.
- Wang et al., [RDB-PFN v5](https://arxiv.org/html/2603.03805v5), synthetic relational prior and Table9 context512 driver-dnf comparison. Full30-run saved-evidence audit in L167, no new inference.
- Molnar, [Tabular Foundation Models introduction](https://tabularfoundationmodels.com/introduction), conceptual reading map; primary papers govern mechanism/benchmark claims. Four source snapshots and hashes: `labs/sources/l167/source-ledger.json`.


## Lesson 168 · cross-database evaluation

- [RDB-PFN v5](https://arxiv.org/html/2603.03805v5): §6.1,AppendixA.3,C.1.1andTable9. Source-pinned trial512-context experiment; explicit synthetic-predictor versus real-schema-generator boundary.
- [Griffin v1](https://arxiv.org/html/2505.05568v1): compare supervised target adaptation with labeled-context prediction. L164fresh experiment remains budget-blocked.
- [Pinned RelBench study-outcome source](labs/sources/l139/relbench__tasks__trial.py): outcome horizon and primary-outcome semantics; new released labels are its complement by full query key.

## Lesson 169 · Scaling laws and open questions · checked 2026-10-01

- [Dwivedi et al., RDL survey v1 §5.2](https://arxiv.org/html/2506.16654v1#S5.SS2): historical foundation-model convergence argument; no fitted relational scaling law in this section.
- [Wang et al., RDB-PFN v5, Appendix A.3 and Tables6–10](https://arxiv.org/html/2603.03805v5#A6): contexts64–1024, ten support seeds; selected full two-task/three-model scope, not the whole benchmark or fresh pretraining.
- [Pinned RDBPFN evaluation source](https://github.com/MuLabPKU/RDBPFN/blob/a95378225478daa262b85f180d482da7516b0af6/model_pretrain/src/eval.py): stable task/seed key, independent sampling per context; same seed does not guarantee nested supports.
- [Modal pricing](https://modal.com/pricing): L4+2physical CPU+16GiB at USD0.00028372/s; USD10aggregate lesson limit including retries and validation, USD3overhead reserve. Source snapshots and hashes in labs/sources/l169/.


## Lesson 170 · evidence-backed FM design checkpoint · checked 2026-10-01

- [Vogel et al., vision v1](https://arxiv.org/html/2305.15321v1): motivation and reusable relational representations; conceptual context, not a completed benchmark.
- [Griffin v1 §§3–4](https://arxiv.org/html/2505.05568v1#S3): cell attention, relational messages and supervised transfer; L164 selected reproduction remains budget-stopped.
- [RDB-PFN v5 Appendix A.3 / Tables 6–10](https://arxiv.org/html/2603.03805v5#A6): source protocol for the complete selected 300-evaluation evidence replay.
- [RDBLearn v1 §§3–5](https://arxiv.org/html/2602.18495v1#S3): relational featurization plus an existing tabular ICL predictor. L170 compares the approach conceptually; the RDB-PFN DFS+TabICL arm does not reproduce its pipeline.
- Versioned primary snapshots and SHA256: `labs/sources/l170/source-ledger.json`. Named scope, executable commands and deviations: `labs/l170-reproduction.md`.

## Lesson 171 · corpus of databases

- Primary reading: [RelBench v1, Robinson et al.](https://arxiv.org/html/2407.20060v1), benchmark construction and dataset descriptions. [Current project](https://star-project.stanford.edu/relbench/) is context for corpus expansion, not a replacement for the fixed lesson scope.
- Pinned installed RelBench1.1.0 sources, registry hashes, loader semantics and complete F1 archive: [source ledger](labs/sources/l171/source-ledger.json). Seven original source definitions declare50tables/62FKcolumns; only F1 receives a complete row audit. [Protocol](labs/l171-reproduction.md).
- [Griffin](https://arxiv.org/html/2505.05568v1) supplies pretraining-corpus motivation; no Griffin training is executed in L171. Unknown lineage, rights and availability remain separate from byte identity and key integrity.

### Lesson 172 · Schema tokenization · 2026-10-02

- [Relational Transformer v1 §3.1](https://arxiv.org/html/2510.06377v1#S3.SS1): primary input-representation comparison. This course audit uses deterministic typed payloads, not learned RT embeddings or a paper-performance experiment. Exact differences are pinned in labs/sources/l172/source-ledger.json.
- [Pinned RelBench F1 source](labs/sources/l171/relbench/datasets/f1.py): key and time roles; source and complete archive reused by SHA256 from L171. L172 semantic kinds are explicit course policies.
- [Full F1 tokenization protocol](labs/l172-reproduction.md): all67columns /866,746cells, train-only fit, all-cell masks and independent scalar verification.

## Lesson 173 — multi-task pretraining objective

- [Relational Transformer v1 §3.3 and §4.1](https://arxiv.org/html/2510.06377v1): datatype-specific Huber/BCE, masked-cell mean and full training protocol. The course uses a small pooled encoder and multiclass CE; it does not reproduce RT.
- [KumoRFM-2 v1 §3](https://arxiv.org/html/2604.12596v1): row, column, FK and cross-sample information processing, not four mandatory additive losses. The course model implements bounded row/FK context only.
- [Source ledger](labs/sources/l173/source-ledger.json), [frozen protocol](labs/l173-reproduction.md), [complete six-fit evidence](labs/evidence/l173/report.md), [reference](reference/multi-task-pretraining.html). Whole-paper reproduction NOT_RUN; fresh selected course training COMPLETE.

## Lesson 174 · fine-tuning protocol · pinned 2026-10-02

- Houlsby et al., [Parameter-Efficient Transfer Learning for NLP](https://proceedings.mlr.press/v97/houlsby19a.html), ICML 2019: fixed-backbone adaptation with small task modules. L174's single post-MLP residual adapter is a course mechanism, not their BERT architecture or benchmark reproduction.
- [Relational Transformer v1 §4.2 and Appendix D](https://arxiv.org/html/2510.06377v1#S4.SS2): pretrained/untrained relational fine-tuning comparisons and validation selection. Reported 1.5 hours on eight A100 GPUs per fine-tuning run exceeds the $10 aggregate course cap at the [pinned pricing snapshot](https://modal.com/pricing).
- Snapshots and hashes: `labs/sources/l174/source-ledger.json`. Twelve fresh course fits use all 21 inherited F1 autocomplete tasks and later temporal windows; prior L173 test exposure is explicitly retained. Full paper reproduction NOT_RUN.

## Lesson175 zero-shot evaluation · 2026-10-02

- [RT-v1 primary paper §4.1–4.3](https://arxiv.org/html/2510.06377v1#S4): database-held-out pretraining, target validation selection and context-label access.
- [Original source pin](https://github.com/stanford-star/relational-transformer/tree/8d83590b5ae7fba9e40e8df463ed2dd9066ce5fb), [fixed release card](https://huggingface.co/stanford-star/rt-v1/blob/299701dedae451f3dfa40717b831d9dc17c0e4e7/README.md) and [original preprocessing revision](https://huggingface.co/datasets/hvag976/relational-transformer/tree/e8b48dc2cfb0a3c9171a8fddaaef14b6240f18ee). Current RT-J quickstart is a different model/protocol.
- [Full contract](labs/l175-reproduction.md): complete native-context audit; six model evaluations NOT_RUN after event-time gate failure. Race schedule attributes may be known earlier; no historical arrival proof is available.


### Lesson176 · Few-shot ICL evaluation · 2026-10-02

- [RDB-PFN v5 AppendixA.3 and Tables6–10](https://arxiv.org/html/2603.03805v5#A3): global labeled support per task, five context sizes, ten support seeds. L176 replays the full selected L169 evidence, then declares nested prefixes as a separate course intervention.
- [Original model at a953782](https://github.com/MuLabPKU/RDBPFN/blob/a95378225478daa262b85f180d482da7516b0af6/model_pretrain/src/models.py): support normalization, target-mean query placeholders, feature/row attention and context-only classifier fit. Visible inline in the notebook.
- [Modal resource pricing](https://modal.com/pricing), checked2026-10-02: L4 .000222USD/s,CPU .0000131/core/s,memory .00000222/GiB/s. Aggregate rate .00028372USD/s; timeout reservations and overhead included.
- [Frozen L176 contract](labs/l176-reproduction.md), [independent results](labs/evidence/l176/report.json), [shared-support consistency](labs/evidence/l176/shared-1024-comparison.json). Complete selected course experiment is distinct from whole-paper reproduction, pretraining and learner mastery.

## Lesson 177 · Compute budget realism · 2026-10-02

- [Modal pricing and billing FAQ](https://modal.com/pricing): base GPU/physical-CPU/GiB rates, loading/idle billing and requested-versus-used resources. Pinned HTML and rates in labs/sources/l177/; rates do not constitute an invoice.
- [Relational Transformer v1 §4.1](https://arxiv.org/html/2510.06377v1#S4.SS1): approximate2hpretraining/1.5hfine-tuning on8A100s; used only for explicit rental-price scenarios above the course cap.
- [RDBLearn v1 §3](https://arxiv.org/html/2602.18495v1#S3): deterministic relational featurization plus existing tabular ICL predictor; conceptual route, no L177model reproduction.
- [PyTorch v2.5.1 CUDA memory source](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/cuda/memory.py): max_memory_allocated reports allocated tensor peaks, not full VRAM requirement.
- [L177 full reproduction contract](labs/l177-reproduction.md): authenticates300inference/18fitrecords and all3L175reservations, preserving failed attempts and missing human/memory/invoice evidence.

## Lesson 178 · matched information and a stopped fresh comparison

- [RDB-PFN v5 Table9](https://arxiv.org/html/2603.03805v5): complete selected published F1 replay, three arms × ten seeds ×702queries; no fresh model inference in L178.
- [RDBLearn v1 §3/§5](https://arxiv.org/html/2602.18495v1): relational featurization plus a tabular predictor; its published depth/backend search differs from the approved L178 course protocol.
- [RDBLearn pinned estimator](https://github.com/HKUSHXLab/rdblearn/blob/b5b03ebf8091547285a6e06cba53d2d1a40cb171/rdblearn/estimator.py): full target-history augmentation precedes downsampling. Actual FastDFS0.2.1 preflight preserved; no RDBLearn model evaluation.
- [RelGNN source](https://github.com/snap-stanford/RelGNN/tree/cffdb8b54627e92c7dd112c1243dde739c90d35b): local CPU numerical-encoder prerequisite fails with256nonfinite gradients on actual support history. Independent original-encoder and arithmetic checks agree; full graph fit and historical CUDA reproduction NOT_RUN.
- [Pinned F1 task SQL](https://github.com/stanford-star/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/relbench/tasks/f1.py):30-day DNF target; all12679release labels are exact complements under independent raw reconstruction. Historical availability remains unestablished.


## Lesson 180 · public encoder fine-tuning checkpoint · checked 2026-10-02

- [RT-v1 paper §4.1 / Appendix D](https://arxiv.org/html/2510.06377v1): full supervised schedule, reported runtime, and per-task fine-tuning results. L180 reproduces saved context/source/cost audits only; fresh fit NOT_RUN.
- [Pinned source example](https://github.com/stanford-star/relational-transformer/blob/8d83590b5ae7fba9e40e8df463ed2dd9066ce5fb/scripts/example_finetune.py): 32769 steps, per-rank batch32, checkpoint saving disabled by default; original trainer selects validation metrics while also logging test.
- [Fixed RT-v1 release card](https://huggingface.co/stanford-star/rt-v1/blob/299701dedae451f3dfa40717b831d9dc17c0e4e7/README.md): public initialization provenance; weight bytes not downloaded after scientific/budget stop.
- [Modal pricing](https://modal.com/pricing): one reported eight-A100 1.5-hour run implies USD25.185600–29.980800 GPU-only. Full-run forecast exceeds the USD10 aggregate ceiling; no paid work authorized.

## Lesson 181 · RelBench v2 autocomplete

- [Gu et al., RelBench v2 v1](https://arxiv.org/html/2602.12606v1): §§2–4 and Tables2/5/14/15/21. Primary task and result source; complete two-F1-task aggregation baselines reproduced, fresh GNN NOT_RUN_TRAINING_HEALTH_GATE.
- [Publication-date code snapshot](https://github.com/stanford-star/relbench/tree/0d47fe0c8a1a51aaf97ab485f4a028e290f97c67): task registry, AutoCompleteTask, global database column removal, baseline train+validation refit and GNN model/trainer. Snapshot identity does not establish historical training identity.
- [Jiang, RelGT-AC v1](https://arxiv.org/html/2606.03040v1): paper-described seed masking, TF-IDF and local/global graph model. Validation comparisons and differing mask policy require care. No authenticated model/checkpoint release located in bounded search of paper/author pages and repository search; NOT_RUN_SOURCE_GAPS. Text projection dimensionality differs between general equation and experimental description.
- Sources and exact environment: `labs/sources/l181/source-ledger.json`, `protocol-audit.json`; measured comparison `labs/evidence/l181/paper-comparison.json`.


### Lesson 182 · RDB-PFN + composite message passing · 2026-10-02

- [RDB-PFN v5 §§5–6, Appendix C and Table 9](https://arxiv.org/html/2603.03805v5): prior/predictor separation and the complete selected 512-support F1 comparison. Fresh L182 run retains fixed source/checkpoints and all 30 evaluations; whole-paper/pretraining NOT_RUN.
- [RelGNN v2 §4, Equations 3–5](https://arxiv.org/html/2502.06784v2): atomic routes, fusion and destination attention. One-head identity specialization checked against original source and finite-difference input gradients; no new full RelGNN fit.
- [Frozen contract](labs/l182-reproduction.md), [fresh scores](labs/evidence/l182/report.json), [independent verification](labs/_verify_l182_results.json). Three means match rounded paper values, but hybrid superiority is an unrun proposal. Label orientation, DFS provenance and historical identity limits remain explicit.

## Lesson 185 · Causal & relational data

- [Pearl 2009, Causal inference in statistics: An overview](https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf), DOI10.1214/09-SS057: §§2,3.2.1,3.3.1,3.4. Primary definitions and assumptions; no empirical result reproduced. Pinned bytes in `labs/sources/l185/`.
- [Maier et al.2013, relational causal discovery](https://arxiv.org/abs/1309.6843v1): optional abstract-level context; RCD not implemented or reproduced.
- [L185 protocol](labs/l185-reproduction.md): full original synthetic five-seed shortcut/adjustment/intervention experiment; real-world effectiveness NOT_ESTABLISHED.

## Lesson 184 — GelGT

- [GelGT v2, §§3 and Appendix B](https://arxiv.org/html/2605.15575v2): structural sampling, semantic refinement, Gaussian temporal bias and Table 2 driver-position target.
- [Official pinned source](https://github.com/USTC-DataDarknessLab/GelGT/tree/1997b2c2f480ce5d3cbdb48d46f33cc303f5feb4): original sampler counterexample executed; full trainer NOT_RUN.
- [Local protocol and evidence](labs/l184-reproduction.md): 8,712 raw labels verified; query-cache identity stop; no paper-MAE claim.


## Lesson 186 · production constraints

- [Chip Huyen: Real-time machine learning, stages 1–3](https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html): primary reading for prediction and feature refresh choices. Conceptual guidance, not a published numerical reproduction target.
- [Chip Huyen: Data distribution shifts and monitoring](https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html): natural label delays and separate operational/predictive quality signals.
- [Google SRE: Monitoring distributed systems](https://sre.google/sre-book/monitoring-distributed-systems/): latency, traffic, errors and saturation; tail latency motivation.
- [L186 frozen protocol](labs/l186-reproduction.md) and [source snapshots](labs/sources/l186/source-ledger.json): all81course simulations/810000requests, separate300batch-receipt replay. Hypothetical durations; no model scores or real serving benchmark. Fresh inference/production NOT_RUN; learner PENDING_WRITTEN_DEFENSE.


## Lesson 187 · Relational contribution privacy

- [Dwork and Roth, The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/privacybook.html): Definition 2.4, global sensitivity, Laplace mechanism Theorem 3.6, basic composition Corollary 3.15. Primary reading for the ideal mechanism; finite-precision public seeded simulation is separate.
- [GAP](https://arxiv.org/abs/2203.00949): graph-aggregation privacy proposal, cited context only; no benchmark reproduction in L187.
- [Xiang, Wang and Wang v2](https://arxiv.org/html/2311.06888v2): Section VI critique of private instance embeddings and GAP noise accounting; attributed findings, not a locally reproduced attack.
- RelBench F1 schema pinned at `0d47fe0c8a1a51aaf97ab485f4a028e290f97c67`, authenticated through L181; all nine tables retained. Source hashes: `labs/sources/l187/source-ledger.json`. Course experiment `L187-F1-ENTITY-PRIVACY` is complete; private-GNN paper reproduction NOT_RUN.

## Lesson 183 · Graph-Transformer pretraining gap

- [Dwivedi et al., RelGT v1](https://arxiv.org/html/2505.10960v1): §§3–4 and Tables1/6; five-element tokenization, local/global attention and supervised benchmark protocol. Code19e423ca3e7cac761130aba790857f2dc3a46ef7 retained from L145.
- [Wang et al., Griffin v1](https://arxiv.org/html/2505.05568v1): §§3–4/Table12; unified cell/task interface, message passing and selected transfer protocol. Codeb9d0e1fa8d89dfb1cd8bd5976b71de8a3b515427 retained from L164.
- [Authors' relational graph-transformer overview](https://docs.nvidia.com/sdgm/research/relational-graph-transformers): existing connection to KumoRFM prevents a broad absence/novelty claim. Bounded source review does not establish originality of the proposed experiment.
- `labs/sources/l183/source-ledger.json` pins primary HTML; `labs/evidence/l183/input-manifest.json` authenticates inherited data and code. The notebook's complete saved replay is separate from fresh training, inference and raw-label reconstruction.


## Lesson188 · Systematic literature tracking

Primary: [arXiv API manual](https://info.arxiv.org/help/api/user-manual.html), [API terms](https://info.arxiv.org/help/api/tou.html), [RSS guide](https://info.arxiv.org/help/rss.html). Query/window/pagination and version semantics; source bytes retained in labs/evidence/l188/packet.

Watch sources: [RelBench board](https://star-project.stanford.edu/relbench/leaderboard/), [TabArena board](https://huggingface.co/spaces/TabArena/leaderboard). Page retrieval is separate from score extraction; no historical rank changes established.

[Q3 paper log](labs/evidence/l188/paper-log.md):30unique API records,7abstract-screened candidates,23deferred. Two failed phrase queries leave quarterly collection INCOMPLETE. Supplemental primary-page examples: [JEPA recipe](https://arxiv.org/abs/2609.25541), [physics limitations](https://arxiv.org/abs/2609.02766), [Xiaomi-TabLDM](https://arxiv.org/abs/2609.03880). Reading queue only; methods/artifacts not audited and results not reproduced. No core-paper promotion.

## Lesson 189 · Research-problem selection · 2026-10-02

Primary starting reading: [Dwivedi et al., RDL survey, 2506.16654v1](https://arxiv.org/html/2506.16654v1), §§2.4/5. Treat a dated survey as a map rather than a novelty certificate. Closest-work checks use [RelGNN v2](https://arxiv.org/html/2502.06784v2), [RDB-PFN v5](https://arxiv.org/html/2603.03805v5), [RT v1](https://arxiv.org/html/2510.06377v1), [Temporal Heterogeneous Graph Pretraining v1](https://arxiv.org/html/2609.35219v1), §4.4, and [RelArena-α v2](https://arxiv.org/html/2608.16319v2). The temporal paper explicitly leaves unseen-database transfer and matched computational cost open; RT means that cross-database transfer itself is not new. Selected-source coverage only; candidate novelty NOT_ESTABLISHED. Exact source bytes and retrieval receipts: `labs/evidence/l189/packet/sources.json`.
