# Relational Deep Learning Resources

Organized by curriculum year. Full sequencing in [CURRICULUM.md](./CURRICULUM.md).

## Knowledge — Year 1 (Tabular foundations)

- [XGBoost — Chen & Guestrin, KDD 2016](https://arxiv.org/abs/1603.02754)
  Canonical gradient boosting. Use for: the baseline every RDL result must beat fairly.
- [LightGBM — Ke et al., NeurIPS 2017](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
  Fast tree training. Use for: large tabular baselines.
- [CatBoost — Prokhorenkova et al., NeurIPS 2018](https://arxiv.org/abs/1706.09516)
  Ordered boosting + categoricals. Use for: high-cardinality categoricals without target leakage.
- [Random Forests — Breiman, Machine Learning 2001](https://doi.org/10.1023/A:1010933404324)
  Bagging + per-split feature subsampling to decorrelate trees; OOB error. Use for: Lesson 012 primary reading; the variance-reduction baseline (contrast with boosting's bias reduction).
- [Greedy Function Approximation: A Gradient Boosting Machine — Friedman, Annals of Statistics 2001](https://doi.org/10.1214/aos/1013203451)
  Boosting as stagewise additive modelling / gradient descent in function space; fit the negative gradient (residual for squared error), shrinkage as regularization. Use for: Lesson 013 primary reading; the bias-reduction mechanism behind XGBoost/LightGBM.
- [Why trees beat DL on tabular — Grinsztajn et al., NeurIPS 2022](https://arxiv.org/abs/2207.08815)
  Three inductive biases (irregular targets, uninformative features, rotation non-invariance). Use for: Lesson 010 Q1-checkpoint primary reading + year-1 exit exam; explains why a competently-built GBDT is the baseline the thesis must beat.
- [Designing Machine Learning Systems — Huyen](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)
  Evaluation, leakage, deployment. Use for: year-1 evaluation lectures.
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

## Knowledge — Year 2 (Advanced tabular)

**Architectures**
- [TabNet — Arik & Pfister, 2019](https://arxiv.org/abs/1908.07442)
  Sequential attention on features. Use for: interpretable neural tabular.
- [NODE — Popov et al., 2019](https://arxiv.org/abs/1909.06312)
  Differentiable oblivious trees. Use for: piecewise / irregular-function modeling.
- [TabTransformer — Huang et al., 2020](https://arxiv.org/abs/2012.06678)
  Contextual embeddings for columns. Use for: bridge to transformers on tables.
- [FT-Transformer / Revisiting DL — Gorishniy et al., NeurIPS 2021](https://arxiv.org/abs/2106.11959)
  rtdl baselines. Use for: FT-Transformer reproduction; strongest *classic* neural single-table model.
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
- [GraphSAGE — Hamilton et al., NeurIPS 2017](https://arxiv.org/abs/1706.1016)
  Inductive neighbor sampling. Use for: scaling GNN training.
- [GAT — Veličković et al., ICLR 2018](https://arxiv.org/abs/1710.10903)
  Attention over neighbors. Use for: adaptive aggregation.
- [R-GCN — Schlichtkrull et al., ESWC 2018](https://arxiv.org/abs/1703.06103)
  Relation-specific convolutions. Use for: heterogeneous graphs; direct precursor to REG edges.
- [HGT — Hu et al., WWW 2020](https://arxiv.org/abs/2003.01396)
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

- [Deep Learning on Relational Data — Cvitkovic, 2019](https://arxiv.org/abs/1903.06430)
  Early relational DL line. Use for: historical context.
- [Position: Relational Deep Learning — Fey et al., ICML 2024](https://proceedings.mlr.press/v235/fey24a.html)
  REG blueprint. Use for: north-star thesis and vocabulary.
- [RelBench beta — arXiv 2312.04615](https://arxiv.org/abs/2312.04615)
  First benchmark release. Use for: package architecture history.
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

- [Towards Foundation Models for Relational DBs — Zahradník et al., 2023](https://arxiv.org/abs/2305.15321)
  LM + GNN pre-training vision. Use for: foundation-model thesis and open problems.
- [Griffin — Wang et al., ICML 2025](https://arxiv.org/abs/2505.05568)
  Graph-centric RDB foundation model; unified encoder/decoder, cross-attention, pretrained on 150M+ nodes. Use for: the first serious open RDB FM.
- [RDB-PFN — Wang et al., 2026](https://arxiv.org/abs/2603.03805)
  First relational FM trained *purely on synthetic data* (Relational Prior Generator + PFN). Use for: open, reproducible relational ICL.
- [RDBLearn — Zhang et al., 2026](https://arxiv.org/abs/2602.13697)
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
- Lesson HTML published through 014 (XGBoost); agent produces lessons as you progress through CURRICULUM.md rows.
- Fast-moving frontier: re-run an arXiv search each quarter (sort by `submitted`) and add only papers that set SOTA, expose a failure mode, or are a baseline to beat. Full verified ID index + exhaustive registry in [CURRICULUM.md](./CURRICULUM.md#exhaustive-paper-registry-july-2026).
- Resources are tagged **core** (the default entries) vs **◆ optional / extension** (read only after the year's core papers and lab are done). Optional papers are ~2 h skims, never a reason to skip reproduction or exit exams. Full optional index: [CURRICULUM.md → Optional / extension reading](./CURRICULUM.md).
- **July 2026 merge:** PyTorch Frame ID corrected to `2404.00776`; TALENT to `2407.00956`. See [Research merge status](./CURRICULUM.md#research-merge-status-july-2026-deep-research-pass) for what's already solved vs newly added.
