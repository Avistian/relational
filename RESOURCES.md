# Relational Deep Learning Resources

Organized by curriculum year. Full sequencing in [CURRICULUM.md](./CURRICULUM.md).

## Knowledge — Year 1 (Tabular foundations)

- [XGBoost — Chen & Guestrin, KDD 2016](https://arxiv.org/abs/1603.02754)
  Canonical gradient boosting. Use for: the baseline every RDL result must beat fairly.
- [LightGBM — Ke et al., NeurIPS 2017](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
  Fast tree training. Use for: large tabular baselines.
- [CatBoost — Prokhorenkova et al., NeurIPS 2018](https://arxiv.org/abs/1706.09516)
  Ordered boosting + categoricals. Use for: high-cardinality categoricals without target leakage.
- [Why trees beat DL on tabular — Grinsztajn et al., NeurIPS 2022](https://arxiv.org/abs/2207.08815)
  Three inductive biases. Use for: year-1 exit exam; explains the bar neural methods must clear.
- [Designing Machine Learning Systems — Huyen](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)
  Evaluation, leakage, deployment. Use for: year-1 evaluation lectures.

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
- [TabArena — Erickson et al., NeurIPS 2025](https://arxiv.org/abs/2506.16791)
  Living, maintained tabular leaderboard. Use for: ground truth on what actually wins, with ensembling.
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
- [LoCalPFN — Thomas et al., 2024](https://arxiv.org/abs/2406.05207) · [Drift-Resilient TabPFN — Helli et al., NeurIPS 2024](https://arxiv.org/abs/2411.10634)
  Retrieval+fine-tuning; distribution shift. Use for: making PFNs practical.
- [Realistic eval of TabPFN v2 in open environments — Cheng et al., 2025](https://arxiv.org/abs/2505.16226)
  **Critical counterweight:** where tabular FMs fail (covariate shift, scale, imbalance) and trees still win.
- [PyTorch Frame — Hu et al., 2024](https://arxiv.org/abs/2402.05964)
  Deep tabular row encoder. Use for: RDL stack (encoder → GNN); read in Y2, apply heavily in Y4.

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
- [RDL survey — arXiv 2025](https://arxiv.org/abs/2506.16654)
  Challenges + next-gen architectures. Use for: year-4/5 frontier mapping.

## Knowledge — Year 5 (Foundation relational models)

- [Towards Foundation Models for Relational DBs — Zahradník et al., 2023](https://arxiv.org/abs/2305.15321)
  LM + GNN pre-training vision. Use for: foundation-model thesis and open problems.
- [Griffin — Wang et al., ICML 2025](https://arxiv.org/abs/2505.05568)
  Graph-centric RDB foundation model; unified encoder/decoder, cross-attention, pretrained on 150M+ nodes. Use for: the first serious open RDB FM.
- [RDB-PFN — Wang et al., 2026](https://arxiv.org/abs/2603.03805)
  First relational FM trained *purely on synthetic data* (Relational Prior Generator + PFN). Use for: open, reproducible relational ICL.
- [GelGT (Gaussian Relational Graph Transformer) — 2026](https://arxiv.org/abs/2605.15575)
  Long-range dependency fixes for relational graph transformers. Use for: tracking architecture frontier.
- [RelBench v2 — arXiv 2602.12606](https://arxiv.org/abs/2602.12606) · [RelGT-AC — arXiv 2606.03040](https://arxiv.org/abs/2606.03040)
  Expanded benchmark + autocomplete tasks. Use for: year-5+ experiments.
- KumoRFM (2025) — proprietary in-context relational FM. Track via vendor tech report; not reproducible but sets a commercial bar.

## Wisdom (Communities)

- [RelBench mailing list](https://groups.google.com/forum/#!forum/relbench/join)
  Benchmark authors and practitioners. Use for: reproduction questions, leaderboard norms.
- [PyTorch Geometric Discussions](https://github.com/pyg-team/pytorch_geometric/discussions)
  GNN implementation. Use for: PyG + RelBench integration.
- [RelBench GitHub Issues](https://github.com/snap-stanford/relbench/issues)
  Bug reports and baseline discrepancies. Use for: when numbers don't match papers.
- [r/MachineLearning](https://reddit.com/r/MachineLearning)
  New paper alerts. Use for: spotting RDL papers; filter for signal.

## Gaps

- No canonical textbook for relational foundation models — curriculum is paper-driven.
- Proprietary FM weights (KumoRFM) limit full reproduction; plan ablations on open components (Griffin, RDB-PFN).
- Lesson HTML exists only for lecture 001 so far; agent produces lessons as you progress through CURRICULUM.md rows.
- Fast-moving frontier: re-run an arXiv search each quarter (sort by `submitted`) and add only papers that set SOTA, expose a failure mode, or are a baseline to beat. Full verified ID index lives in [CURRICULUM.md](./CURRICULUM.md#verified-paper-index-arxiv-ids).
