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

- [TabNet — Arik & Pfister, 2019/2020](https://arxiv.org/abs/1908.07442)
  Sequential attention on features. Use for: interpretable neural tabular.
- [TabTransformer — Huang et al., 2020](https://arxiv.org/abs/2012.06678)
  Contextual embeddings for columns. Use for: bridge to transformers on tables.
- [Revisiting DL for tabular / FT-Transformer — Gorishnaya et al., NeurIPS 2021](https://arxiv.org/abs/2106.11959)
  rtdl baselines. Use for: FT-Transformer reproduction; strongest neural single-table model.
- [TabPFN — Hollmann et al., ICLR 2023 / Nature 2024](https://arxiv.org/abs/2207.01848)
  Tabular foundation model via in-context learning. Use for: FM concepts before relational FM.
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
- [RelGNN — Chen et al., ICML 2025](https://proceedings.mlr.press/v267/chen25ad.html)
  Composite message passing on REG. Use for: current SOTA architecture study.
- [RDL survey — arXiv 2025](https://arxiv.org/html/2506.16654v1)
  Challenges + next-gen architectures. Use for: year-4/5 frontier mapping.

## Knowledge — Year 5 (Foundation relational models)

- [Towards Foundation Models for Relational DBs — Zahradník et al., 2023](https://arxiv.org/abs/2305.15321)
  LM + GNN pre-training vision. Use for: foundation-model thesis and open problems.
- [RelBench v2 — arXiv 2602.12606](https://arxiv.org/abs/2602.12606)
  Expanded benchmark. Use for: year-5+ experiments.
- RelGT (Dwivedi et al., 2025), KumoRFM, Griffin — track via survey 2025 and RelBench leaderboard.
  Use for: reproduction targets when code is public.

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
- Proprietary FM weights (KumoRFM) limit full reproduction; plan ablations on open components.
- Lesson HTML exists only for lecture 001 so far; agent produces lessons as you progress through CURRICULUM.md rows.
