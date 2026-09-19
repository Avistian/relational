# Worked L079 guide — consult after writing your own

This is a defensible example, not the only correct shortlist. Family priorities are hypotheses. No new model training was performed for L079.

| Regime | Baseline and challenger | Evaluation and reason to change |
|---|---|---|
| Mixed features | Tuned GBDT → RealMLP/TabM: test a strong training recipe and shared-parameter ensemble | Same frozen split, metric and selection budget; retain a neural model only for validation improvement within measured cost |
| Few labels | GBDT → pinned TabPFN v2; pinned 2025 TabICL for classification | Check task/feature support and memory; use deployment-matching validation; reject if shift or calibration removes the gain |
| Large context | Trees/TabM → feasible TabICL classification run | Record context size, hardware, batching and p95; reject if total prediction path exceeds budget |
| Feature interactions | Trees/TabM → FT-Transformer | Feature-token attention is a challenger hypothesis; drop if extra compute fails to improve validation |
| High-cardinality IDs | Native-category CatBoost → trained embeddings | Chronology plus cold-start subgroup; train-only vocabularies; reject identity memorization exposed on unseen entities |
| Names/text | No-text baseline → fixed text representation or appropriate CARTE transfer | Availability-safe text, pinned encoder, equal downstream selection; reject if text ablation shows no incremental value |
| Complementary errors | Best feasible single model → OOF blend | Fit every preprocessing/selection step within folds, preserve test freeze, measure combined serving; reject if gain does not justify cost |
| Missing history | Flat baseline → point-in-time aggregates → relational challenger | Match target, cutoff, split and information access; retain relational complexity only beyond strong eligible aggregates |

For the fraud case I start with native-category CatBoost on features known at transaction arrival and prespecify log loss plus calibration inspection. I split by time with fraud-label delay respected, evaluate unseen merchants separately, and select with a fixed search budget on validation only. I compare RealMLP/TabM and a pinned ICL classifier only if complete-path p95 on the intended batch-one hardware fits 5 ms with margin. I change my recommendation when a feasible challenger improves validation; one locked test checks the frozen choice, while future labels or a revised shortlist require a new evaluation boundary.

Reproduction footer: L079 reanalyzes the complete corrected L060 v2 artifact, SHA256 `99387c9ac29f941ec17ad1a9593a57c780551db6371c458db9197d5eab73f266`, using seeds 0/1/2 and the saved train/validation/test IDs. The manifest pins its course origin and analysis sources. Run `.venv/bin/python labs/_verify_l079.py --output /tmp/l079-replay.json`; use the observed analysis environment in `labs/requirements-l079-observed.txt`. This checks 210 frozen records, not a fresh training or paper reproduction. The latency numbers in the lesson are synthetic and cannot certify the fraud system. FT-Transformer/ICL/OOF serving gains remain proposed and unrun here. See `labs/l079-reproduction.md` for training commands and missing paper-protocol alignment.

Primary sources and qualified claims are linked in the lesson. The local ranks support only a convenience-sample comparison of reduced historical recipes; they cannot rank missing families or decide tomorrow's fraud model.
