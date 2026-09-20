# L091 · R-GCN on AIFB: reproduction contract

Named target: Schlichtkrull et al., *Modeling Relational Data with Graph Convolutional Networks*, [Table 2](https://arxiv.org/abs/1703.06103), AIFB R-GCN mean accuracy **95.83% over ten runs**. This package reconstructs the complete named AIFB experiment, not every experiment in the paper.

## Protocol fixed before execution

Source: `tkipf/relational-gcn` commit `4bec1341dd46b72bf482f7ed26c2dca4533577f6`. Author recipe:

```sh
python prepare_dataset.py -d aifb
python train.py -d aifb --bases 0 --hidden 16 --l2norm 0. --testing
```

The release's AIFB command uses unrestricted relation matrices. Preserve it even though the general paper prose emphasizes basis decomposition. A separately labeled four-basis extension teaches sharing without silently changing the target.

| Dimension | Paper/release | Local implementation |
|---|---|---|
| Graph | AIFB, 8,285 entities, 45 predicates, 29,043 triples | Complete mirrored RDF, exact statistics asserted |
| Targets | 176 people, four affiliations | 140 train / 36 test; original TSV order and membership |
| Leakage removal | Remove employs and affiliation | Assert both absent before supports |
| Directions | Original adjacency and transpose | Same; `[subject,object]` receives object into subject |
| Self path | One identity adjacency | Same; never double-add self |
| Pruning | Keep rows of labeled roots and one-hop neighbors | Same; unpruned output and gradient oracle |
| Normalization | Row mean independently per support | Same; zero-degree rows remain zero |
| Inputs | Featureless identity | Implicit identity; forward/gradient oracle |
| Model | Two layers, width16, no biases, ReLU then softmax | Same logits; cross-entropy incorporates softmax |
| Initialization | Separate Glorot uniform per matrix | Same distribution, different random stream |
| Final recipe | 50 epochs, Adam .01, dropout0, L2=0 | Same; Keras1.2.1 optimizer equation implemented visibly |
| Masked objective | Training entities only | Mean CE over selected entities only |
| Hyperparameters | 20% validation used for historical tuning | Released final choices frozen; historical search NOT_RUN |
| Final evaluation | `--testing`: all140 labels, final epoch | Same; no best-test checkpoint |
| Repetition | 10 runs, no recoverable seeds | Fresh declared seeds0–9, all retained |
| Metric | Mean test accuracy | Mean, sample SD, predictions, correct counts, loss traces |

## Exact deviations and limits

- PyTorch CPU replaces Keras1.2.1/Theano0.9. Numerical backend and RNG differ. Adam's historical epsilon placement is preserved; CE uses stable log-softmax rather than the historical probability clipping implementation. Historical runtime execution is NOT_RUN.
- Document-local blank-node labels are explicitly preserved; otherwise RDFLib generates fresh identifiers on each parse. Node order is canonical RDF `n3()` order, retaining literal datatype/language distinctions. Release node IDs were built from unordered sets; the original ordering and seeds are unavailable. Relation frequency ties and class IDs are sorted deterministically. This is equivalent graph structure but not identical random initialization assignment.
- The graph archive comes from DGL's public mirror and is SHA256-pinned. All three TSV files exactly match the pinned author's bytes. Graph counts and forbidden predicates are checked. The original Dropbox graph's historical byte identity is not established.
- Training is transductive. Both train and test **identities** participate in the release's memory-pruning roots. Test target values are used only for final scoring.
- AIFB is one graph with36 test entities. Seeds measure initialization variability, not independent dataset uncertainty. No model winner or temporal RDL superiority is inferred.
- No tolerance was fitted after seeing scores. The package reports numerical gap, without converting proximity into a historical-identity verdict.
- Full MUTAG/BGS/AM classification and all link-prediction benchmarks are NOT_RUN. The basis extension is not one of these paper experiments.

## Execution

From `relational/`, using the existing author environment:

```sh
.venv/bin/python labs/_verify_l091.py
.venv/bin/python labs/_run_l091.py
.venv/bin/python labs/_run_l091.py --basis-extension
.venv/bin/python labs/_audit_l091.py
.venv/bin/python labs/_figures_l091.py
.venv/bin/python labs/_build_l091.py
.venv/bin/python labs/_execute_l091.py
.venv/bin/python labs/_delivery_l091.py
```

Author target result: **95.8333% mean, 1.4640 percentage-point sample SD**, ten complete50-epoch runs. The separate B=4 extension reached96.2963% across three runs. The standalone inline solution exactly matched the author target predictions and loss traces after a fresh download. A fresh Python3.13.11 / PyTorch2.8.0 environment completed both full tracks and obtained the same means.

Portable runtime pins are in `requirements-l091-runtime.txt`. The observed author environment is recorded separately; portable pins do not claim numerical identity. Both notebooks contain all model/data/training definitions inline and a hash-checked data manifest. Set `RUN_FULL_REPRO=True` to execute all ten target runs and three basis-extension runs. The prepared solution has this enabled and is executed in a fresh working directory with its own verified download.

Raw evidence: `_paper_l091_results.json`, `_teaching_l091_results.json`, `_verify_l091_results.json`, `_audit_l091_results.json`, `_execution_l091_results.json`, `_delivery_l091_results.json`. Scores in prose are author-reference results, not automatically evidence of learner execution.

Original author code is archived under `sources/l091/` with its MIT license. `keras-1.2.1-optimizers.py` is an audit reference from the Keras1.2.1 release (MIT); its license is retained separately. The modern visible port is written for this lesson. `sources/l091/` and `_sources_l091.json` preserve the provenance needed to inspect each decision.

Live Colab and post-deployment checks remain NOT_CHECKED unless a separate report establishes them. Creating this lesson does not certify learner mastery.
