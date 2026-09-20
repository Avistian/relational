# L097: negative sampling — complete experiment and claim boundaries

## What full reproduction means here

The curriculum assigns **Negative sampling — training unit**, not a new model paper. This package reconstructs the complete GroupLens ML-100K release contract and executes the complete declared course experiment: five official folds × three seeds × three samplers × ten epochs, without row, catalog, fold or epoch caps. It does **not** claim reproduction of BPR's historical results or Krichene & Rendle's sampled-metric experiments.

- Full source-release reconstruction: **MATCH** (100,000 unique rating pairs, 943 users, 1,682 items, minimum 20 ratings/user, five 80,000/20,000 partitions; their test sets partition the release).
- Full predefined course experiment: **MEASURED**, all **45/45 fits** complete.
- Full BPR paper reproduction: **NOT_RUN**; paper-result parity **NOT_ESTABLISHED**.
- Learner mastery: **PENDING_WRITTEN_DEFENSE**.
- Live Colab and deployment: **NOT_CHECKED**. Local browser, standalone notebook and second-environment evidence are separate files.

## Frozen protocol

| Field | Exact choice | Provenance / deviation |
|---|---|---|
| Data | Entire MovieLens 100K archive | Original GroupLens release |
| Archive identity | SHA-256 `50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229` | Verified before parsing; member hashes saved |
| Partitions | u1 through u5 base/test; all rows | Source-defined static splits; not temporal forecasting |
| IDs | Separate zero-based 943-user / 1,682-item catalog | Transductive known identity spaces; cold embeddings remain randomly initialized unless trained |
| Positive | Base rating ≥4 | Course binary relevance transformation, same as L095 |
| Negative support | All items minus all base-rated items for that user | Course unobserved-item task; lower base ratings excluded too |
| Training information | Base data only for positives, masks, degrees and optimization | Test labels never passed to fit; held-out true links can be sampled |
| Scorer | 32-dimensional user/item factors plus item bias | BPR §5.1 motivates factorized score; no GNN encoder |
| Initialization | Torch normal(0,.1) factors, zero bias; seeds 0/1/2 | Same initial hash across sampler arms per fold/seed |
| Positive order | NumPy permutation per epoch, seed 9700+model seed | Every positive visited once each epoch; differs from LearnBPR bootstrap SGD |
| Uniform | Conditional uniform over unobserved items | With replacement, one selected negative per positive |
| Degree | Conditional (base-like-degree +1)^.75 | Explicit course exponent/smoothing; no importance correction |
| Hard | Four uniform draws, select maximum detached current score | Course hard-mining policy; extra score evaluations; duplicates allowed |
| Negative RNG | NumPy default_rng(19700+model seed), state evolves each epoch | Draws separate from positive-order RNG |
| Data objective | Mean softplus(s_neg−s_pos) | BPR pairwise log-sigmoid form; objective distribution changes with q |
| Regularizer | .0001 × mean(sum-squares of sampled user, positive/negative item factors and both item biases) | Course scaling, not historical regularization reconstruction |
| Optimizer | Adam lr=.01, default betas/epsilon; batch 2048 | Course choice, historical LearnBPR SGD not replayed |
| Budget / selection | Exactly 10 epochs; final weights; no validation tuning or early stopping | Declared fixed-budget comparison, not a best-model search |
| Main candidates | All items except that user's base-rated items | Identical evaluation rule across training arms |
| Relevance / metrics | Test ratings ≥4; binary NDCG@10 and Recall@10; macro user mean | Users without any test like excluded and counted |
| Ties | Descending score, ascending item ID | Deterministic ranking |
| Candidate diagnostic | All test likes plus min(99, available) other items sampled without replacement | Same scores; seed (970+fold)*10000+user; same candidates across arms and model seeds |
| Summary | Seed mean within fold, then unweighted five-fold mean; sample fold SD | Descriptive, correlated folds; not independent datasets |
| Compute | CPU, Torch one thread, deterministic algorithms | Author Python/package versions recorded |
| Resume | None | Every runner/notebook invocation starts fresh |

The lab uses BPR's objective as a teaching ingredient. Its ML-100K task, threshold, masks, Adam trainer, positive iteration, regularization scaling, hard/degree sampling, ten-epoch budget, and NDCG/Recall protocol do not reconstruct the original paper's complete experimental conditions. A close score would not remove those deviations. The sampled-metric paper is conceptual grounding; its datasets/models/corrections are not an experiment target in this lesson.

## Results and audit

Full-catalog mean NDCG@10: **uniform .31339**, **degree .25811**, **hard .34054**. Same-score sampled-candidate NDCG@10: **.73861**, **.69444**, **.76258** respectively. These describe one dataset and fixed training budget. Hard mining uses extra candidate scoring; uniform versus hard is not equal wall-clock compute. Lower degree-weighted performance here does not establish a general rule.

`_experiment_l097_results.json` stores all runs, complete configuration, source/data hashes, package environment, per-epoch losses, collision counts, initialization hashes, negative-stream hashes, metrics and aggregation. `results/l097/*.json` contains every relevant item and saved full/sampled top-ten ranking for each evaluable user. The independent `_audit_l097.py` reconstructs candidate pools from original files and checks **33,453** user metric records, macro means, paired initialization and coverage. It recomputes metric arithmetic from saved rankings; it does not independently reconstruct learned score matrices or certify training generalization.

The standalone solution executes all **15 code cells**, reruns all 45 fits from a temporary directory, and matches all author run records and summaries exactly. It contains the complete implementation and hash-checks the input archive. The temporary execution copies the cached original archive only; it does not use cached model/results. The second environment instead performs a fresh network download and runs the same inline cells under NumPy 2.2.6 / Torch 2.8.0. Every run record matched exactly in that second environment (maximum per-run metric difference 0.0). See `_portable_l097_results.json`; its explicit portability threshold is .01 absolute per-run metric difference, not a paper-parity tolerance.

Behavioral checks cover source-preserving typed draws, zero probability on blocked pairs, finite support for zero-degree items, saturated-row failure, invalid strategy failure, frequency/replay checks, pairwise values/gradient signs, independently worked ranking arithmetic and the complete PyG 2.6.1 bipartite complement.

## Run from repository root

For a minimal experiment environment, install `labs/requirements-l097-runtime.txt`; author versions are in `requirements-l097-observed.txt`. Notebook build/execution also needs nbformat, nbconvert, nbclient and a Jupyter kernel. Data are downloaded from GroupLens and are not redistributed. Acknowledge GroupLens and Harper & Konstan (2015), DOI 10.1145/2827872.

```bash
.venv/bin/python labs/_verify_l097.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python labs/_run_l097.py
.venv/bin/python labs/_audit_l097.py
.venv/bin/python labs/_sources_l097.py
.venv/bin/python labs/_figures_l097.py
.venv/bin/python labs/_build_l097.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python labs/_execute_l097.py
.venv/bin/python labs/_build_l097.py
.venv/bin/python labs/_delivery_l097.py
```

The notebook executor uses the existing `relational-labs` kernel. You may instead run the standalone solution in your own kernel. `_portable_l097.py` executes the notebook's exact inline code under the invoking Python without repository imports. Student and solution have identical supplied code; only the three sampler/loss functions are blanked in the student. Rebuilds preserve outputs only when all executable cell sources agree.

## Primary sources

- Rendle et al., [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618), §4.1–4.3 and §5.1.
- Krichene & Rendle, [On Sampled Metrics for Item Recommendation, extended abstract](https://www.ijcai.org/proceedings/2021/0651.pdf), introduction and inconsistency result.
- [PyG 2.6.1 negative sampling source](https://raw.githubusercontent.com/pyg-team/pytorch_geometric/2.6.1/torch_geometric/utils/_negative_sampling.py).
- [GroupLens ML-100K README](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt), SUMMARY and file descriptions.

Snapshots and hashes live in `_sources_l097.json`. All course protocol choices are stated above; published-source authority does not extend to the measured course results.
