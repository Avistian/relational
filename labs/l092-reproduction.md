# L092 — HAN / meta-paths reproduction contract

Target: Wang et al. WWW2019 HAN, arXiv1903.07293v2, Table3 ACM HAN classification, all four KNN training fractions. Targets (macro/micro F1):20% .8940/.8922;40% .8979/.8964;60% .8951/.8933;80% .9063/.9054. Descriptive tolerance±2pp frozen in `plan/lesson-092-delivery.md` before training. Numerical proximity cannot override protocol incomparability.

## Exact commands

From the course root:

```bash
.venv/bin/python labs/_verify_l092.py
.venv/bin/python labs/_run_l092.py --seeds 0 --output labs/_paper_l092_results.json
.venv/bin/python labs/_figures_l092.py
.venv/bin/python labs/_build_l092.py
.venv/bin/python labs/_execute_l092.py --full
.venv/bin/python labs/_audit_l092.py
.venv/bin/python labs/_delivery_l092.py
```

The CLI always trains from scratch. `--epochs 2 --output labs/_smoke_l092_results.json` is a diagnostic, not the target. `--mode paper_global` is an explicit alternative to released semantic attention and must use a separate output file. `--seeds 0 1 2` adds encoder initializations; it must not be represented as three independent datasets.

## Source and data

- Paper: https://arxiv.org/html/1903.07293v2
- Authors: https://github.com/Jhy1993/HAN/tree/71bac29a07fb8fab908d50a806a7bc38aa6c6611
- Snapshot: `sources/han-l092/`; SHA256 per file in `_sources_l092.json`. The upstream snapshot has no LICENSE file; provenance is recorded without inventing a license grant.
- DGL loader: archived `dgl_utils.py`, hash-pinned; resolves `https://data.dgl.ai/dataset/ACM3025.pkl`. Hash is checked before unpickling.
- Data SHA256: `b21cb3cb3c5b1f8562493942ddbc65c900a01fb58338adaf17966e4c8959161b`.
- N=3025, F=1870, C=3; PAP29281 and PSP2210761 directed endpoint entries including self. File calls PSP `PLP`; upstream README explicitly explains the name.
- Raw features; fixed600/300/2125 split; no new feature normalization or resampling. All split IDs and metrics are auditable. Transductive graph/features are permitted; gradients use training labels only.
- Original authors' SharePoint link returned an access-denied HTML page on2026-09-20 (HTTP200, not a MAT payload). Original preprocessed MAT bytes were not acquired; numerical/byte identity of the DGL conversion to the authors' original file is NOT_ESTABLISHED. No identity claim follows from matching shapes.

## Mechanism and protocol audit

| Element | Paper / original release | Modern reconstruction |
|---|---|---|
| Neighbors | Meta-path endpoints | binary PAP/PSP; no multiplicity weighting; self retained |
| Projection/head | Released Conv1D with no projection bias; two affine scalar scorers | equivalent linear layers; independent per path/head |
| Node score | LeakyReLU slope.2; receiver-row softmax | sparse equivalent, independently checked against dense outputs/gradients |
| Dropout | input, coefficients, projected values; rate.6 | all three sites preserved; PyTorch random masks differ |
| Head output | learned post-aggregate bias; ELU; eight heads concatenated |64 dimensions per path |
| Semantic scoring | W,b,q normal sd.1; tanh; width128 | preserved |
| Semantic reduction | Paper Eq7 averages scores over nodes; release `SimpleAttLayer` does not | primary lane uses released per-node softmax; paper-global mode implemented and separately checked |
| Classifier | dense64→3, default Glorot, zero bias | preserved shapes/initialization distribution, different RNG |
| Objective | masked cross-entropy +.001×tf.nn.l2_loss | training-only CE +.001×½sum squared parameters |
| L2 scope | exact-name filter does not match scoped TF names | all learned parameters included |
| Optimizer | TensorFlow1 Adam, lr.005 | PyTorch Adam, lr.005, epsilon1e-8; update implementation not historically identical |
| Schedule | release200 epochs; patience100 | full ceiling and patience; natural early stop retained |
| Selection | OR reset / AND save, equality allowed | preserved, recorded checkpoint epoch |
| Evaluation | `ex_acm3025.py` extracts test embeddings; imports missing `jhyexps`, repository has `jhyexp.py` | visible `knn_probe` follows present `my_KNN`; fixes filename issue by direct implementation |
| KNN | k5, uniform Euclidean, four fractions, ten cumulative shuffles | same operations, declared independent probe seed1000 |
| Historical randomness | unseeded source; original checkpoint/seed provenance absent | declared encoder seed0; no historical seed parity claim |

## Evidence boundaries

Read `_paper_l092_results.json` for measured results and `_execution_l092_results.json` for standalone solution coverage. The primary target runs one complete encoder fit plus40 probe evaluations. The ten repetitions in each fraction estimate split variability conditional on that fit; they are not an encoder-seed error bar. Repeating all40 evaluations from a second execution with identical seeds verifies portability of the visible code, not independent statistical replication.

`_audit_l092_results.json` checks source/data hashes, exact split reuse, stored predictions/F1, the checkpoint rule, fixed-weight global/local intervention and notebook task wiring. `_verify_l092_results.json` checks computation and gradients. These checks cannot prove historical result parity.

**Historical parity: INCOMPARABLE. Full-paper parity: NOT_ESTABLISHED.** Unresolved: mirror-to-original identity; semantic and stopping discrepancy; original checkpoint and RNG; exact historical backend; which released revision produced Table3. Remaining full-paper work: DBLP, IMDB, baselines, ablations, clustering, and parameters experiments are NOT_RUN. The lesson makes no state-of-the-art or database-generalization claim.

Portable dependency pins and observed author versions are separate. `requirements-l092-portable-lock.txt` records the complete isolated smoke interpreter, including harmless packages inherited from its L091 setup; Python3.13.11 and CPU torch2.8.0 were checked. The author full run uses the versions in `requirements-l092-observed.txt`. A clean portable smoke check, if present, only establishes import/forward/backward compatibility; it does not establish full portable-run scores. Live Colab and deployment are NOT_CHECKED unless separate evidence says otherwise. Authoring does not imply learner mastery: PENDING_WRITTEN_DEFENSE.

## Measured author result

One complete encoder fit executed all200 epochs; selected checkpoint35. Training time930.86s on the observed CPU runtime. Direct head accuracy87.4353%. Mean released per-node semantic weights over all3025 nodes: PAP.7095408 / PSP.2904592.

| Probe fraction | Macro-F1 mean ± split SD (%) | Micro-F1 mean ± split SD (%) | Descriptive ±2pp gate |
|---|---:|---:|---|
|20%|87.8782 ±1.0036|87.7176 ±1.0678|within|
|40%|88.0379 ±.7731|87.8353 ±.8079|within|
|60%|88.5251 ±1.3458|88.3176 ±1.4356|within|
|80%|88.1767 ±1.6471|87.9765 ±1.6765|outside|

All four historical protocol verdicts remain INCOMPARABLE. Scores were not used to retune parameters or choose the seed. See raw records for all40 evaluations.

## Independent inline replay and saved outputs

A fresh-directory full inline notebook replay completed200 epochs and all40 probes. Direct predictions, selected epoch and every KNN split/prediction/metric matched the author run exactly. Bitwise trace equality FAILED: maximum train-CE difference9.80e-7 and validation-CE difference6.57e-6; validation accuracies were equal. A post-hoc absolute1e-5 numerical diagnostic passes; this does not change the predeclared paper-score tolerance. See `_replay_l092_results.json` and `_inline_paper_l092_results.json`.

The initial executor compared traces before saving notebook outputs, so the failed exact assertion prevented that notebook save. Full raw results had already been retained. The executor now saves before comparison. The distributed solution was separately re-executed with the full switch disabled and honestly contains diagnostic outputs; the independent full replay remains a distinct evidence artifact. Run `_execute_l092.py --full` to regenerate full saved outputs. Default `_execute_l092.py` executes all definitions/tasks and the two-update diagnostic.
