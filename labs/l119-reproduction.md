# Lesson 119: selected RelBench reproduction and synthesis evidence

## Result and exact scope

**COMPLETE:** five freshly executed full-data, ten-epoch RDL runs for RelBench v1 Table7, `rel-f1/driver-position`. L119 uses the unchanged pinned L117 model/trainer, with a new Modal application, volume, budget reservations, timestamps and run UUIDs. None of the L117 trained checkpoints or predictions supplies these new results.

Validation MAE **3.18950124 ± 0.04008558**; test MAE **4.01457097 ± 0.12673210** (mean ± sample seed SD). Published RDL means:3.193 and4.022. Both meet the predeclared descriptive absolute-mean tolerance .2MAE, labeled **CLOSE**. This is not a statistical equivalence test. Full source/current-runtime replay accompanies every final validation/test prediction. Maximum raw-output difference:2.861023e-6.

This reproduces the complete selected released-protocol experiment. It is not whole-paper reproduction, a fresh LightGBM comparison, historical environment identity, or learner mastery. L118 Home Credit remains NOT_RUN. Fresh repeated runs need not be bitwise identical under GPU/sampling execution; L119 results differ from the earlier L117 results despite the same declared seed list. No post-result selection among the two lessons' runs is performed.

## Frozen protocol and deviations

| Item | Executed contract |
|---|---|
| Paper | https://arxiv.org/html/2407.20060v1 Table7 RDL F1; AppendixB/Table9 |
| Release | RelBench1.1.0 commit `9aa346267c2e1c560bd92da07d6f4ad1ca2f0639`; original files and MIT license in `sources/l117/` |
| Canonical visible implementation | `relkit/rdl_l117.py`, SHA256 `8417d096b73354c0b08c4378e705bf4ba98be467b39a47c109513d13107029c2`; runner `_run_l117.py` |
| Database | Nine tables,74,063 release-censored rows at2010-01-01; no row cap |
| Database identity | `https://relbench.stanford.edu/download/rel-f1/db.zip`; SHA256 `ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482` |
| Task identity | `https://relbench.stanford.edu/download/rel-f1/tasks/driver-position.zip`; SHA256 `775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e` |
| Queries | All7,453 train /499 validation /760 test; published time split |
| Encoders | Four-block table-specific Frame ResNet, width128; numerical/categorical/timestamp encodings and frozen GloVe sentence vectors; relative-age encoding |
| Text | `sentence-transformers/average_word_embeddings_glove.6B.300d`, revision `e5e8fec6971be8960cfaa853a77a6ddc62a265d7`; CPU text embedding |
| Graph | Row-to-node PK/FK mapping with separately typed reverse edges; every forward relation checked against key lookups; independent SQL census |
| GNN/head | Two typed sum-GraphSAGE layers, width128, relation-specific root maps, summed relations, node-wise LayerNorm/ReLU; scalar seed-node head |
| Sampling | Native temporal NeighborLoader, disjoint query context, batch512, uniform fanout[128,64], num_workers0 |
| Optimization | Adam .005, mean L1 loss, ten complete epochs; default released initialization/reset with each declared seed |
| Selection | First lowest validation-MAE epoch; restore full state; validation/test resample neighborhoods after selection |
| Evaluation | Clip to train-target2nd/98th percentiles; official MAE and independent scalar scorer; compare separate original Model on identical sampled inputs |
| Seeds |0–4; five fresh runs; seed100 one-epoch pilot is excluded from reported scores |
| GPU runtime | Python3.11, torch2.5.1+cu124, PyG2.6.1, Frame0.2.3, RelBench1.1.0, pyg-lib0.4.0+pt25cu124; remaining pins in `requirements-l117-runtime.txt` |
| Preprocessing deviation | Released full test-censored materialization; statistics are NOT train-only. Type inference fixed seed42 before per-run training reset |
| Paper/source deviation | Source fanout[128,64] versus paper Table9 neighbor count128 |
| Unknowns | Historical seeds/random state/package identity; real ingestion/revision histories; all other paper tasks and fresh tabular baselines |

The course dual-clock graph walk uses explicitly known event/availability timestamps. The released benchmark only establishes its available event-time semantics. Do not transfer the course fixture's stronger availability guarantee to the benchmark.

## Commands used

Run from the `relational` root. `.venv/bin/python` is the local analysis environment; the Modal image builds the pinned GPU environment.

```bash
.venv/bin/python labs/_check_l119.py
.venv/bin/python labs/_source_check_l119.py
.venv/bin/modal run --detach modal/l119_repro.py --mode pilot
.venv/bin/python labs/_collect_l119.py --mode pilot
# Inspect the new pilot's full-query, original-model, checkpoint and budget evidence.
# The shipped ledger records the successful author review and approval for all five runs.
.venv/bin/modal run --detach modal/l119_repro.py --mode paper
.venv/bin/python labs/_collect_l119.py --mode paper
.venv/bin/python labs/_analyze_l119.py
.venv/bin/python labs/_audit_l119.py
.venv/bin/python labs/_verify_l119.py
.venv/bin/python labs/_figures_l119.py
.venv/bin/python labs/_build_l119.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_execute_l119.py
node labs/_check_pedagogy.js
.venv/bin/python labs/_delivery_l119.py
```

The completed cloud ledger deliberately rejects duplicate dispatch and the remote worker refuses to overwrite completed evidence. Do not delete reservations to force another run. The independent local/Colab route below reruns the complete experiment into a new directory using the same visible implementation; it does not reuse author weights. A separately budgeted cloud rerun requires a new ledger and isolated evidence volume.

For direct execution on a compatible GPU host (no automated monetary cap):

```bash
# Choose a new directory; refuse to overwrite an earlier independent run.
test ! -e /tmp/l119-independent || exit 1
for seed in 0 1 2 3 4; do
  .venv/bin/python labs/_run_l117.py --seed "$seed" --epochs 10 --output "/tmp/l119-independent/seed-$seed" || exit 1
done
# Preserve all five traces, predictions and checkpoints, then rescore by query identity.
```

The notebook also embeds the full model and an OFF-by-default five-seed gate. It requires the pinned GPU runtime and native pyg-lib; its default solution execution only rescored stored predictions. Live Colab is NOT_CHECKED.

## Budget and provenance

At checked Modal base prices on2026-09-27: T4$.000164/s + two physical CPU cores$.0000262/s +16GiB$.00003552/s = **$.00022572/s**. Reserve at most eight one-hour worker slots: **$6.500736**, plus **$3.499264** for build/startup/storage overhead. Each worker timeout is3600s and automatic retries are disabled. One pilot plus five full runs consumed six reservations; projected worst-case reservations stay below$10. The new23.037s pilot projected about$.260 for all five ten-epoch workers.

Recorded worker resource estimate, including pilot: **$0.05843653**. This is not an itemized account bill; build/startup/storage costs are not itemized. Credits do not expand the cap. The budget ledger rejects changed pinned source and duplicate modes. Completed remote results cannot be overwritten.

- Pilot: https://modal.com/apps/pszar92/main/ap-Ju6CwPnoPznuRQIOLMMUv2
- Five full runs: https://modal.com/apps/pszar92/main/ap-BQKcBFqoHMSzMhXTqwXf1S
- Fresh compact artifacts: `evidence/l119/paper/seed-*/`; each includes audit, trace, query predictions, run timestamp and UUID.
- Selected weights: `results/l119/paper/seed-*/selected.pt` locally and the Modal evidence volume; excluded from public site payload. Their hashes are verified before aggregation.
- Source/data audit: `_audit_l119_results.json`; source/current-CPU forward/gradient/Adam parity: `_source_check_l119_results.json`.

## Three evidence levels

**COURSE_ONLY:** authored collision ceilings, typed sums, dual-clock eligibility and cycle/triangles calculations. No synthetic outcome is presented as a real accuracy measurement.

**COMPLETE selected release replay:** all five full-data fits, source parity, 6,295 rescored query predictions, and recorded deviations. Published LightGBM values remain cited, not freshly reproduced.

**NOT_ESTABLISHED / NOT_CHECKED:** whole-paper parity, historical identity, fresh matched tabular superiority, live Colab and deployment. Learner status remains **PENDING_WRITTEN_DEFENSE**.
