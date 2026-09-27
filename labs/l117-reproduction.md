# Lesson 117 reproduction contract

The complete selected experiment is **RelBench v1 (arXiv:2407.20060v1), Table 7, rel-f1/driver-position, RDL column**: five fresh full-data fits, ten epochs each. This is the empirical companion to Fey et al.'s ICML 2024 RDL blueprint, not a replay of the earlier beta tables in that position paper. Historical identity and whole-paper parity remain **NOT_ESTABLISHED**.

Measured validation MAE **3.18179877 ± 0.04348035**, test **4.13391958 ± 0.15660422** (mean ± sample seed SD). Published means 3.193 / 4.022. Both means are **CLOSE** under the predeclared descriptive 0.2 MAE tolerance; this is not a statistical equivalence claim. No post-result tuning. All 6,295 final query predictions were independently rescored; largest original-model raw-output discrepancy was 2.861023e-6 on GPU, within `rtol=atol=1e-5`. The CPU fixture's outputs, gradients and Adam update match exactly in the current local runtime.

## Frozen protocol and deviations

| Item | Executed contract and boundary |
|---|---|
| Upstream | RelBench v1.1.0 commit `9aa346267c2e1c560bd92da07d6f4ad1ca2f0639`; original sources, MIT notice and fresh byte audit in `sources/l117/` |
| Data | `https://relbench.stanford.edu/download/rel-f1/db.zip`, SHA256 `ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482`; task archive `tasks/driver-position.zip`, SHA256 `775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e` |
| Scope | Nine tables, 74,063 rows after release censoring at 2010-01-01; all 7,453/499/760 train/validation/test queries |
| Graph | One row per node; original key remapping; each non-null valid FK produces two separately typed edges; explicit mapping audit of every edge during each preparation; independent SQL census afterward |
| Encoders | Table-specific four-block Frame ResNet, width128; numerical/categorical/timestamp embeddings and frozen GloVe sentence vectors; relative-age sinusoid and learned linear map |
| Text | `sentence-transformers/average_word_embeddings_glove.6B.300d` revision `e5e8fec6971be8960cfaa853a77a6ddc62a265d7`; encoded on CPU; original script selects the available device |
| Preprocessing | Original `get_stype_proposal` and full test-censored materialization. Type inference uses seed42 before resetting training RNG to each run seed. Full inferred types and table iteration order captured per run. Statistics are **not train-only** |
| GNN | Two heterogeneous SAGE layers; sum neighbors and sum relations, relation-specific root transforms, node-wise LayerNorm/ReLU, width128; scalar linear head |
| Sampling | Native pyg-lib temporal NeighborLoader, disjoint query subgraphs, batch512, uniform, fanout `[128,64]`, num_workers0; original source halves fanout at second hop although paper Table9 says128 |
| Training | Adam .005, mean L1, all 7,453 training queries per epoch, ten complete epochs; no capped subset or early stopping |
| Selection | First minimum validation MAE over ten epochs; restore all model state. Selection-time and final validation can differ because neighborhoods are sampled again |
| Evaluation | Clip to training-target 2nd/98th percentiles; official evaluator and independent scalar MAE. Validation/test labels never enter model forward inputs. Every evaluation batch checks dated sampled rows against its owning query cutoff |
| Seeds | Five declared seeds0–4. Paper reports five runs but does not provide a recoverable historical RNG/type-inference state; our sequence is not claimed identical |
| GPU runtime | Python3.11, torch2.5.1+cu124, PyG2.6.1, pyg-lib0.4.0+pt25cu124, Frame0.2.3, RelBench1.1.0, NumPy1.26.4, pandas2.2.3, sentence-transformers3.3.1. Remaining pins in requirements file |
| Source reuse | Model and included graph/encoder classes retain upstream definitions (AST checked); Frame/PyG operators are explicit pinned dependencies, with ResNet/SAGE source also visible in the notebook. We do not claim an independently reimplemented numerical kernel |
| Original replay | Separate original Model loaded with selected weights; all final val/test batches compared before clipping. Preserve RNG around reference construction to avoid changing evaluation samples |
| Unestablished | Real ingestion/revision histories, original historical environment/seeds, all other RelBench tasks, Fey beta experiments, LightGBM/user-study reproduction, whole-paper parity |

## Run the published author experiment

From the repository root, with a configured Modal account:

```bash
.venv/bin/python labs/_check_l117.py --red
.venv/bin/python labs/_check_l117.py
.venv/bin/python labs/_source_check_l117.py
.venv/bin/modal run --detach modal/l117_repro.py --mode pilot
.venv/bin/python labs/_collect_l117.py --mode pilot
# Inspect pilot correctness, timing and projected cost; then set
# pilot_approved_for_full=true in the ledger. The shipped ledger records this run.
.venv/bin/modal run --detach modal/l117_repro.py --mode paper
.venv/bin/python labs/_collect_l117.py --mode paper
.venv/bin/python labs/_analyze_l117.py
.venv/bin/python labs/_audit_l117.py
.venv/bin/python labs/_mutation_l117.py
.venv/bin/python labs/_figures_l117.py
.venv/bin/python labs/_build_l117.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_execute_l117.py
.venv/bin/python labs/_delivery_l117.py
```

The recorded budget ledger rejects duplicate dispatch. For a deliberate new experiment, copy its budget settings to a new JSON ledger with empty `reservations` and `pilot_approved_for_full=false`, and pass `--budget-file path/to/new-ledger.json` to both Modal commands. A new replay writes the corresponding seed paths on the named volume; collect/archive an existing run first. Keep its outcomes separate from the published evidence.

To run without Modal on a compatible NVIDIA CUDA environment, create a separate Python3.11 environment, install:

```bash
python3.11 -m venv .venv-l117
.venv-l117/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
.venv-l117/bin/pip install -r labs/requirements-l117-runtime.txt
.venv-l117/bin/pip install pyg_lib==0.4.0+pt25cu124 -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
# Repeat for seeds 0,1,2,3,4, preserving each output directory.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 3600 .venv-l117/bin/python labs/_run_l117.py --seed 0 --epochs 10 --output labs/results/l117/fresh/seed-0
```

The notebook contains all model/trainer/graph definitions and an optional five-seed gate. Its default run is a separate three-table real-data teaching exercise. It downloads less than1MB of archives, practices three live tasks, and leaves full training off. Existing notebook packages are reported rather than falsely claimed pinned. Live Colab **NOT_CHECKED**.

## Budget and execution record

Aggregate plan USD10, all seeds/pilots/retries/validation. [Modal prices](https://modal.com/pricing) checked 2026-09-27: T4 .000164/s + two physical CPU cores .0000262/s +16GiB .00003552/s = **.00022572/s**. Eight maximum3600-second worker reservations cap worker resources at USD6.500736, leaving USD3.499264 for overhead. No automatic retries. One one-epoch pilot plus five ten-epoch full runs were actually dispatched. Their measured worker-resource estimate totals **USD0.05768676**; image build, startup, idle reservation and storage are not itemized invoice costs.

The first full-run command was blocked locally before any worker dispatch because an approval assertion demanded exact GPU output identity. The experiment's existing numerical comparison is `rtol=atol=1e-5`; observed pilot differences1.91e-6 passed it. The local approval assertion was corrected to honor that tolerance, and one five-worker batch was launched. No fit was retried or replaced after inspecting its score.

## Delivery and learning

`_execution_l117_results.json`: standalone solution execution in an empty directory, three live tasks, real archive hashes and future-row checks. `_delivery_l117_results.json`: canonical notebook alignment, deterministic build, desktop/mobile/native controls/keyboard/no-JS/print and realistic copied Pages navigation. `labs/_check_pages_checkout.py` additionally runs the actual Pages build from Git-index files only. This caught the ignored L110 solution that broke GitHub deployment; fix8107c76 deployed successfully in run36298946460, with live bytes checked.

Lesson117 publication is separately recorded in `reviews/l117-publication.json` after deployment. Learner status remains **PENDING_WRITTEN_DEFENSE**. Nothing in the author run constitutes learner mastery.
