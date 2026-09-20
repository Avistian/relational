# L100 · Heterogeneous checkpoint: reproduction and evidence contract

Learner status **PENDING_WRITTEN_DEFENSE**. Complete course experiment **MEASURED**. Full-paper parity **NOT_ESTABLISHED**. AIFB historical identity **INCOMPARABLE**. HGT CS full training **NOT_RUN**. Prepared artifacts do not establish mastery, live Colab or deployment.

## Complete declared course experiment

The frozen pre-scoring plan is `plan/lesson-100-delivery.md`. The visible canonical implementation is `relkit/checkpoint_l100.py`; both notebooks inline its model, data loader, native sampler integration, trainer, validation selection and metrics. Source/data/environment identities are in `_sources_l100.json`. The full CS archive was freshly SHA256-verified without deserializing or training it.

| Component | Protocol |
|---|---|
| Source | DGL ACM.mat, SHA256 `0ccd838e545e8f16e3dc84356da2f51dfd2290c32e37a784e374dd510c76578d` |
| Preprocessing | Conference filter `[0,1,9,10,13]`, targets `[0,1,2,2,1]`; all4025 papers retained; compact connected7167 authors/60subjects; raw IDs saved |
| Features | Paper1903 word counts normalized per row; auxiliary constant ones; conference incidence excluded |
| Edges | Paper→author13407 and reverse13407; paper→subject4025 and reverse4025 |
| Access | Static transductive; full graph features/edges visible, labels used only through specified masks |
| Split | L099 NumPy RandomState99 stratified split; per-class floor(.2n) train, next floor(.3n)−floor(.2n) validation, rest test; 804/403/2818 IDs saved |
| Arms | R-GCN, HGT, uniform HGT, feature-only MLP; visible L099 operators unchanged |
| Architecture | Type adapters, width32, depth2, heads4, dropout0, paper classifier3; no RTE |
| Sampling | Native PyG NeighborLoader/pyg-lib, two directional hops, fanout8 per relation/hop, no replacement, batch256, workers0 |
| Sample pairing | Reset RNG100000+1000*seed+epoch_index; training seed permutation then native sampling; actual node/edge/seed fingerprints match across all arms/rates |
| Initial pairing | Uniform HGT copies every common initial tensor from HGT; different families are not weight-identical |
| Optimization | Adam, rates .003/.01, weight decay .001,40epochs, batches256/256/256/36; one mean-loss update per batch,160updates/fit |
| Loss | Three-class cross-entropy on firstB seed papers; context output rows have no direct label loss |
| Logging | Epoch mean is weighted B/804; optimizer steps use batch means without this weight |
| Validation | Full-graph CE every epoch; earliest strictly lowest CE saves checkpoint |
| Selection | Choose LR per arm by mean best validation CE across3seeds; tie smaller LR; freeze before scoring test |
| Test | Full graph, full2818 test targets; accuracy and macroF1; one selected checkpoint per seed/arm |
| Budget | 4arms×2rates×3seeds=24fits,960epochs,3840updates; equal update budget is not equal capacity/compute |
| Uncertainty | Descriptive sample SD over3 initialization/sampling seeds on one fixed graph/split; no dataset-level inference |
| Saved | All loss curves, selected checkpoints and SHA256, sample fingerprints, test logits/labels/predictions, parameter counts, elapsed seconds, source hash |
| Resume | No automatic resume; each suite invocation trains fresh and overwrites its named output/checkpoint directory |

All24 fits completed. Mean accuracy: R-GCN49.5387%, HGT88.8100%, uniform HGT87.1067%, MLP70.9605%. R-GCN's selected models predict only the majority class. This failure is retained; the fixed budget is not an optimized architecture benchmark. HGT−uniform mean gap1.7033pp is conditional on this one protocol. All three paired gaps and SDs are visible in the lesson.

The MLP is also passed sampled batches to equalize target exposure; its timing includes redundant sampler work. Full-graph validation/test fit locally. No peak-memory, asymptotic-scaling or isolated model-speed claim is made. L099 differs in number of updates, sampling and epoch budget; cross-lesson deltas are not a clean sampler ablation.

## Correctness and independent evidence

- `_verify_l100.py`: typed endpoint reconstruction; seed-only gradients and context-label intervention; invalid batch weights; all-neighbor logit/gradient equivalence on a fixture with isolated node/unequal degrees/empty stores; held-out-label training intervention for all four arms; paired samples and common ablation weights.
- `_audit_batches_l100.py`: complete real ACM graph,23 fixed training seeds, batch sizes1/7/23, all4arms. Float64 output and gradient tolerance2e-6; all gaps below7e-16. Probes are not exhaustive all-seed/all-batch proofs.
- Full-neighbor gradient parity holds at fixed weights with B/N weighting and no optimizer steps between batches. It does not imply sequential Adam trajectory equality or unbiased finite-fanout attention.
- `_audit_l100.py`: restores every checkpoint, recomputes validation loss and all12 selected prediction sets, independently reconstructs sklearn metrics/selection and raw incidence counts, checks all actual sample fingerprints.
- `_execute_l100.py`: independent bounded downloads into an empty temporary working directory, notebook hash checks, no repository imports; fresh full24-fit notebook replay plus fresh ten-run AIFB appendix. Its result file states actual completion; same-author-runtime portability is distinct from clean installation.

## Paper protocol audit

| Item | R-GCN AIFB named replay | HGT CS named track |
|---|---|---|
| Published target | Schlichtkrull Table2 AIFB95.83% mean/10runs | Hu Table2 CS Paper–Field L2, NDCG.403±.041, MRR.439±.078 /5runs |
| Model | Release featureless R-GCN, width16,2layers,91supports, no basis compression | Modern pyHGT port; full-setting width256,3layers,8heads,RTE |
| Objective | Train-only4class CE | Multi-label field KL, release field candidates/masks |
| Optimizer | Reconstructed Keras Adam equation,lr.01, noL2/dropout | AdamW .001,decay.01,clip.25, cosine schedule; full audit in L093 |
| Data | Full8285nodes,45raw relations,29043triples; mirrored bytes hashed | Exact8,606,617,611-byte CS download hashed; historical snapshot identity unknown |
| Split | Original140train/36test; original TSV bytes match pinned source | Released time/task boundaries and removal of target edges |
| Training | All50updates×10declared seeds0–9; fixed release final epoch | 200epochs×32batches×2updates; seed batch256, depth6,width128;5seeds0–4 |
| Selection | Fixed50epochs, no test selection | Validation KL for paper-settings port; modern release default uses validation NDCG |
| Evaluation | Original36test nodes, accuracy, mean across10 | 10sampled test batches per run; source ranking conventions, NDCG/MRR |
| Source | `4bec1341dd46b72bf482f7ed26c2dca4533577f6` | Modern `85eaccd482bc1d1af56c2de297b6e3a88b96d5cd`; publication-era `fd4a244db8efc72410537f3effec3b0c432892f7` archived |
| Executed here | Fresh10runs:95.833333% mean,1.464017pp sample SD | NOT_RUN; definitions/config and launch syntax checked only |
| Deviations | PyTorch vs Keras/Theano; declared RNG/canonical node order; graph historical byte identity unestablished; historical tuning unrun | Modern operator differs from publication-era prior/normalization/dropout/time encoding; serial RNG vs historical multiprocessing; exact table-producing revision/data identity unknown |
| Verdict | Numerical CLOSE, historical INCOMPARABLE; full paper NOT_ESTABLISHED | Full training NOT_RUN; historical INCOMPARABLE; full paper NOT_ESTABLISHED |

Full visible source/trainer for each named lane is included in notebook appendices. The core course implementation cannot substitute for them. The publication-era trainer remains archived in `labs/sources/hgt-l093/publication-2020/`, with its license and historical environment requirements; it has not been executed here. See `l091-reproduction.md` and `l093-reproduction.md` for inherited source investigations.

## Re-run commands (from repository root)

```bash
.venv/bin/python labs/_verify_l100.py
.venv/bin/python labs/_audit_batches_l100.py
.venv/bin/python labs/_run_l100.py
.venv/bin/python labs/_audit_l100.py
.venv/bin/python labs/_paper_l100.py --target aifb
.venv/bin/python labs/_figures_l100.py
.venv/bin/python labs/_build_l100.py
.venv/bin/python labs/_execute_l100.py
.venv/bin/python labs/_build_l100.py
.venv/bin/python labs/_delivery_l100.py
```

Tested runtime pins are in `requirements-l100-runtime.txt`; full observed environment in `requirements-l100-observed.txt`. Native source build:

```bash
MAX_JOBS=2 .venv/bin/python -m pip install --no-build-isolation 'git+https://github.com/pyg-team/pyg-lib.git@edc9e2a88d1c5d0953b5f69c98b8365597c6b699'
```

Install matching torch, PyG and build tools first. The existing native extension was used; L100 does not claim a fresh rebuild or clean environment validation. Live Colab **NOT_CHECKED**.

HGT CS, on a suitably provisioned host (full training unrun):

```bash
.venv/bin/python labs/_fetch_l093.py --dataset CS
.venv/bin/python labs/_paper_l100.py --target hgt-cs --device cuda
# Same complete canonical HGT source on Modal; fresh L100 evidence volume:
.venv/bin/modal run --detach modal/l100_paper_repro.py --preset paper
```

Author data link: https://drive.usercontent.google.com/download?id=1RYrGoBAIVMuYnFxznBgsmhxRLZTWtsM9&export=download&confirm=t . Expected CS SHA256: `bf054d93f45d385691f89902e7a021c53328c2ffefe50eb4d91c9dc51d05877f`. The standalone notebook exposes `CS_PATH` and `RUN_HGT_CS=False` with the complete visible port and five-run loop. The checksum is mandatory before deserializing the trusted release.

L093 previously measured MemoryError under a10GiB address-space guard and an account-level A10G rejection requiring a payment method. Those findings are referenced, not rerun in L100. This host currently has about15GiB RAM. No local unguarded CS load or cloud GPU job was launched here. The remote operator requests64GiB/A10G; full-epoch capacity and historical parity are still unverified. Prior source checks and successful small NN work do not prove CS execution.

## Delivery and mastery

The lesson, notebooks, reference, portable figures, manifest and Pages staging are checked by `_delivery_l100.py`; actual status is in `_delivery_l100_results.json`. Author package creation never marks the learner complete. No publication requested; deployment **NOT_CHECKED**. Notebook execution and browser delivery are independent of paper-result reproduction.
