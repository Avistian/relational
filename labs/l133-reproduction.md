# Lesson133 reproduction contract

## Named target and scope

RelBench arXiv2407.20060v1 Table7, rel-f1/driver-position, basic RDL. Five fresh seeds0–4; all released data and ten full epochs. Published validation/test MAE means3.193/4.022, reported SDs.024/.119. Predeclared descriptive tolerance: absolute mean difference≤.2 for each split. CLOSE is not a statistical equivalence test. Primary paper: https://arxiv.org/html/2407.20060v1.

Actual execution: `evidence/l133/summary.json`. Local behavioral evidence: `_verify_l133_results.json`. Default solution execution: `_execution_l133_results.json`. Whole-paper/historical parity NOT_ESTABLISHED; other29tasks NOT_RUN here. Learner PENDING_WRITTEN_DEFENSE.

## Measured outcome

All five fresh ten-epoch fits completed. Validation MAE **3.195830 ± 0.042147**; test **4.170602 ± 0.292558** (mean ± sample seed SD). Both means are descriptively CLOSE under the frozen0.2 tolerance. All6295final predictions independently rescored;403895sampled query occurrences audited. The largest original-GPU-model replay discrepancy was3.81469727e-06. All five first-layer CPU float64 shadow checks passed.

Successful recovery-pilot plus five-fit worker estimate: **USD0.063732**. Including the failed pilot's full reserved upper bound givesUSD0.876324, before the separateUSD2.686672overhead reserve. Billing is not itemized; this is an estimate with conservative reservations, not an invoice.

All24default solution code cells executed independently in an empty directory. Browser tests cover54interactive states at1200px and375px, keyboard/reset, no-JS and print; four portable figures and copied-Pages links passed. All24portable code cells also passed in a separate pinned GPU Python namespace, including five fresh ten-epoch fits and five actual-layer checks. All6295validation-run predictions were independently rescored. The code path is verified; Google Colab frontend remains NOT_CHECKED.

The separate portable validation returned validation3.186997 ±.021048 and test3.962961 ±.086268. These are verification runs, not replacements for the primary table or ten independent seeds. GPU sampling/reductions need not replay bit-for-bit. Their worker estimate wasUSD.044592; total successful worker estimateUSD.108324. With the failed-pilot reserved bound, worker estimate/bound isUSD.920916, plus the separateUSD2.686672overhead reserve. All remain belowUSD10.

## Frozen protocol

| Axis | Contract and deviations |
|---|---|
| Source | RelBench9aa346267c2e1c560bd92da07d6f4ad1ca2f0639; upstream files and licenses in sources/l117; hashes in _sources_l133.json |
| Database | Full archived rel-f1 database through2010-01-01; nine tables,74063rows,338842directed edges |
| Database SHA256 | ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482 |
| Task SHA256 | 775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e |
| Queries | 7453train /499validation /760test, keyed by entity and nanosecond cutoff |
| Cohort | Released future-participation conditioning, not a purely past-only operational cohort |
| Features | Keys removed; released full-database statistics through test cap, not train-only; type inference seed42 |
| Text | average_word_embeddings_glove.6B.300d at e5e8fec6971be8960cfaa853a77a6ddc62a265d7 |
| Encoders | Separate four-block Frame ResNets,128channels; table-specific relative-time encoders |
| Graph layer | Two layers; per-directed-relation sum-SAGE with neighbor bias and root transform; outer relation sum; node LayerNorm then ReLU |
| Sampling | Uniform temporal fanouts[128,64]; paper table says128; all batches audited |
| Training | Adam.005, batch512, mean L1, ten full epochs; five fresh seeds0–4 |
| Selection | First strict validation MAE improvement; final evaluation of frozen selected checkpoint |
| Scoring | Clip to training-label2nd/98th percentiles; independent scalar MAE and original model replay |
| Validation | Final evaluation resamples contexts; can differ from checkpoint-selection score |
| Runtime | Python3.11,Torch2.5.1+cu124,PyG2.6.1,Frame0.2.3,RelBench1.1.0,pyg_lib0.4.0+pt25cu124; requirements-l117-runtime.txt |
| Unknown | Original training seeds, exact historical runtime/commit, ingestion histories and mutable-feature histories |

No model architecture, optimizer, split, schedule or selection rule was tuned to improve the numerical verdict. The primary model remains original PyG. The notebook makes the complete graph builder, model and trainer visible and provides a standalone full-training gate in the pinned runtime.

## Layer audit and failed pilot

The visible `relkit/hetero_l133.py` implements source gathering, destination scatter-add, relation-specific root and bias transforms, destination grouping and complete relation-layer composition. Three learner functions drive the synthetic optimization and the optional full-run shadow audit.

The first remote pilot failed before its first optimizer update: 38of65536 GPU output coordinates exceeded rtol=atol1e-5; the reported largest absolute discrepancy was3.814697265625e-05. Source snapshots, original budget and failure record are retained in `sources/l133/failed-pilot/`. Its full3600s cost reservation remains charged conservatively. No failed run is counted as a completed fit.

Recovery compares two cloned first-layer modules on CPU float64, preserving actual sampled input values and edges. This separates arithmetic from GPU scatter-add ordering. It checks all layer outputs, input-feature gradients and convolution-parameter gradients. Primary training still runs unchanged on GPU; final predictions also replay through the original GPU model. Each successful run retains `layer-trace.json`. This shadow check does not validate row-encoder gradients: L131 independently documented nonfinite numerical-encoder gradients in the released algorithm, which is retained here. No claim of healthy whole-model gradients is made.

The local verifier independently checks PyG outputs/gradients, an Adam update, separate source/destination row permutations, edge order, duplicate edges, empty versus absent relations and rejection of three semantic mutants. The tiny150-step synthetic fit is a course demonstration, not a paper experiment. The default portable notebook runs it and independently rescores6295 recorded author predictions. The full-training notebook code path was separately executed in the pinned GPU runtime; `_notebook_gpu_l133_results.json` records five fresh complete fits and five layer checks. Live Colab remains NOT_CHECKED.

## Commands

From the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_check_l133.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_verify_l133.py
.venv/bin/python labs/_source_check_l133.py
.venv/bin/python labs/_prepare_l133.py
.venv/bin/modal run --detach modal/l133_repro.py --mode pilot
.venv/bin/python labs/_collect_l133.py --mode pilot
.venv/bin/python labs/_pilot_check_l133.py
# Only after the fresh pilot admission gate passes:
.venv/bin/modal run --detach modal/l133_repro.py --mode paper
.venv/bin/python labs/_collect_l133.py --mode paper
.venv/bin/python labs/_analyze_l133.py
.venv/bin/python labs/_audit_l133.py
.venv/bin/python labs/_figures_l133.py
.venv/bin/python labs/_build_l133.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_execute_l133.py
.venv/bin/modal run --detach modal/l133_notebook_check.py
.venv/bin/python labs/_collect_notebook_l133.py
.venv/bin/python labs/_delivery_l133.py
```

For an independent repeat use a fresh Modal volume and a new cost ledger; preserve all completed evidence and old reservations. Alternatively, inside the pinned runtime:

```bash
for seed in 0 1 2 3 4; do
  python3 labs/_run_l133.py --seed "$seed" --epochs 10 --output "labs/results/l133/new-seed-$seed"
done
```

## Aggregate budget

USD10 maximum across pilots, seeds, retries and validation. Current2026-09-27 rate: T4.000164 +2CPU*.0000131 +16GiB*.00000222 =USD.00022572/second (https://modal.com/pricing). Reserve nine3600sworker slots =USD7.313328; retainUSD2.686672for overhead. No automatic retries. Each dispatch reserves worst-case time before launch. Completed runs cannot be overwritten, source fingerprints must match, and the full fit gate requires projected runtime below3600s per worker plus aggregate feasibility.

The recovery pilot took23.0374s; conservative pilot-plus-five-fit projection (charging ten times total pilot duration per full fit) isUSD.26520. Failed-pilot upper bound is an additionalUSD.812592; overhead reserve remains untouched by these estimates. Actual worker estimates and reservations are retained separately; startup/build/storage overhead is not itemized, so these are not invoices.

## Delivery

Desktop/mobile/keyboard/no-JS/print, notebook execution and copied Pages staging have separate checks. Working-tree delivery does not prove a clean committed checkout. No deployment requested; live Colab and deployment NOT_CHECKED. Author execution does not establish learner mastery.
