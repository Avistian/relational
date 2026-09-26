# Lesson 106 reproduction contract

## Declared target and boundary

Poursafaei et al., *Towards Better Evaluation for Dynamic Link Prediction*, NeurIPS 2022, arXiv v2 Appendix B Wikipedia rows: Tables 2/3 (random AP/AUROC), 4/6 (historical), 8/10 (inductive). Two EdgeBank variants, three samplers, five original loop iterations, every test event. This is a full **selected Wikipedia released-code replay**, not the entire paper benchmark. Other datasets and all neural-model training/evaluation are NOT_RUN.

Targets (AP, AUROC): unlimited random (.90,.91), historical (.50,.49), inductive (.48,.43); window random (.87,.87), historical (.71,.77), inductive (.46,.40). Before execution, CLOSE was declared as both means within .015 absolute of the rounded published values. FAIL is retained when this is not met. This numerical criterion compares means only. The lesson also displays the paper’s reported SD and replay population SD; differing variability is retained, not treated as a successful SD reproduction. The criterion never establishes exact historical protocol identity.

## Exact commands

From repository root; author environment Python 3.12.3:

```bash
python3.12 -m venv /tmp/l106-replay
/tmp/l106-replay/bin/python -m pip install -r labs/requirements-l106-authoring.txt
/tmp/l106-replay/bin/python labs/_check_l106.py
/tmp/l106-replay/bin/python labs/_provenance_l106.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/l106-replay/bin/python labs/_run_l106.py --raw labs/data/l102/wikipedia.csv
/tmp/l106-replay/bin/python labs/_audit_l106.py
/tmp/l106-replay/bin/python labs/_figures_l106.py
/tmp/l106-replay/bin/python labs/_build_l106.py
/tmp/l106-replay/bin/python -m ipykernel install --prefix /tmp/l106-replay --name python3
/tmp/l106-replay/bin/python labs/_execute_l106.py
/tmp/l106-replay/bin/python -m playwright install chromium
/tmp/l106-replay/bin/python labs/_mutation_l106.py
/tmp/l106-replay/bin/python labs/_delivery_l106.py
```

In the existing workspace use `.venv/bin/python`. Runtime replay needs the runtime requirements, visible module, runner and complete `labs/sources/l106` directory. The provenance check authenticates source before extracting just the original `reindex` function, avoiding its top-level command-line execution. Browser execution may require Playwright OS dependencies.

`_run_l106.py` downloads the raw file when absent and always hashes it. URL: https://snap.stanford.edu/jodie/wikipedia.csv . SHA256: `a6b73e09c0d1e5b9db11e7e7aa416f2e87838a745273e0446a79952cf4cfae09`. Size: 559,937,473 bytes. No existing processed cache is trusted. All candidate generations execute anew; there is no partial-run resume. A 45-minute cutoff protects the local replay; it does not silently downgrade the scope. The run uses local CPU and $0 paid compute.

## Protocol audit

| Dimension | Executed protocol | Evidence boundary / deviation |
|---|---|---|
| Source | DGB commit `7793e9449f5321c7e39b24c0585e3c3de7cf9f5e`, 2022-09-09 | Publication-era source, not proof it produced the paper tables |
| Data | All 157,474 authenticated JODIE Wikipedia events | Historical publication bytes not independently available for identity proof |
| Projection | User+1, page+8228, original timestamps; directed pairs | Labels/features unused by EdgeBank; original reindex independently checked |
| Split | Timestamp .70/.85 quantiles; seed-2020 sample of 922 held nodes removes their early training edges; validation retained | Compare complete split IDs to original loader. Python 3.9 set sampling adapted to tuple on Python 3.12 |
| Initial memory | Released filtered train + full validation; 104,650 events | Not all events before test cutoff; withholding changes initial history |
| Test | Full 23,621 interactions, batches of 200 including final 21 | Source labels “Old nodes” but actually uses the complete test split, including unseen nodes |
| Unlimited model | Directed pair-set membership | No embeddings, optimization, initialization, pretraining, loss or learned parameters |
| Window model | History timestamp .85 quantile, inclusive retention, recomputed every batch | Paper describes a fixed duration; source window duration changes |
| Update | Score entire batch then append observed positives | Strict per-event/tie-group causality is a separate teaching contract; audit tied boundary exposure |
| Random sampler | Original complement of positive pairs; then caller replaces sampled sources | Can reintroduce collisions; no silent fix |
| Historical sampler | Original interval-set exclusion and random fallback | Full endpoint universe known; no operational availability record |
| Inductive sampler | Historical pool excludes train/validation pair set; random fallback | Pair novelty is not new-node induction |
| Randomness | Seed 2; five original iterations | Random resets local state; adversarial methods consume global state. No invented independent-seed claim |
| Metrics | Equal positive/negative counts; arithmetic mean of batch AP/AUROC | Pooled scores saved separately; no test-selected thresholds used |
| Source validation | Every batch/run/memory prediction checked against original model; sklearn oracle checks every batch metric | Original ancillary threshold metrics not replayed; not needed for selected AP/AUROC targets |
| Environment | Recorded Python/NumPy/platform; pinned current packages | Modern runtime and set ordering can differ from historical run |
| Notebook | Authenticated embedded original candidate arrays; all predictions freshly recomputed | Candidate regeneration switch defaults OFF; author runner separately executes fresh generation |

Original loader feature files are replaced by tiny dummy arrays only during the split oracle: the splitter and EdgeBank never use them. Zero labels supplied to that oracle are likewise unused. These are explicit I/O adaptations, not trained-model feature parity claims.

The two memories use identical candidate arrays. Original code launches each memory separately with the same sampler initialization; memory computation does not consume RNG state. This pairing preserves that sequence while avoiding duplicate candidate generation. Random iterations may be byte-identical. Population SD over the five released iterations is descriptive, not an uncertainty estimate for future datasets.

## Evidence and replay integrity

`_sources_l106.json`: pinned source hashes and predeclared targets. Original code and MIT license are retained under `sources/l106/`.

`_analysis_l106_results.json`: complete per-run/per-batch AP/AUROC, means, SD, target deltas, numeric statuses, collisions, tie exposure, split identity, runtime and code/artifact hashes. No numerical conclusion is inferred from a partial run.

`evidence/l106/`: original negative edges, complete split/held-node IDs, and positive/negative predictions for each condition and run. The notebook embeds the split and candidate bytes with SHA256 checks; no unpublished URL or repository import is required for the default scoring replay. The optional visible original-sampler cell regenerates candidates. A regenerated stream that differs under another Python/set environment must be reported as such, not silently called historical identity.

`_audit_l106_results.json`: persisted positive scores fixed across samplers, independently encoded collision counts and pooled sklearn metric checks.

`_provenance_l106_results.json`: independent parser + source reindex comparison. `_check_l106_results.json`: causal, memory, tie and invalid-input fixtures. `_mutation_l106_results.json`: checks reject incorrect implementations. `_execution_l106_results.json`: actual solution execution. `_delivery_l106_results.json`: actual browser and copied-site checks.

Student checks and source replay serve different purposes. A stricter causal gate is taught and tested, but not substituted into the published replay. Correcting sampler collisions or window semantics would require an explicitly named new experiment.

Learner status: PENDING_WRITTEN_DEFENSE. Live Colab and deployment: NOT_CHECKED. No publication was requested or performed.
