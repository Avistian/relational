# Lesson149 — weakest RelBench tasks

Approved2026-09-30: published weakness catalog plus fresh complete F1 basic RDL and manual-FE replay, aggregate USD10 cap. Author work is distinct from learner completion.

## Named experiments and boundaries

RelBench arXiv2407.20060v1 Table7 basic RDL rel-f1/driver-position: five fresh seeds0–4, ten complete epochs each,7453training queries/epoch,499validation/760test. Validation3.172598±.021711;test4.123071±.235929MAE(sampleSD). Paper3.193/4.022, predeclared descriptive tolerance±.2: CLOSE/CLOSE. This is not equivalence or historical identity.

Released user-study SQL +LightGBM: five fresh ten-trial searches plus selected train-only refits,50trials total; validation2.777330/test3.948917meanMAE. Full selected released pipelines COMPLETE. Original FE scalar/randomness unavailable; historical identity NOT_ESTABLISHED. Human feature ideation not repeated; whole-paper training NOT_RUN. Figure3 regression uses GNN+LightGBM, whereas this replay uses the Table7 basicGNN. Do not substitute this replay into that figure.

Catalog: all15Figure3user-study mean bars extracted from original SVG endpoints against printed ticks; eight classification and seven regression, separate rankings. Largest negative point gaps: driver-top3 (about−4.1AUROC points) and item-sales (about−.35normalizedMAE). Source labels checked visually. Figure user-votes vs Table7 post-votes unresolved: preserve literal figure label. Status PLOT_DERIVED; no original scalar/error-bar recovery, no significance ranking. Regression normalization divides by RDL score and changes cross-task magnitudes. All other RelBench tasks excluded from this particular catalog. H&M diagnosis is source-grounded hypothesis only; fresh H&M training/intervention NOT_RUN.

## Source and protocol

GNN source9aa346267c2e1c560bd92da07d6f4ad1ca2f0639; full visible model relkit/rdl_l117.py, trainer relkit/tuning_train_l135.py, preprocessing _run_l117.py. Two128-channel GraphSAGE layers, typed FrameResNet encoders, relative-time encoding, sum neighbor/relation aggregation, Adam.005, fanout128/64,batch512,L1loss. First strict validation minimum. Evaluation clips to train-label2nd/98th percentiles; source-matching evaluation resampling can make final validation differ from checkpoint-selection validation. Source forward replay checks every saved held-out prediction; maximum original-model error1.90735e-6. PyG/Frame kernels remain pinned dependencies. Gradient health was not independently audited in L149; forward parity does not establish it.

FE source445bb7a3b1230f49f8e5890ae81754d3e365680f; Frame0.2.2 release with LightGBM4.3.0; visible relkit/fe_experiment_l129.py and original sources/l129. All50engineered features plus numeric driverId; date ignored. Train-fitted mappings, seeded TPE10trials/run,2000round cap,50round early stopping, train-only selected refit. Explicit seed and4threads adapters; released SQL row order frozen to archive order. Original Frame trainer two-trial check matches predictions exactly. Independent Python tree traversal checks every primary FE prediction; SQLite independently scores MAE. Fresh SQL outputs checked across1/4threads.

Complete source model and trainers appear inline in both notebooks, with independent FE and GNN full gates OFF by default. Default execution audits saved author predictions. Historical FE/GNN dependency stacks differ and are validated separately. Main runtime pins: requirements-l117-runtime.txt and requirements-l129-runtime.txt. Transitive dependencies not fully historically recoverable.

Archives: dbSHA256ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482;taskSHA256775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e. Each GPU run reconstructs graph from verified archives; FE freshly materializes original SQL. FE raw archives/source files reused under checksums. Original staging endpoint unavailable; v1 archives replace it. Labels independently reconstructed for all8712queries from raw future results in(cutoff,cutoff+60days]. Graph census74063rows; independent SQL edge counts. Upstream bytes verified in _upstream_l149.json and evidence/l149/sources.json.

GNN preprocessing uses the database snapshot through2010-01-01, not training-only rows. FE history uses strict past, GNN sampling inclusive cutoff. Every sampled query retains its own cutoff. Future scheduled dates in FE have unknown publication timestamps. Static attribute mutation/arrival histories unavailable. Actual historical availability NOT_ESTABLISHED. Complete-pipeline comparison, not isolated causal architecture experiment.

## Frozen slice profile

Loss penalty D=abs(y−GNN)−abs(y−FE): positive means GNN loses. This is the negative of MAE advantage used in catalog. All12590primary held-out predictions matched by full(entity,cutoff) keys; common archive float64 target; GNN internal float32 difference disclosed. Six complementary slices: history≤22(trainingmedian),recency>180days,recent-slot missingness≥.5 plus complements. Require30queries/10drivers. Nominate largest supported positive validation penalty with lexical tie break; no eligible slice means none. Frozen nomination high_history before test-slice analysis; trained test artifacts already existed. Test reused in earlier lessons: exploratory, not pristine confirmation.

High-history validation+.554406;test+.079944,conditional95%driver-cluster interval[−.397691,.638169]. Global test+.174155,interval[−.085850,.449621]. Both test intervals cross zero. Low history is not the largest validation weakness; cold start/no-history is not equivalent to low history. Empty/unsupported slices retained, no fabricated intervals.

Bootstrap2000whole-driver draws,seed137 retained from prior protocol, query-weighted mean after averaging five run losses per query. Conditional on fitted models/splits; shared race/time dependence and across-training/database uncertainty not covered. Matched integer run labels do not create common random numbers. Proposed explanatory interventions NOT_RUN.

## Execution commands

From repository root, in a fresh isolated output/volume namespace for another reproduction; operators refuse duplicate dispatch/overwrite. `_new_run_l149.py --directory /tmp/l149-repeat-001` prepares an isolated GNN replay without launching paid work. Existing reservations are consumed, not reusable authorization.

```bash
.venv/bin/python labs/_prepare_l149.py
.venv/bin/modal run --detach modal/l149_repro.py --mode pilot
.venv/bin/python labs/_collect_l149.py --mode pilot
.venv/bin/python labs/_pilot_check_l149.py
.venv/bin/modal run --detach modal/l149_repro.py --mode final
.venv/bin/python labs/_collect_l149.py --mode final
.venv/bin/python labs/_analyze_l149.py
.venv/bin/python labs/_audit_l149.py
# In isolated pinned FE runtime:
# _prepare_fe_l149.py → _audit_fe_l149.py → _slices_l149.py
# _run_fe_l149.py --seed N --output fresh/path (N=0..4; default ten trials)
# _analyze_fe_l149.py after collecting GNN fits
.venv/bin/python labs/_freeze_l149.py
.venv/bin/python labs/_errors_l149.py
.venv/bin/python labs/_catalog_l149.py
.venv/bin/python labs/_figures_l149.py
.venv/bin/python labs/_build_l149.py
.venv/bin/python labs/_execute_l149.py
# Pinned FE: _execute_fe_l149.py; GPU:
.venv/bin/modal run --detach modal/l149_notebook_check.py
.venv/bin/python labs/_collect_notebook_l149.py
.venv/bin/python labs/_verify_l149.py
.venv/bin/python labs/_delivery_l149.py
```

Fresh author training lanes do not overwrite old lessons. Full standalone notebook can run in an empty directory with its full gates enabled in their separate pinned environments. Those gates expose the same model/trainer code but do not enforce dollar limits; paid operator reservations/timeouts do. Live Colab frontend NOT_CHECKED.

## Cost and delivery

T4.000164/s+2physicalCPU*.0000131/s+16GiB*.00000222/s=.00022572USD/s, checked2026-09-30 at https://modal.com/pricing. Pilot and five primary900s slots reserveUSD1.218888; full GPU notebook900s reserveUSD.203148. After strengthening the notebook ranking-unit contract, a second900s full-GPU validation reservation addsUSD.203148. Both first and final notebook evidence are retained; training code unchanged. Total reservation+USD3overhead=USD4.625184;USD10aggregate cap, no automatic retries. Pilot26.036s projected390.545s per full fit and passed cutoff. Primary measured worker estimateUSD.065666; final total in _verify_l149_results.json includes notebook validation. Reservations are conservative resource allowances, not invoices; startup/build/storage not itemized. FE runs on local CPU,USD0cloud; main50trial search completed well within3600s cutoff. Two-trial source parity and two full50trial notebook checks are separate verification, not extra primary seeds.

Delivery reports record exact executed code hashes, desktop/mobile/keyboard/noJS/print/copy-Pages checks. Live Colab/deployment NOT_CHECKED. Learner PENDING_WRITTEN_DEFENSE. No push or deployment requested.

Final verification PASS: both final25-code-cell notebook runtimes match codedigestdb517c2d04df4ca8e1bd3d1e7a391a9af7c47094c573094088d4bdeb1be765da. Final GPU validation five complete fits,6295independently rescored predictions; final FE notebook50trials/five refits,6295exactly compared predictions. First revision validations retained and charged; full training appendix unchanged. Measured primary+both GPU notebook worker estimateUSD.150529941; conservative reservations+overheadUSD4.625184. Primary FE search35.953s, final FE notebook40.440s; local CPU cloud spendUSD0. See _verify_l149_results.json.
