# L110 — full selected TGN-attn Wikipedia reproduction and clean-state checkpoint

## Frozen scope

Rossi et al., arXiv:2006.10637v3 Table 2, TGN-attn Wikipedia all-event AP 98.46 ± 0.1% and new-node AP 97.81 ± 0.1%. Ten **fresh** release-compatible fits, explicit seeds 0–9, complete released populations. Two mean scores are numerically CLOSE when their absolute published-target gap is at most **0.5 percentage point**, fixed before launching. This is descriptive, not an equivalence test.

Ten additional fresh fits use a course clean-state policy. They are not assigned published targets. The intervention combines atomic selected-state restoration, timestamp-group batching, and best-validation selection at the epoch cap. No component-specific accuracy attribution is justified. Earlier L102 results are not counted among these new seeds.

Current execution status and independent metrics: `evidence/l110/summary.json` (generated when evidence is collected). Full-paper reproduction NOT_ESTABLISHED; historical identity INCOMPARABLE. Student mastery PENDING_WRITTEN_DEFENSE.

## Exact commands

Run from the repository root; use a separate environment for the pinned GPU-equivalent libraries. Existing `.venv` CPU libraries are recorded by the execution report and are not silently replaced.

```bash
python3 -m venv /tmp/l110-runtime
/tmp/l110-runtime/bin/pip install -r labs/requirements-l110-runtime.txt
OMP_NUM_THREADS=1 /tmp/l110-runtime/bin/python labs/_check_l110.py
OMP_NUM_THREADS=1 /tmp/l110-runtime/bin/python labs/_source_check_l110.py
OMP_NUM_THREADS=1 /tmp/l110-runtime/bin/python labs/_audit_l110.py
OMP_NUM_THREADS=1 /tmp/l110-runtime/bin/python labs/_protocol_l110.py
OMP_NUM_THREADS=1 /tmp/l110-runtime/bin/python labs/_causality_l110.py
OMP_NUM_THREADS=1 /tmp/l110-runtime/bin/python labs/_state_check_l110.py
# Small teaching run, never used for a paper comparison:
OMP_NUM_THREADS=1 /tmp/l110-runtime/bin/python labs/_run_l110.py --preset smoke --arm both --seeds 19 --output labs/results/l110/new-smoke
# Full twenty-fit schedule; use an external runtime/cost limit if using your own GPU:
OMP_NUM_THREADS=1 /tmp/l110-runtime/bin/python labs/_run_l110.py --preset paper --arm both --seeds 0,1,2,3,4,5,6,7,8,9 --device cuda --output labs/results/l110/new-full
# Author cloud schedule: one pilot, then ten bounded paired workers; no automatic retries.
.venv/bin/modal run --detach modal/l110_repro.py --mode pilot
.venv/bin/modal run --detach modal/l110_repro.py --mode paper
# Download existing artifacts without submitting more training:
.venv/bin/python labs/_collect_l110.py --checkpoints
OMP_NUM_THREADS=1 .venv/bin/python labs/_replay_l110.py
.venv/bin/python labs/_analyze_l110.py
.venv/bin/python labs/_figures_l110.py
.venv/bin/python labs/_build_l110.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_execute_l110.py
.venv/bin/python labs/_mutation_l110.py
.venv/bin/python labs/_delivery_l110.py
```

The complete model, preprocessing, trainer and evaluator are inline in the student/solution notebooks. The default runs two short real-data fits after full-data split/frontier checks. `RUN_FULL_REPRODUCTION = True` executes both full ten-seed arms with the live student functions. Author scientific checks and source replay are repository commands, not silently substituted notebook outputs.

## Provenance and protocol

| Dimension | Frozen choice |
|---|---|
| Source | twitter-research/tgn `e38cdf85998c6ca077167610dc4e769a688efa95`; unmodified archive and Apache-2.0 license in `sources/l102/` |
| Model | Complete one-layer TGN-attn; 172 memory/time dimensions; two heads; ten most-recent strict-past neighbors |
| Message/update | 688-wide identity concatenation; latest message per node; GRU updater |
| Features | 172 interaction features; zero node features; padding ID 0 |
| Inert source computation | Released global inter-event normalization statistics are unused by this graph-attention embedding; the port omits that inert computation |
| Initialization | Source-equivalent constructors, cosine frequencies and Xavier-normal merge weights; no pretrained checkpoint |
| Raw data | `https://snap.stanford.edu/jodie/wikipedia.csv` |
| Raw SHA-256 | `a6b73e09c0d1e5b9db11e7e7aa416f2e87838a745273e0446a79952cf4cfae09` |
| Populations | 157474 full; 81029 train; 23621 val/test each; 12016 new-val; 11715 new-test |
| Split | 70%/85% time quantiles; 922 held nodes; Python split seed 2020; exact source event arrays checked |
| Visibility | Training-only adjacency for fitting; full adjacency for evaluation with strict-past lookup; event=availability assumption |
| Objective | Mean positive BCE plus mean negative BCE; one destination negative per event |
| Negatives | Preserve unused source RNG draw; no positive/existing-edge rejection; full destination pool for val/all test, new-test pool for new test |
| RNG | NumPy global training stream; val seed0; test seed2; new test seed3; explicit independent model seeds0–9 |
| Optimization | Adam lr1e-4, dropout0.1, nominal batch200, maximum50 epochs; state detaches between batches |
| Selection | All-event validation batch AP; patience5; relative tolerance1e-10 |
| Release selection | Restore best state_dict only on early stop; retain stopping queue; final epoch if maximum reached |
| Clean selection | Always restore best weights + memory + clocks + queue; ties kept together in train and all evaluation populations |
| Epoch/branch reset | Zero memory/clocks/queue each epoch; validation branches share train snapshot; test branches share selected post-validation snapshot |
| Metrics | Published comparison uses equal mean of batch AP; separately report pooled AP; predictions include batch IDs |
| Uncertainty | Sample SD across seeds on one fixed split; paired clean-minus-release differences are descriptive |
| GPU | Python3.12, torch2.8.0, numpy2.2.6, pandas2.3.2, scikit-learn1.7.1, T4; one Torch CPU thread |
| CPU | Actual installed runtime recorded in standalone notebook; separate engineering/teaching evidence |

Saved `epoch` and `selected_epoch` values are zero-based; `epochs_completed` is a count.

The clean arm preserves released evaluation population and candidate-pool definitions, not exact per-event negative arrays. Changed batch sizes interleave otherwise-unused source draws and destination draws differently. Changed batching also changes dropout and training RNG trajectories. A pooled AP difference remains a comparison of these full policies, not an isolated checkpoint effect.

An indexed future edge cannot enter the strict-past neighbor result. Node-universe size and negative pools are nevertheless benchmark-defined and may contain identities first occurring later. This is a closed released evaluation universe, not proof of deployability with unknown future identities. Real ingestion timestamps are absent. The two-clock task and late-arrival fixtures test a generalization of the release's event-time contract.

## Audits

- `_check_l110.py`: next-prediction recovery after state/parameter changes; tie grouping; two-clock eligibility; five rejected mutants.
- `_source_check_l110.py`: original-source output, gradient, initialization and temporal-state comparison in the actual pilot GPU runtime.
- `_audit_l110.py`: original preprocessing on every raw row, all six exact split arrays, 720 independently enumerated temporal neighborhoods, snapshot recovery and negative-message exclusion.
- `_protocol_l110.py`: full batch-frontier census; original/new release-trainer exact probability regression; forced clean early-stop branch.
- `_state_check_l110.py`: fixed-weight current/future feature exclusion and incomplete-snapshot counterexample.
- `_causality_l110.py`: actual equal-time feature intervention (weights/candidates fixed), clean invariance, later legal effect and evaluation-order independence.
- `_resume_l110.py`: exact completed-seed reuse and rejection after changing a called eligibility helper; no partial optimization resume.
- `_provenance_l110.py`: authenticate all 21 archived source files, scientific source bytes and recorded environments.
- `_replay_l110.py`: replay all downloaded fresh checkpoint probabilities in the distinct local CPU runtime.
- `_metrics_l110.py`: 500 tied-score randomized AP checks and the explicit batch-versus-pooled counterexample.
- `_analyze_l110.py`: reconstruct batch AP independently using score-tie groups, check against saved metrics, pooled AP, data/seed identity and artifact hashes.
- `_replay_source_l110.py`: complete fresh release seed0 prediction replay through original model/evaluator on GPU.
- `_mutation_l110.py`: student blanks are live; broken solutions fail the notebook's actual CHECK cells.
- `_delivery_l110.py`: canonical code alignment, executed solution, images, browser interaction/accessibility and realistic copied Pages links.

Every clean preceding batch ends strictly before the next. Counts and maximum groups are recorded in `_protocol_l110_results.json`. These checks establish the stated event-time contract; they cannot establish missing ingestion history. A snapshot-only inference artifact is not a mid-epoch optimizer resume. Partial runs have progress traces but never count as completed seeds.

## Budget and run identity

The user approved the combined scope with a USD10 aggregate ceiling. Prices checked 2026-09-26 at https://modal.com/pricing: T4 .000164/s + two CPU cores .0000262/s + 8GiB RAM .00001776/s = .00020796/s. One pilot has timeout600; ten paired workers have timeout3500 each. Maximum allocated function-resource time is35600 seconds, USD7.403376; reserve USD2.596624 for startup/build/storage and incidental overhead. No automatic retries. The pilot actually took171.170263631 seconds, about USD0.03560. Completed-call resource estimates are not a billing invoice.

Pilot app: https://modal.com/apps/pszar92/main/ap-s9EDA34nOJPHBvabMpsfbN

Fresh full app: https://modal.com/apps/pszar92/main/ap-7k4fDUBAhgpqzrEmgje9RF

Persistent volume `l110-checkpoint-evidence`, source-hash path recorded in `_sources_l110.json`. Both arms have per-seed weights/temporal snapshots, predictions, trace, and results; run identity records input and environment. No source code changes are permitted within one scientific run identity. Local CLI reuse requires matching source/data/runtime identity and completed artifacts; partial epochs restart and do not establish optimization resume.

## Unrun and evidence boundaries

Reddit, Twitter, dynamic node classification, all competitor training, the ablation suite, historical torch1.6 runtime and original continuously consumed RNG streams are NOT_RUN. The original data's historical identity is not established solely by hashing the present download. Close scores do not change that boundary. Real arrival histories are NOT_RECORDED. Live Colab and deployment remain NOT_CHECKED. Browser/copy checks are recorded separately from experiment results. Prepared learner status remains PENDING_WRITTEN_DEFENSE.
