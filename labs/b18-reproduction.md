# B18 reproduction contract

## Executed experiment: B18-CONTEXT-SUFFICIENCY

Tier C synthetic information-loss diagnostic, not an Animus reconstruction or next-month forecast. Seeds0/1/2 × degrees8/64/512/4096 × uniform/concentrated amounts × budgets8/32/128/512 ×100 paired repetitions. Total USD1200, uniform amounts or80% in one credit and20% evenly elsewhere. Sample uniformly without replacement; pair budgets through one random prefix. Four estimates: full total, sample sum, N/k-corrected sum, complete monthly aggregation. Report signed bias and RMSE per degree/shape/seed/budget. No fitting, model weights, optimizer, checkpoint, split selection or hyperparameter tuning applies to this exact-sum task. Event availability equals event day in the course fixture; real arrivals/backfills require an additional availability cutoff. Full counts and aggregates use all eligible history; they are not equal-access comparisons to an unassisted truncated model.

Run from repository root, with NumPy, nbformat, nbclient, nbconvert and matplotlib installed:

```
.venv/bin/python labs/_budget_b18.py .venv/bin/python labs/_test_b18.py
.venv/bin/python labs/_budget_b18.py .venv/bin/python labs/_run_b18.py
.venv/bin/python labs/_budget_b18.py .venv/bin/python labs/_verify_b18.py
.venv/bin/python labs/_budget_b18.py .venv/bin/python labs/_build_b18.py
.venv/bin/python labs/_budget_b18.py .venv/bin/python labs/_execute_b18.py
.venv/bin/python labs/_reproduce_b18.py --phase audit
.venv/bin/python labs/_reproduce_b18.py --phase paper
```

The last command MUST exit nonzero: it is a source gate, not a hidden benchmark launcher. The portable notebook runs the same complete finite experiment without repository imports. A successful audit authenticates saved bytes, not the paper results.

## Published target: B18-ANIMUS-RT-RAW-AGG

Primary source: https://arxiv.org/html/2609.00460v1 (archived bytes and dated receipt under sources/b18).

Retain the full matrix: raw/Agg-Simple × num_blocks6/12 × learning_rate3e-5/1e-4/3e-4 =12 configurations. Published fixed values: d_model256, seq_len1024, max_steps32769, batch_size32, weight_decay0. Original100,000-customer Animus generator, full original task population and10,000-customer test split; train/validation/test cutoffs2024-03-31,2024-06-30,2024-09-30. Target is next-month income; future transactions may never be input. No replacement dataset, invented seeds, smaller schedules or substitute RT checkpoint authorized.

| Field | Current evidence / gap |
|---|---|
| Architecture / initialization | RT description and dimensions; exact source commit, initialization/checkpoint bytes not authenticated |
| Objective / optimizer | Income regression; exact loss, optimizer/schedule and checkpoint-selection implementation unresolved |
| Data / preprocessing | Paper generator narrative and aggregation description; generator bytes, generation seed, complete relational/task keys and preprocessing identities unresolved |
| Seeds / aggregation | Original repeated-run seed recipe and uncertainty aggregation unresolved |
| Selection | Appendix captions say best test R²; must distinguish reporting reconstruction from prospective validation-only selection |
| Targets | Tables1/3: raw+.1836, agg+.6541; Appendix B lists raw test scores−.039,−.552,−.184,−.118,−.126,−.086 and agg−.042,−.025,−.595,−.084,−.251,−.654 |
| Degree analysis | Tables2/9 use474 selected customers defined by post-cutoff behavior; not the complete10,000-test-customer degree distribution |
| Temporal reasoning | Paper §3 uses future-month summation to motivate truncation; it cannot authorize observing future-month events in prediction |

No signs are silently repaired. Original run logs are needed to reconcile the main-table/appendix inconsistency. The source's 'same aggregate information' is query-specific: monthly sums erase event order and extremes.

## Scope and stop conditions

Paid spendUSD0. All local numerical attempts, failed tests, notebook execution and validation share3600seconds; wrapper kills the process group at cutoff. Standing user ceilingUSD10 includes preparation/retries/validation; no paid run is authorized here. Full benchmark cost is NOT_ESTABLISHED because source identity is unresolved, not presumed affordable.

Paper execution: NOT_RUN / INCOMPLETE_SOURCE_PROTOCOL_GATE. Whole-paper RT/Griffin/RelGT/GraphSAGE runs and pretraining: NOT_RUN. No full reproduction claim. Release searches are dated limited queries; an unrelated GitHub name match and inaccessible OpenReview API are not evidence that no code exists. New evidence requires a reviewed frozen protocol and cost estimate before dispatch. Live Colab/frontend and deployment are separate from portable local execution. Learner: PENDING_WRITTEN_DEFENSE.
