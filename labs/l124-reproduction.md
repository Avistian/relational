# Lesson 124: task tables and full selected F1 reproduction

## Executed evidence

Five fresh full-data ten-epoch RDL fits for RelBench v1 Table7 F1 driver-position. Validation MAE3.179652907 ± .040714021; test4.018438246 ± .102606666 (mean ± sample seed SD). Targets3.193/4.022; both CLOSE under predeclared descriptive .2 MAE tolerance. All6295final predictions independently rescored and replayed through original model; max discrepancy2.861023e-6. No historical/whole-paper parity claim.

Independent pandas reconstruction and original pinned SQL agree with every released train7453/validation499/test760 task row. Full timestamp grids332/30/40 include empty windows. Exact keys and zero observed label discrepancy (tolerance1e-12). All26080raw result events included in portable task payload. Train labels mature before validation start; source split boundaries and entity filtering retained.

Source cohort: date > t - one calendar year has no upper bound. Adding date <= t excludes955/33/42labeled rows. This is a separately named course intervention, not a change to benchmark training. Even the past-only intervention retains future-participation conditioning; no future event is not a zero mean. Label-end maturity assumes no reporting delay; real availability histories are absent.

Current CPU original-source output/gradient/Adam checks pass. Six semantic mutations rejected. Every sampled training/evaluation batch passes the inherited L123 audit:403895query occurrences, including repeated epochs. Input graph census74063rows/338842edges independently checked by SQL.

## Protocol and deviations

| Component | Executed choice / boundary |
|---|---|
| Paper target | Robinson RelBench v1 Table7 driver-position; not Fey beta or whole paper |
| Upstream source | 9aa346267c2e1c560bd92da07d6f4ad1ca2f0639, pinned files in sources/l117 |
| Database | Complete released rel-f1 database censored at2010-01-01; sha256 ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482 |
| Task archive | sha256 775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e |
| Splits |7453 train,499 validation,760 test queries; released IDs and seed times |
| Initialization | Fresh random model per seed0–4, no checkpoint reuse; preprocessing seed42 |
| Encoder | Released per-table Frame feature encoders and four-block ResNet to128 |
| GNN / head | Two typed sum-GraphSAGE layers, relative-time encoding, scalar seed head |
| Training | L1 mean loss, Adam .005, batch512, ten full epochs, uniform fanouts[128,64] |
| Selection | First lowest epoch validation MAE; training-label2nd/98th percentile output clipping |
| Final scoring | Original and visible models share each final sampled batch; independently score saved predictions |
| Aggregation | Five seeds, sample SD; .2 MAE descriptive tolerance, not equivalence inference |
| Paper/source gap | Released fanouts[128,64] vs paper table128; historical seeds/binaries unknown |
| Preprocessing gap | Released statistics use database through test cutoff, not training-only |
| Temporal boundary | Dated node time <= own root query; node-based release does not supply independent edge arrival histories |
| Missing data | Ingestion and feature-version histories; static-table creation times |
| Notebook | Task functions reconstruct all 8,712 released rows and rescore all final queries; full native training gate OFF by default |

Pinned runtime: Python3.11, Torch2.5.1+cu124, pyg-lib0.4.0+pt25cu124; remaining exact dependencies in requirements-l117-runtime.txt. Sentence-transformer model/revision in sources/l117/text_model.json. Full library primitive source is shown in notebooks and source files. Local check environment is newer and is reported separately; it is not the training runtime.


## Exact commands

From the repository root, using the existing local verification environment:

```bash
.venv/bin/python labs/_check_l124.py
.venv/bin/python labs/_mutation_l124.py
.venv/bin/python labs/_source_check_l124.py
.venv/bin/python labs/_task_audit_l124.py
.venv/bin/modal run --detach modal/l124_repro.py --mode pilot
.venv/bin/python labs/_collect_l124.py --mode pilot
.venv/bin/python labs/_pilot_check_l124.py
.venv/bin/modal run --detach modal/l124_repro.py --mode paper
.venv/bin/python labs/_collect_l124.py --mode paper
.venv/bin/python labs/_analyze_l124.py
.venv/bin/python labs/_audit_l124.py
.venv/bin/python labs/_figures_l124.py
.venv/bin/python labs/_build_l124.py
.venv/bin/python labs/_execute_l124.py
.venv/bin/python labs/_delivery_l124.py
.venv/bin/python labs/_verify_l124.py
```

These record the actual initial execution. Repeating the remote run requires a NEW volume and source-hashed budget ledger. Completed runs cannot be overwritten. Do not erase reservations. No automatic retries. The notebook includes complete visible model/trainer plus full five-seed gate OFF by default. That gate does not enforce a money cap and live Colab remains NOT_CHECKED.

For another local full run, provision the exact GPU runtime in modal/l124_repro.py (Python3.11, Torch2.5.1+cu124, requirements-l117-runtime.txt and native pyg-lib0.4.0+pt25cu124), then run:

```bash
for seed in 0 1 2 3 4; do
  python3 labs/_run_l124.py --seed "$seed" --epochs 10 --output "labs/results/l124/my-new-run/seed-$seed"
done
```

This local command does not impose a monetary limit. The fresh process downloads checksum-pinned data and the pinned text model; archived checkpoints are not required.

## Compute

Current rate from https://modal.com/pricing: T4 .000164 +2CPU*.0000131 +16GiB*.00000222 = USD.00022572/s. Eight worker reservations each3600s cap worker resources atUSD6.500736; overhead reserveUSD3.499264; aggregateUSD10. Six reservations consumed, no retries. Pilot conservative projectionUSD.25036960. Actual function-body resource estimateUSD.05386633 excludes unitemized setup/storage/other billing and is not an invoice total.

Pilot: https://modal.com/apps/pszar92/main/ap-dXljHGDf7deaLaONsaAX16
Five fits: https://modal.com/apps/pszar92/main/ap-sLIGvqiOrvAEY1t9KbzUD4

## Artifacts and boundaries

- evidence/l124/paper/seed-*/: full epoch trace, selected epoch, compact predictions, source/runtime audits, per-batch audit counts and completion UUID.
- results/l124/paper/seed-*/selected.pt: local weights, intentionally not published; training regenerates them.
- evidence/l124/task-inputs.json.gz: complete task inputs and archive oracle; checksum in _task_audit_l124_results.json.
- L123 complete graph payload is embedded with checksum in the notebook; one real task row is joined to its99-node/245-edge input neighborhood.
- _sources_l124.json and _upstream_l124.json: pinned task/split/database/model sources. MIT license retained.
- Visible task code: relkit/tasks_l124.py. Full reused trainer: relkit/rdl_l117.py. Batch audit: relkit/batch_audit_l123.py.

Task-table equality, temporal visibility, original-model parity and score closeness are separate evidence. Author execution leaves learner PENDING_WRITTEN_DEFENSE. Local delivery checks are recorded separately in _delivery_l124_results.json. Live Colab/deployment NOT_CHECKED; no publishing requested.
