# Lesson 123: temporal heterogeneous graphs and selected reproduction

## What was actually run

Five fresh full-data ten-epoch RelBench v1 Table7 F1 driver-position RDL fits. Validation MAE3.17118733 ± .05090126; test4.06491279 ± .06990434 (mean ± sample seed SD). Published means3.193/4.022, both CLOSE under predeclared descriptive .2 MAE tolerance. All6295 final query predictions independently rescored and matched to archived entity/time/target rows. Maximum original-model output difference3.8146973e-6. Historical identity and whole-paper parity NOT_ESTABLISHED.

Full graph74,063 rows/338,842 directed edges.72,918 dated rows in six tables; drivers/circuits/constructors are undated. Nine selected exhaustive two-hop neighborhoods agree exactly with SQL(node sets) and native PyG(typed nodes/edges); a separate same-entity two-time query test passes. This is not an exhaustive proof over all roots.

Every batch in five training runs is audited, including all training epochs, each epoch's validation, and final validation/test.403,895 query occurrences are audited (including repeat queries across epochs). Audits check node timestamp identity, root-specific <= cutoff, original edge identity and same-query endpoints. They do not observe ingestion times or mutable-feature histories.

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
| Notebook | Teaching sampler and audits operate on complete graph topology/times; full native training gate OFF by default |

Pinned runtime: Python3.11, Torch2.5.1+cu124, pyg-lib0.4.0+pt25cu124; remaining exact dependencies in requirements-l117-runtime.txt. Sentence-transformer model/revision in sources/l117/text_model.json. Full library primitive source is shown in notebooks and source files. Local check environment is newer and is reported separately; it is not the training runtime.

## Exact first-run author commands

From the relational repository. The supplied budget ledger records the completed run and rejects duplicate launch modes. These commands describe the recorded initial execution, not authorization to overwrite evidence or spend again.

```bash
.venv/bin/python labs/_check_l123.py
.venv/bin/python labs/_mutation_l123.py
.venv/bin/python labs/_source_check_l123.py
.venv/bin/python labs/_temporal_audit_l123.py
.venv/bin/modal run --detach modal/l123_repro.py --mode pilot
.venv/bin/python labs/_collect_l123.py --mode pilot
.venv/bin/python labs/_pilot_check_l123.py
.venv/bin/modal run --detach modal/l123_repro.py --mode paper
.venv/bin/python labs/_collect_l123.py --mode paper
.venv/bin/python labs/_analyze_l123.py
.venv/bin/python labs/_audit_l123.py
.venv/bin/python labs/_figures_l123.py
.venv/bin/python labs/_build_l123.py
.venv/bin/python labs/_execute_l123.py
.venv/bin/python labs/_delivery_l123.py
.venv/bin/python labs/_verify_l123.py
```

A local fresh run in the exact pinned runtime does not need Modal or the original ledger. This downloads the full public dataset and pinned text model, starts a fresh model, audits every batch, and stores outputs in a new directory. Selected training is GPU-sized; the CPU fallback exists but has no promised runtime. These commands do not enforce a monetary limit; run them only on compute you intend to use.

```bash
for seed in 0 1 2 3 4; do
  .venv-l117/bin/python labs/_run_l123.py --seed "$seed" --epochs 10 --output "labs/results/l123/my-new-run/seed-$seed"
done
```

Use the Modal image definition to provision the pinned environment if .venv-l117 is unavailable. The standalone notebook has a separately gated full five-seed operator using its visible definitions, dataset hashes and runtime checks. It defaults OFF. Live Colab execution is NOT_CHECKED. To repeat the budgeted remote workflow, use a new volume namespace and a new source-hashed budget ledger; never erase the recorded reservations or overwrite completed seeds. No automatic retries or spending above USD10.

## Compute accounting

Current published rate: T4 .000164 +2 CPU cores*.0000131 +16GiB*.00000222 = USD.00022572/second. Eight workers, each capped3600seconds, reserve USD6.500736; overhead reserve USD3.499264. Pilot and five full fits consume six reservations, no retries. Recorded function-body resource estimate USD.059597827 excludes unitemized setup/storage/other billing; it is not an invoice total. Pilot conservative projection includes50 pilot-equivalent units for five full fits before launch. Source hashes freeze the training runner and audit.

Pilot: https://modal.com/apps/pszar92/main/ap-FZOYL9f6pEIuRYPTo2uDH0
Full fits: https://modal.com/apps/pszar92/main/ap-01opDycIm3dObrrz9w9qDs

## Evidence and regeneration

- `evidence/l123/paper/seed-*/`: source/data/runtime audit, every-batch temporal counters, full epoch trace, compact predictions and completion UUID.
- `results/l123/paper/seed-*/selected.pt`: local selected weights (not published). Re-running training regenerates them; archived predictions alone do not reproduce training.
- `_audit_l123.py`: fresh released archive queries and independent SQL relation census.
- `_temporal_audit_l123.py`: constructs full graph, validates dated rows and compares SQL/PyG query sets; exports portable graph and exact query oracle.
- `_mutation_l123.py`: rejects six temporal/identity faults; checks equality/root-time semantics and no audit RNG consumption.
- Notebook input/oracle are hash-embedded and load without repository paths. All three student functions feed actual toy and real graph computations. Default prediction rescoring replays author evidence; it is not learner training.

Visibility audit ≠ neighborhood completeness ≠ historical pipeline correctness ≠ predictive score reproduction. Author execution does not demonstrate learner mastery: PENDING_WRITTEN_DEFENSE. Live Colab and deployment remain NOT_CHECKED. No publishing requested.
