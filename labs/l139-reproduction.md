# L139 — complete selected clinical-trial reproduction

Approved scope: RelBench v1 Table6 **rel-trial/study-outcome**, basic RDL, five full-data runs. Published validation68.18±.49 and test68.60±1.01 AUROC percentage points. Descriptive tolerance1pp declared before fitting; not equivalence. Source9aa346267c2e1c560bd92da07d6f4ad1ca2f0639. The source manifest records URLs and SHA256 hashes, including paper HTML and the Apache license.

## Protocol

- Historical database archive `9fb5ba14f7cbca8115f3dfe0800415f98d6ddc15561e56c35ee614da6b89552a`; task archive `20eb922c1a8f894563f4b4c900c912e396688d2bd71eeb6b13f19429aa74a649`. Full rows/features; no teaching caps.
- Train11994 / val960 / test825 queries. Independent pandas reconstruction and released SQL check inclusion and every label. The first audit falsely rejected nullable `Int64` versus `int64` MultiIndex metadata; original failing source and diagnostic evidence retained. The fix checks non-null keys and normalizes only integer storage dtype, preserving all values.
- Study starts by cutoff; qualifying primary-outcome analysis in `(t,t+365days]`; p in[0,1], modifier not `>`; positive iff minimum numeric p≤.05. No qualifying analysis means no query. The source's handling of other modifiers is preserved, including `<.051` as numeric.051.
- The source constructs a retrospective actual-completion cohort with starts from2000 onward. It infers analysis/outcome dates from completion and several design/association dates from start; these are not recorded publication/arrival timestamps.
- Full raw archive supports label audit. Model database cutoff2021-01-01; validation2020-01-01. Timestamped nodes must be at or before their owning query cutoff. Untimestamped metadata lacks historical availability guarantees. Released snapshot-level feature statistics are preserved.
- Full graph/preprocessing from L132 reused after archive, table, edge and graph checksum checks. Pinned GloVe300 revision`e5e8fec6971be8960cfaa853a77a6ddc62a265d7`; preprocessing seed42. Preparation reuse is disclosed, not claimed as fresh materialization or historical RNG recovery. `_full_l139.materialize` supplies the complete fresh path.
- AppendixB.2 exception: two128-channel layers, **mean** neighbors, sum relation outputs, fanout64/32 following released halving rule, uniform temporal sampling, batch512, Adam.0001, BCE logits,20epochs. Each fit traverses all24 training batches each epoch. No source-defined2001-batch cap is reached.
- Seeds0–4. First strict maximum validation AUROC selects weights, followed by re-evaluation of validation/test with freshly sampled neighborhoods. Therefore selected-epoch validation and final validation may differ. No test-based tuning.
- Frozen Py3.11 runtime: `requirements-l117-runtime.txt`, Torch2.5.1/cu124, pyg-lib0.4.0+pt25cu124. Historical seeds/environment identity remain unestablished.

## Commands

From repository root:

```bash
.venv/bin/python labs/_check_l139.py
.venv/bin/python labs/_verify_l139.py
.venv/bin/python labs/_source_sql_l139.py
.venv/bin/python labs/_figures_l139.py
.venv/bin/python labs/_build_l139.py
.venv/bin/python labs/_execute_l139.py
.venv/bin/python labs/_delivery_l139.py
```

Author operators (each reserves cost before dispatch; completed phases refuse repeat dispatch):

```bash
.venv/bin/modal run --detach modal/l139_repro.py --mode prepare
# The recorded first preparation failed at the independent dtype comparison.
.venv/bin/modal run --detach modal/l139_repro.py --mode prepare2
.venv/bin/python labs/_collect_l139.py prepare
.venv/bin/modal run --detach modal/l139_repro.py --mode pilot
# The first pilot trained, then failed an overly strict bitwise GPU check.
.venv/bin/modal run --detach modal/l139_repro.py --mode pilot2
.venv/bin/python labs/_collect_l139.py pilot
.venv/bin/modal run --detach modal/l139_repro.py --mode full
.venv/bin/python labs/_collect_l139.py full
.venv/bin/modal run --detach modal/l139_notebook_check.py --attempt 1
.venv/bin/modal run --detach modal/l139_full_notebook.py --mode prepare
.venv/bin/modal run --detach modal/l139_full_notebook.py --mode check
.venv/bin/python labs/_collect_notebook_l139.py
```

The preparation operator reads prior volume`l132-identity-evidence`; all new artifacts are written to`l139-trial-evidence`. The pilot is seed99/one epoch and never enters the reported five-run aggregate. The full operator projects20-epoch runtime with margin before dispatch. Every attempt, including failures and checks, counts toward USD10; USD3 reserved for overhead. Do not erase reservations or markers to retry. No automatic retries.

Fresh preparation and one selected fit, in the pinned runtime:

```bash
python labs/_full_l139.py --materialize --prepared-root /path/to/new/prepared --output /path/to/new/seed-0 --seed 0 --epochs 20
python labs/_full_l139.py --prepared-root /path/to/new/prepared --output /path/to/new/seed-1 --seed 1 --epochs 20
# Repeat seeds2–4; never overwrite an existing output directory.
```

The portable notebook includes the same materialization, model, graph and trainer code. `RUN_FULL_REPRODUCTION=True` with the default `PREPARED_ROOT=None` prepares a fresh graph with seed42, then runs five fresh20-epoch fits. The author full-notebook validator executes fresh preparation in its own phase using the identical source, checks its SHA256 against the final notebook source module, then supplies that verified root to the notebook gate. These additional five validation fits remain separate from the primary aggregate. Its default is false; the default checks the complete preserved prediction packet,36real label examples, a real sampler trace, and a synthetic full neural forward/backward. Full preparation and cached training require substantial host memory (author workers64GiB); the raw graph artifact is about16.7GiB. Local/notebook execution has no billing guard. Live Colab is NOT_CHECKED.

## Evidence boundaries

Fresh full materialization independently produced byte-identical graph SHA256`4f683211d33360e9ad293c0189fddad940223788ce1d616db800ed72766b9eb7` (17,883,804,206bytes). See`evidence/l139/preprocessing_parity.json`. This verifies current preprocessing reuse, not historical paper graph identity.

`evidence/l139/training.json` contains measured means/SDs, individual runs, predictions/checkpoint hashes and declared verdicts. `_verify_l139_results.json` independently checks source AST, labels, selection, query keys, score arithmetic and temporal counts. Each final checkpoint is also loaded into the original source Model and compared on512real query roots with rtol1e-5/atol1e-6. Both same-model repeats are measured to contextualize GPU reduction nondeterminism. The first pilot's bitwise assertion failure and original source are preserved. The visible Model omits only the unused recommendation method `forward_dst_readout`; the classification path and all graph/encoder primitives match the pinned source. This checks model output agreement on that batch, not historical training identity.

The collector waits for all five complete seeds and preserves every output before aggregation. Selected checkpoints are retained in the cloud and ignored local `labs/results/l139/checkpoints`; large graphs/checkpoints are not published. Paper-data matches, source parity and numerical closeness do not prove historical RNG identity, whole-paper reproduction, clinical causal benefit or learner mastery. LightGBM figures are cited context; this lesson does not retrain that baseline. Live Colab/deployment remain NOT_CHECKED.
