# L075: PyTorch Frame row-encoder contract

## Scope

This is the curriculum's **tool/API lesson**, using the lab-authoring skill's explicit API exception. The deliverable is an encoded row tensor plus semantic-type configuration, not a new architecture or an accuracy claim. The notebook exposes the full local composition and a reconstructed numeric map. It uses the actual library as the object being learned.

- Paper: Hu et al., *PyTorch Frame*, arXiv 2404.00776v2, Figure 1 / §3, four-stage abstraction.
- Executed release: `pytorch-frame==0.3.0`, commit `d998aae368db6a4e36139ccc56bd54579a70874b`. This is a release-specific demonstration, not a publication-era environment claim.
- Five-type mechanism fixture: four training and two query rows; deterministic bag-of-words text adapter, width4; stored vectors width3; encoder width8, MLP 40→16→6. Synthetic, Tier C, for exact interventions only.
- Real-table demonstration: OpenML31 credit_g, existing cache with SHA256 in `_verify_l075_results.json`; seed75 permutation, 128 fitting and32 query rows. Actual numeric/categorical columns, no synthetic modalities added. Width8, MLP 160→16→16 for20 columns. Random weights, no labels or accuracy estimates.
- Materialization fits statistics on training rows only. Query converter reuses fitted state. Numeric mean imputation, default categorical zero-padding, default median timestamp imputation. Text adapter is frozen and has no learned language semantics.

## Executed behavior and source checks

Selected installed Python files must byte-match the pinned release. Tests cover unknown/missing category IDs, finite token/row output, all-parameter finite gradients, DataFrame column reordering, row locality, numeric forward parity and a hand-computed oracle. The split-column counterexample holds training rows fixed and changes fit scope: mean20 versus265. Neither result is a predictive metric.

The release's default numerical missing policy can have finite forward values without guaranteeing finite parameter gradients through NaNs. This lab explicitly selects `NAStrategy.MEAN`, and the finite-gradient check includes missing numeric inputs. This is a configuration choice, not a patch to the upstream framework.

## Commands (repository root)

```bash
.venv/bin/python -m pip install -r requirements-labs.txt
.venv/bin/python labs/_prepare_l075.py
.venv/bin/python labs/_verify_l075.py
.venv/bin/python labs/_build_l075.py
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python labs/_execute_l075.py
.venv/bin/python labs/_browser_l075.py
.venv/bin/python labs/_delivery_l075.py
node labs/_check_pedagogy.js
```

The solution runs with `labs/` as its working directory. Student TODOs remain blank. `real_table_demo(export=True)` writes `l075-row-encoder.pt` and `l075-schema.json` there; only load your trusted local Torch artifact. State contains fitted column metadata and row identity, not just anonymous vectors. The verifier does not overwrite student artifacts.

## Evidence ledger

| Evidence | Status | Interpretation |
|---|---|---|
| Local five-type API and real-table contracts | PASS after `_verify_l075.py` | This pinned configuration produces the promised shapes and preserves tested boundaries |
| Numeric reconstructed forward vs configured library | MATCH within1e-7 | Primitive parity on fixture; not full model-result parity |
| Paper Figure1 dataflow | Demonstrated | Typed blocks → columns → local MLP readout; no claim to reproduce a benchmark experiment |
| Paper §5.1 single-table benchmarks | NOT_RUN | No benchmark architecture/trainer/tuning/data-suite replay in this API lesson |
| Paper §5.2 text experiments | NOT_RUN | Fixed four-word adapter is not the paper's pretrained text model |
| Paper §5.3 relational experiment | NOT_RUN | No GNN, foreign keys, temporal sampler or supervised optimization here |
| Comparing local tensor values to paper accuracy | INCOMPARABLE | Different quantity and protocol |
| Live Colab / deployment | NOT_CHECKED | Portable images and local execution are separate checks |

There is no `paper` preset or cloud scale-up operator: this is the tool/API exception, not a downscaled benchmark reproduction with an unimplemented scale-up claim. A future paper-results exercise must audit the selected model, objective, initialization, text checkpoint, dataset/split, preprocessing, tuning, schedule, seeds and aggregation before claiming a replay. See the authors' pinned source and the paper §5 for the experiments intentionally outside this unit.

## Delivery boundary

Inspect `_execution_l075_results.json`, `_browser_l075_results.json` and `_delivery_l075_results.json` for checks actually performed. The pandas3 timestamp conversion emits an upstream read-only NumPy warning in the audited environment; the tested forward/backward path completes and the notebook does not mutate those views. Future package upgrades require re-running source and behavior checks. A live Colab runtime and deployment have not been exercised.
