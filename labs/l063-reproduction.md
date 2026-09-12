# L063: reproducible generator audit and original prior-fitting route

[Lesson](../lessons/0063-synthetic-scm-prior.html) · [Notebook](0063-synthetic-scm-prior.ipynb) · [Source manifest](_sources_l063_v2.json) · [New measured evidence](_verify_l063_v2_results.json) · [Source checks](_check_l063_v2_results.json)

The executable track reconstructs the numerical historical SCM/BNN generator's central operations. It does not pretrain a PFN. Its logistic probes, finite two-world Bayesian oracle and known synthetic edge interventions have distinct interpretations. Original Table 4 and final benchmark reproduction are **INCOMPARABLE / NOT_RUN**.

## Three executable budgets

From the repository root, using `requirements-labs.txt`:

```bash
.venv/bin/python labs/_check_l063_v2.py
.venv/bin/python labs/_verify_l063_v2.py --preset smoke --output labs/data/cache/l063-v2/smoke-new.json
.venv/bin/python labs/_verify_l063_v2.py --preset lab --output labs/data/cache/l063-v2/lab-new.json
.venv/bin/python labs/_verify_l063_v2.py --preset closer --output labs/data/cache/l063-v2/closer-new.json
.venv/bin/python labs/_build_l063.py
```

Every measured output path must be new. There is no resume and no automatic overwrite. The latest notebook EXIT submission uses `labs/data/cache/l063-v2/student-l063-exit.json`; it is not a training cache. The CLI source hash labels author code. The notebook instead records actual live functions, helper implementations, defaults, nested bytecode, presets, configuration and versions; EXIT rejects changes made after measurement.

| Preset | Worlds per family | Rows / context | Exact-oracle tasks | Interpretation |
|---|---:|---:|---:|---|
| smoke | 2 | 128 / 64 | 500 | Wiring and artifact check |
| lab | 12 | 256 / 128 | 500 | Measured generator diagnostic |
| closer | 64 | 1024 / 512 | 5000 | Broader generator audit, still no PFN prior fitting |

Each generated world receives an ordinary and a context-label-shuffled fixed logistic probe. The two family panels do not share latent worlds, so family means are not a matched causal comparison. Scaling and logistic coefficients fit on context only. Requested classes absent from context retain the stated clipped/renormalized probabilities. Array payloads are losslessly compressed where large; `decode_array` restores them. World parameters, root/noise draws, observations, targets and predictions allow independent reconstruction.

The notebook defaults to new seeds 1630–1641, separating student execution from the saved author seeds 630–641. It includes an OFF-by-default closer gate that calls the same live functions. The existing shared unattended operator can run the same local experiment in Modal:

```bash
modal run --detach modal/foundation_repro.py --lesson 63 --preset closer
```

Packaging this command does not claim that a cloud job or live Colab browser session was run. The generic route is `labs/_run_foundation.py --lesson 63 --preset smoke|lab|closer`; it delegates to the versioned generator. There is deliberately no `paper` preset for this heterogeneous original training protocol.

## Source identity and exact selected computation

The original files are vendored with license/notice under `labs/sources/l063-v2/`, pinned to the last repository commit before the paper v6 date:

`44f60d83c545238c551f1481a5f6f031bbf376bf`.

- [`mlp.py`](https://github.com/automl/TabPFN/blob/44f60d83c545238c551f1481a5f6f031bbf376bf/tabpfn/priors/mlp.py): `get_batch` nested `MLP.__init__`, `generate_module`, `MLP.forward`, `GaussianNoise.forward`, `causes_sampler_f`.
- [`flexible_categorical.py`](https://github.com/automl/TabPFN/blob/44f60d83c545238c551f1481a5f6f031bbf376bf/tabpfn/priors/flexible_categorical.py): `class_sampler_f`, `MulticlassRank.forward`; broader `FlexibleCategorical.forward` refinements are inspected and explicitly excluded.
- [`differentiable_prior.py`](https://github.com/automl/TabPFN/blob/44f60d83c545238c551f1481a5f6f031bbf376bf/tabpfn/priors/differentiable_prior.py): `DifferentiableHyperparameter` / `meta_trunc_norm_log_scaled`.
- [`utils.py`](https://github.com/automl/TabPFN/blob/44f60d83c545238c551f1481a5f6f031bbf376bf/tabpfn/priors/utils.py): `trunc_norm_sampler_f`, `randomize_classes`.

`_sources_l063_v2.json` also records selected metadata from the actual historical checkpoint, SHA256 `3c9aadaeddbf51462af8c0ee4b3ca3c697890f77e92318abbb0821b75261c392`. No checkpoint predictions or trained weights are used by this lesson's measurements. The pinned repository default `model_configs.py` is not identical to that saved configuration. The serialized activation lambda is not resolved as a source-parity claim: the lab uses the four functions explicitly named by paper Appendix C.1.

The checker executes the original MLP class, records actual causes/noise draws and copies weights into the live numerical implementation. It compares every affine output for 16 cases. It separately executes original MulticlassRank for 20 random streams. This is **conditional operator parity**, not complete sampler/RNG parity. NumPy initialization changes random streams. The first affine map has neither activation nor explicit GaussianNoise; later maps execute activation → affine → additive noise. Independent-edge scaling preserves `1−sqrt(p)` for the selected flag; the block branch uses `sqrt(keep_fraction)` and also sparsifies the first affine matrix.

## Remaining generator differences

- Depth, width and root count are capped at 5, 32 and 16, with raw draws recorded.
- Root sampling conditions on the Gaussian branch; the actual checkpoint uses mixed Gaussian, categorical and Zipf inputs.
- Hyperparameter magnitudes use the released relative-standard-deviation TNLU construction, differing from Table 5's simplified independent μ/σ description.
- Activation and boolean choices are uniform, rather than reproducing all hierarchical meta-choice parameters.
- The lab omits feature-scale refinements, categorical conversion, missingness and full flexible preprocessing.
- The inner strict-bound classification operator is reproduced; context/query class-support repair, contiguous class remapping and final random label shift from the complete wrapper are omitted.
- Source blockwise observation can overlap the forced final target; the lab rejects and redraws such selections, recording counts.
- Context/query target bounds are jointly sampled from the synthetic episode, as a task-definition step. This is not a deployable rule for thresholding unknown real test labels.

## Concrete route to the paper experiment

Start with the pinned [historical training notebook](https://github.com/automl/TabPFN/blob/44f60d83c545238c551f1481a5f6f031bbf376bf/tabpfn/PriorFittingCustomPrior.ipynb), [model builder](https://github.com/automl/TabPFN/blob/44f60d83c545238c551f1481a5f6f031bbf376bf/tabpfn/scripts/model_builder.py) and [training loop](https://github.com/automl/TabPFN/blob/44f60d83c545238c551f1481a5f6f031bbf376bf/tabpfn/train.py). The notebook is a runnable source entry point to adapt, not an assertion that its displayed defaults reproduce the final checkpoint or Appendix B.4.

Before launching a reproduction, reconcile the saved final checkpoint configuration with this source revision and original dependencies. Restore full mixed roots, uncapped size distributions, full categorical wrapper and preprocessing. The historical numerical predictor architecture is traced in lesson 062. For the prior ablation, keep that predictor, optimizer, synthetic-task count and evaluation procedure matched across BNN-only, SCM-only and the 50/50 mixture; change the family probability alone. Retain independent training seeds, per-dataset predictions, failures and original meta-validation boundaries.

Paper Appendix E.3 reports final-model training at 18,000 updates, batch size 512 datasets, 1024 rows per dataset: 9,216,000 tasks, 20 hours on eight RTX 2080 Ti GPUs. Appendix B.4 explicitly uses less compute than this final run; it must not silently inherit the final run's budget. Its Table 4 means are CE .811/.771/.776 and ROC AUC .865/.881/.883 for BNN/SCM/mixture. Exact ablation budgets and uncertainty semantics need resolution before claiming a faithful table reproduction; current local diagnostics do not fill that gap.

**Status:** fresh author lab and executed teacher notebook are local measured evidence. Full prior-fitting, full Table 4 reproduction, live Colab and cloud jobs remain NOT_RUN/NOT_CHECKED. Copied Pages, browser and post-publication byte checks are separate parent-owned delivery checks, not inferred from notebook execution. Historical `foundation_core.py`, `foundation_experiments.py` and `_verify_l063_results.json` remain untouched as prior operator/evidence records.
