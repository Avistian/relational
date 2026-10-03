# B02 — numerical embeddings, TabM and TabPack

## Approved scope and outcome

User approved B02-TABPACK-CALIFORNIA on 2026-10-03. COMPLETE_SELECTED_RELEASE_PROTOCOL: one complete fresh 64-member search followed by all five evaluation seeds, original full California rows. Independent scalar and vector scoring authenticates 44,586 final ensemble validation/test rows and 780,255 retained member prediction values. Mean test RMSE 0.4172312882, sample SD 0.0006877176. Released five-seed mean 0.4175778958, sample SD 0.0028366839. Maximum independent versus fresh-report score difference 1.923e-8 (declared tolerance 1e-6 for float32 scoring).

This is source-protocol execution, not identical historical training. The fresh observed search selected 13 distinct configurations; the release selected 12. Numerical proximity does not prove exact reproduction, causal embedding/ensemble improvement, broad superiority or learner mastery. No pass threshold was selected after observing the test scores: we report differences, not a post-hoc score-parity verdict.

## Frozen sources and data

- TabPack paper v1: https://arxiv.org/html/2607.05380v1, §3, Appendix B/F/G.
- Source commit: `05a89e21b955f12de84889d662e15ca534019aaa` at https://github.com/yandex-research/tabpack.
- Primary config: `experiments/tabpack-cosine/california/main/config.json`; original generator `experiments/tabpack-cosine/make.py`.
- Official archive: https://huggingface.co/datasets/Yura52/tabpack-data/resolve/main/tabpack-data.tar.gz. All 189856645 bytes authenticated to SHA256 `89338c628fed24af03084c9348ca0a5c8ca4f12f8b988a576d9d1711cc661558` before extracting California.
- `evidence/b02/source-lock.json` pins every relevant source and data file. `evidence/b02/compact/compact-lock.json` pins the portable subset. Raw row identity is the index into the authenticated `x_num.npy` / `y.npy`, with original split indices. The three disjoint splits cover all 20640 rows: 13209/3303/4128. This is the released random split, not a temporal or historical availability experiment.
- Environment uses upstream `uv.lock` with Python 3.12, torch 2.7.1, NumPy 2.3.1 and sklearn 1.7.0; original training runs on Modal A100-80GB, four requested CPU cores and 24GiB host RAM. Local arithmetic audits use a separate CPU environment (recorded in mechanism-audit.json). A lock and configured image are not a complete historical environment certificate.

## Exact selected procedure

Source owns noisy-quantile feature normalization fitted on training features, training-only target standardization, batch size 256, 64 packed members, width 384, depth 1–3, sampled embedding size 8–32 step 4, source dropout/scale/optimizer ranges, MuonAdamWPack, bfloat16 and no artificial epoch ceiling. Member patience is 16; online greedy ensemble patience 32 with maximum ensemble size 32. Members can appear repeatedly at different training steps. Selection uses validation; upstream also logs test scores during training but they are not passed to the selector. No human choice between runs was based on test performance: the observed rerun was required for missing evidence.

The complete search derives its own selected configurations. Evaluation follows the upstream `_evaluate_ensemble` function with five seeds 0–4, including per-seed online selection and stopping. Copying the release's 12 configurations would skip the initial search; we did not do that. Seed SD is conditional on one split and one selected configuration set.

The paper's Table 14 caption says ten seeds, while the generator, evaluation configuration and released report contain five. Preserve `UNRESOLVED_SEED_COUNT`; do not silently run ten or call five paper-exact.

## Execution history and deviations

1. `pilot`: stopped before training because upstream atomic rename crossed `/tmp` and the output volume.
2. `pilot2`: stopped in preprocessing because the cache rename crossed the volume boundary.
3. `pilot3`: complete unmodified search, worker body 64.189 seconds; original 12-configuration selection. Move temporary directory and cache to the same volume is a filesystem adapter only.
4. `full`: evaluation interrupted by Modal client deadline/cancellation; persisted receipt exit 124 at 219.030 seconds, not the 1800-second worker limit. Retain partial output, never mark complete.
5. `audited`: complete new search and all five evaluations, 183.514 seconds. Original model/optimizer/selection source bytes remain unchanged. A wrapper calls the original online-update function, retains the ensemble object, and after the original main returns saves its final member arrays/IDs/steps. It returns the original report unchanged. The fresh result differs from the earlier search; deterministic trajectory identity is NOT_ESTABLISHED.

The wrapper is necessary because original `online_ensemble_predictions.npz` enumerates dictionary keys, saving `"greedy"` rather than the numerical arrays. Also, the local variable intended to hold those arrays is not populated. We preserve the bug and original files; we do not silently repair the pinned release. The extra output `observed_ensemble.npz` contains selected member predictions with IDs and checkpoint steps. The observer is not a new selector or trainer. Byte parity of source and arithmetic checks do not by themselves prove observer non-interference with hardware-level numerical behavior.

The audited worker succeeded, but returning a large archive stalled the client. Files and receipts were recovered directly from the persisted Modal volume, and the idle app was stopped. No further training was required. Commands and logs retain both failure and success.

## Cost and stop conditions

USD10 all-in cap, USD8 admission stop, USD2 overhead reserve. Conservative rate 0.00080076 USD/second bounds the listed A100-80GB + 4 CPU + 24GiB rates at https://modal.com/pricing (checked 2026-10-03). Each dispatch reserves its whole timeout plus startup allowance before running; retries are never refunded in the admission ledger. Five reservations total USD4.80456; with USD2 overhead, the conservative all-in envelope is **USD6.80456**. This is a reserved estimate, not an itemized invoice. The overhead allowance covers image builds, transfers, shutdown and bookkeeping. Actual remote worker-body seconds are separately recorded; credits are not treated as free compute.

See `evidence/b02/budget.json`, `admission.json` and raw phase receipts. Local numerical execution has a separate 3600-second aggregate cutoff in `local-budget.json`, including failed checks and conservative allowances. No paid work remains running.

## Reproduce and inspect

From repository root, local no-cloud verification:

```bash
.venv/bin/python labs/_audit_b02.py
.venv/bin/python labs/_verify_b02.py
.venv/bin/python labs/_source_b02.py
```

The portable student/solution notebook embeds `evidence/b02/replay.zip`: all final selected-member arrays, original full dataset/split identities, source-derived configuration lineage and independent scorer. It performs saved replay, not fresh training. Three live TODO functions govern the mechanism exercises. Full source is accessible in [the source archive](evidence/b02/source.zip) and at the pinned upstream URLs; extract the source archive at repository root to restore `labs/sources/b02/upstream` before the original source audit or training.

Fresh training is a separate, paid lane. The guarded Modal operator is `modal/b02_repro.py`, the observer/filesystem adapter is `labs/_worker_b02.py`. The spent ledger and existing volume are immutable evidence: do not clear them to bypass the cap. A new run needs a new named volume and separately authorized ledger; current approval is consumed by this completed experiment. For an independently provisioned source checkout, the unmodified original command is:

```bash
uv sync --frozen --no-dev
uv run experiments/tabpack-cosine/make.py
# Work in a NEW experiments directory; copy the original California main config.
uv run scripts/run_tabpack_experiment.py experiments/reproduce/california \
  --eval-online-ensembles greedy --eval-online-ensembles-n-seeds 5
```

The upstream generator creates configurations for all datasets. Execute only the frozen California command for this named scope. Avoid `--clean` if retaining raw evidence. Configure temporary/cache/output paths on the same filesystem. The unmodified command retains the known online-prediction serialization bug; use the provided observer to obtain independent final prediction evidence.

## Evidence boundaries and learner work

Full multi-dataset TabPack benchmark, numerical-embedding paper benchmark, fresh TabM benchmark, four-cell ablation and fresh tuned-tree/RealMLP comparisons: NOT_RUN. A four-cell/baseline design is learner work, not measured evidence. Hardware runtime differences are not a speedup comparison. Operator parity and independent rescoring are author checks. Learner remains PENDING_WRITTEN_DEFENSE; live Colab and deployment are NOT_CHECKED.
