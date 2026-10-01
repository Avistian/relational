# L158 Year 4 thesis evidence replay

Approved 2026-10-01. CPU-only; USD0 additional cloud spend. No new fits, paid workers or downloads. Full reproduction here means complete replay of the declared evidence set; it does not mean fresh model training, the original human study or the whole RelBench paper.

## Frozen inclusion rule

Include L151 reference seeds0–4 and selected seeds10–14, L152 seeds0–4, L153 validation-only ranking pilot, L155 five complete FE searches and five GNN fits, and L156/L157 five seeds per released/fixed-horizon lane. Exclude notebook-validation reruns from primary estimates. L137/L149 are conceptual callbacks, not added statistical samples. L157b is absent and contributes no result. No new test-led selection.

`evidence/l158/input-manifest.json` pins220 source/evidence files. Verify existing hashes; never re-freeze to bypass a mismatch. `_freeze_l158.py` refuses changed inputs. L154/L155 adapters are pinned as code dependencies; the portable notebook exposes them inline. Training source identities remain in the inherited protocol manifests and result files.

## Execution

From the repository root:

```bash
.venv/bin/python labs/_replay_l158.py
.venv/bin/python labs/_verify_l158.py
.venv/bin/python labs/_build_l158.py
.venv/bin/python labs/_execute_l158.py
.venv/bin/python labs/_delivery_l158.py
```

The standalone solution embeds the complete input packet and visible reporting implementations. Python3.10+ and numpy are sufficient for the replay; notebook execution uses the existing locked workspace environment. No remote data import or package installation occurs in the notebook. Output JSON should exactly equal the repository report.

## Protocol and checks

Recompute98,918 prediction rows: L154's61,148 plus L155's12,590 and L156/L157's25,180. These count repeated populations and validation/test rows, not independent observations. Audit45 validation selection decisions (15 portfolio fits,10 FE/GNN decisions,20 temporal fits). Preserve full entity/cutoff keys, reject missing/duplicate keys, and explicitly convert temporal label seconds to prediction nanoseconds. Confirm complete declared epochs/trials/seeds and original frozen source identities. The ranking pilot retains its full53,241candidate protocol and remains validation-only.

Independent sklearn AUROC/MAE and a separate vectorized ranking oracle check every saved score. A loop-based driver bootstrap reproduces the L1552000-draw interval with RNG155. It is conditional on fitted models and this test split; race/time, training and cross-database dependence are not covered. No equivalence or broad superiority inference is made. Mutation checks reject missing/changed files, wrong hashes, mismatched query keys, duplicate evidence IDs, pilot promotion and test-based selection in future specifications.

## Evidence boundaries

Fresh replay: frozen bytes, metric calculations, query alignment, selection histories, the L155 interval and claim/lineage report. Inherited only: raw SQL reconstructions, sampling, historical availability assessments and numerical-gradient diagnostics. No new source-SQL, training or ingestion-history audit is performed. Source nonfinite-gradient concerns persist. Policy FAIL/NOT_ESTABLISHED is copied from pinned upstream audits and labeled as inherited. Hash equality does not prove correctness or historical identity.

Two completed tasks on two databases; one matched manual-FE task. Local effort NOT_OBSERVED; recommendation INCOMPLETE/test NOT_RUN; graph-construction experiment NOT_RUN; whole paper and original human study NOT_RUN; historical identity NOT_ESTABLISHED; local contribution PENDING_PUBLICATION. Live Colab/deployment NOT_CHECKED. Learner PENDING_WRITTEN_DEFENSE.

## Primary reading and upstream runs

[RelBench v1 Section6](https://arxiv.org/html/2407.20060v1#S6), AppendixC and Tables6–8. Fresh execution commands and individual protocol deviations remain in [L151](l151-reproduction.md), [L152](l152-reproduction.md), [L153](l153-reproduction.md), [L155](l155-reproduction.md), [L156](l156-reproduction.md), [L157](l157-reproduction.md). The basic GNN F1 comparison does not reproduce Figure3's boosted regression head or the paper's human-work study.
