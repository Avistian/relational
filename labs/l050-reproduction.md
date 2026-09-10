# L050 reproducibility contract

Local run: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_verify_l050.py` from the repo root. Expected duration on the author's CPU: about 91 seconds; hardware affects timing. Dependencies are in `requirements-labs.txt`; exact measured versions and source hashes are in `_verify_l050_results.json`.

- Numeric-only FT-Transformer with ReGLU, omitted first attention LayerNorm, 2 blocks, d=32, 4 heads, attention and FFN dropout .1, residual dropout 0, no compression; binary head. Copied-weight numeric eval output/input-gradient parity with `rtdl_revisiting_models==0.0.2`; not original-environment or stochastic training parity.
- OpenML 37 diabetes, 1464 blood transfusion, 1489 phoneme. All cached rows; positive class is the loader's lexicographically last string class, or existing numeric 1. Hashes and encoded labels are recorded. These are substitutes, not paper benchmark tasks.
- Stratified 60/20/20; both split calls seed 50. Median imputation and mean/SD fit only on training rows. Constant feature scale maps to 1; entirely missing training columns fail explicitly. Numeric zeros in diabetes retain their supplied values; no clinical reinterpretation is inferred.
- Each arm/seed gets two candidates: neural lr .001/.0003, tree depth 3/6. Same validation AUROC selection; exact ties first. Neural caps 35 epochs/patience 8/batch 256; tree caps 160/patience 20/lr .05/subsample .8/column sample .8. Seeds 0,1,2. No test-based selection or final train+validation refit.
- Sample SD and 95% t intervals reflect training-seed variability conditional on one split. Dataset mean ranks enter Friedman/Nemenyi with N=3, k=3. Report exact two-sided sign test for the primary pair. This convenience suite is too small to establish universal superiority or equivalence.

The notebook's preprocessing, ReGLU, candidate selector and paired summary are live student code. The complete model, neural trainer and experiment loop are visible. The executed teacher solution runs the comparison again and checks its scores against the recorded author run.

## Required next step

`python labs/_paper_repro_l050.py --preset closer --out labs/data/cache/l050-your-run`

Or `modal run --detach modal/l050_paper_repro.py --preset closer`, then `modal volume get relational-artifacts l050 ./l050-artifacts`. The Colab cell passes the live notebook experiment into the same artifact/resume operator. It is gated off until the student elects to run it. No Modal job was launched during authoring.

Presets: smoke = 800 rows/2 epochs/20 trees/one seed; closer = 6000 rows/d64/3 blocks/50 epochs/400 trees/3 seeds; paper = full Higgs Small/d192/3 blocks/100 epochs/2000 trees/3 seeds. The `paper` name describes resource scale, **not a complete replication**. All select on AUROC and also record accuracy at probability .5. Completed seeds resume; interrupted seeds restart. A changed source/operator/data/configuration rejects reuse, and custom notebook implementations need an explicit identity.

Paper target: Gorishniy et al., arXiv 2106.11959v5, Table 2 FT-T Higgs Small accuracy .729, prospective absolute tolerance .01 **only if protocols align**. Current verdict is INCOMPARABLE for any measured preset: new splits, standard rather than quantile preprocessing, two candidates, different settings and repetition/selection. The paper's Table 4 tree comparison is an ensemble experiment, still NOT_RUN here. No universal tree/neural claim follows from either local track.

Source: https://arxiv.org/html/2106.11959v5 ; https://github.com/yandex-research/rtdl-revisiting-models . The larger-run summary gives its actual execution status. Notebook PNG data-URL integrity and prepared HTML are checked; live Colab UI and browser interaction are separate verification claims.


## Measured larger attempt

The author completed `closer` on CPU: 3 seeds, about 1814 seconds of fitting. Accuracy: FT-T .68306 ± .00394, XGBoost .68472 ± .00210, MLP .64833 ± .00546 (sample SD). This is INCOMPARABLE to the .729 paper target. Full per-seed predictions, validation histories and split/source identities are in `_scaleup_l050_evidence.json`. `_paper_repro_l050_measured.py` preserves the exact hash-verified operator used for these measurements; the current operator additionally requires an identity for a custom notebook implementation. The measured cache intentionally fails identity validation under the newer operator; use a new output directory. This guard addition does not change model training.


## Audit repair, 2026-09-10

The original `relkit/checkpoint.py`, `_verify_l050_results.json`, `_paper_repro_l050_measured.py` and `_scaleup_l050_evidence.json` are preserved. The teacher solution is freshly executed after the notebook repairs; the original selected AUROCs are required to match exactly in this environment. Added code is diagnostic and artifact bookkeeping, not a replacement for those historical measurements.

The notebook now has five live tasks. `mean_predictions` combines the three already selected models' test probabilities before computing AUROC. This is a **post-hoc diagnostic**, not a new independent evaluation or Table 4 reproduction. No confidence interval is borrowed from single-model seed scores for the one resulting ensemble. The synthetic CHECK shows mean seed AUROC 0.50 versus ensemble AUROC 0.75.

`_audit_l050.implementation_identity` covers live preprocessing, ReGLU, selection, paired summary, trainer, predictor, experiment, and constructors/forward methods for FT-T, its blocks/attention and MLP. It includes callable defaults semantically and the data harness source. It is recomputed when the scale-up gate opens, so edits after EXIT change the cache identity. Package versions, data, device and configuration remain recorded by the operator. Arbitrary external library monkeypatches are outside this identity contract. `_audit_check_l050.py` checks fresh-process stability, execution stability, actual smoke execution, unchanged resume and rejection after a changed live predictor.

Explicit remaining recipe differences: the paper fixes eight attention heads; all current presets use four. The paper evaluates a tuned configuration over 15 seeds and creates three ensembles with five models each; this lab reselects two candidates separately for each of three seeds. The scale-up selects on AUROC and then reports accuracy. It does not reconstruct accuracy-based selection. Merely increasing the resource preset cannot establish protocol fidelity.

Pinned source provenance is in `../reviews/lesson-quality-audit-047-070/050-sources.json`. The modern package source hash equals the installed 0.0.2 reference hash used in the copied-weight check. The original `bin/ft_transformer.py` was inspected separately for residual wiring, final-query optimization and training/optimizer distinctions.
