# L054 — TabM reproduction and evidence contract (audit v2)

The taught model is now **corrected v2 numeric TabM-mini**. The previous implementation kept an extra first output adapter, member-specific backbone biases, and applied PyTorch Kaiming initialization to transposed/three-dimensional weight layouts with the wrong fan-in. Those are real mechanism/initialization differences, not just a smaller training budget.

## Three identities that must remain separate

1. **Historical v1:** `relkit/tabm.py`, `relkit/tabm_experiment.py`, `_verify_l054.py`, `_verify_l054_results.json`, and its old logs remain unchanged. Some later lessons use these operators. Original JSON labels say “TabM-mini”, but they identify the older custom variant, not the corrected architecture. Do not overwrite its source identity or treat its numbers as v2.
2. **Corrected v2 fixed-budget suite:** `relkit/tabm_v2.py`, `relkit/tabm_experiment_v2.py`, `_verify_l054_v2.py`, `_verify_l054_v2_results.json`. The lesson and regenerated evidence figures use this track. Exact source hashes, selected raw row indices, array hashes, library versions, prediction arrays and single-MLP/TabM validation histories and first-best checkpoint epochs are recorded. Independent deep-ensemble member checkpoints/histories are not retained.
3. **Closer procedure:** `_paper_repro_l054.py`. Same corrected model and live learner functions, with independent member batches, train-only normal quantiles, AdamW, gradient clipping and patience. A bounded smoke was run. Larger presets and cloud execution remain NOT_RUN. This is not a paper benchmark reproduction.

## Primary sources and actual reference check

Paper: [arXiv 2410.24210v3](https://arxiv.org/html/2410.24210v3), especially §3.1–3.5; §4.2–4.3; §5.1–5.4; Appendices A.1–A.4, C, D.1–D.9.

Pinned official repository: [28e47ae301c92ec37787dde1ce923a0793f405b4](https://github.com/yandex-research/tabm/tree/28e47ae301c92ec37787dde1ce923a0793f405b4). Paper experiment configuration and routing are inspected in `paper/bin/model.py` (`Model.__init__`, `Model.forward`, `_init_first_adapter`, training batches and clipping). Copied-weight validation uses the released standalone `tabm.py` v0.0.3 (`LinearEnsemble`, `LinearBatchEnsemble`, `MLPBackboneMiniEnsemble`, `TabM.make`). The standalone release is a validation reference, not a claim of identical historical training code.

`_source_check_l054.py` checks equal parameter counts and complete numeric forward/input-gradient/every-parameter-gradient parity for mini and full TabM; scalar regression and two-logit classification; shared 2-D inputs and distinct 3-D member inputs. Dropout is zero for deterministic copied-weight checks. Eight output comparisons are exactly zero in float64. This validates the implemented operators and layouts. It does **not** match the authors' random-number draw sequence, training trajectories, optimized runtime or benchmark scores.

For reference validation only, `rtdl_num_embeddings==0.0.12` was installed into `/tmp/tabm-audit054/deps` without modifying workspace dependencies. Re-run with an environment containing that package, or `--deps` pointing to an equivalent temporary install. Reference source URL and SHA256 are saved in `_source_check_l054_results.json`.

## Architecture and initialization

Mini: first learned R:[k,p] initialized at random ±1; shared W and ordinary shared bias in every backbone layer; ReLU/dropout; independent heads. There is no S adapter, later R, or member-specific backbone bias. Adapters learn unrestricted real values after initialization.

Full TabM: R,S and member bias in every backbone layer; first R random ±1, all other multiplicative adapters ones; all member biases begin at the same random linear-style bias. Shared and head weights use U(−1/√fan_in,+1/√fan_in). Full TabM is source-checked but not an arm of the measured suite. Neither model implements numerical embeddings, categorical handling or a multiclass training harness.

The paper column-vector equation has W:[out,in] and mask srᵀ. Local row-vector storage has W:[in,out] and mask rsᵀ. The rank-one mask does not make the effective matrix rank one. MLP×k and TabM-packed share an unshared architecture but differ in member-wise versus collective tuning/stopping. Packed training is not implemented by the independent deep-ensemble baseline.

## Data and fixed-budget v2 procedure

California Housing, House 16H and Higgs Small from the verified TabR release (`_fetch_l052.py`, `_data_l052.json`). Paper Appendix C explicitly inherits these datasets and released boundaries; no independent byte-level second-archive comparison was performed. Caps are 1,200 train / 600 validation / 600 test; label-blind selection seeds 53/54/55; model seeds 0/1/2. Train-only median imputation, z-score features and regression target scaling. The released full sizes in paper Table 4 are respectively 13,209/3,303/4,128, 14,581/3,646/4,557 and 62,751/15,688/19,610.

Neural arms: width 64, depth 3, 64 epochs, batch 256 with partial final batches dropped, dropout 0.1, Adam 0.002, no gradient clipping or patience, first strictly best validation checkpoint. The old suite used last-best ties, so a v1/v2 score difference is not an isolated initialization effect. The independent ensemble fits 32 MLPs and selects each member independently; TabM-mini uses shared batches and collective validation. XGBoost selects among six depth/rate candidates with 150-tree ceilings and validation early stopping. Trees consume the same imputed raw arrays. Equal data boundaries do not imply equal tuning or compute budgets.

The notebook's separate live track uses k=8 / 32 epochs and three seeds; it exercises the learner's functions. Its scores are not the k=32 author-reference scores. EXIT saves `labs/artifacts/l054/exit.json` with protocol, predictions and an explicit interpretation/completion field. An empty interpretation is not learner completion.

Metrics: original-unit RMSE or classification error, not AUROC. Summaries use sample SD and conditional 95% Student-t intervals over seeds. Ranks first average seeds within each dataset and then rank datasets. Ordinary local ranks and approximate Friedman/Nemenyi summaries are not the paper's SD-based tiers (D.3). Three tasks have low power; p>.05 is not equivalence.

## What the paper experiments actually test

- Main benchmark:46 datasets,37 random / 9 domain-aware splits; usual modified quantiles; AdamW, clipping 1, patience 16; model-specific 50–100 trial TPE (some 25), mostly 15 evaluation seeds (D.2–D.3, D.9).
- Figure 5: depth 3,width 512; no dropout/weight decay;25 learning-rate grid points tuned separately for k=1 / k=32; training continues beyond optimal epochs. The local checkpoint-only comparison does not recreate these training dynamics (D.5).
- Figure 7:17 datasets, five seeds, dropout 0.1, learning rate retuned at each k. A one-seed fixed-rate California k-sweep cannot identify a cause such as shared bias or reproduce this figure (D.7).
- Greedy pruning: choose by improvement to current validation aggregate; stop when no next member helps AND current validation already beats full-k. Mean retained 8.8 ± 6.6 is over 46 datasets (D.6). Not run here.
- Dead neurons: whether a shared ReLU ever activates on 2,048 training objects across member paths, at the selected checkpoint; .29±.17 for k=1 versus .14±.09 for k=32 across 46 datasets (§5.4). This is evidence consistent with utilization, not a causal proof. Not run here.
- Efficiency: asterisk variants additionally use AMP/compile; Table 2's large datasets reuse small-training configurations and common evaluation sets, mostly three seeds but one for FT-T (D.4). Our CPU fit times do not reproduce those comparisons. They are diagnostic wall times recorded under varying editor/background CPU contention, not controlled latency or speedup evidence.

Ensemble MSE/RMSE and probability cross-entropy cannot exceed the average member value by convexity. This alone does not establish learned improvement over a separately trained MLP. Classification error can worsen: probabilities (0.51, 0.51, 0.01) on a positive row give 1/3 mean individual error and 1 collective error. The lab records observed class-error ordering rather than asserting a desired winner. Stored minimum individual test error is an oracle diagnostic, not validation-selected deployment evidence.

## Concrete published target

Appendix E Table 18, California, single-model numeric TabM-mini: RMSE **0.4479 ± 0.0022** (mean ± seed SD). This single model contains 32 implicit branches. The separate five-whole-model ensemble column reports 0.4461 ± 0.0011; the † mini with embeddings reports 0.4275 ± 0.0024. Neither should be substituted for the plain numeric mini target. Our fixed-recipe result remains INCOMPARABLE regardless of numerical proximity; matching requires the actual selected preprocessing/recipe and evaluation protocol.

## Executable closer track

From repository root:

```
.venv/bin/python labs/_paper_repro_l054.py --preset smoke
.venv/bin/python labs/_paper_repro_l054.py --preset closer --device cpu
modal run --detach modal/l054_paper_repro.py --preset closer
```

The gated post-EXIT Colab cell passes the actual notebook `TabM`, `member_mean_loss`, `ensemble_predict` and visible `fit_closer`. Independent shuffles supply B distinct rows per member, not B/k rows; input and label axes are flattened consistently for the live loss. AdamW uses learning rate 0.002 and zero decay, clipping 1, dropout 0.1 and patience 16; incomplete final batches are included. Train-only sklearn normal quantiles approximate but do not exactly match the released modified quantiles/jitter.

`closer`:6,000 California train rows, full validation/test, width 256, k=32, 150-epoch ceiling, 3 seeds. `paper`:full train, width 512, k=32, 1,000-epoch ceiling, 15 seeds. Both retain a fixed recipe; no TPE, feature embeddings or multi-dataset benchmark. The name is a resource preset, never a fidelity certificate. Resume checks live transitive code, data, configuration, operator and environment. The smoke checker confirms unchanged resume and rejection after a live helper edit.

## Rebuild and verification

Run `_verify_l054_v2.py` from `labs/` to regenerate its separate evidence, then `_figures_l054.py`, `_build_l054.py`, execute the solution notebook, and `_render_l054.py`. Use the project venv. Run `_source_check_l054.py`, `_check_repro_l054.py`, and root's `_viz_check_l054.js`. Parent integration separately checks browser desktop/mobile, copied Pages staging and deployment. Teacher solution remains gitignored. Live Colab UI and cloud runs are NOT_CHECKED/NOT_RUN.

**Paper results reproduced: NO.** Corrected numeric operators are validated; local measured procedures and the bounded closer smoke are separate evidence, all INCOMPARABLE to the main benchmark.
