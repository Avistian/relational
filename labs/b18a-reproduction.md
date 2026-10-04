# B18a reproduction contract

Approved 2026-10-04. See [frozen design](../docs/plans/2026-10-04-b18a-design.md).

## B18A-CONTEXT-STATE: complete finite release experiment

All 569 breast-cancer rows, sklearn1.6.1. train_test_split(test_size=.5,stratify=y,random_state42):284 support/285 query rows; original integer identities retained. Fixed stratified71-row subset (25% of support). All seeds0/1/2 × POT-full/POT-selected/TACO4 × fit_preprocessors/fit_with_cache =18 fits. Eight estimator ensemble, float32, TF32 disabled; six sequential query batches50/50/50/50/50/35; four full passes, first pass cold prediction and three repeated passes. No seed, arm or checkpoint selection. Seeds vary inference recipe, not dataset identity. All metrics use unchanged285 query labels. No trained distilled student in this experiment.

Compressor output reuse is present in released TACO even in fit_preprocessors; the toggle specifically controls predictor KV caching. Both full and selected POT use the released POT checkpoint. TACO uses a separately jointly trained compressor/predictor checkpoint. This is not a matched-weights causal test of compression.

Time definitions: model construction/load+fit, first complete query pass, subsequent three query passes, individual synchronized batches. Download/image preparation excluded from these runtime comparisons, included separately in budget accounting. GPU allocated peak is not total process VRAM or isolated KV cache size. AUROC/log loss and max absolute prediction differences are independently recomputed. Probability agreement tolerance1e-5 fixed before results. Any failed agreement remains FAIL, not relabeled successful caching.

Additional seed0 cached POT/TACO interventions use283 baseline support rows, reserve one support row for addition, delete a different baseline row, deliberately flip one baseline label, rescale feature0 by10 consistently for support/query and refit. Query missingness masks feature0 on first57/285 queries. Every support change creates a fresh estimator. Cached baseline predictions are the stale comparator; no unsafe in-place mutation needed. Query-only changes retain support cache identity but invalidate any memoized query outputs. Separate foreground16 queries/backgrounds first8 and last8 baseline support rows define f(x)-mean_b f(x with feature0=b0). This is a replacement contrast, not SHAP, model-retraining LOCO or a causal effect. All forward rows and latency retained.

## Source identity

- [TACO v2 paper](https://arxiv.org/html/2602.05649v2), including archived Figure3 SVGs.
- Git revision002f83bdb5b1776ca69b7916f993a82344a3aef7; full source/licenses archived in sources/b18a/taco.tar.gz.
- Hugging Face revision d38ed9517764698a0b0064a7a8cb4197016349a3. Both checkpoint hashes independently checked before loading.
- The sparse model card does not identify which paper training stage/table run each checkpoint represents. Released-checkpoint results cannot establish historical paper identity.

## B18A-TACO-FIG3: historical result gate

Target: Figure3 cumulative inference cost,15000 synthetic support rows,500 features,100 batches50 queries, POT and the figure's TACO compression ratios, with/without predictor KV caching. Original generator/data hashes, dataset random seeds, historical checkpoint mappings, complete executable figure runner, timing/hardware identity and numerical reference curves must be authenticated before result reproduction. The archived release contains training scripts but no Figure3 evaluation runner. Paper figure and prose are source evidence, not execution evidence. Gate must refuse when any required identity is unresolved; no substituted generator or resized workload is paper reproduction. See sources/b18a/paper-contract.json for audited fields.

Full pretraining (reported80k steps,1024 global batch, eight H100s over about20days), continuation/adaptation, full26/36-task TabArena evaluations and TabPFN2.5 distillation benchmarks: NOT_RUN. Full-paper reproduction: NOT_ESTABLISHED.

## Commands

From repository root:

```bash
.venv/bin/python labs/_reproduce_b18a.py --phase audit
.venv/bin/python labs/_reproduce_b18a.py --phase paper  # refuses unresolved historical identities
.venv/bin/python labs/_test_b18a.py
.venv/bin/modal run labs/_cloud_b18a.py --phase pilot
# Inspect saved pilot and complete-matrix cost projection before these:
.venv/bin/modal run labs/_cloud_b18a.py --phase matrix
.venv/bin/modal run labs/_cloud_b18a.py --phase updates
.venv/bin/python labs/_verify_b18a.py
```

Runner refuses evidence overwrite and outstanding reservations. Each remote call has1800-second timeout, retries0. All attempts share US$10 cap, US$8 main envelope, US$2 reserve,18000 aggregate GPU-second cutoff. The operator must reconcile interrupted reservations. A pilot failing fit/caching or a cost projection exceeding remaining cap blocks subsequent dispatch. Local numerical execution has a3600-second aggregate bound. No unaffordable run is silently shrunk.

## Observed cache discrepancy

All three TACO cached/uncached pairs fail1e-5 probability agreement (max differences0.02807/0.03192/0.02798). All six POT pairs pass. See [source-path audit](sources/b18a/cache-path-audit.md): encoder statistics useN versusN−K in these release paths. No patch/retest causal attribution was performed. Preserve timings as mode-specific measurements, not function-preserving speedup.

## Evidence boundaries

Finite experiment completion, individual agreement tests, source audit, historical result identity, paper reproduction, rendered packaging, live Colab, deployment and learner mastery are separately reported. Student TODOs must run; merely reading author outputs does not establish mastery. No live Colab or publication is implied by local notebook/browser checks.
