# L183 · Graph-transformer pretraining: evidence and reproduction contract

Approved2026-10-02: complete lesson and executable audits; no paid training. USD10 aggregate ceiling, plannedstop8/reserve2; numerical local execution<=3600seconds including failures and notebook verification. No new hybrid model result is claimed.

## Executed lane

L146 six saved course fits, GNN/reduced RelGT × seeds0/1/2,499validation/760test each:7554predictions. All full(entity,cutoff) populations are matched to saved preparation targets, complete10epoch histories/7453queries per epoch and first strict validation-minimum checkpoint selection checked. Independent SQLite joins/MAE agree with live Python scoring. This is replay, not raw SQL label reconstruction, fresh inference or training. Original model/trainer sources match original ledgers; copied evidence matches Git HEAD at pin time. SHA256 proves byte identity, not historical experimental identity. `input-manifest.json` pins the complete portable packet.

L145 temporal counts are authenticated saved report fields; one stored witness is checked. Full cache rescan NOT_RUN. L164/L145 cost arithmetic is recomputed from saved pilot observations at inherited rates; early-stopping and deeper-architecture costs remain uncertain. No new pilot or price quote. Synthetic visibility/factorial fixtures teach contracts only. The transfer gate consumes documented evidence flags; it cannot prove a flag is true.

## RelGT full named protocol (retained, NOT_RUN)

Primary https://arxiv.org/html/2505.10960v1 Tables1/6; source19e423ca3e7cac761130aba790857f2dc3a46ef7. F1 driver-position, all7453/499/760train/validation/test queries. Complete depths1/4/8 ×dropout.3/.4/.5, seed0,100epochs each, width512,K300,4096centroids,batch256,Adam1e-4,weight_decay1e-5,L1loss,gradientclip1,training-label2nd/98th percentile clipping. Within-fit source selection retains the last tied validation minimum. Historical cross-configuration selection NOT_ESTABLISHED. Published headline3.917MAE matches displayed minimum test; displayed validation-minimum configuration has test4.6316. Printed values and stochastic reevaluation do not establish selection intent.

Original source token caches violate per-query temporal ownership (entity-key overwrite and unfiltered fallback); saved audit569502train/21368validation future-token occurrences. Eval local-attention dropout and structural randomness remain source observations. Preserve these defects as evidence, never silently repair and call it the original experiment. Cost forecast80.4115008914USD assumes all nine run at measured shallow speed, excludes final evaluation/overhead and leaves deeper costs unmeasured. Temporal failure independently blocks the clean experiment.

Complete model: `relkit/relgt_l145.py`; trainer: `_full_l145.py`; source: `sources/l145/`; preparation, archive hashes, runtime and operator: `l145-reproduction.md`. Existing `_reproduce_l146.py --audit` gives the inherited gate; the guarded full search requires prepared source data and is currently blocked. The notebook includes the complete canonical model and trainer as visible source, not an executed full fit.

## Griffin full named protocol (retained, NOT_RUN)

Primary https://arxiv.org/html/2505.05568v1 Table12. Codeb9d0e1fa8d89dfb1cd8bd5976b71de8a3b515427; processed datae0c54ceada75317b06f11f8dcda7aa8304fbb593; Others-2FULL checkpointbd8c5be5130f34e7faa31099d0bd81d95d0aa995. F1 driver-dnf, full11411training population with source subsets512/4096,566validation/702test. `RandomState(seed).permutation(train_rows)[:n]`, subsetseeds42–46, fixed modelseed42; two arms random/Others-2 =20fits. Same selected subset every epoch, batch shuffling. D512,8heads,4message layers,forward/reverse,use_gate=False; hop2/fanout20/fewshotfanout3; batch256,AdamW3e-4,weight_decay2e-4,max200epochs,validation every2epochs,stop epoch−best_epoch>10; highest strict validation score. Source also logs test on each improvement. Final best weights/full test; original2class macroAUROC, independently check positive-classAUROC. Mean/sampleSD over all5subsets per arm/budget, paired differences. Paper means .6558/.7176(no-pretrain),.7098/.7275(Others-2); inherited ±.02 descriptive tolerance does not establish equivalence.

Source trainer expects missing `model.device`; retained wrapper adds read-only parameter-device property. Checkpoint config says gates enabled but43weight tensors match transfer script's disabled setting. Original numerical encoder/decoder and metadata/task conditioning preserved. Processed label SQL identity, preprocessing fit horizon and historical feature arrival remain unestablished. Source fewshot helper ignores timestamp; narrow F1 static-root audit does not prove general temporal legality.

Inherited conservative200epoch scenario: max pilot batch time,100validations,101test equivalents scaled702/566,all20fits,1.25margin →63.8910407376USDcompute. Source16worker throughput is unmeasured; pilot used0workers. Estimated cost including prior reservation and reserve66.7592007376USD; not an invoice. Zero full fits; final testNOT_RUN. Complete visible model `relkit/griffin_l164.py`, original `sources/l164/upstream/hmodel.py`, original full trainer `hmaintask_downsample_absolute_eval_sample.py`, wrapper `_run_l164.py`, full20fit managed operator `modal/l164_repro.py`. Commands/environment/preparation and full lineage: `l164-reproduction.md`.

## Local commands

Run from repository root. Default notebook embeds every needed audit input and is offline, CPU-only; Python3 and NumPy suffice. Source appendices require separate historical GPU environments to execute and are displayed only.

```bash
.venv/bin/python labs/_budget_l183.py .venv/bin/python labs/_prepare_l183.py
.venv/bin/python labs/_budget_l183.py .venv/bin/python labs/_check_l183.py
.venv/bin/python labs/_budget_l183.py .venv/bin/python labs/_audit_l183.py
.venv/bin/python labs/_budget_l183.py .venv/bin/python labs/_verify_l183.py
.venv/bin/python labs/_run_l183.py --lane relgt
.venv/bin/python labs/_run_l183.py --lane griffin
# --run deliberately exits nonzero before any cloud or training call.
.venv/bin/python labs/_budget_l183.py .venv/bin/python labs/_figures_l183.py
.venv/bin/python labs/_build_l183.py
.venv/bin/python labs/_budget_l183.py .venv/bin/python labs/_execute_l183.py
.venv/bin/python labs/_budget_l183.py .venv/bin/python labs/_delivery_l183.py
```

Original full trainers remain runnable in their declared environments with resolved inputs/gates and separate authorization; L183 does not bypass them. `--run` is tested only as a refusal. No new hybrid trainer is supplied or implied. Notebook post-EXIT cells expose the full protocols and source implementations, with paid training disabled.

## Proposed probe, not execution authorization

Four arms(backbone×initialization),two F1 label budgets,five subset/model seed blocks =40targetfits,plus independent source pretraining for each backbone. Candidate source corpus is pinned Griffin Others-2 group, not an authenticated new pretraining run. Shared semantic encoder and decoder; exclude all F1 rows, edges, labels and fitted statistics from every source-training stage; forbid access to target validation/test during source selection. Audit duplicate databases/entities and temporal/arrival histories. Match target subsets, permissible contexts and selection policy; measure parameter counts, updates and all-in compute. Primary interaction is(GTpretrained−GTscratch)−(MPpretrained−MPscratch) in AUROC per seed/block/budget. Transformer pretraining gain<=0 falsifies its benefit in this probe; interaction<=0 falsifies extra benefit over MP. Five seed blocks are not five databases.

Exact adapter, source schedule, dependencies, source data eligibility and affordable all-in bound remain unresolved. Future execution requires a frozen protocol and separate approval. One F1 database is feasibility evidence only; a generalization study must hold out multiple entire databases and report per-database results, avoiding pooled raw metrics across tasks. All hybrid fits/pretraining NOT_RUN. Novelty and transfer superiority NOT_ESTABLISHED.

## Delivery boundaries

Author checks are recorded in `_verify_l183_results.json`, `_execution_l183_results.json`, `_delivery_l183_results.json` and `_checkout_l183_results.json`. The latter verifies a local Git-index Pages build, not deployment. LiveColab NOT_CHECKED; no push or deployment requested. Learner PENDING_WRITTEN_DEFENSE; earlier practical exits are unchanged.
