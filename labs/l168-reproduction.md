# L168 — cross-database generalization reproduction

Approved 2026-10-01. **Complete selected released-checkpoint experiment:** RDB-PFN arXiv2603.03805v5, Table9, rel-trial/study-outcome,512supports,10seeds,3arms. All30fresh evaluations and24,750probabilities complete. F1's30runs/21,060probabilities are reused from L166 and independently rescored. Whole-paper benchmark and fresh foundation-model pretraining NOT_RUN. This is not a proof that the exact target schema was unseen in all pretraining components.

## Frozen protocol

- Paper: https://arxiv.org/html/2603.03805v5 ; official code a95378225478daa262b85f180d482da7516b0af6. Original62model-pretrain source files authenticated against L166's original source ledger; no upstream edits.
- Data: yamboo/RDB_PFN revision d6a88c0a8cce79607cfc0fca0dcba78ba262ffad, model_pretrain/rdb_datasets/rel-trial-dfs-2/study-outcome. [Download URLs/hashes](sources/l168/downloads.json); [frozen inputs](evidence/l168/input-manifest.json).
- Complete11994train/960validation/825test rows;176sorted float/category features, excluding outcome; nct_id and timestamp are excluded by metadata dtype. Do not encode label or identity columns as features. All feature values preserve released numeric category codes. No validation tuning/refit.
- For seed s, SHA256 of `rel-trial-dfs-2:study-outcome:{s}`, first4bytes big-endian → NumPy default_rng →512indices without replacement. These match original downsample_split exactly. Same supports across all3arms. Model global seeds42; no target optimizer updates.
- Impute missing features with selected support medians,0for an entirely missing support column. RDB-PFN additionally applies original support-only population mean/variance and clamp±100. Original numeric model:6blocks,width96,4heads,MLP192,binary head. Float32inputs. Query chunk2000does not split825rows. No50,000row cap binds.
- Fixed RDBPFN model_eval00528.pt and RDBPFN_single model_eval00360.pt. No checkpoint-directory search, which would choose using test metrics. Initialization ablation received a different training history; this is not matched-compute causal isolation of the prior.
- TabICL0.1.3,v1.1-0506weights at eaf789a9b25ee8486d6f48997ba076f850bbc30b:32estimators, default preprocessing, shuffling and temperature,random_state42,CUDA. No lite substitution. Model hashes remain identical to L166.
- All825test rows identified by complete `(nct_id,timestamp)` pairs. Every raw prediction, label, support key, metric, package environment and file hash retained. Support seeds0–9in separate pilot/full dispatches preserve the stateless per-support evaluation; both phases authenticate the same input manifest.
- Paper AUROC targets:.5986RDBPFN,.5961single-table,.5926TabICL. Descriptive tolerance.02fixed before new test scores. Report every seed and sampleSD; tolerance is not equivalence or provenance proof.

## Data and exposure audit

All13,779released labels equal **one minus** the raw RelBench task labels, after exact full-key matching. The task archive hash20eb922c1a8f894563f4b4c900c912e396688d2bd71eeb6b13f19429aa74a649matches the L139historical artifact. We re-downloaded and aligned this task archive; L168did not re-run the full raw-database SQL reconstruction from L139. Both y and probability must be complemented to express the raw primary-outcome direction. AUROC then remains unchanged.

All12exposed MAXtimestamp features precede each owner cutoff. Both timestamps and these float32feature values use nanoseconds. Every candidate training label's365-day target horizon ends before the earliest test query. This checks represented clocks and horizon readiness, not actual publication/arrival dates or all176feature lineages. Raw DFS regeneration NOT_RUN; historical availability and exact historical preprocessing identity NOT_ESTABLISHED. The full raw database is not loaded by this lesson. The prior L139retrospective-cohort and inferred-date cautions remain applicable.

The paper reports synthetic predictor training. Its LayerDAG schema generator is trained on real schema corpora, citing Spider and BIRD. Pinned configs show synthetic dataset paths and the single-table checkpoint initialization; this is not a reconstruction of all training contents. Independent checkpoint lineage and exact target-schema exclusion remain NOT_ESTABLISHED. The teaching explorer distinguishes verified exclusion, known inclusion and incomplete evidence; hypothetical0-label mode is not measured RDB-PFN inference.

Griffin's supervised adaptation protocol is explained using its paper and L164. Its selected fresh reproduction remains INCOMPLETE_BUDGET_GATE. No cited Griffin metric is represented as a fresh measured arm.

## Execute

From the repository root, the saved-evidence audit needs NumPy; author verification also needs pandas, pyarrow and scikit-learn:

```sh
.venv/bin/python labs/_audit_l168.py
.venv/bin/python labs/_verify_l168.py
.venv/bin/python labs/_figures_l168.py
.venv/bin/python labs/_build_l168.py
.venv/bin/python labs/_execute_l168.py
.venv/bin/python labs/_delivery_l168.py
```

Fresh complete inference in a separate Python3.11environment: install torch2.5.1and `labs/l166-requirements.txt`, then:

```sh
python labs/_fetch_l168.py --out /tmp/l168-fresh-input
python labs/_run_l168.py --input /tmp/l168-fresh-input --out /tmp/l168-fresh-results
```

This runs all30evaluations with full test data and original model code. Output files are immutable: use a new output directory. The fetcher authenticates the prepared packet and all weights; no paid API service is required. Local execution has no billing guard. Original raw-preparation prerequisites are listed in the preparation script and frozen downloads ledger; the shipped fresh-inference lane does not need local raw database caches. The notebook embeds the full evidence and original model source, exposes the audit/learner/runner code, and provides an explicit opt-in fresh-inference lane. Default solution execution performs only CPU rescoring.

Author paid lane: `modal/l168_repro.py`,pilot seed0then remaining seeds1–9. Every worker attempt reserves its full timeout+30startup seconds before dispatch. The completed ledger rejects duplicate phase names. Any later paid rerun needs a separately approved ledger; do not erase this one.

## Results and cost

Trial means/sampleSD:

| Arm | Measured AUROC | Paper | Mean difference |
|---|---:|---:|---:|
| RDBPFN | .598555568 ± .021570196 | .5986 | −.000044432 |
| RDBPFN_single | .596100759 ± .021897228 | .5961 | +.000000759 |
| TabICLv1.1 | .592820215 ± .030356550 | .5926 | +.000220215 |

RDBPFN−TabICL paired mean gain:.005735353on trial,.004369297on reusedF1;6/10positive draws on each. Equal-database descriptive mean:.005052325. Two selected tasks do not establish broad superiority or database-level statistical certainty. All45,810probabilities independently rescored. [Full report](evidence/l168/report.json).

USD10aggregate cap. L4+2physical cores+16GiB at baseUSD.00028372/second. USD3overhead reserve;600secondpilot and7200secondfull-phase reservations plus30seconds startup each = **USD5.2300392conservative total reservation**. Worker body84.614352seconds≈USD.0240068excludes unitemized startup/build/storage; **invoice NOT_ITEMIZED**, so neither number is a claimed actual bill. All3apps confirmed already stopped. No inference retries. One local entrypoint failed before worker dispatch because budget creation used the wrong working directory; log retained. One local collection destination error was fixed by recollecting and checking hashes; no rerun. [Budget](evidence/l168/budget.json), [timing decision](evidence/l168/cost-decision.json), [cost](evidence/l168/cost.json).

## Delivery and evidence boundaries

Notebook/browser/source/link/Pages checks have separate receipts. Live Colab and deployment NOT_CHECKED. No push/deployment requested. Author execution does not imply learner mastery: PENDING_WRITTEN_DEFENSE.

Delivery completed: standalone 17-code-cell solution executed from an empty directory with exact report parity; three incorrect learner functions and six evidence corruptions rejected. All 24 desktop/mobile explorer states, keyboard/reset, print/no-JS, three portable figures, 38 copied-site links, manifest navigation, source parity and deterministic generation pass. Actual rendered figures and screenshots inspected. The full Pages workflow passes from a clean Git-index checkout; every frozen audit input is in the index. Receipts: `_verify_l168_results.json`, `_execution_l168_results.json`, `_delivery_l168_results.json`, `_checkout_l168_results.json`. These are local author/delivery checks, not live Colab or deployment evidence.
