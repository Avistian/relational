# L167 — tabular-to-relational transfer audit

Approved 2026-10-01. Complete new audit of **existing** L166 checkpoint inference: Wang et al., arXiv 2603.03805v5 Table 9, rel-f1/driver-dnf, 512 supports, seeds 0–9, RDBPFN / RDBPFN_single / TabICLv1.1. All 702 test queries and 72 features; 30 evaluations, 21,060 probabilities. No new training, inference or paid work is authorized by this audit.

## Reproduce this audit

From the repository root, using Python 3 with NumPy:

```sh
python3 labs/_audit_l167.py
```

The portable solution embeds the same raw NPZ files, original receipts, prepared arrays, selected original source files and both input manifests. It unpacks into `l167-portable`, checks SHA256 and executes the identical visible audit. Default notebook execution needs NumPy only beyond Python's standard library. The author checker additionally uses sklearn as an independent metric oracle.

```sh
.venv/bin/python labs/_verify_l167.py
.venv/bin/python labs/_execute_l167.py
.venv/bin/python labs/_delivery_l167.py
```

The new [input manifest](evidence/l167/input-manifest.json) freezes 41 existing files. The audit checks raw predictions against the **original** run-receipt hashes; prepared arrays against the original input manifest; original manifests against both run receipts; selected code against the original source ledger. These establish local artifact continuity, not an independently signed historical chain of custody.

Support indices regenerate using SHA256(`rel-f1-dfs-2:driver-dnf:{seed}`), first 4 bytes big-endian → NumPy default_rng → 512 indices without replacement from 11,411 training rows. Test keys and labels must match the entire prepared population. Every arm uses the same support order. No validation/test-driven checkpoint or hyperparameter selection is performed by this lesson.

AUROC is recomputed independently using average ranks and half-credit ties; test probabilities are aligned by complete `(driverId,date)` keys. Mean and sample SD use ten support draws. Paired differences preserve seed correspondence. The original descriptive mean tolerance 0.02 is retained; no statistical equivalence or multi-database claim follows. Primary published targets are .7219/.6640/.7176. The original inference protocol, fixed checkpoints, packages, transformations and deviations remain in [L166's contract](l166-reproduction.md). TabPFN v2 is conceptual comparison only.

## Full selected fresh-inference lane (separate)

The complete original lane remains runnable, using the pinned environment in [L166 requirements](l166-requirements.txt): Python 3.11, torch 2.5.1, tabicl 0.1.3, original source commit `a95378225478daa262b85f180d482da7516b0af6`, data revision `d6a88c0a8cce79607cfc0fca0dcba78ba262ffad`. From repository root:

```sh
python3 labs/_fetch_l166.py --out /tmp/l167-fresh-input
python3 labs/_run_l166.py --input /tmp/l167-fresh-input --out /tmp/l167-fresh-results
```

These commands evaluate all 30 selected checkpoint runs; they are **not executed by L167**. They do not pretrain foundation models. Check destination availability before running: the runner refuses existing outputs. Remote paid execution requires a new costed plan; the L166 cloud cost ledger must never be reset or reused to hide spend.

## Boundary ledger

| Claim | Status / provenance |
|---|---|
| L166 original 30 evaluations | COMPLETE; existing evidence |
| L167 raw-file hash, support and metric audit | PASS; complete [report](evidence/l167/report.md) and [verifier](_verify_l167_results.json) |
| Course cutoff/label-ready interventions | Synthetic mechanism checks, no benchmark accuracy claim |
| Fresh L167 inference / pretraining / whole-paper suite | NOT_RUN |
| Historical preprocessing identity | NOT_ESTABLISHED |
| Raw SQL target reconstruction / full DFS regeneration in L167 | NOT_RUN; L166 target audit inherited explicitly |
| Available-event history in the real database | NOT_ESTABLISHED |
| TabPFN v2 benchmark result here | NOT_RUN |
| Learner mastery | PENDING_WRITTEN_DEFENSE |
| Live Colab / deployment | NOT_CHECKED |

Released labels complement the current raw DNF target documented by L166. No label relabeling is silently applied. Support-label readiness is demonstrated with a separate synthetic fixture; it is not inferred from task timestamps in the benchmark packet. Frozen weights do not remove temporal availability obligations.

Budget: **USD 0 additional paid compute**, no cloud dispatch; standing USD 10 aggregate cap retained. Local author commands single-threaded where numerical libraries permit, each bounded at 600 seconds. All CPU audit evidence is new; model inference evidence is reused.
