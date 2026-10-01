# L166 — RDB-PFN reproduction contract

Approved 2026-10-01. **Complete selected experiment using released checkpoints**: Wang et al., [arXiv 2603.03805v5, Table 9](https://arxiv.org/html/2603.03805v5), rel-f1 / driver-dnf, 512 support rows, seeds 0–9, three arms. AUROC targets: RDBPFN 0.7219, RDBPFN_single_table 0.6640, TabICLv1.1 0.7176. Whole-paper benchmark and fresh foundation-model pretraining NOT_RUN. Descriptive mean-distance tolerance 0.02 was fixed before evaluation; it is not statistical equivalence or a provenance test.

## Exact selected protocol

- Code commit `a95378225478daa262b85f180d482da7516b0af6`; paper v5. Archived source bytes and hashes: [ledger](sources/l166/source-ledger.json).
- Dataset `yamboo/RDB_PFN`, revision `d6a88c0a8cce79607cfc0fca0dcba78ba262ffad`, `model_pretrain/rdb_datasets/rel-f1-dfs-2`. Original download hashes in [downloads](sources/l166/input-downloads.json). Prepared feature and query packet [here](evidence/l166/prepared.npz), authenticated by [manifest](evidence/l166/input-manifest.json).
- 11,411 train, 566 validation, 702 test queries. Upstream evaluator uses train and test; no validation tuning. Features: all metadata float/category columns except target, lexicographically sorted. Keep released numeric category codes. No query label or entity key is an input feature.
- For seed s, SHA256 of `rel-f1-dfs-2:driver-dnf:{s}`, first 4 bytes big-endian → NumPy default_rng → 512 indices without replacement. Same supports for all models. Complete 702 test rows; upstream 50,000 cap does not bind. Chunk size 2000 does not split this task. Full `(driverId,date)` identities retained.
- Fill each missing feature with the **selected support's median**, zero when the support column is entirely missing. RDB-PFN then normalizes with support-only mean/population variance and clamps to ±100. Float32 feature input, ordinary numeric encoding, six blocks, width 96, four heads, MLP 192, binary decoder; 692,738 parameters.
- Fixed RDBPFN `model_eval00528.pt`; fixed single-table `model_eval00360.pt`. Both known before test access. Do not use upstream directory-mode checkpoint search, which ranks checkpoints on test metrics.
- TabICL 0.1.3, v1.1 `0506` weights at pinned Hugging Face revision/hash in manifest. **32 estimators**, default normalization/shuffling/temperature, random_state=42. No lite substitution. Device only changes to CUDA; no hyperparameter changes.
- Each source-model arm initializes the global random state 42, matching upstream; support randomness is separately seeded. The split dispatch (seed 0 pilot, 1–9 full) preserves the stateless per-support prediction and TabICL's explicit random_state=42. No optimizer steps on the real task.

## Observations and limits

All 12,679 released labels equal **one minus** a fresh reconstruction of the current raw RelBench 30-day `MAX(statusId != 1)` target on the same driver/cutoff keys. Preserve released orientation for this numerical replay. For the current raw DNF interpretation, both labels and probabilities must be complemented. A model probability cannot silently be called DNF risk. Raw database hash and counts are recorded. The earlier Griffin labels match current SQL and are opposite to this release.

The three available MAX historical timestamp features pass every owner-cutoff comparison. Both the task date and these float32 features use **nanoseconds**. A preliminary check incorrectly assumed seconds; corrected before dispatch. Source DFS config uses cutoff time and `include_cutoff_time=False`. This does not independently regenerate every feature, certify historical availability, or prove the exact preprocessing population used in the paper. The processed test graph appears frozen before later query dates; full raw DFS regeneration NOT_RUN. Historical identity and real-world availability remain NOT_ESTABLISHED even though all three means match the paper at four decimals.

Portable visible model matches both released checkpoint logits exactly on the checked CPU input; gradient max differences below 4e-15. Query intervention and chunk invariance pass. This is base numeric-path parity, not an audit of every optional upstream variant.

One released generator draw completed: default small sizing, seed 42, selected schema 197, 14 tables, 13 foreign-key relationships, 14 tasks. All generated PKs unique and all 16,547 FK cells resolve. This reuses the released schema pool and runs without row-GNN; LayerDAG training and the full synthetic corpus are NOT_RUN. Output archive, raw generation log and audit are in `evidence/l166/`. The tiny notebook two-table SCM is a separate course illustration.

## Run the entire selected experiment

Use a fresh environment with Python 3.11, torch 2.5.1 and `l166-requirements.txt`. The existing course environment can rescore and run teaching mechanisms. From the repository root:

```sh
python labs/_fetch_l166.py --out /tmp/l166-input
python labs/_run_l166.py --input /tmp/l166-input --out /tmp/l166-fresh-results
```

This executes all 30 evaluations with the original model implementation and preserves each query probability/support key. Destination files are immutable: choose a new output directory for another run. The fetcher verifies hashes and requires no API key. The archived `upstream/model_pretrain` code is used directly; unrelated AutoGluon/tabpfn dependencies are not needed because the runner invokes the same classifier constructors directly. No implementation parameters differ.

Author cloud execution used `modal/l166_repro.py`, pilot seed 0 then remaining seeds. The shipped ledger rejects repeated attempt names; a future paid run needs a new costed ledger. It must never erase the completed run's reservations. Re-score this delivery without cloud access:

```sh
python labs/_report_l166.py
python labs/_verify_l166.py
```

To recreate original data preparation, `_prepare_l166.py` additionally needs `/tmp/l166-data` populated with the original release files and the cached raw RelBench F1 database for the independent SQL audit. These are preparation prerequisites, not hidden dependencies of the executable inference lane. The notebook embeds complete measured prediction evidence for offline rescore, a real checkpoint for the visible model, and meaningful learner tasks; its solution does not silently rerun paid inference.

## Cost and evidence

USD 10 aggregate approved. L4, 2 physical cores, 16 GiB, base rate USD 0.00028372/sec. USD 2 reserved for build/preparation/storage/overhead. Pilot 1 failed remote import before worker code; stopped and its entire reservation retained. Pilot 2 and full 1 completed. One premature local full entrypoint failed before dispatch because its decision file was absent. Full reservations including 30-second startup allowance per dispatch + overhead: **USD 4.408783**. Known successful worker-body time 52.768803 seconds ≈ USD 0.014972; **invoice NOT_ITEMIZED**, so this is not an actual billed total. See [cost](evidence/l166/cost.json) and [budget](evidence/l166/budget.json).

Selected released-checkpoint evaluation COMPLETE; all 30 runs and 21,060 predictions independently checked. Means/SD, paired support differences and deltas: [report](evidence/l166/report.json). These are ten support draws on **one** task and one test population, not ten databases. No claim of broad superiority or pretraining replication. Learner PENDING_WRITTEN_DEFENSE. Browser/notebook/build receipts are separate; live Colab and deployment NOT_CHECKED.
