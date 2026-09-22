# Lab datasets

Approved datasets for labs, by tier (see `.agents/skills/lab-authoring/SKILL.md`).

| Tier | When | Source |
|------|------|--------|
| **A — Real, small, open** | Default Q2+ training/eval | OpenML, UCI |
| **B — Real, relational** | Y3+ RelBench preview | RelBench / PyTorch Frame |
| **C — Synthetic** | Mechanism isolation only | Generated in-notebook |

## Tier A — pinned for Year 1 Q2

| Key | OpenML id | Use | License |
|-----|-----------|-----|---------|
| `credit_g` | 31 | Imbalanced binary classification (German credit) | OpenML |
| `adult` | 1590 | Mixed types, prevalence ~24% | OpenML |
| `bank_marketing` | 1461 | Imbalanced marketing response | OpenML |

Later labs also register `diabetes`, `blood_transfusion`, `kc1`, `phoneme`, `churn` via `relkit.data.SPECS`.

### Paper-results scale-up table

| Key | OpenML id | Use | License |
|-----|-----------|-----|---------|
| `higgs_small` | 23512 | ~98k of UCI Higgs (NODE/TabNet paper tables). **Not** the paper's 10.5M — ledger must say INCOMPARABLE on the absolute number. OpenML leaves **1 incomplete row** (NaN jet/mass features); L044's paper-repro drops it before scaling or sklearn raises `Input contains NaN.` | OpenML |

`higgs_small` is for the **paper-results scale-up** (standard #25), not the CPU learning lab. Fetch on
demand via `relkit.data.load_tier_a("higgs_small")`; do not add it to the default `fetch_datasets.py`
list (it is much larger than the Q2 cache).

## Fetch

From repo root:

```bash
source .venv/bin/activate
python labs/data/fetch_datasets.py
```

Caches parquet files under `labs/data/cache/` (gitignored).

Each lab documents which tier and dataset key it uses in the intro markdown.

## Paper-mirror datasets (standard #24)

When the lesson mirrors a paper, **prefer that paper's own datasets, splits, and preprocessing** if
they are open and affordable. If you substitute (Tier A OpenML stand-in, subsample, synthetic for
mechanism isolation):

- Name the paper datasets you are *not* running.
- State which paper claim therefore cannot be reproduced here.
- Do not present the substitute run as a full reproduction without that gap.

## Paper-results scale-up (standard #25)

A downscaled learning lab is a *different experiment* from the paper's table. After EXIT, paper-mirror
labs try a closer run (same from-scratch code) and print a ledger with three buckets — verified here /
paper claim / scale-up. Until that job has been run, the paper number stays **cited, not reproduced**.

Comparative claims still need ≥3 datasets + seeds + rank/Friedman/CD (standard #23).


## L048 — DCNv2

Local Tier-A substitutes: `credit_g`, `diabetes`, `blood_transfusion`, fixed stratified split seed 5 (65/15/20), model seeds 0/1/2. All vocabularies and numeric scaling fit training rows. These tables do not reproduce the paper's recommendation benchmark.

Paper-dataset track: GroupLens MovieLens 1M, `relkit/dcnv2_data.py`; archive MD5 `c4d9eecfca2ab87c1945afe126590906`. Remove 3-star ratings, map 1/2→0 and 4/5→1 (739,012 rows). Six single-valued categorical fields: user, movie, gender, age, occupation, ZIP. Random 80/10/10 split seed 48; original rating row IDs retained. Genres omitted; targets/timestamps excluded from inputs. Field mapping is our explicit interpretation of the paper's six fields. Actual closer run saved under `cache/l048-closer`; raw archive/checkpoints/predictions stay gitignored. The raw-rating TFRS tutorial is a different task. Criteo/private production data remain outside the run.


## Lesson 051 — authors' processed numerical suite

Tier A: OpenML 44120 electricity (38474 × 7), 44125 MagicTelescope (13376 × 10), 44126 bank-marketing (10578 × 7), excluding target columns from feature counts. The January 2023 release belongs to suite 337; it is later than the cited 2022 paper v1. `_fetch_l051.py` caches exact IDs under `cache/l051/`; `_data_l051.json` records file hashes. Label-blind subsampling and stratified 60/20/20 seed-51 partitions are local choices. The complete original suite, upstream split IDs and tuning protocol are not reconstructed. These datasets replace generic small-table substitutes for the L051 intervention study; they do not make its fixed-budget results paper-comparable.

### L055 — TabReD released temporal tasks (Tier A)

`_fetch_l055.py` downloads and verifies Ecom Offers, Homesite Insurance and Sberbank Housing against the official pinned registry. About 60 MB compressed; cache is gitignored. Local row caps1500/600/600 retain released random-0/sliding-window-0 membership, with numeric/binary features only. Shared boundary timestamps and omitted categories are disclosed. Five other benchmark tasks are skipped; this cannot reproduce the full benchmark. See `../l055-reproduction.md` and `../_data_l055.json`.

## L073
Tier A: offline sklearn Wine, Breast Cancer Wisconsin Diagnostic, and Digits. Digits is flattened image data. Full built-in datasets, stratified 70/10/20 splits; label-blind nested training subsets. Data bytes and row IDs recorded in `_verify_l073_results.json`. No original SCARF benchmark table reproduced.

## L074 CARTE

Tier A: three 384-row subsets of released wine tables. Fixed real FastText sentence-vector cache and YAGO checkpoint accompany source IDs and SHA256 digests in `l074/manifest.json`. No full string-model download is needed for the lab; see `../l074-reproduction.md`.

## L075 PyTorch Frame

Tier C: four training/two query rows generated visibly in `relkit/frame_l075.py`, isolating five semantic types. Tier A: existing OpenML31 credit_g cache, actual numeric/categorical columns, seed75 fixed 128/32 rows, labels unused. Cache hash and IDs are recorded in `_verify_l075_results.json`; no new download required when cached.

## L076 two-table fixture

TierC generated by `relkit/stack_l076.py`: four customers, five events, noncontiguous keys and two excluded records. Fixed cutoff10; arbitrary diagnostic labels; no benchmark accuracy claim. Optional release replay uses original rel-f1 task data with upstream hashes, no substituted toy dataset. Dataset download/training NOT_RUN locally.

## L077 paired histories

Tier C, generated entirely in `relkit/ceiling_l077.py`. Full experiment uses 1000 pairs/seed and seeds0–4; 2000 customers and10000 event rows per seed. Pair-group 60/20/20 splits, cutoff10 with event/availability masks. Target is a known eligible historical pattern. Synthetic data is intentional for an exact impossibility construction; no external dataset substitute is presented as a paper reproduction. Source and per-seed dataset hashes are saved.

## L078 Cora

Full raw released Cora files from tkipf/gcn revision39a4089fe72ad9f055ed6fdb9746abdcfebc4d81; `_sources_l078.json` pins every file hash. Load only after verification. Fixed140/500/1000 train/validation/test label split, all2708 nodes visible under the transductive protocol. Separate four-node synthetic fixture exists only for manual arithmetic.

## L079 decision evidence

`l079/l060-v2.json.gz` is a deterministic gzip copy of the complete corrected `_verify_l060_v2_results.json`, including all 210 prediction records. Uncompressed hash and source-course revision are pinned in `_sources_l079.json`; no sampling or rounding was introduced. Reanalysis only, not new training.

## L080 raw subsets

`l080/{smoke,exam}.npz` stores complete float32 numeric/binary raw subsets and int64 targets from hash-pinned TabReD Ecom Offers/Homesite release archives. Matching JSON contains exact original IDs, full split boundaries, omitted categorical counts and archive/NPZ hashes. No preprocessing is fitted before packaging. `_prepare_l080.py` rebuilds the extraction.

## L081 QM9

Tier C toy graph for routing; Tier B real molecular graphs for named-target reconstruction. Hash-pinned QM9 SDF/CSV and public uncharacterized list download into ignored data/cache/l081. RDKit sanitization failures and molecule IDs are retained. Smoke takes first512 valid molecules, then seeded splits; the file-order cap is not representative. Full historical population/IDs unavailable.

## L083 PPI

Tier B, original complete Stanford GraphSAGE PPI archive. Download on demand into data/l083/ppi.zip; SHA-256 checked before reading.56,944 nodes,50 features,121 labels; split44,906/6,514/5,524. Standardization fits training nodes only. Original graph split, full ten-epoch schedule; see l083-reproduction.md for historical gaps.

## L084 Cora attention

Tier B: reuses pinned Planetoid Cora bytes in data/l078, independently checked against all eight files in the original GAT release. Duplicate adjacency entries coalesce to5278 unique undirected citation pairs; self-loops yield13264 directed messages. Fixed140/500/1000 label splits. The complete graph/features are visible under the transductive protocol. See ../l084-reproduction.md.

## L085 · over-smoothing

Karate graph: full34 nodes and78 binary undirected edges, identity features; serialized in `../sources/l085/karate.json`, pinned in `../_sources_l085.json`. Li et al. Figure2 setup, untrained, labels only color points. Cora: reuse L078/L082 full pinned data and fixed label masks for a separate depth extension; not Li et al. classification-table protocol.

### L090: Cora checkpoint
Tier B, full real citation graph; shared hash-pinned release data with L078/L082. Full fixed-split100-initialization GCN reconstruction and separate inductive context extension. See [contract](../l090-reproduction.md).

## L093 · OAG NN and CS

Tier B, real heterogeneous academic graphs from the authors. Both archives acquired locally; excluded from git/Pages. `_fetch_l093.py` verifies pinned bytes. NN is the teaching subset, not a paper Table2 dataset. CS full-target loading/training has explicit resource blockers. See `../l093-reproduction.md`.

## L094 · Reuse complete NN, synthetic routes separately

Tier B: the entire hash-pinned L093 NN archive for Table1 statistics. Tier C: four-author matrices for route interventions only. No sampled graph is labeled as the complete NN target.

## L095 · Full MovieLens 100K

Tier B relational interaction data: 100,000 ratings, 943 users and 1,682 items. Downloader verifies the GroupLens archive and all official split members. Data are downloaded directly, not redistributed. Tier C two-user synthetic graph is used only for mechanism checks. Full five-fold ranking is a course protocol, not a claimed published-model result.

## L096 · Complete synthetic relational fixtures

Tier C: all input generation is visible in `relkit/schema_l096.py`; no download. Worked example has 13 rows across four tables, plus 32 declared generated databases. Per-input hashes and full query records are in `_experiment_l096_results.json`. No public benchmark score is claimed.

## L098 · three-type synthetic batching fixture

Tier C data, intentionally synthetic as specified by the roadmap: 24 customers, 144 timed orders, 12 products, four directed FK/reverse stores. `relkit/batching_l098.py:make_graph` generates all arrays from the declared seed. There is no real dataset or published-score mapping. Run the full protocol in `../l098-reproduction.md`; do not interpret the four-customer test metric as benchmark evidence.

## L102 · complete JODIE Wikipedia event stream

Tier B: 157,474 user–page edit events, 9,227 distinct nodes and 172 event features. The loader downloads the full raw CSV, verifies SHA-256 and preserves input order. User/page ID spaces are offset; zero is padding. Paper/release mapping: Rossi et al. v3 Table 2, TGN-attn Wikipedia all-event and new-node AP. The 600-event/two-epoch teaching run is a separately labeled subset, not paper-result evidence. The scalar timeline widget is Tier C mechanism illustration. See `../l102-reproduction.md`.
