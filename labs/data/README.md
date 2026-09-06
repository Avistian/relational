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
