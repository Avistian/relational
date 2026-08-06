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

Comparative claims still need ≥3 datasets + seeds + rank/Friedman/CD (standard #23).
