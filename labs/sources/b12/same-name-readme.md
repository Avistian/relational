# OpenRFM

**An open reproduction and extension of a relational foundation model (RFM):
pre-train *once* on synthetic relational data, then evaluate *in-context* (no
fine-tuning) on RelBench.**

OpenRFM is a fully runnable, from-scratch implementation of the core ideas in
KumoRFM-2 (a database-native relational foundation model). It implements the
synthetic structural-causal-model (SCM) task generator, a hierarchical
relational encoder (column → row → graph → cross-sample attention), and a
strictly leak-free in-context-learning (ICL) evaluation harness against the
RelBench benchmark suite.

This repository is research code. It reproduces the *methodology* faithfully and
reaches a meaningful fraction of the paper's transfer performance on consumer
hardware (2× RTX 3090 Ti), while being honest about where the gap remains. See
[Results](#results) and [Honest assessment of the gap](#honest-assessment-of-the-gap).

---

## TL;DR

| | RelBenchV1 classification (12-task avg AUROC) |
|---|---|
| **Paper target (KumoRFM-2)** | **79.60** |
| 5% tolerance band | 75.62 – 83.58 |
| **OpenRFM (this repo, leak-free ICL, no tricks)** | **~69.0** |
| Strict root-only logistic baseline | ~62–70 (task-dependent) |

The result above is obtained by pre-training **once** on synthetic SCM tasks and
evaluating **in-context with no per-task fine-tuning**, under strict temporal
splits (no data leakage) and with no benchmark-specific tuning or evaluation
tricks. The remaining gap is attributed to a specific, validated architectural
cause (see below), not to a bug in the evaluation.

---

## Why this exists

The paper's claim is a *foundation model* for relational databases: train on
synthetic + real relational data, then answer predictive queries on unseen
databases via in-context learning, with no task-specific training. Reproducing
this honestly requires getting three things right:

1. **The training regime** — pre-train on synthetic data, *not* fine-tune
   per-task. Per-task training would beat the benchmark but would not be a
   reproduction of the paper's contribution.
2. **No data leakage** — context rows and all relational neighbors must be
   filtered strictly before the prediction anchor (`< t`), and train/eval splits
   must be by unique timestamp with `train_max_time < eval_min_time`.
3. **No evaluation tricks** — report the root-only baseline alongside the model,
   run context-target ablations to confirm the model actually uses in-context
   labels, and never hand-pick task-specific structure.

OpenRFM implements all three and treats them as non-negotiable.

---

## Results

Pre-train → ICL pipeline, all numbers leak-free and trick-free:

- **Context scaling** (the validated lever): ctx24 = 65.8 → ctx48 = 68.0 →
  ctx96 = **69.0** (best) → ctx160 = 67.4 (saturates / undertrained at this
  compute budget).
- **Model scaling** d_model 256 → 384: flat (~68.0) at this data/compute scale.
- **Diversity / heterogeneity ingredients**: no broad gain on the full 12-task
  suite (several help 2-task screens but wash out at full coverage).
- **Column-permutation ensembling**: +0.2 AUROC.
- **Real-database LODO co-training**: no gain at the achievable scale.

Where OpenRFM matches or beats the paper per-task (small context / chain schema
suffices): `avito-visits` 0.71 (paper 0.69), `f1-dnf` 0.82, `event-ignore` 0.84,
`stack-engagement` 0.83. The gap concentrates on data-rich, star-schema,
domain-mismatched tasks (`trial`, `stack-badge`, `amazon-item`, `event-repeat`).

The best leak-free, no-trick checkpoint is released — see
[Pretrained weights](#pretrained-weights).

## Honest assessment of the gap

The residual ~6–7 points was traced to a definite cause, documented in
[`progress_plan/reproduction_status.md`](progress_plan/reproduction_status.md):

- **The 3-table-chain model is the hard ceiling (~68–69).** Three principled,
  non-task-specific child/aux selection strategies (row-count, connectivity
  coverage, two-sibling) all land at ~68–69 because a fixed root→child→aux chain
  holds only **one** child and **one** aux table, while different RelBench tasks
  need *different and multiple* sibling tables at once. No single selection rule
  can satisfy all tasks within three slots. (Proof point: feeding the chain model
  two strong siblings *hurts*, because it was never trained for that.)
- **The fix is the paper's actual contribution**: a *database-native* model over
  **all** FK-connected tables (N typed tables, not a fixed 3-table chain) plus a
  semantic 2-hop subgraph sampler. This is implemented here (`multitable*.py`)
  and structurally validated (it captures the full 5-table `rel-stack` star
  schema), but reaching parity requires porting the full transfer-ingredient
  suite into the N-table generator **and** substantially more pre-training
  compute than 2× consumer GPUs provide.
- **The synthetic-to-real gap is fundamental.** SCM mechanisms with IID-normal
  features only approximate real relational data patterns so far; closing the
  rest is a simulation-quality + scale problem, not a tuning problem.

This is presented as a faithful negative/partial result. The methodology is
correct; the remaining distance is compute and the N-table architecture, both
documented as the concrete next campaign.

---

## Architecture

`implementation/kumorfm_repro/`

| Module | Role |
|---|---|
| `data.py` | `RelationalTaskGenerator` — synthetic SCM tasks (24/36 mechanisms), task-conditioned inputs (visible target, query mask, lagged targets, temporal features), entity-history trajectories, feature heterogeneity, column permutation. |
| `model.py` | `RelationalFoundationModel` — hierarchical attention: `TableEncoder` (column→row), `GraphCrossSampleEncoder` (FK-graph + cross-sample), plus an improved stack (RoPE, attention pooling, typed pairwise graph attention) and optional typed task-conditioning roles. |
| `train.py` | Synthetic pre-training: DDP, AMP, grad accumulation, cosine LR, non-finite guards, periodic validation, best-checkpoint selection. |
| `eval.py` | Mechanism / robustness / latent-probe evaluation on synthetic tasks. |
| `icl_eval.py`, `icl_suite.py` | **Leak-free in-context evaluation** on RelBench (single task / full 12-task suite), with root-only baselines and context-target ablations. |
| `transfer_sweep.py` | Fixed-protocol checkpoint screening for transfer-aware model selection. |
| `relational_io.py`, `benchmark_adapter.py` | File-backed parquet/CSV RelBench adapter with strict `< anchor` temporal filtering and RelBench export. |
| `multitable*.py` | The N-table database-native model + generator + eval (the paper's core contribution; implemented, structurally validated, not yet at parity). |
| `targets.py` | Paper-target registry and tolerance-aware comparator. |

See [`CLAUDE.md`](CLAUDE.md) for the detailed module map, stability rules, and
the proven production training command, and
[`implementation/README.md`](implementation/README.md) for exhaustive CLI usage.

---

## Installation

```bash
git clone https://github.com/T-Lab/OpenRFM.git
cd OpenRFM/implementation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m compileall kumorfm_repro   # fast syntax/import check
```

Requires Python 3.10+ and PyTorch ≥ 2.5 (CUDA strongly recommended).

## Quickstart

All commands run from `implementation/`.

**1. Pre-train on synthetic data (proven stable 2-GPU config):**

```bash
torchrun --standalone --nproc_per_node=2 -m kumorfm_repro.train \
  --steps 2000 --batch-size 4 --context-size 96 \
  --rows-per-child 12 --rows-per-aux 6 \
  --d-model 256 --heads 8 --layers 3 \
  --task-type mixed --max-classes 8 \
  --entity-history-frac 0.5 --entity-drift-scale 0.3 \
  --history-label-frac 0.75 --root-shortcut-dropout 0.5 \
  --column-permutation --no-improved \
  --checkpoint-every 500 --validate-every 500 \
  --save-best-checkpoint --save-checkpoint \
  --output-dir runs/pretrain_ctx96
```

**2. Export a RelBench task:**

```bash
python -m kumorfm_repro.benchmark_adapter relbench-export \
  --dataset rel-f1 --task driver-dnf --split train --download \
  --output-dir data/relbench_exports/rel-f1_driver-dnf_train --max-rows 200000
```

**3. Evaluate in-context (no fine-tuning), leak-free:**

```bash
python -m kumorfm_repro.icl_suite \
  --checkpoint runs/pretrain_ctx96/checkpoint_step_1000.pt \
  --output-dir runs/icl_eval --data-dir data/relbench_exports \
  --context-size 96 --batch-size 32 \
  --preset relbenchv1_classification \
  --context-mode entity_mixed --local-context-frac 0.5
```

**4. (Recommended) Confirm the model genuinely uses in-context labels:**

```bash
python -m kumorfm_repro.icl_suite ... \
  --context-ablation zero_targets --context-ablation shuffle_targets
```

A large positive `ablation_*_delta` means real ICL dependence; a small delta
means the score is mostly static/root-feature driven.

---

## Pretrained weights

The best leak-free checkpoint (3-table model, ctx96, ~69.0 AUROC) is released as
a **GitHub Release asset** (it is ~224MB, too large for the source tree).

After the release is published:

```bash
# via gh CLI
gh release download v0.1.0 -R T-Lab/OpenRFM -p "openrfm-pretrain-ctx96.pt"

# or via curl
curl -L -o openrfm-pretrain-ctx96.pt \
  https://github.com/T-Lab/OpenRFM/releases/download/v0.1.0/openrfm-pretrain-ctx96.pt
```

See [`MODEL_CARD.md`](MODEL_CARD.md) for the exact training config, architecture
hyperparameters, intended use, and limitations. To publish the release, see
[`scripts/release_weights.sh`](scripts/release_weights.sh).

---

## What is *not* in this repo

To keep it lean and lawful, the `.gitignore` excludes:

- `runs/` — 35GB of training runs and checkpoints (curated weight is released separately).
- `data/` — exported RelBench datasets (regenerable with `relbench-export`).
- `technical_report/` — the source paper PDF (distributed by its original authors).

---

## Citation & acknowledgments

This is an independent reproduction of KumoRFM-2 for research and educational
purposes. Please cite the original KumoRFM technical report for the method, and
the RelBench benchmark for the evaluation data. This repository contains no
code or weights from the original authors.

## License

Code is released under the MIT License (see `LICENSE`). The source paper and the
RelBench datasets are property of their respective authors and are **not**
redistributed here.
