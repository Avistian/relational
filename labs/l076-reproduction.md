# L076 reproduction contract

## Two explicit experiments

1. **Executed teaching composition:** actual PyTorch Frame 0.3.0, two table-specific encoders, visible one-hop mean, concatenation and binary head. Tier C generated fixture, fixed cutoff10, seed76, Adam0.03, 100 full-batch updates. Full local code and trainer are inline in both notebooks. No train/test split: this is a differentiability and overfit diagnostic, not predictive evidence.
2. **Historical released replay:** RelBench v1 Table6, rel-f1 driver-dnf. Target test AUROC72.62% (five runs), primary source https://arxiv.org/html/2407.20060v1. This remains **NOT_RUN**; no claimed paper-result reproduction. The full publication-day model/trainer and supporting heterogeneous neural layers are readable inline after EXIT. They are separate from the teaching architecture.

## Source and protocol audit

Pinned RelBench commit: `5894184f3d1b2432feb9a208a8aaf18b106fbdf4`, July29 2024. Original MIT license and source hashes are in `sources/l076-relbench/` and `_sources_l076.json`. The operator checks downloaded source against these hashes. Upstream package is imported directly from the pinned checkout.

| Dimension | Teaching composition | Historical released replay |
|---|---|---|
| Data | 4 customers,5 events | rel-f1 with author's registered download hashes |
| Inputs | age/region; amount/kind | original database schema and stype proposal |
| Encoder | typed tokens→flatten→linear→tanh | per-table PyTorch Frame ResNet, width128,4 layers |
| Context | one-hop mean; fixed cutoff | heterogeneous GraphSAGE,2 layers,sum; relative time encodings |
| Initialization | random seed76 | released reset_parameters; no pretrained GNN |
| Text checkpoint | none | upstream Glove sentence-transformer, external download |
| Objective | BCE with logits | released BCE with logits for driver-dnf |
| Optimizer | Adam,0.03 | Adam,0.005 |
| Schedule | 100 full-batch steps | 10 epochs,max_steps_per_epoch2000; preserves released loop |
| Sampling | all eligible events | temporal uniform; fanouts128,64; batch512 |
| Split | no held-out split | released train/val/test task tables |
| Selection | none | largest validation roc_auc; deep-copied checkpoint |
| Metric | training loss | validation/test AUROC |
| Seeds | 76 | chosen0–4; original five IDs not located |
| Aggregation | single diagnostic | mean and sample SD over requested seeds |
| Main result | executed; see verifier | NOT_RUN |

The paper's maximum128-neighbor description is not an instruction to silently edit released `[128,64]` fanouts. Preserve source and flag the ambiguity. Release preprocessing/materialization is reproduced as written; do not replace it with the toy fixture's conservative fit policy and still claim original-protocol replay. The released graph builder uses its database snapshot's column statistics. A separate point-in-time preprocessing audit is required before generalizing the toy's stronger fit-invariance guarantee to the release.

Remaining gaps: the authors did not provide a full dependency lock or exact five seed IDs in the inspected sources. Our Python3.10 and package versions are reconstruction pins; torch2.3.0 and pyg-lib0.4.0 follow the repository's publication-era CPU CI. PyTorch Frame0.2.3 follows the then-required minimum. The text model is not immutable in the upstream helper; archive and hash its resolved checkpoint files before asserting identical initialization. The runner archives data/materialization file hashes, environment, raw logs, commands and final predictions. It does not claim exact determinism across CPU/GPU libraries or sampling backends.

## Observed runtime boundary

`_replay_l076_preflight.json` is the actual local attempt: ARM64, Python3.12, torch2.13 CPU, Frame0.3.0 and PyG2.8.0.post1; pyg-lib and sentence-transformers missing. Preflight blocks training rather than silently changing historical code. A real pip download probe against the torch2.3.0 CPU wheel index failed with “No matching distribution found for pyg-lib==0.4.0”; the inspected index contains no aarch64 wheel for that release. Building a compatible backend from source was not attempted. The isolated environment installation and actual cloud training are **NOT_CHECKED / NOT_RUN**, not validated operators. No paid cloud job has been submitted.

## Run the teaching experiment

From repository root:

```bash
.venv/bin/python labs/_verify_l076.py
.venv/bin/python labs/_build_l076.py
.venv/bin/python labs/_execute_l076.py
.venv/bin/python labs/_browser_l076.py
.venv/bin/python labs/_delivery_l076.py
```

The canonical solution is `labs/solutions/0076-encoder-predictor-stack.ipynb`. The student has three live TODO functions, not a hidden imported model. Browser, copied Pages delivery and notebook execution have separate reports. Live Colab/deployment are NOT_CHECKED.

## Run the historical release on a supported x86-64 host

Use a separate Python3.10 environment; do not downgrade the course environment. CPU commands:

```bash
python3.10 -m venv /tmp/l076-paper-env
/tmp/l076-paper-env/bin/pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
/tmp/l076-paper-env/bin/pip install --no-index pyg-lib==0.4.0 -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
/tmp/l076-paper-env/bin/pip install -r labs/requirements-l076-replay.txt
/tmp/l076-paper-env/bin/python labs/_replay_l076.py --preflight --output /tmp/l076-paper-results
/tmp/l076-paper-env/bin/python labs/_replay_l076.py --run --seeds 0 1 2 3 4 --output /tmp/l076-paper-results
```

For a CUDA12.1/T4 environment, replace the two CPU wheel indexes above with `cu121` and `torch-2.3.0+cu121.html`. The notebook's gated Colab cell creates an isolated Python3.10 interpreter using uv and invokes this operator. These commands are provided but not installation-tested here. Download sizes, memory and run duration were not measured. A successful preflight only checks the listed versions/backend; it does not certify every dependency or data download.

For unattended execution:

```bash
modal run --detach modal/l076_paper_repro.py
modal volume get relational-l076-replay / ./l076-paper-results
```

The Modal job retains evidence even if training fails. Cloud image build and account access are NOT_CHECKED. Inspect per-seed logs, missing seeds, checkpoint provenance, task-table hashes and metrics before comparing with72.62%. A single seed is not a five-run reproduction. Do not tune on test AUROC or rerun selectively until the number looks good. The operator keeps comparison status `INCOMPARABLE_PENDING_PROTOCOL_AUDIT` after execution so a completed process cannot automatically become a paper-parity claim.

## Exit ledger

- Verified here: precise key routing, temporal fit/output invariance, mean forward/backward oracle, PyG primitive parity, identity permutations, gradient support and executable toy trainer.
- Paper claim: relational stack evaluated on real RelBench tasks; named numerical target above.
- Full release run: NOT_RUN. Runtime blocker and remaining source/protocol gaps are explicit.
