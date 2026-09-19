# L079 reproduction contract

[Lesson](../lessons/0079-neural-tabular-decision-guide.html) · [Notebook](0079-neural-tabular-decision-guide.ipynb) · [Read lab](html/0079-neural-tabular-decision-guide.html) · [Sources](_sources_l079.json)

## What is reproduced

L079 is the curriculum's **writing/synthesis unit**, not a new model paper. Its executable result is a complete audit/reanalysis of the corrected L060 v2 predictions: 14 dataset/regime cells × five arms × three seeds = 210 records. The compressed artifact includes every saved target, prediction, split ID, validation candidate score and selected candidate index. It is not a reduced score-only substitute. SHA256 of decompressed bytes: `99387c9ac29f941ec17ad1a9593a57c780551db6371c458db9197d5eab73f266`.

Freshly executed: hash verification; split-ID disjointness/order; aligned targets; validation-only candidate choice; recomputed binary log loss/RMSE; full declared coverage; seed-mean within-dataset ranks; all versus matched cohorts; synthetic latency selection. Author and executed solution summaries agree. Independent count-based rank oracle and corruption tests supplement source agreement. This cannot prove that historical preprocessing or feature availability was correct merely from predictions.

**Fresh training: NOT_RUN in L079. Published paper reproduction: NOT_RUN in L079.** The table does not include FT-Transformer, TabPFN, TabICL or an external ensemble. These are literature-backed candidates in the essay, not evaluated arms here. The latency fixture is synthetic. Matching underlying dataset names is not a causal intervention on time.

## Exact offline analysis replay

From the course root, with Python 3.12 and the pinned analysis dependencies:

```bash
python3 -m venv /tmp/l079-analysis
/tmp/l079-analysis/bin/pip install -r labs/requirements-l079-observed.txt
/tmp/l079-analysis/bin/python labs/_check_l079.py
/tmp/l079-analysis/bin/python labs/_verify_l079.py --output /tmp/l079-replay.json
```

The existing `.venv/bin/python` was used for author execution. The clean-environment install command is provided but not separately executed. There is no model training, network data access, checkpoint or random draw in this analysis. The frozen original model seeds are 0/1/2. Floating-point loss reconstruction tolerates `atol=1e-12, rtol=1e-10`; ranks and saved cohort identities agree exactly. The manifest pins the complete input and both analysis modules. The notebook embeds these input bytes and inlines all audit and decision functions; it can run without a repository clone. Its standalone Colab bootstrap pins only the analysis dependencies. Live Colab remains NOT_CHECKED.

Build and verification in the full course environment:

```bash
.venv/bin/python labs/_build_l079.py
.venv/bin/python labs/_execute_l079.py
.venv/bin/python labs/_integrate_l079.py
.venv/bin/python labs/_browser_l079.py
.venv/bin/python labs/_delivery_l079.py
```

Builder reruns preserve executed solution outputs only when every code cell is unchanged. Student notebooks retain two unfilled TODOs. The writing task is assessed by the rubric, not by executable checks.

## Full local training replay — separate from this unit's audit

Use the **corrected** [L060 notebook](0060-broad-model-comparison.ipynb), with its visible models and trainer, and [L060 contract](l060-reproduction.md). That contract specifies loader/data requirements, row caps, two candidates, 24 epochs/100 tree rounds, validation checkpoint selection, 14 cells and seeds 0/1/2. All preprocessing, saved IDs and protocol variants must match before comparing a new run with the historical one.

```bash
.venv/bin/python labs/_fetch_l055.py
.venv/bin/python labs/_verify_l060.py --preset lab --output /tmp/l060-l079-fresh.json
```

Choose a new output path; the training operator refuses overwrite/resume. These commands are not executed in L079. They rerun the complete **local** comparison, not a paper benchmark. Python/environment, raw data provenance and original model sources are recorded in L060's manifest and result, independently of L079's analysis environment.

## Published experiment boundaries

| Source lane | Named target / required alignment | L079 status |
|---|---|---|
| TabM and RealMLP | Original benchmark rosters, full variants, preprocessing, search budgets, training schedules and aggregation; L060 reduced arms do not match | NOT_RUN; INCOMPARABLE to L079 ranks |
| TabArena v1 | Full original 51-task protocol and ensemble/tuning budgets; see L056/L060 contracts | NOT_RUN |
| TabReD v4 Figure 2 | Eight tasks, three split pairs, 15 initializations and original four-method study; L060 uses three tasks and five arms | NOT_RUN; INCOMPARABLE |
| Historical TabPFN v2 / TabICL 2025 | Exact checkpoint, task support, context/preprocessing and complete published benchmark | Not measured by L079 |
| L078 GCN, Table 2 Cora | Existing 100-seed release-protocol port with explicit framework/seed deviations | Previously executed in L078; not re-executed or evidence for the tabular decision table |

The primary paper versions in `_sources_l079.json` are historical course sources, not a current leaderboard. No inference-only check or frozen result audit substitutes for full pretraining/benchmark reproduction. The user-requested reproduction boundary remains visible wherever numerical evidence appears.

## Delivery boundaries

Machine reports separately record local notebook execution, browser interaction/mobile layout, and copied Pages staging. Live Colab and remote publication are NOT_CHECKED. No learner completion or mastery is inferred from author verification.
