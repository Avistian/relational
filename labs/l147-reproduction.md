# Lesson 147 — survey synthesis and evidence reanalysis

Approved 2026-09-30. Source snapshot: Dwivedi et al., arXiv2506.16654v1, 19June2025. The survey reviews methods and opportunities; there is no new survey training table to reproduce. This package operationalizes an open-problems map and independently reanalyses saved author evidence. It is not a 2026 frontier census.

## Executed scope

Fresh CPU audit of REUSED L146 predictions: two reduced course arms × three seeds × (499validation +760test) =7,554held-out predictions. Check unique complete (entity,cutoff) key sets, recorded target equality to saved prepared targets, one-dimensional finite predictions, and score invariance under row shuffling. Compute MAE independently, pair GNN−RelGT by seed, report mean and sample SD. Reconcile all per-split arm aggregates and paired differences with L146's summary to absolute1e-12. Raw database label reconstruction, checkpoint inference and training are not repeated. Matching stored targets cannot independently prove raw-label correctness. Hashes identify the reused artifacts, not historical paper identity.

Validation paired difference +.149340±.125959 MAE; test −.323904±.253784 (sample seed SD, not confidence intervals). This split reversal is conditional on one task and these model variants. It does not identify a causal architecture effect or demonstrate cross-database transfer. Prior test access makes research questions exploratory.

The embedded portable `audit-inputs.npz` contains only saved query keys, targets and predictions; source preparation caches are referenced under their own SHA256. All six fits remain REUSED, never called freshly trained. NumPy1.26.4 is the portable target; author environment version is recorded in sources.json. No new dependencies beyond existing course NumPy/notebook/plot/browser tools.

## Research map protocol

Five question cards carry source, mechanism, gap, proposed test, fixed quantities, falsifier, effort/cost/readiness and provenance. Effort values are author planning estimates, not worker timings. Rank individually feasible candidates by evidence level descending, hours ascending, ID ascending. Evidence levels0–3 are an explicit ordinal course rubric, not survey measurements or probabilities. No portfolio total is implied. Baseline4hours/$10: selection,ownership,routes. At1hour: selection. At8hours: add coverage. Transfer remains unready. Its40hours/$10 entry is a placeholder, not a training quote.

Three live learner functions are visible in `relkit/survey_l147.py` and inlined in solution cells. Student cells remain blank. Checks exercise real key permutations, wrong labels, missing/extra/duplicate keys, nonfinite values, unmatched seed sets and planning-boundary cases. Final report status does not certify the written EXIT defense. Learner PENDING_WRITTEN_DEFENSE.

## Named full reproduction retained

RelGT v1 Table1 rel-f1/driver-position, published target3.9170test MAE. Source revision19e423ca3e7cac761130aba790857f2dc3a46ef7. Full model `relkit/relgt_l145.py`, full data/preparation/trainer `_full_l145.py`, pinned source `sources/l145/`, complete protocol `l145-reproduction.md`. Canonical RDL comparator/source and five-seed recipe remain in `l117-reproduction.md` and `_run_l117.py`; not freshly run here.

Nine configurations: depths1/4/8 × dropout.3/.4/.5, seed0,100epochs each, width512, K300,4096centroids, batch256, Adam1e-4, weightdecay1e-5, gradientclip1, L1, train2/98percentile clipping. Last tied within-fit validation minimum. Historical cross-configuration selection NOT_ESTABLISHED. All architecture, objective, preprocessing, split, initialization, versions, selection and deviations are documented in L145; the L147 audit cannot resolve them.

Full selected reproduction INCOMPLETE. Nine complete fits NOT_RUN. Source temporal audit FAIL. Whole paper NOT_RUN. Historical identity NOT_ESTABLISHED. Inherited pilot projects ~USD80.41 at shallow speed before overhead; deeper runtime unmeasured. No new pilot, price quotation or billing claim. The current clean guard must not bypass these blockers or silently replace source contexts with corrected course contexts.

Safe preflight from repo root:

```bash
.venv/bin/python labs/_reproduce_l146.py --audit
```

Retained full command (currently stops at its guard, not invoked for L147):

```bash
.venv/bin/modal run modal/l145_repro.py::full
```

## Local reproduction commands

```bash
.venv/bin/python labs/_check_l147.py
.venv/bin/python labs/_verify_l147.py
.venv/bin/python labs/_figures_l147.py
.venv/bin/python labs/_build_l147.py
.venv/bin/python labs/_execute_l147.py
.venv/bin/python labs/_delivery_l147.py
.venv/bin/python labs/_check_pages_checkout.py
```

Standalone notebook runs in an empty temporary directory with embedded evidence and PNG diagrams. Rebuilding local raw-input reanalysis requires the hash-identified L146 evidence available in this repository. Builder retains executed solution outputs only when cell sources match exactly.

## Cost and delivery

Planned and actual L147 cloud spend USD0; local CPU only. Standing USD10 aggregate cap unchanged; no cloud launches/retries/paid validation. Local electricity/time not priced. Browser and packaging results are recorded by `_delivery_l147_results.json`; standalone execution by `_execution_l147_results.json`. Live Colab and deployment NOT_CHECKED. No publication or learner mastery claim.
