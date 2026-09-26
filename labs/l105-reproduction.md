# L105 reproduction contract: continuous time

## What is fully reproduced

All 157,474 interactions in the pinned Wikipedia release, all three predetermined window widths, the event projection and every weighted snapshot edge, independent group/pair/access checks, and the complete standalone solution notebook. This is a course-defined representation audit. It has no published predictive-score target. TGN/JODIE full-paper reproduction is NOT_APPLICABLE to this lesson; their model scores are not rerun or inherited as fresh evidence.

The declared protocol predates implementation in `docs/plans/2026-09-26-lesson-105-design.md`. Nothing is selected on outcomes. There is no training, optimizer, model selection, negative sampling, train/test split, GPU, seed variation, or paid computation. All results are deterministic census statistics of the fixed input under the declared transformations. Do not attach model-confidence intervals to them.

## Exact commands

From the repository root (Python 3.12.3 was used):

```bash
python3.12 -m venv /tmp/l105-replay
/tmp/l105-replay/bin/python -m pip install -r labs/requirements-l105-authoring.txt
/tmp/l105-replay/bin/python labs/_check_l105.py
/tmp/l105-replay/bin/python labs/_run_l105.py --raw labs/data/l102/wikipedia.csv
/tmp/l105-replay/bin/python labs/_provenance_l105.py
/tmp/l105-replay/bin/python labs/_mutation_l105.py
/tmp/l105-replay/bin/python labs/_figures_l105.py
/tmp/l105-replay/bin/python labs/_build_l105.py
/tmp/l105-replay/bin/python -m ipykernel install --prefix /tmp/l105-replay --name python3
/tmp/l105-replay/bin/python labs/_execute_l105.py
/tmp/l105-replay/bin/python -m playwright install chromium
/tmp/l105-replay/bin/python labs/_delivery_l105.py
```

The runtime-only replay needs only `requirements-l105-runtime.txt` and `_run_l105.py` plus `relkit/stream_l105.py`. The runner defaults to `labs/data/l102/wikipedia.csv`; if missing it downloads 559,937,473 bytes from `https://snap.stanford.edu/jodie/wikipedia.csv`. Raw bytes must hash to:

`a6b73e09c0d1e5b9db11e7e7aa416f2e87838a745273e0446a79952cf4cfae09`

Use `.venv/bin/python` for the existing workspace environment. The notebook is standalone and downloads/authenticates this file itself when `l105-data/wikipedia.csv` is absent. No unpublished course URL or `relkit` import is required to execute it. The delivered local execution reused authenticated raw bytes rather than downloading them anew. Browser runtime dependencies may need Playwright's documented OS setup. See actual runtime and execution records, rather than inferring compatibility with every Python/OS combination.

## Representation contract

- Raw record projection: user ID, item/page ID, event time, raw zero-based row identity. Availability is set equal to event time for this audit. Raw labels and edge features are excluded, not silently aggregated. Typed identity is preserved by separate user and page columns.
- Source cross-check: TGN commit `e38cdf85998c6ca077167610dc4e769a688efa95`, original `reindex` function, all 157,474 records. Our typed columns map bijectively to its user+1 / item+8228 global IDs. This is only preprocessing identity, not model parity.
- Fixed widths: 3,600, 86,400, 604,800 seconds; origin 0. Bins are `[start,end)`. Daily/weekly refer to elapsed-time bins, not a calendar/timezone policy. The observed range is 0 through 2,678,373 seconds. Zero-event windows would count as empty; none occur at these widths.
- Snapshots are **window interaction graphs**, not cumulative relationship-state graphs. Group by `(bin,user,page)`. Binary mode sets every grouped edge to 1; weighted mode keeps the count. Raw event IDs, exact times, and per-event availability remain in `events.npz`.
- Snapshot artifacts also include first/last times and release times for audit. First/last are NOT features supplied to the count-only representation studied here; retaining them would define a different information contract.
- Snapshot release is max(window end, latest availability over the WHOLE window). Under a=t, this is window end. Production completeness needs a policy; a finite-file oracle cannot certify unobserved future arrivals.
- Query population: every event time, retaining duplicates. Legal event history is `t < query` and `a <= query`. Read only snapshots with `release <= query`. The final observation window retains its scheduled end; it is never declared complete early at the last file row.
- Hidden strict pairs: sum C(window event count,2) minus sum C(timestamp tie count,2). Global pairs may involve unrelated endpoints; this is not the count of lost causal paths.
- Withheld past: strict global history minus records in completed windows, per query. Nonpast exposure: records at/after the query in the retrospectively completed current window. These global quantities are not sampled neural dependencies or empirical AP changes. Summed withheld = hidden strict pairs; exposure minus withheld = N + 2 × tied pairs.
- Delay: scheduled window end minus event time under a=t. It is not observed ingestion latency or model runtime. Last-window close can be later than the observation horizon.

## Protocol/deviation ledger

| Dimension | Executed | Boundary |
|---|---|---|
| Data | Every raw record hashed; endpoint/time projection checked by two parsers | Historical byte identity with the publication is not established |
| Preprocessing | Typed-ID bijection matches pinned original TGN reindex | Only reindex, not upstream model/trainer |
| Aggregation | All rows, all widths; SQLite counts/min/max agree | Count/binary window graphs; no feature aggregation |
| Order | All strictly ordered within-window global pairs | Does not recover order within tied timestamps |
| Availability | a=t for measured data; delayed-arrival adversarial fixtures | Real ingestion history is unavailable |
| Query access | Every timestamp; independent release and pair counts | Global record exposure, not actual GNN neighborhoods |
| Paper scores | No model benchmark target applies | No new TGN/JODIE score reproduction |
| Notebook | Full fresh raw-data computation, all checks, clean workdir | Prepared material is not learner mastery |
| Delivery | See `_delivery_l105_results.json` | Live Colab and deployment NOT_CHECKED |

## Artifacts and verification

`_analysis_l105_results.json` stores all results, data/runtime identities, code hashes and compressed-artifact hashes. `evidence/l105/events.npz` contains the complete studied projection. Three `snapshots-*.npz` files hold every grouped edge and audit metadata. The runner recalculates from raw bytes; it does not resume or trust a processed cache. A raw identity mismatch fails before computation. `_provenance_l105.py` verifies the original preprocessing source hash before executing its isolated reindex function.

`_mutation_l105.py` proves the notebook checks reject wrong bin boundaries, binary count loss, ignored late arrival, and a strict release boundary. `_check_l105.py` covers 100 generated streams plus exact bin boundaries, ties, delayed arrival across different edges in one window, empty input, typed IDs, invalid inputs, partial final windows, and an order-collision witness. `_run_l105.py` uses SQLite aggregation and an independent timestamp walk. `_execute_l105.py` executes the entire visible solution. `_delivery_l105.py` verifies artifact hashes, notebook seams, browser controls and copied Pages assets.

Primary sources: [TGN v3 §2](https://arxiv.org/html/2006.10637v3#S2), [JODIE authors' data](https://snap.stanford.edu/jodie/), [JODIE format](https://github.com/claws-lab/jodie#dataset-format), [pinned TGN preprocessing](https://github.com/twitter-research/tgn/blob/e38cdf85998c6ca077167610dc4e769a688efa95/utils/preprocess_data.py). Cite Kumar, Zhang & Leskovec (KDD 2019) for the data. See `_sources_l105.json` for provenance and deviations.
