# Lesson 126: beta reproduction and API-tour contract

## Scope and outcome

Primary target: Fey et al., arXiv:2312.04615v1 (2023-12-07), §4.4.1
`rel-stackex-engage`: reconstruct the complete historical data/task/evaluation
contract. **Full historical reconstruction NOT_RUN / BLOCKED_DATA.** Seven fresh
HTTP probes returned 404; the web-archive query timed out. No matching original
archive was recovered. This paper version reports no numerical predictive-results
table: numerical paper-score target NOT_APPLICABLE. Whole-paper/historical identity
NOT_ESTABLISHED. No cloud training; USD0 spent.

Pinned beta source: `0433616ee94003fb15a4ac4d633e499d0f129077` (2023-11-27,
package0.1.1). Full archive hashes and availability probes are in
`_paper_audit_l126_results.json`; source hashes/URLs in `_sources_l126.json`.
The beta db hash `dfb84faa...` differs from L125's later `deb00ccd...`.

## Protocol/deviation ledger

| Axis | Historical beta target | Executed evidence |
|---|---|---|
| Database | 2023 Stack Exchange release, seven tables | Original bytes unavailable; complete F1 tour is separate |
| Task | Existing users active through cutoff; any future post/comment/vote | 82 synthetic original-class + independent-SQL cases |
| Window | Paper two years; source exactly730 days, `(t,t+730d]` | Boundary/leap-year tests; no calendar substitution |
| Splits | validation2019-01-01, test2021-01-01, backwards730-day training grid | Original Dataset cap, grid and test-mask path exercised on fixture |
| Metric | Average precision | Visible tied-threshold kernel; 200 released-metric comparisons |
| Sentinel | Original filters −1 in posts/comments; votes drops nulls | Normalized helper drops −1 everywhere; full recovery rejects any disagreement |
| Source prose | EngageTask docstring says3 years | Executable constant730 days agrees with paper's two-year description |
| Model/trainer | Blueprint and example code; no reported score/seed ensemble | Original code preserved, not trained; stale `rtb` imports unresolved |
| Availability | Event times only; mutable/static histories missing | Same limitation disclosed; no historical availability claim |
| Historical runtime | Not pinned by paper | Current author versions in `_environment_l126.json`; binary identity not claimed |
| Modern tour | Not a beta dataset | RelBench1.1.0 F1, nine tables, all74063rows through2010-01-01 |
| Modern task | Not a beta task | driver-position, 7453train/499val/760test rows |
| Predictor | No historical numerical target | Training-label median13.333333333333334, no tuning/training randomness |
| Modern metric | Separate regression demonstration | MAE4.135571142284569val /4.444671052631579test |
| Metric compatibility | Default RMSE helper incompatible with installed sklearn | Explicit `metrics=[mae]`; MAE independently rescored |

F1 archive has97,606 raw rows; the API exposes74,063 after its test cap. The independent
SQL audit reconstructs this cap and checks every declared FK. Static rows are retained
because creation histories are absent. Source archives originate from the existing
released F1 cache used in L125; hashes are embedded and reported. This is a full
available-data API exercise, not a fitted GNN or historical beta result.

## Exact commands

From the repository root, using the existing author environment:

```bash
.venv/bin/python labs/_check_l126.py
.venv/bin/python labs/_source_check_l126.py
.venv/bin/python labs/_check_recovery_l126.py
.venv/bin/python labs/_run_l126.py
.venv/bin/python labs/_audit_l126.py
.venv/bin/python labs/_figures_l126.py
.venv/bin/python labs/_build_l126.py
.venv/bin/python labs/_execute_l126.py
.venv/bin/python labs/_delivery_l126.py
.venv/bin/python labs/_verify_l126.py
```

Optional fresh network audit (source pin is immutable, availability is time-dependent):

```bash
.venv/bin/python labs/_prepare_l126.py
```

Recovery lane, after obtaining checksum-matching `db.zip` and `engage.zip`:

```bash
.venv/bin/python labs/_recover_l126.py \
  --archive-dir /path/to/original-archives \
  --report /tmp/l126-recovered-report.json
```

Missing or mismatched archives return exit2 without launching anything. Once hashes
match, the operator loads every table, regenerates all train/validation/test labels,
compares every entity/time key and target against both original source and archived
tables, and checks the public test mask. Its logic and corrupt-label rejection were
tested on synthetic source-generated archives. **The full historical-data path is
NOT_RUN.** Hash agreement alone is not a paper-identity attestation. Any mismatch
is a failure to investigate, not a reason to overwrite the archived labels.

The standalone notebook embeds the F1 archives and all teaching functions. It requires
RelBench1.1.0; its optional Colab installer is not a claim of live Colab validation.
The original beta API is audited in a separate process to avoid importing two different
`relbench` packages into one interpreter. Reproduction scripts are meant for a checkout;
notebook teaching execution needs no checkout. No Modal runner is needed for this
CPU-only API/evaluation scope with unavailable historical data and no score table.

## Compute and delivery

Approved cap: USD10 aggregate. Conditional resource ceiling: T4 +2physicalCPU +16GiB
USD0.00022572/s, max8aggregatehours USD6.500736 plus USD3.499264 reserve.
No paid workers launched. Actual paid spendUSD0. Source/API/SQL execution is localCPU.
Notebook/browser/copied-Pages outcomes are recorded separately; live Colab and
post-deployment checks NOT_CHECKED. Learner PENDING_WRITTEN_DEFENSE.
