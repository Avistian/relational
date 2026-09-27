# Lesson 129 · manual feature-engineering reproduction contract

## Target and status

COMPLETE_RELEASED_PIPELINE_REPLAY: F1 driver-position manual features, released user-study SQL, complete 7453/499/760 queries, PyTorch Frame0.2.2 preprocessing and LightGBM4.3.0, five independent ten-trial searches (seeds0–4),2000-tree cap,50-round early stopping, train-only refit of selected configuration. All6295 final predictions independently replayed as sums of saved tree leaves and scored in SQLite. Source-parity pilot uses unmodified pinned original Frame trainer with seeded TPE and four threads; exact outputs.

This targets §6/Figure3 of https://arxiv.org/html/2407.20060v1, NOT Table7's raw-entity LightGBM. Figure3 uses normalized regression bars; exact historical task scalar/seeds/models unavailable. Historical paper-score parity NOT_ESTABLISHED; full human study NOT_RUN; other tasks NOT_RUN. No invented numeric tolerance or exact historical target. New learner mastery PENDING_WRITTEN_DEFENSE.

## Provenance

User study commit445bb7a3b1230f49f8e5890ae81754d3e365680f. Original SQL, notebook, schema and trainer in sources/l129; source hashes/URLs in manifest.json and _sources_l129.json. PyTorch Frame0.2.2 at56f687ddf4bf1c4a7d7b72ab0ef4117493256199; MIT license preserved. The user-study repository does not include a license file in its pinned tree; its public release is attributed, no new license is asserted for those files.

Raw database SHA256 ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482.
Task archive SHA256775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e.
Both archives are included. Source loader filters dated tables through2010-01-01; actual table and query identities are recorded in evidence/l129/preparation.json. Historical relbench0.2.0 staging URL returns404. Using released v1 archives is a disclosed data adapter, not proof of historical identity. All50 SQL features and train/validation labels match independent Python calculation; every target also matches raw future positionOrder events.

## Exact execution

From repository root. Python3.11.14/aarch64 used for author training. All package versions in requirements-l129-runtime.txt; core versions match released requirements. Secondary dependencies and OS are contemporary, not historical identity. No GPU or cloud used.

```bash
uv venv /tmp/l129-runtime --python 3.11
uv pip sync --python /tmp/l129-runtime/bin/python labs/requirements-l129-runtime.txt
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /tmp/l129-runtime/bin/python labs/_prepare_l129.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /tmp/l129-runtime/bin/python labs/_order_check_l129.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /tmp/l129-runtime/bin/python labs/_audit_l129.py
# Fresh pilot directory is required; never overwrite completed evidence.
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 timeout 600 /tmp/l129-runtime/bin/python labs/_run_l129.py --seed 129 --trials 2 --output labs/evidence/l129/pilot
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /tmp/l129-runtime/bin/python labs/_source_check_l129.py
for seed in 0 1 2 3 4; do
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 timeout 600 /tmp/l129-runtime/bin/python labs/_run_l129.py --seed "$seed" --output "labs/evidence/l129/paper/seed-$seed"
done
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /tmp/l129-runtime/bin/python labs/_analyze_l129.py
.venv/bin/python labs/_check_l129.py
.venv/bin/python labs/_mutation_l129.py
.venv/bin/python labs/_figures_l129.py
.venv/bin/python labs/_build_l129.py
.venv/bin/python labs/_execute_l129.py
.venv/bin/python labs/_delivery_l129.py
```

These are the original author commands, whose output directories now exist. For new runs use new --output directories and point the analysis to those artifacts (or use a clean checkout before restoring evidence). The operator refuses existing directories and does not resume or overwrite. Preparation regenerates matrices; analysis fingerprints inputs and source files to reject stale evidence.

## Protocol and deviation ledger

| Item | Released behavior / replay choice | Consequence |
|---|---|---|
| SQL | Original template, full queries, strict ASOF past standing | Matches every independently computed SQL value |
| Race slots | IDs relative to last global race; two-calendar-month lookback | Not driver's last three starts; preserved |
| Schedule | Next three IDs, up to one calendar month ahead | Publication/arrival histories unknown; not certified point-in-time |
| Snapshot | Dated database rows through2010-01-01 for all splits | No recent race/upcoming slots on any test query |
| Static fields | Released final attributes | Creation/mutation history absent |
| Cohort | Future participation required by task | Not a deployed past-only eligible-driver universe |
| Features | 50 engineered + numeric driverId; timestamp ignored | 51 model columns, not50; source's unused drop_cols preserved |
| Mapping | Categories fitted on train; numerical NaNs retained | Validation/test do not fit preprocessing |
| Tuning |10 trials per run;L1 objective;2000 rounds;patience50 | No reduced-trial run presented as full |
| Row order | Freeze SQL output to task archive query order; source SQL had no ORDER BY | Necessary for repeatable seeded row sampling; independently checked with1/4 SQL threads |
| Seeds | TPE seeds0–4; original default unseeded | Repeatable new searches, not historical draw recovery |
| Threads |4 CPU threads | Runtime adapter, exact pilot parity under matched threads |
| Test | Read only after selecting/refitting each run | No feature or tuning choice uses test |
| Refit | Training rows only, validation early stopping | No train+validation refit |
| L127 comparator | Existing basic RDL evidence with identical keys/labels | Separate preprocessing; not necessarily Figure3 boosted-RDL comparator |
| Human work | New learner log supplied; original data scientist work not repeated | Machine seconds cannot reproduce96% savings |

## Cost and verification

USD10 aggregate ceiling honored; paid cloud spendUSD0. Local pilot2 trials, five full searches50 trials plus five refits, source-parity pilot3 fits, audits and notebook course tree. An initial full notebook replay exposed unstable SQL row order. Its initial50-trial run and first50-trial notebook verification are retained as diagnostic work, not mixed into the final aggregate. All final author searches and the full notebook were rerun after freezing query order. A second full50-trial notebook verification is separately recorded. No automatic retries. Each author search constrained by600-second timeout; all final searches finished within11seconds. Exact measured totals in summary.json; rates for unused cloud contingency were verified2026-09-27 at https://modal.com/pricing. The one-hour aggregate training cutoff was not approached.

Source hashes, individual search traces/models/predictions and complete matrices are included. Author/reference execution and browser checks are separate reports. Default notebook executes a small course tree and replays author evidence; full50-trial replay is an explicit gate requiring pinned runtime. Live Colab and deployment NOT_CHECKED. No publication requested.
