# L096 reproduction contract · SQL FK semantics

## Target and scope

The Year 3 roadmap explicitly assigns a **bridge unit**, not a model paper. The complete target is the course's four-table schema and declared query suite: all 13 worked-example rows and 32 generated databases (seeds 0–31). All rows in every database are processed. `relkit/schema_l096.py` contains the complete schema, inputs, construction, graph query, independent SQLite oracle, generator and runner. No external data, learned model, optimizer, train/test split or score tolerance is involved.

Full paper reproduction: **NOT_APPLICABLE_NO_MODEL_PAPER**. Historical predictive-score parity: **NOT_ESTABLISHED**. A fully executed semantic experiment does not reproduce Fey et al.'s predictive results. Generalization to arbitrary schemas or production data is not established.

## Exact protocol

| Field | Contract |
|---|---|
| Input | Four explicitly declared tables; integer PKs; line_item composite (order_id,line_no) |
| Node order | Lexicographic complete PK tuples, independently within each type |
| FK roles | orders buyer/referrer; line_item order/product |
| NULL | Optional FK omitted; required FK rejected; composite nullable uses MATCH SIMPLE |
| Invalid keys | Duplicate/noninteger/NULL PK and non-null orphan FK rejected |
| Reverse | Exact swapped forward edge tensor, separate role |
| Isolated rows | Explicit num_nodes; no drop by inferred edge count |
| Attributes | Positive integer line quantities; original keys audit metadata only |
| Query | Every line→order→buyer and line→product path; grouped SUM(quantity); all-customer order counts |
| Oracle | Fresh SQLite connection, FK enforcement enabled before DDL/inserts, native joins and aggregations |
| Equality | Exact Python integer tuples; no floating-point tolerance |
| Generated extension | Seeds 0–31; 7 customers, 6 products, 12 orders each; 0–5 lines/order; quantities 1–9 |
| Hashes | Each complete input plus source and results; JSON serialization defined in source |
| Resume | None; every command recomputes every declared database |
| Selection | No tuning, model selection or seed filtering |

There are 13 worked-example nodes, four forward roles with 12 edges total and four derived reverse stores with 12 edges. Full SQL result: (10,10,5), (10,50,1), (30,50,4). Customer-order counts: (10,2), (30,1), (90,0). Full path output has four rows, including two distinct line identities reaching customer 10/product 10.

## Evidence files

- `_experiment_l096_results.json`: all 33 full query outputs, edge/node counts, per-input SHA-256, environment and implementation hash.
- `_verify_l096_results.json`: independent oracle and adversarial checks, including 20 row permutations, orphan/NULL distinction, duplicate PK, composite optional FK, empty edge store, reverse content and intervention.
- `_execution_l096_results.json`: entire solution executed in a temporary directory without repository imports; exact fresh-record equality.
- `_portable_l096_results.json`: same inline code executed under a second pinned Python environment.
- `_delivery_l096_results.json`: browser and copied-Pages checks; separate from live deployment.
- `_sources_l096.json`: archived primary-source page hashes and locators. Documentation describes semantics; the schema and measurements are original course work.

## Regenerate from the repository root

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_verify_l096.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python labs/_run_l096.py
.venv/bin/python labs/_figures_l096.py
.venv/bin/python labs/_build_l096.py
.venv/bin/python labs/_execute_l096.py
.venv/bin/python labs/_build_l096.py
.venv/bin/python labs/_delivery_l096.py
```

The author executor uses the existing `relational-labs` Jupyter kernel. A separate interpreter can execute the notebook directly using the portable runner:

```bash
uv venv --python 3.12 /tmp/l096-portable-env
uv pip install --python /tmp/l096-portable-env/bin/python -r labs/requirements-l096-portable-lock.txt
/tmp/l096-portable-env/bin/python labs/_portable_l096.py
```

The student and solution contain all load-bearing code inline. Three student tasks feed the full run. Images are embedded PNG data URLs in notebooks. Rebuilding preserves solution outputs only if the complete ordered code cells are unchanged. The frozen author reference is a labeled SHA-256 of the canonical complete record list; SQLite recomputes the oracle on every run.

## Boundaries and deviations

This is a declared integer-key converter, not an SQL schema introspector. Alternate UNIQUE-key references, collations/coercion, cascading changes, keyless tables and automatic feature encoders are outside its scope. Static connectivity says nothing about temporal feature availability. PK metadata is not fed into a learned model. The course example deliberately preserves line rows; collapsing association tables into binary edges is a different transformation with a different information contract.

Complete defined course suite: **PASS**. Live Colab and deployment: **NOT_CHECKED**. Learner mastery: **PENDING_WRITTEN_DEFENSE**. No claim about unseen production databases or predictive superiority follows from these checks.
