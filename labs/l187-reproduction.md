# L187-F1-ENTITY-PRIVACY

Approved 2026-10-02. Complete named **course** experiment, not a published private-GNN result.

## Reproduce from repository root

```bash
.venv/bin/python labs/_budget_l187.py .venv/bin/python labs/_check_l187.py
.venv/bin/python labs/_budget_l187.py .venv/bin/python labs/_run_l187.py
.venv/bin/python labs/_budget_l187.py .venv/bin/python labs/_verify_l187.py
.venv/bin/python labs/_budget_l187.py .venv/bin/python labs/_figures_l187.py
.venv/bin/python labs/_build_l187.py
.venv/bin/python labs/_budget_l187.py .venv/bin/python labs/_execute_l187.py
.venv/bin/python labs/_budget_l187.py .venv/bin/python labs/_delivery_l187.py
```

The complete nine-table packet is already present. `_prepare_l187.py` authenticates the inherited L181 snapshot and retrieves primary sources; it is preparation, not needed for replay. The standalone student/solution notebooks embed all inputs and all numerical implementations and run without the repository or network. Install numpy, pandas, pyarrow and duckdb before running the notebook. Author versions are pinned in `evidence/l187/environment.json`; renderer/browser dependencies are recorded there too. No new mandatory course dependency was needed.

## Data and exact scope

All nine tables from the pinned public F1 snapshot, with SHA256 per file in `evidence/l187/input-manifest.json`; inherited source revision `0d47fe0c8a1a51aaf97ab485f4a028e290f97c67`. This is not a claim of historical paper-experiment identity. Source schema and primary reading receipts are in `sources/l187/source-ledger.json`.

One declared owned set consists of a driver row and every results, qualifying and standings row with that driverId. Audit all 857 drivers, all 13 foreign-key columns and all 227,716 logical FK edges (count each reference once, not twice for GNN reverse-edge storage). Owned sets contain 70,876 rows and touch 175,933 edges. Full graph-node deletion and SQL constraint semantics differ; no SQL deletion is executed on user data. The audit constructs filters on public in-memory tables.

The histogram reads all 26,080 results, bins by the fixed public constructor domain of 211 IDs, and bounds each driver's **total** retained events by ascending immutable resultId. All other owners' IDs and ordering remain fixed when an owner is removed. Caps `[1,5,20]`; epsilon `[0.5,1,2]`; seeds `0..29`; 270 vectors and 56,970 coordinates. NumPy PCG64 uses SeedSequence `[187,cap,int(epsilon*10),seed]` separately for every configuration. No HPO, train/validation/test selection or row subsampling. Dates are descriptive snapshot metadata, not a forecasting information contract.

## Claim, proof and implementation boundary

Add/remove-one-owner adjacency; public category domain and shared metadata fixed. A retained event contributes one one-hot count. Per-owner stable clipping changes only the removed owner's vector, with L1 norm at most C. Thus ideal independent Laplace noise of scale C/epsilon per bin gives the **whole vector** epsilon-DP. Replacement adjacency may require 2C. Clipping separately per bin, renumbering rows globally, allowing unknown categories or using the observed maximum as a global bound would change the argument.

The 2,571 actual histogram neighbors and all 857 direct-owner filters are implementation checks, not an exhaustive proof over possible databases. Independent SQL/scalar calculations verify ownership, every histogram, every release and all 30-seed summary metrics (tolerance 1e-10); RNG replay is exact under the recorded environment. The strongest synthetic tests reject unknown owners, incomplete public bins, duplicate IDs, invalid privacy parameters and unstable order. Wrong per-bin clipping, row-scale accounting and profile-only ownership are explicitly rejected in the executed notebook.

This simulator publishes raw public data, seeds and float64 outputs. It is **not a production private release**: known noise can be subtracted and finite-precision privacy needs its own implementation analysis. The ideal basic composition bound if all 270 mechanisms were genuinely privately released would be 315; that number is not this simulator's guarantee. Raw and clipped query errors are distinct; reported SD is sample SD across seeds, not a confidence interval across databases.

## Budget and evidence

No paid service permitted; cloud/API USD0. Aggregate numerical wall-clock cap 3600 seconds, including checks, failures, notebook execution and figure generation. The wrapper logs attempts, reserves the remaining budget before execution and kills the process group at cutoff. An initial 60-second conservative allowance covers preliminary schema inspection and RED checks. An interrupted active reservation requires reconciliation before another run. No silent downscaling or additional budget.

`report.json`: full scientific report. `driver-audit.csv`: one row per driver. `releases.json`: every float64 output. `_verify_l187_results.json`: independent verification. `_execution_l187_results.json`: empty-directory notebook execution. `_delivery_l187_results.json`: browser, notebook and link checks. `artifact-manifest.json`: sealed delivery hashes. `_checkout_l187_results.json`: actual clean Git-index Pages build.

Status: named course experiment COMPLETE; published private-GNN reproduction NOT_RUN; production DP and complete erasure NOT_ESTABLISHED; live Colab and deployment NOT_CHECKED; learner PENDING_WRITTEN_DEFENSE. The retained 24,501 driver/constructor-aggregate links are lineage evidence, not demonstrated membership attacks. No personal mastery or prior practical-exit state is changed.

Primary reading: Dwork/Roth Definition 2.4, Definition 3.4, Theorem 3.6 and Corollary 3.15. GAP and Xiang et al. v2 Section VI supply attributed research context; no model/trainer or benchmark reproduction is claimed. A future private-GNN experiment needs a separately audited privacy accountant and a source-pinned full protocol.
