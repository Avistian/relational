# L121 reproduction contract: historical claims and Home Credit GCN

**Full selected experiment NOT_RUN. Full reproduction is not complete.** The renewed attempt verifies deterministic caching, reruns original-module checks, audits raw keys and tests the released Neo4j query on a bounded real subset. The fresh optimized timing projection still exceeds the approved USD10 aggregate cap. No five-fold job was launched and no new AUROC is reported.

## Frozen target and provenance

Expanded Cvitkovic arXiv2002.02046v1 (February2020), Table4, Home Credit GCN: AUROC .780±.004 across five folds. Descriptive mean tolerance .01, fixed before training; no statistical-equivalence claim. Workshop2019 is a separate paper. The lesson cites its own historical primary sources in `_sources_l121.json`.

Released code: `mwcvitkovic/Supervised-Learning-on-Relational-Databases-with-GNNs`, commit `57195ccab62d23dcbcac1a317f8a9811a9fd6cb5`. File hashes and MIT/Apache licenses are retained in `sources/l118/` and `_sources_l118.json`. The complete protocol audit is in [L118](l118-reproduction.md); the notebook visibly includes the same complete modern-port model and trainer. Runtime pins: [requirements-l118-runtime.txt](requirements-l118-runtime.txt).

Preserved choices: all307511 labeled applications; KFold5 shuffle seed14; inner validation15%; model seed1234 per fold; table MLPs d→4d→256; one shared GCN; gate/value graph pooling; dropout.5; batch1024; AdamW1e-4, weight decay0; max300epochs/patience50; validate before training; first-strictly-best validation AUROC; full held-out AUROC per fold and sample SD. Released global metadata is not fold-local preprocessing. Historical DGL binary, original stochastic trajectory, original hyperparameter search, and historical ingestion times are not reconstructed.

## Fresh extraction work

`_prepare_l121.py` scans all eight raw CSV files. It selects the first32 training identities of L118's target-blind fold0 pilot and writes an ignored local Neo4j fixture. It scans every one of1670214 previous-application keys. In this sample,14 payment references are globally absent, and zero references point to a previous application owned by another applicant. No licensed rows are published.

`_neo4j_l121.py` executes the original loader with only import-URI relocation, then the exact released `MATCH r=(a:Application)-[*0..2]-(n)` query. It compares type/feature node multisets, directed typed relationship multisets, and labels against the prior CSV reconstruction. Identical-feature duplicate rows retain multiplicity; this comparison is not a proof of global graph isomorphism. The first comparison detected empty-string versus null differences. L121 normalizes empty cells to Neo4j nulls in a new ignored verified fixture; it preserves the original L118 fixture. The audit checks that every encoded tensor remains exactly unchanged by this correction. See `_neo4j_l121_results.json` for mismatch/correction counts and actual executed status.

This checks32 applicants, not all1024 pilot graphs or all356255 prepared applications. The original loader creates the applicant edge before matching the previous application: a missing previous key leaves that edge in place. The fresh sample resolves that ambiguity for its14 cases; it does not resolve all5148 references from the earlier larger pilot. Full-population graph equivalence remains NOT_CHECKED.

## Recreate the isolated historical query engine

The original amd64 image did not start usefully under emulation on this ARM host. We ran its unchanged Neo4j3.5.4 jars with native TemurinJava8 instead. This is historical query-engine version verification, not the original whole runtime. The runner binds HTTP only to localhost.

```bash
docker create --platform linux/amd64 --name l121-source neo4j:3.5.4@sha256:1eb754a81e7e48431be4f846dfc5ab820e7b7c841d99e7604ef62c3420a2a886
docker cp l121-source:/var/lib/neo4j /tmp/l121-neo4j
# Replace /tmp/l121-neo4j/conf/neo4j.conf with the settings below.
docker run -d --name l121-query --memory 2g -p 127.0.0.1:17474:7474 \
  --mount type=bind,source=/tmp/l121-neo4j,target=/neo4j \
  --mount type=bind,source="$PWD/labs/data/l121/neo4j-input",target=/neo4j/import,readonly \
  eclipse-temurin:8-jre@sha256:66a7b358772dfdcc1486919a5ede66d6c7275f203a5fefddb39565c2a151a485 /neo4j/bin/neo4j console
```

Minimal `neo4j.conf` for the isolated audit:

```text
dbms.security.auth_enabled=false
dbms.connectors.default_listen_address=0.0.0.0
dbms.connector.http.enabled=true
dbms.connector.http.listen_address=:7474
dbms.memory.heap.initial_size=256m
dbms.memory.heap.max_size=512m
dbms.memory.pagecache.size=256m
dbms.directories.import=import
dbms.directories.data=/neo4j/localdata
```

Wait for the server to report Started, then run the audit once on this fresh store. Use `--compare-only` to repeat comparisons after loading; rerunning CREATE statements on populated data can duplicate rows. Stop only this task container with `docker stop l121-query` when finished. Local fixture and store are not public artifacts.

## Optimization and budget

The cache in `relkit/cache_l121.py` stores deterministic feature tensors and graph indices. Learned embeddings and dropout remain live. On8 real graphs, uncached and cached batches agree exactly on tensors, training outputs, and gradients with the same RNG state. The1024-graph GPU pilot also checks input equality before timing.

Fresh T4 pilot: batch preparation1.4895s versus cached collation median.04338s; training median approximately.1586s and evaluation median.05011s. Maximum-schedule five-fold projection including final tests: **USD15.70298**, excluding full cache construction, disk IO and overhead. GPU compute alone projectsUSD11.63698. Reusing one batch for timing measures throughput, not held-out performance.

Current Modal rates checked2026-09-27: T4 .000164USD/s +2CPU×.0000131 +16GiB×.00000222 = **.00022572USD/s**. One1800s worker reservation .406296USD; retries0; overhead/retry reserve2USD; lesson cap10USD. Function-body estimate .002409USD excludes unitemized startup/build/storage costs. Pilot app: https://modal.com/apps/pszar92/main/ap-3ggXnkGdVLFUYsjWA4UCvN . Credit balances do not increase the cap.

**Stop decision:**15.70298 +2 +.406296 exceeds10USD. No full launch. Disk-backed caching is not integrated into the full trainer: the new cache is a validated timing prototype. `_run_l121.py` exposes that distinction and retains the runnable, source-checked L118 full trainer with its older conservative uncached cost gate. It rejects execution under the current cap and rejects command-line budget overrides. Lowering epochs, folds or population would define another experiment, not complete this one.

## Exact commands

From the repository root, local checks and default notebook:

```bash
.venv/bin/python labs/_check_l121.py
.venv/bin/python labs/_verify_l121.py
.venv/bin/python labs/_cache_check_l121.py
.venv/bin/python labs/_source_check_l121.py
.venv/bin/python labs/_prepare_l121.py
# Start the isolated Neo4j server as described below, then:
.venv/bin/python labs/_neo4j_l121.py
.venv/bin/python labs/_run_l121.py
.venv/bin/python labs/_figures_l121.py
.venv/bin/python labs/_build_l121.py
.venv/bin/python labs/_execute_l121.py
.venv/bin/python labs/_delivery_l121.py
```

The paid pilot was executed once with `.venv/bin/modal run modal/l121_pilot.py`. It reserves before dispatch and refuses an existing evidence file. Do not delete the ledger to bypass the aggregate limit. `_budget_l121.json` pins the exact implementation bytes used by that job.

For full-data preparation follow the pinned author's Neo4j loader/extractor in the L118 protocol; the32-applicant fixture cannot substitute for it. Then inspect the full-run plan:

```bash
.venv/bin/python labs/_run_l121.py --prepared "$HOME/RDB_data/homecreditdefaultrisk/preprocessed_datapoints"
```

The default plan checks all356255 expected prepared graph files. `--execute` explicitly requests training but currently refuses due to the budget. Any future optimization must be integrated, independently checked, and timed across representative disk-loaded batches before changing the recorded estimate. Any larger budget needs explicit authorization and an updated contract.

## Evidence categories

- Historical claims: primary publications, not local measurements.
- Course calculations: rule truth values, both-clock SQL features,68/100 grouping and gradients; no predictive benchmark.
- Current-source agreement: controlled original Python modules, dense graph adapter, full fold identities; historical DGL binary NOT_CHECKED.
- Real sample extraction: bounded Neo4j comparison with explicit limits above.
- Timing pilot: real reconstructed inputs, current T4 throughput; no AUROC.
- Selected five-fold result: NOT_RUN. Whole-paper/historical training parity: NOT_ESTABLISHED.
- Browser/local notebook/copied Pages: see delivery evidence. Live Colab and deployment: NOT_CHECKED.
- Learner mastery: PENDING_WRITTEN_DEFENSE.
