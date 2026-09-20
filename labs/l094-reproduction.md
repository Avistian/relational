# L094 reproduction contract: survey taxonomy and OAG accounting

Primary source: Dong, Hu, Wang, Sun and Tang (2020), *Heterogeneous Network Representation Learning*, Table 1 and §§1,3–4. The earlier “Sun & Han 2020” roadmap entry was an ambiguous placeholder, corrected in this package. The original PDF and verbatim numeric table transcription are archived under `sources/hin-l094/`; SHA-256 hashes are in `_sources_l094.json`.

This is a survey/evaluation lesson, not a new-model reproduction. It has no new model architecture, objective, optimizer, initialization, train split, checkpoint, or predictive metric to reproduce. For the selected statistics target those fields are **not applicable**, not missing training work. The portable notebook exposes every counting, taxonomy, protocol-comparison, loader and intervention function.

## Named target and full coverage

**Table 1, complete NN row**, all five node types, all relation maps, no sampled subgraph. A fresh pass recounts the complete 656,262,985-byte author release. Every forward relation is checked against its reverse content, including timestamps. The audit reports:

- 66,211 nodes: all five type counts and the total match.
- 422,958 forward adjacency entries, including 36,696 field-hierarchy entries.
- 386,262 entries in the five listed forward edge families.
- 845,916 stored entries including verified reverse relations.
- Paper Table 1 prints 3,638,702 total edges; P–V and P–P columns also mismatch the released counts.

“Edges” here means distinct entries in typed adjacency maps. Original repeated events, if collapsed during preprocessing, cannot be recovered. Mapping affiliation → institute is explicit. Full original-graph identity is **NOT_ESTABLISHED**. Historical comparison is **INCOMPARABLE**; the audit deliberately reports mismatches rather than repairing the printed target.

All three printed rows receive a separate arithmetic audit. NN listed edge counts sum to 1,843,795; CS node-type counts sum to 11,747,265 versus 11,732,027 printed. We do not assert which entries or conventions are mistaken. CS/OAG **data-level audits are NOT_RUN**. This is not full reproduction of all survey datasets or of the surveyed models.

## Run locally

From the repository root, use an environment with `labs/requirements-l094-runtime.txt`; author versions are separately recorded in `requirements-l094-observed.txt`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python labs/_verify_l094.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python labs/_run_l094.py --download
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python labs/_build_l094.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python labs/_execute_l094.py --full-graph
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python labs/_build_l094.py
```

The downloader verifies exact bytes before deserialization. Existing `labs/data/l093/graph_NN.pk` is reused. NN cannot substitute for another dataset. The runner accepts `--dataset CS|OAG --data PATH --sha256 HASH --output PATH` for separately sourced full graphs; those paths are unexecuted extensions, not evidence of availability or historical identity. The runner defaults to a 3 GiB address-space cap; a separately provisioned host can choose `--memory-gib N` for those larger graphs. The larger-data paths remain untested.

The student notebook defaults the full-data switch OFF, includes frozen evidence and portable figures, and needs no repository imports. Set `RUN_FULL_GRAPH=True` in its explicit cell to download and audit NN. The local executor uses the same switch through an environment variable and validates exact agreement with the fresh runner. A memory cap of 3 GiB address space and one BLAS thread protects local execution; cell progress is saved incrementally. Browser/Colab runtime identity is a separate check.

## Existing named model experiments

These are complete model/trainer tracks in earlier notebooks. L094 inspects pinned prior results; it performs **zero fresh model fits**. Do not compare the different tasks as a leaderboard.

| Lesson | Named target | Existing runnable command from repo root | Boundary |
|---|---|---|---|
| L091 | R-GCN Table 2 AIFB, 10 release-port seeds | `.venv/bin/python labs/_run_l091.py` | Historical framework/seed identity INCOMPARABLE |
| L092 | HAN Table 3 ACM KNN, full release schedule | `.venv/bin/python labs/_run_l092.py --seeds 0 --epochs 200 --mode release_node --output labs/_paper_l092_results.json` | Paper-global versus released per-node attention; one encoder, not 40 independent training runs |
| L093 | HGT Table 2 CS Paper-Field L2, five seeds | See exact command below | Full CS target NOT_RUN in the prior evidence; NN is not CS |

```bash
.venv/bin/python labs/_run_l093.py --preset paper \
  --data labs/data/l093/graph_CS.pk \
  --sha256 bf054d93f45d385691f89902e7a021c53328c2ffefe50eb4d91c9dc51d05877f \
  --seeds 0 1 2 3 4 --device cuda --output labs/_paper_l093_results.json
```

Consult [L091](l091-reproduction.md), [L092](l092-reproduction.md), [L093](l093-reproduction.md) before launching; they record model/data/selection deviations. The HGT Modal operator remains [l093_paper_repro.py](../modal/l093_paper_repro.py); this lesson does not retry the previously recorded account/resource blocker or claim it is still current.

## Evidence and delivery

- `_paper_l094_results.json`: fresh complete NN statistics, all-row arithmetic and provenance.
- `_verify_l094_results.json`: behavioral checks, including unsupported-comparison rejection and reverse-content corruption.
- `_execution_l094_results.json`: actual executed notebook coverage, memory limit and kernel.
- `_delivery_l094_results.json`: browser, links, portable figures and copied Pages evidence.
- `sources/hin-l094/prior-evidence.json`: exact parsed records and source byte hashes for previous lessons; not fresh training.

WSL interrupted the first notebook execution during authoring. The saved statistics and source files survived; no completed notebook claim was retained from that attempt. The available logs did not establish the cause. Subsequent execution is bounded and recorded separately. Live Colab and deployment are NOT_CHECKED unless explicitly updated by a corresponding delivery check.

Learner mastery remains **PENDING_WRITTEN_DEFENSE**.

## Final verification of this package

The guarded notebook completed all 16 code cells and the full NN recount, peaking at approximately 1.33 GiB resident memory. A second, clean Python 3.12 interpreter with NumPy 2.2.6, pandas 2.3.2 and dill 0.3.8 replayed every solution cell and matched the complete graph statistics exactly (approximately 1.27 GiB peak). Its transitive pins are in `requirements-l094-portable-lock.txt`. Recreate it with `uv venv --python 3.12 /tmp/l094-portable-env`, `uv pip install --python /tmp/l094-portable-env/bin/python -r labs/requirements-l094-portable-lock.txt`, then `/tmp/l094-portable-env/bin/python labs/_portable_l094.py`.

Desktop 1200 px and mobile 375 px browser checks passed, including all six route/intervention states, keyboard/reset controls, print layout, notebook images and no-JavaScript gallery navigation. The actual copied Pages build resolved 33 local lesson/reference links. Builder regeneration preserved identical lesson, student notebook, executed solution and prepared HTML bytes. These checks do not establish live Colab or deployment.
