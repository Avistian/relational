# L161 Relational Foundation-Model Scope Audit

Approved 2026-10-01. USD0 cloud spending. CPU-only, no training or paid dispatch. Each execution/check has a 600-second timeout; stop and investigate failure without reducing the declared 15-record scope. All cases, retries and checks stay within the zero-cloud-spend authorization.

## Source and experiment boundary

Bommasani et al., *On the Opportunities and Risks of Foundation Models*, arXiv2108.07258v1, 16 August 2021. The mutable arXiv page now defaults to v3 (12 July 2022); the lesson cites v1. The CRFM report overview supplies accessible navigation. This is a broad synthesis report: the approved concept scope selects no relational model/table or published numerical experiment. Architecture, optimizer, initialization, training corpus/splits, epochs, seeds and numerical target are therefore NOT_APPLICABLE_TO_SELECTED_CONCEPT_SCOPE. Do not invent a historical trainer or call the local metadata exercise a reproduction of report-wide claims.

Source references and claim mapping: sources/l161/source-ledger.json. The course-defined evaluation rules extend the conceptual reading; they are explicitly not a Bommasani benchmark. Later model-specific lessons must audit their own training protocol and cost. This lesson provides no cloud scale-up switch because no such experiment has been specified.

## Executable local target

All 15 synthetic protocol fixtures, identified by SHA256 in evidence/l161/input-manifest.json. No real corpus is loaded. Database IDs A/B/C illustrate canonical identities; real aliases/copies require external provenance investigation. Adaptation labels are legal target examples, not test outcomes. Record completion and audit statuses are supplied assertions, not authenticated artifacts. A plan with unknown provenance remains REVISE; it cannot silently become known disjoint.

The three learner functions are visible inline and used by audit161. All diagnostics accumulate. The fixed initial checkpoint is reused across tasks even if adaptation changes copies of its weights. No per-task scores, randomness, seeds, rankings or confidence intervals are computed. The reproducible outcome is the audit table; all claimed model performance is NOT_ESTABLISHED. Full paper numerical reproduction is not claimed; training NOT_RUN. Author execution does not establish learner mastery.

## Commands

From the repository root, Python3.10+:

```bash
timeout 600 .venv/bin/python labs/_run_l161.py
timeout 600 .venv/bin/python labs/_verify_l161.py
timeout 600 .venv/bin/python labs/_build_l161.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 600 .venv/bin/python labs/_execute_l161.py
timeout 600 .venv/bin/python labs/_delivery_l161.py
```

The audit and standalone notebook use the standard library. Authoring additionally uses installed nbformat/nbclient/nbconvert/matplotlib; delivery uses Playwright. Runtime versions and executable hashes are recorded in receipts. The builder validates immutable fixture hashes; it never refreezes changed inputs. Notebook inputs are embedded with the same hashes. The independent verifier compares fixture expectations, a 27-case adaptation lookup table and all 256 combinations of eight disqualifiers; three deliberately incorrect learner functions must be rejected. PRETRAINING_OVERLAP is checked separately from PRETRAINING_UNKNOWN.

Checks do not authenticate corpus completeness, historical timestamps, empirical benefit, the foundation-model designation or prose quality. The two route diagrams are schematic, not architectures of a trained model. Live Colab and deployment remain NOT_CHECKED unless separately verified. No push or publication is authorized by this task.
