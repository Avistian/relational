# Lesson 172 — Schema tokenization implementation plan

User approved this scope on 2026-10-02.

**Goal:** turn every column of the pinned nine-table F1 snapshot into explicitly typed tokens with reproducible fit and transform boundaries.

**Architecture:** a visible deterministic typed tokenizer produces scalar payloads, state codes and schema descriptors, preserving keys as relation metadata. It is a course pipeline contract, not a learned RT embedding or performance reproduction. Full-table transformation is distinct from fit admission and query-time availability.

**Tech stack:** Python, pandas, PyArrow, NumPy, nbformat/nbclient; existing course HTML/CSS and Playwright.

## Frozen scope

L172 Full F1 Schema-Tokenization Audit. Reuse exact L171 archive and nine Parquet members with authenticated hashes. Every column receives an explicit policy; every row is transformed without sampling. Numerical/category fitting uses date < 2005-01-01 for time-bearing tables. Untimed tables have no admitted fit rows; use declared neutral numerical defaults and empty category vocabularies, preserving availability NOT_ESTABLISHED. Text stays UTF-8 strings; keys stay identifiers and never become numerical magnitudes. Timestamp payloads are UTC epoch days, a course choice. Missing and deliberately masked states are distinct and carry neutral payloads; heldout categories map to UNKNOWN. Unknown semantic types or mismatched schema fail closed.

No learned encoder or objective; optimizer/seeds/model selection are not applicable. No claim of RT parity. Whole-paper reproduction/fresh pretraining NOT_RUN. USD0 cloud/API, 1800 aggregate seconds numerical execution including failures, tests and notebook verification. Authoring/source retrieval/rendering/browser/Pages checks are delivery work. No push or deployment.

## Steps and acceptance

1. `labs/_check_l172.py`: write adversarial behavior contracts before implementation; observe failure. `labs/_budget_l172.py` accounts all numerical attempts and kills process groups on timeout.
2. `labs/_prepare_l172.py`: authenticate L171 sources and metadata, write explicit `evidence/l172/schema.json`, input manifest and reading provenance. Inspect every column policy.
3. `labs/relkit/tokenization_l172.py`: implement `fit_column`, `encode_column`, `tokenize_table` with clear errors, schema identity and no mutation. Unit checks include fit leakage, masked-value erasure, unknown categories, key identity and reordered columns.
4. `labs/_audit_l172.py`, `_verify_l172.py`: complete all-row audit, independently reconstruct payload/state outputs, compare deterministic output hashes and heldout interventions. Save per-column fit counts, missing/unknown/masked counts and exact receipts. Fail without downscaling.
5. `labs/_figures_l172.py`, `_build_l172.py`, `lessons/content/0172-schema-tokenization.md`, `assets/schema-tokenization.*`: worked numerical trace, type/key distinction and fit-boundary figure; interactive unknown/missing/masked cases with fixed baseline. Three live learner functions, portable source/data payload, immediate CHECK and written EXIT. Add quick reference.
6. `_execute_l172.py`: execute solution from empty directory and require exact audit parity. `_delivery_l172.py`: source parity, notebook blanks, desktop/mobile controls, keyboard/reset/print/no-JS, screenshots, links and deterministic build.
7. Update manifest, curriculum, resources, notes, glossary and thesis ledger; preserve learner PENDING_WRITTEN_DEFENSE. Stage only intended changes and run `_checkout_l172.py` against complete real Git-index Pages build. Retain previous staged L171 work.
