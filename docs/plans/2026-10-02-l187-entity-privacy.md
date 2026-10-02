# Lesson 187 implementation plan and approved contract

User approved the scope on 2026-10-02. Execute locally; no cloud/API spending.

**Goal:** teach the difference between row, graph-node and declared person contribution privacy, with a complete F1 contribution audit and bounded histogram simulation.

**Architecture:** hash-authenticated nine-table L181 F1 snapshot; explicit primary/foreign key schema; canonical visible Python functions; independent SQL/scalar verifier; portable notebooks and shared browser components. This is a course experiment, not a published private-GNN benchmark.

**Stack:** Python, numpy, pandas, pyarrow, duckdb, matplotlib, nbformat/nbclient, existing HTML/CSS widgets, Playwright.

## Frozen experiment L187-F1-ENTITY-PRIVACY

Audit every driver and all nine tables, without subsampling. Owned rows are the driver plus all results, qualifying and standings rows whose driverId matches. Deleting a graph driver node removes only that node and its incident edges; SQL parent-only deletion either violates/rejects foreign keys or leaves orphan child records without enforcement. The course entity-removal operation deletes the declared owned set and its incident edges, retains shared races/constructors/circuits and constructor aggregates, and is explicitly NOT certified erasure/anonymization.

Histogram query: number of results per constructor over the full results table, fixed public constructor-ID domain from this public snapshot. Per-driver deterministic prefix by resultId; caps 1/5/20. Add/remove-one-driver-and-owned-rows adjacency, fixed shared metadata, stable ownership and row identities. Epsilon 0.5/1/2; 30 seeds 0..29 each, 270 vector releases. Float64 seeded Laplace simulation; no production DP claim. Report signed clipping bias, L1 clipping loss, MAE relative to clipped and raw queries, mean/sample SD over seeds. No HPO, selection or train/test split: descriptive release simulation. Analytical scale C/epsilon, expected absolute noise C/epsilon. All raw data, seeds and outputs are public demonstrations.

Proof: deleting one owner leaves other owners' prefixes unchanged and removes at most C one-hot contributions, hence L1 global sensitivity at most C over the declared domain. Prove ideal real-arithmetic Laplace DP separately from empirical checks and finite-precision code. Repeated independent releases compose; 30*(.5+1+2)*3 = 315 is a basic upper bound if all these mechanisms were actually privately released on the same population. Public seeds invalidate that interpretation for this simulation.

## Budget and stop

USD 0 cloud/API. 3600 aggregate wall-clock seconds for numerical subprocesses, including failures, notebook validation and numerical plotting. Reserve 60 seconds for initial unwrapped schema inspection/accounting overhead. Wrapper records every attempt and terminates process groups at remaining cutoff. Stop as INCOMPLETE on budget exhaustion, corrupted provenance or failed proof assumptions; do not shrink scope.

## Ordered work

1. Write adversarial checks for ownership, stable contribution bounding, full vector sensitivity and accounting. Observe RED with explicit learner stubs, then implement `labs/relkit/privacy_l187.py`.
2. Prepare hash-pinned packet and source ledger in `labs/evidence/l187` and `labs/sources/l187`; preserve inherited input bytes. Implement `_run_l187.py` and `_verify_l187.py`: independent full-population deletion and SQL/FK checks, saved simulation reconciliation, wrong-policy rejection.
3. Author `lessons/content/0187-ethics-privacy-reg.md`, `assets/entity-privacy.js/css`, portable figures and reference. Explain prerequisites inline because L185/L186 are planned. Primary reading Dwork/Roth definitions, Laplace theorem and composition; GAP and later critique attributed separately.
4. Build student/solution notebooks with three live TODOs, immediate CHECKs, all full experiment functions inline, embedded authenticated data/figures, independent verifier, EXIT defense. Execute all solution cells from an empty directory and compare complete reports.
5. Update curriculum, manifest, resources, notes, reference glossary, retrieval pool and thesis ledger. Author-prepared learning record only; no learner mastery changes.
6. Check browser states, keyboard/reset, mobile/desktop, no-JS/print, real notebook images, local links, source parity and deterministic builder. Stage only lesson-owned changes, seal evidence and run actual Pages build from Git index. No push or deployment requested.

## Evidence boundaries

Complete named course experiment is possible. Whole-paper reproduction/private-GNN training NOT_RUN; operational DP and complete personal-data erasure NOT_ESTABLISHED; live Colab/deployment NOT_CHECKED; learner PENDING_WRITTEN_DEFENSE. Accuracy or attack failure does not prove privacy. Ethics includes purpose, consent, group harms and recourse; no legal compliance claim.
