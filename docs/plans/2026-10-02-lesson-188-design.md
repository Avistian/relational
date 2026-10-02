# Lesson 188 · Systematic literature tracking

Approved by the user in conversation, 2026-10-02. Named experiment: L188 Q3-2026 Literature Tracking Replay. USD0 new cloud/API; aggregate 1800 local execution seconds including collection, attempts, tests, notebook and delivery checks. Paper-result reproduction NOT_RUN; learner PENDING_WRITTEN_DEFENSE.

## Frozen protocol

Calendar window [2026-07-01T00:00:00Z, 2026-10-01T00:00:00Z). Four arXiv queries: exact phrases relational foundation model and tabular foundation model; terms RelBench and TabArena; each searched across all fields with submittedDate bounds 202607010000 through 202609302359. Sort submittedDate ascending; page size100; drain every page through totalResults. Preserve every raw response, HTTP status, request URL, retrieval UTC timestamp, SHA256 and errors. Two attempts per failed request, timeout30seconds; API requests separated by at least3seconds. Cap100pages per query; any cap, malformed response, inconsistent total, duplicate identity within a query or missing page makes collection INCOMPLETE. No silent alternate corpus. Search completeness is only relative to these queries, not all relevant literature. New submissions and revised older papers must be distinguished.

Deduplicate across queries by unversioned arXiv identity; retain versions, dates and provenance. Primary-source screening log records include/exclude/defer, reason, SOTA/failure-mode/baseline relevance, reported versus reproduced evidence, artifact availability and next action. Unreviewed records remain DEFER. Existing curriculum entries are leads, never verified primary sources. Freeze official arXiv documentation and RelBench/TabArena pages as current observations, not Q3 historical snapshots. No changed-rank claims without comparable older snapshots.

Collect full declared searches and current source snapshots; replay locally from immutable packet. If upstream fails, ship runnable collection plus authenticated failed attempts, an honest incomplete report and separately labelled synthetic fixtures for the learner contracts. No pretending fixtures reproduce quarterly coverage. Prepare local reusable query/RSS configuration and instructions; recurring execution and account subscriptions are not activated.

## Delivery

Short connected HTML lesson, printable reference, source-pinned paper log, runnable collector/replay, three learner functions (canonical identity, complete-pagination contract, evidence-aware triage), student and executed portable solution notebooks, accessible interactive triage, source/status ledger and manifest navigation. Primary sources: arXiv API manual and RSS documentation, official RelBench and TabArena pages. Reuse existing course assets. Provide retrieval/prediction/teachback and one-day/one-week revisits.

## Implementation and validation

1. Tests for identities, version merging, pagination gaps and evidence triage; then visible implementation.
2. Bounded collection and complete deterministic replay, with failed attempts retained.
3. Independent XML/identity/count/hash checks and corrupt-packet/wrong-function rejection.
4. Lesson/reference/notebook construction; execute solution from an empty directory.
5. Browser desktop/mobile, keyboard/reset, print/no-JS, local links, source parity and deterministic build.
6. Stage only intended artifacts and use the real Pages build from the Git index; no push/deployment.

The writing-plans skill referenced by brainstorming is unavailable. This document supplies its plan. Existing changes for other lessons are preserved; do not claim those lessons are complete.
