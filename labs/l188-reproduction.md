# L188 · Q3-2026 Literature Tracking Replay

User-approved 2026-10-02. This reproduces a tracking workflow, not a published model experiment. USD0 new cloud/API; 1800 aggregate local execution seconds, with failed attempts and verification included. The budget wrapper records wall-clock execution and kills the subprocess group at the remaining cutoff. Reading/authoring is not charged as numerical execution.

## Frozen collection

Window [2026-07-01T00:00:00Z, 2026-10-01T00:00:00Z). Exact config: `evidence/l188/packet/config.json`. Four independent all-field arXiv queries: exact phrases relational foundation model and tabular foundation model; terms RelBench and TabArena. submittedDate range202607010000–202609302359, ascending submittedDate, page size100, max100pages/query, at most2attempts/request, timeout30seconds, at least3seconds between requests. Limits are stop gates, never a completeness shortcut. This is query-relative discovery, not exhaustive literature coverage. Older-paper revisions are a separate manual watch lane.

Raw receipt: `packet/collection.json`, every HTTP response/error retained, URLs/timestamps/status/content type/SHA256. Packet bytes are immutable; author annotations are separately stored in `screening-review.json`. The source manifest authenticates original packet contents. Exact version URI, first-submission and update dates remain in Atom. Identity deduplication is not an independent-experiment audit. Published-date filters do not recreate historical versions.

## Observed run

Relational phrase: HTTP429 then read timeout. Tabular phrase: two read timeouts. RelBench: complete9/9; TabArena: complete22/22. 31records,30uniqueIDs; one overlap. Overall collection INCOMPLETE. All saved bytes/records replay; replay COMPLETE. Seven abstract-screened reading candidates,23deferred. Supplemental verified pages2609.25541/2609.02766/2609.03880 are a separate discovery lane; the last is already in the API corpus. Do not sum those counts. Method/code availability audits remain NOT_CHECKED; all paper results REPORTED_NOT_REPRODUCED.

Official arXiv API/manual/RSS sources and current RelBench/TabArena source pages are saved in the packet. HTTP200 is not successful score extraction: RelBench dynamic board shell and TabArena Space README are source-location snapshots. Current score extraction NOT_CHECKED; historical Q3 standings/change NOT_ESTABLISHED. No board result or score delta claimed. The API search metadata is the real collection experiment; synthetic pagination/admission examples only test contract behavior.

## Commands

From repository root:

```bash
.venv/bin/python labs/_budget_l188.py .venv/bin/python labs/_test_l188.py
.venv/bin/python labs/_budget_l188.py .venv/bin/python labs/_replay_l188.py
.venv/bin/python labs/_budget_l188.py .venv/bin/python labs/_verify_l188.py
.venv/bin/python labs/_budget_l188.py .venv/bin/python labs/_build_l188.py
.venv/bin/python labs/_budget_l188.py .venv/bin/python labs/_execute_l188.py
.venv/bin/python labs/_budget_l188.py .venv/bin/python labs/_delivery_l188.py
```

Fresh full search uses a NEW destination (refuses overwrite):

```bash
.venv/bin/python labs/_budget_l188.py .venv/bin/python labs/_collect_l188.py --output /tmp/l188-new-packet
```

The collector intentionally retains the frozen Q3 window. A later quarter needs a separately named protocol/config; do not relabel this historical run. Embedded notebook uses only the standard library for replay, authenticated offline bytes, and visible implementation; the repository relkit package initializer also imports course dependencies. Solution executes in a fresh empty directory. No recurrence or account subscription activated. OPML is ready for learner import.

## Evidence boundaries

Paper-result reproduction NOT_RUN. Model training/inference NOT_RUN. Historical leaderboard shifts NOT_ESTABLISHED. Complete field-wide discovery NOT_ESTABLISHED. Fresh quarterly collection INCOMPLETE. Live Colab NOT_CHECKED. Deployment NOT_CHECKED. Learner PENDING_WRITTEN_DEFENSE. Author preparation does not change course exit gates.
