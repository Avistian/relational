<p class="subtitle">One tangible win: build two representations of the same interactions, then defend which information your prediction needs. Core reading: 15 minutes · lab: 35–50 minutes.</p>

## 1 · Retrieve before reading

Without opening L104, write three short answers:

1. For a prediction just before time 10, can an interaction stamped 10 enter the history?
2. An event happened at 3 but arrived at 12. Can a prediction at 8 use it?
3. If two records have equal timestamps, does their CSV row order prove which happened first?

<details><summary>Check only after committing your answers</summary>

Under our before-event contract: **no**, **no**, and **no**. We require event time < query time and availability time ≤ query time. A row number is a reproducible identity, not evidence of an unrecorded causal order. If you missed one, revisit [L104’s prediction-time boundary](0104-information-leakage-in-time.html) before proceeding.

</details>

**The next problem.** L102 and L103 consume timestamped interactions. L104 asks whether those interactions were knowable. Now suppose your data pipeline hands the model one graph per day. Even a perfectly causal daily graph may have discarded information the task needs. Our mission is to build defensible relational systems: representation choices belong in that argument, before choosing a neural architecture.

## 2 · One stream, two representations

A **continuous-time dynamic graph (CTDG)** records changes at their event timestamps. “Continuous time” means updates need not lie on a fixed grid; recorded timestamps can still be rounded or tied. It does not imply continuous-valued node states or require a point-process model. A **discrete-time dynamic graph (DTDG)** is a sequence of graph snapshots. See [TGN v3 §2, Dynamic Graphs](https://arxiv.org/html/2006.10637v3#S2).

Our interaction stream stores `(event_id, user_id, page_id, time, availability)`. An edge is an occurrence, so repeated edits between the same user and page remain separate records. User 0 and page 0 belong to different identity spaces. Sorting all their raw integers into one node set would merge different entities.

**What does a snapshot mean?** This lesson uses a *window interaction graph*: an edge summarizes interactions occurring in one time interval. This differs from a *state snapshot*, such as all friendships still active at a moment, and from a *cumulative interaction graph* containing all interactions so far. An edit is not a permanent relationship. Deletions would need explicit state-transition semantics; this dataset projection has no such operation.

Choose an origin **o** and width **Δ**. Event time **t** belongs to:

**k = floor((t − o) / Δ), with window [o + kΔ, o + (k + 1)Δ).**

The left boundary is included; the right boundary belongs to the next window. We use origin 0 and nonnegative times. “Daily” here means fixed 86,400-second bins from that origin, not local calendar dates with daylight-saving changes.

**Worked stream, in seconds:** AX@1, AX@3, BY@3, AY@8, BX@12, AX@19. A/B are users; X/Y are pages. With Δ = 10, the first window contains four events but only three distinct endpoint pairs. A binary graph stores one edge per pair. A count-weighted graph stores weights 2, 1, 1.

[[FIG:aggregation]]

**Predict before changing the control:** will increasing the width from 10 to 20 reduce the number of binary edges? Will it change the sum of count weights?

[[AGGREGATION]]

For this stream, width 20 merges the two AX snapshot edges: 5 binary records become 4, while the total count remains 6. Wider windows need not merge anything in another stream. Nested widths make this comparison clean; changing the origin or using nonnested widths can change which events meet.

## 3 · Counts cannot restore a discarded sequence

Keep the same pair counts but exchange the times of two contacts. The resulting snapshot is identical. Yet a time-respecting path—contacts traversed in increasing time order—may disappear. The following is a separate **synthetic directed contact graph**, not the bipartite Wikipedia graph.

[[FIG:order]]

History I has A→B at 1, then B→C at 2: information starting at A can reach C. History II reverses those times: it cannot. Both weighted snapshots contain AB:1 and BC:1. A downstream model receiving only those counts cannot distinguish the histories. Adding first/last times makes a richer summary, but does not generally recover every repeated event’s timing.

[[PREDICT]]

**The right conclusion is task-dependent.** If the question is “how many edits did this pair make in the completed day?”, count weights preserve the answer. If it is “which interaction preceded the complaint?”, those counts may be insufficient. This is an information argument, not evidence that one model has higher AP. A stream encoder can also discard history internally, through sampling, pooling, or memory updates.

## 4 · A window has a publication time

A complete graph for [0,10) contains the event at 8. At a query just before 6, using it would expose the future. Renaming the graph’s timestamp to 0 does not make its contents legal.

We choose **immutable completed windows**. Their earliest permitted release is:

**release = max(window end, latest availability of any constituent event).**

A snapshot is readable when **release ≤ query**. At query 10, [0,10) is legal: every event in it is strictly earlier than 10. An event at 10 belongs to [10,20), which remains unavailable. A late-arriving event can delay the release of its *whole* window.

[[FIG:release]]

[[RELEASE]]

**Try the boundary.** Move the query from 9 to 10. Four records become readable together. Then set query 12 and enable the delayed arrival: the earlier window must wait until 15. The stream can still use individual records that have already arrived.

Completed-window access is causal but stale: it withholds eligible events in the current unfinished window. Reading the retrospectively completed current window is fresher but can expose the target, timestamp ties, and future records. A third design—an incrementally maintained partial window—is possible, but it is a different representation. It needs versioned “as of query” contents, and its count is not the final window count.

> **Operational limit.** The release formula audits a known finite set of records. In a live service, you cannot infer that no more late events will arrive just by taking the maximum arrival observed so far. You need a documented completeness signal, bounded-delay policy, or revisions. The Wikipedia file supplies no ingestion history. Our measured audit assumes availability = event time; delayed arrivals are synthetic tests.

## 5 · Reproduce the complete representation experiment

The [JODIE authors’ Wikipedia release](https://snap.stanford.edu/jodie/) has **157,474 interactions**, involving **8,227 users and 1,000 pages** in the authenticated bytes used here. We retain endpoint IDs, times, and row identities. The original state labels and 172-dimensional edge features are outside this projection. This audit does not claim that the reduced table reconstructs the entire raw record.

**Declared before execution:** all events; widths 3,600 / 86,400 / 604,800 seconds; origin 0; half-open windows; strict-before event queries; one query per event, including timestamp ties; completed-window release. There is no training, tuning, train/test score, or random-seed uncertainty. These are exact census statistics of this file under these rules.

[[RESULTS]]

[[FIG:results]]

**Read the measures carefully.**

- **Binary collapse** = events minus distinct `(window,user,page)` keys. It measures records omitted by the binary projection, not a percentage of all information lost. Count weights preserve multiplicity.
- **Hidden strict pairs** counts pairs of events with distinct times in the same window. Such ordering is not recoverable from count-only snapshots. We count pairs across the entire stream, including unrelated endpoints; this is not a causal-dependency count.
- **Past records withheld per query** counts strictly earlier global-stream records excluded by closed-window access. **Nonpast exposure** counts records at or after the query inside the completed current window. Both are global access diagnostics, not a GNN’s actual sampled inputs.
- **Wait for release** is window end minus event time under the arrival=event assumption. It includes the final partial observation window’s scheduled close; we do not release that window early merely because the file ends.

**An exact identity to explain.** Summed withheld-past counts equal hidden strict pairs. Each pair `(earlier event, later query event)` in one bin contributes once to both. Nonpast exposure minus withheld-past totals equals **N + 2 × tied timestamp pairs**: self-records and simultaneous pairs remain on the nonpast side. This catches implementations that silently order tied timestamps.

The raw file has **152,757 distinct timestamps** and **4,816 unordered pairs tied in time**. We never claim their physical order. Snapshot results agree with an independent SQLite `GROUP BY`; a separate grouped-timestamp walk reconstructs pair counts. All-query release counts are checked independently as well.

**Evidence boundary:** the complete three-condition *course audit* is reproduced. This is not a new reproduction of TGN/JODIE model scores, a full-paper replication, or a measurement of ingestion delays. Source identity and deviations are in the [source manifest](../labs/_sources_l105.json); exact results are in the [audit report](../labs/_analysis_l105_results.json).

## 6 · Build it yourself

[Student notebook](../labs/0105-continuous-time.ipynb) · [Executed solution view](../labs/html/0105-continuous-time.html) · [Solution notebook](../labs/solutions/0105-continuous-time.ipynb) · [Complete reproduction protocol](../labs/l105-reproduction.md)

The notebook is standalone: it includes the canonical implementation in annotated chunks, portable diagrams, immediate checks, and all three full-data conditions. Its default full replay authenticates the original **560 MB** download; an existing matching raw file can be reused. No GPU is needed.

| Live task | Your implementation must decide | Counterexample to pass |
|---|---|---|
| TODO 1: `bin_index` | Half-open boundary and fixed origin | Time 10 moves into bin 1 at width 10 |
| TODO 2: `aggregate_events` | Typed pair grouping, counts, complete-window availability | Two late/early edges in one bin must share its delayed release |
| TODO 3: `visible_snapshot_mask` | Whether the whole snapshot is published | Release at query is legal; release after query is not |

Each function is called by the full-data audit; these are live implementation tasks. Write your predictions before running. Keep the solution closed until you have attempted the checks. After a hint, close it and implement again from memory.

The core release gate is short; the hard part is defining what its timestamp means:

[[CODE:release]]

## 7 · EXIT: defend a representation

Write 150–250 words comparing two services:

1. A daily report predicts tomorrow’s edit volume after yesterday’s data is declared complete.
2. An online service scores each new interaction immediately before it occurs.

For each, specify the prediction clock, retained information, bin/availability policy, and one counterexample that would invalidate your choice. Explain why weighted snapshots can answer a completed-window count question yet fail a temporal-path question. Cite one measured audit quantity without turning it into an accuracy claim.

**Pass criteria:** all three TODO checks; complete three-width replay; a tie/boundary/late-arrival counterexample; and a defensible written choice. Running the notebook alone does not establish mastery. Status remains **PENDING_WRITTEN_DEFENSE** until your work is reviewed.

[[TEACHBACK]]

**Spacing:** tomorrow, reconstruct the two histories with identical snapshots but different temporal paths without looking. In one week, repeat the argument for purchases or messages.

**Read next:** [TGN §2](https://arxiv.org/html/2006.10637v3#S2), then the [event/snapshot reference card](../reference/event-stream-snapshots.html). Ask the tutor about any unclear boundary, or paste your EXIT defense for review. L106 will define future-edge labels and temporal negatives; L107 will put models on the snapshot representation. Neither should silently change the clock contract established here.
