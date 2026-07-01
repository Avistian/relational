---
name: curriculum-currency
description: Keep CURRICULUM.md current with newest tabular DL and relational foundation model research via arxiv MCP and web search at the start of each quarter.
---

Run at the **start of each quarter** and before publishing the first lesson of a new year block.

## Workflow

1. Read current [`CURRICULUM.md`](../../CURRICULUM.md) verified paper index and the quarter's lecture rows.
2. **arxiv MCP** (`user-arxiv`): search quarter topic + `tabular foundation model` / `relational foundation model` / `RelBench` / `TabArena`, sorted by submission date.
3. **Web search** fallback if MCP unavailable: verify arXiv IDs on arxiv.org before adding.
4. **Leaderboards:** skim TabArena and RelBench for new SOTA or baseline shifts.

## Add-only-if (all three tests)

Add a paper only if it:

- (a) sets new SOTA on RelBench/TabArena or a cited benchmark, **or**
- (b) exposes a failure mode the thesis must address, **or**
- (c) is a baseline you'll be measured against

Do **not** add application-only or incremental variant papers.

## Tier assignment

- **★ Core** — required deep read + reproduce or use as baseline
- **◆ Optional** — ~2 h skim after quarter core is done; one paragraph in `NOTES.md`

## Files to sync

After each currency pass, update:

- [`CURRICULUM.md`](../../CURRICULUM.md) — verified index + affected lecture rows
- [`RESOURCES.md`](../../RESOURCES.md) — year-grouped links
- [`reference/curriculum.html`](../../reference/curriculum.html) — student-facing index
- [`NOTES.md`](../../NOTES.md) — "when it wins / when it breaks" for each new ★ paper

## Q2 2026 pass (reference — extend each quarter)

Papers to track (verify IDs before adding):

| Paper | arXiv | Placement |
|-------|-------|-----------|
| TabPFN-2.5 | 2511.08667 | Y2 Q3 after TabPFN v2 |
| TabPFN-3 | 2605.13986 | Y2 Q3/Q4 |
| TabICLv2 | 2602.11139 | Y2 lec 066 extend |
| TabH2O | 2605.18383 | Y2 Q3 optional |
| Relational Transformer | 2510.06377 | Y5 ★ core |
| OpenRFM | 2606.04320 | Y5 ◆ or ★ |
