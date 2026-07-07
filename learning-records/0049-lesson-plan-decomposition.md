# Per-lesson decomposition of the curriculum (031–240)

**What.** Created [`plan/`](../plan/README.md) — a per-lesson decomposition of the expert curriculum for
**every lesson 031 through 240** (Y1 Q4 onward, where `CURRICULUM.md` rows thin to one-liners and
`reference/curriculum.html` collapses whole quarters into paper lists). Six per-year files
(`year-1.md` … `year-6.md`) + a `README.md` defining the schema and how to use it.

**Why.** The published lessons 001–019 all share a fixed anatomy (single skill · specific paper section ·
reusable viz asset · lab with one "crucial fragment" · honest myth-buster · thesis bridge). The
`CURRICULUM.md` rows do not carry that detail for later lessons, so every "build the next lesson" still
required re-scoping from the papers. This plan front-loads that decomposition so future sessions can go
straight to building.

**Schema per lesson.** `Skill` (one testable competency) · `Teach` (2–4 concept beats) · `Lab`
(tier + crucial fragment → deliverable, per `lab-authoring`) · `Viz` (reuse an existing `assets/*-viz.js`
or a one-line spec for a new reusable one, per `lesson-visuals`) · `Bridge` (retrieval callback + thesis
tie + what it sets up). Checkpoints/exit exams use `Deliverable` scored against the CURRICULUM exit
criterion. Year 6 milestones use `Work` instead of `Lab` (research, not concept lessons).

**Grounding.** The 2024–2026 relational + FM frontier papers were verified via the **user-arxiv MCP**
before writing (abstracts/contributions read for): RelGNN `2502.06784` (atomic routes), RelGT `2505.10960`
(5-element tokenization), Desired-graph `2606.08491` (filtering+injection), Universal Row Encoder
`2606.21434` (four pillars, decoupled encoder), ContextGNN `2411.19513`, RDL survey `2506.16654`, RelBench
v1 `2407.20060` + v2 `2602.12606`, Zahradník `2305.15321`, Griffin `2505.05568`, RDB-PFN `2603.03805`,
RDBLearn `2602.13697`, Relational Transformer `2510.06377`, OpenRFM `2606.04320`, KumoRFM-2 `2604.12596`,
RelGT-AC `2606.03040`, GelGT `2605.15575`, TabICLv2 `2602.11139`, BeyondArena `2606.30410`, temporal-shift
`2502.20260`, Operational TTF `2606.29091`, Cvitkovic `2002.02046`, 4DBInfer `2404.18209`. Classic Y1–Y3
papers used from existing curriculum context + parametric knowledge.

**Decisions.**
- **Scope 031–240** (respecting the user's "Y1 Q4 onwards"); 020–030 left to existing rows + precedent.
- **Collapsed ranges expanded** into concrete individual lessons: Y3 `111–114` (OGB → setup/reproduce/
  scale/error-analysis), Y4 `151–154` (portfolio → 3 task entries + synthesis), Y5 `192–194` (FM
  reproduction → setup/full-set/analysis), Y6 `211–218` (ablation matrix → 8 named ablations) and
  `231–234` (external review → solicit/incorporate/validate/polish).
- **New reusable viz assets named** for future authoring (not yet built): `embedding-viz`, `flatten-loss-viz`,
  `arch-family-viz`, `tokenizer-viz`, `mask-viz`, `cross-viz`, `retrieval-viz`, `pfn-prior-viz`,
  `temporal-embed-viz`, `ott-viz`, `cell-graph-viz`, `rdl-stack-viz`, `message-passing-viz`, `hetero-graph-viz`,
  `oversmoothing-viz`, `wl-viz`, `temporal-graph-viz`, `atomic-route-viz`, `mask-pretrain-viz`. Existing
  assets are reused wherever possible.
- **Linked** from `CURRICULUM.md` (intro note + lesson-production-schedule step).

**Status.** The plan is a *starting contract*, not frozen — each entry should be updated if a live
reproduction reveals a better framing (same rule as any planned scope). Next lesson to build remains
**020 (Q2 checkpoint)**; the plan governs 031 onward.
