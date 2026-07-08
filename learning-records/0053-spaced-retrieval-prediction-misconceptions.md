# Pedagogy upgrade — spaced retrieval, prediction-before-reveal, misconceptions ledger

After the Q2 checkpoint the user asked for improvement proposals aimed at learning *as effectively as
possible*, then accepted three (of a larger set) to implement now: **C** (spaced-retrieval bank),
**E** (prediction-before-reveal), and **G** (misconceptions ledger). All three target the `teach` skill's
core distinction — build **storage strength** (durable memory), not **fluency** (feeling fluent while
re-reading). Deferred (proposed, not accepted): **A** real browser viz verification, **B** closing the
EXIT-ticket scoring loop, **D** the community/"wisdom" leg, and batch-2 items **F** (confidence
calibration) and **I** (interleaved GBDT discrimination sets).

## What shipped

- **C — `assets/retrieval-bank.js` + `assets/retrieval-pool.js`.** A warm-up widget that opens a lesson
  with a few questions drawn *only from earlier lessons* (spacing), interleaved across concepts/quarters,
  and **Leitner-scheduled in `localStorage`** (box 1–5; wrong → box 1 due next session, correct → promote
  with expanding intervals 0/1/3/7/16 days). Degrades to in-memory if `localStorage` is blocked (some
  `file://` contexts). Pool is a plain-`<script>` global (no fetch/CORS), one entry per durable idea with
  a **stable id** (Leitner state is keyed on it — never renumber). Seeded with 22 items spanning L001–L019.
- **E — `assets/predict.js`.** Commit-a-prediction-then-reveal widget (pretesting effect). Placed above a
  non-obvious result; the learner locks a choice, reads on, then reveals truth + match/miss framing.
- **G — `misconceptions.md`.** 17 misconceptions mined from records 0001–0051 (e.g. `best_score_` optimism,
  Gini `=1−p²−(1−p)²`, PR baseline = prevalence, ordered TS, GOSS `(1−a)/b`, OOF stacking). 11 are mirrored
  as `"misconception": true` pool items so past mistakes keep returning until genuinely gone.

## Conventions recorded

- New skill **`.agents/skills/lesson-pedagogy/SKILL.md`** — when/how to use all three (open with warm-up;
  predict before non-obvious reveals; keep the ledger and pool in sync).
- **NOTES Preferences standards #10–#12** under the Q2 retrospective.
- Verification: **`node labs/_check_pedagogy.js`** — 23 checks (pool integrity, spacing, interleaving,
  Leitner transitions, storage fallback, predict commit/reveal). All pass. Browser MCP still unavailable →
  headless only (same limitation flagged in proposal A).

## Reference implementation

**L019 ("When trees win")** retrofitted: a spaced-retrieval warm-up (`upTo: 19, count: 3`) opens the lesson,
and a prediction-before-reveal prompt sits above the noise-feature table (predict who wins on clean,
all-informative features — the honest "MLP wins" result). Copy this pattern into L021+ (Q3 opener).

## Open threads (deferred proposals)

- **A** browser viz verification is now more valuable, since warm-up + predict widgets join the viz that
  have only ever been checked headlessly since L013.
- **B** three recent labs (L017–L019) completed with no EXIT ticket → no rubric score; the skills feedback
  loop has lapsed.
- **D** the "wisdom" leg (community/real-world testing) is still entirely absent despite a publish/ship
  mission.
- **F/I** confidence calibration and interleaved GBDT discrimination sets remain on the table.
