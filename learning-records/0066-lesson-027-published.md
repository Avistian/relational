# 0066 — Lesson 027 published: Inductive Bias — Uninformative Features (backfilled)

**Date:** 2026-07-17 (record backfilled 2026-07-17; the lesson itself was published 2026-07-17 in the
merged PR #11 "create-lesson-27", commits `614ac4e`/`f4203b5`/`dd7738b`/`e79ec26`, but no
publish/complete learning record or NOTES session was written at the time — this record and
[[0067-lesson-027-complete.md]] close that gap).
**Status:** Published, and subsequently completed (see [[0067-lesson-027-complete.md]]).
**Curriculum:** Year 1 · Q3 · lecture 027 — the last of the three inductive-bias mechanism lessons
(the Grinsztajn arc 024–027).
**Core paper:** Grinsztajn, Oyallon & Varoquaux 2022, *Why do tree-based models still outperform deep
learning on typical tabular data?* (NeurIPS 2022 D&B, arXiv:2207.08815), **§5.3 (Finding 2: MLP-like
architectures are not robust to uninformative features)**. Theory backing: **Andrew Ng 2004** (ICML),
the rotationally-invariant-learner sample-complexity bound (linked from L026).

## Single skill
Explain why adding uninformative ("junk") features degrades an MLP far faster than a tree; name the
mechanism that makes a tree robust (**greedy, gain-gated split selection = implicit feature
selection**); connect it to Ng's rotation-invariance bound (L026); and recognize the two ablations
that prove it — **adding** junk widens the tree–MLP gap, **removing** it shrinks the gap.

## What shipped
- **Lesson** `lessons/0027-inductive-bias-uninformative-features.html` (~45 min, first lesson under
  the new thoroughness standard #17): vocabulary defbox section (junk feature, Gini/impurity, split
  gain, implicit feature selection, rotational invariance + Ng's bound — all defined in-lesson) →
  the claim + the two-ablation design → predict → add-junk experiment + table → the gate mechanism →
  mechanism viz → Ng (2004) spelled out → teach-back → thesis bridge → subtleties → 3 quizzes →
  primary reading → lab.
- **Two reusable viz** (standard #9): `assets/uninformative-add-viz.js` (add-junk curve, MLP lead
  eroding) and `assets/uninformative-mechanism-viz.js` (the gate: root-split gain vs first-layer
  weight, informative vs junk). Headless check `labs/_viz_check_l022.js`-style.
- **Lab** `labs/0027-inductive-bias-uninformative-features.ipynb` (Tier C, mechanism isolation) —
  crucial fragment `add_junk` (append k pure-noise columns), sweep k, and measure the gate
  (Gini + root-split gain). Built via `labs/_build_l027.py`; verify `labs/_verify_l027.py`.

## Verified live (from _verify_l027.py / executed solution)
- Add-junk sweep (smooth target so the MLP wins clean): k=0 tree 0.885 · GBT 0.945 · **MLP 0.986**;
  k=100 tree 0.865 · **GBT 0.914** · MLP 0.902. MLP fell **0.084**, GBT only **0.032** — the ranking
  **reverses** at k≈100.
- The gate: root-split gain ~**118×** higher on informative than junk columns; the MLP still spends
  ~**56%** as much first-layer weight on junk (a ~1.8× "gate"). Tree selects; MLP dilutes.

## Artifacts synced (at publish time)
- `assets/retrieval-pool.js` +2 (`l027-uninformative` [misconception], `l027-gate`).
- `assets/paper-deck.js` +1 (`grinsztajn2022-uninformative`).
- `misconceptions.md` **M24**; `reference/glossary.html` +4 Q3 terms (uninformative feature, implicit
  feature selection, split gain/gate, root-split gain vs MDI).
- `thesis-dossier.md` +1 ledger line (BAR + FOR, C3/C1).
- `lessons/manifest.json` → 27 entries.

## New teaching standard captured this lesson
**Standard #17 (thoroughness over brevity)** — everything introduced in a lesson must be explained
from first principles before use, even if it lengthens the lesson; no length ceiling. Recorded in
NOTES Preferences and the `lesson-pedagogy` skill. L027 is the reference implementation.

## Next
Lesson 028 (MLP & ResNet tabular baselines — Gorishniy 2021 §3.2), which closes the Grinsztajn arc and
pivots Q3 from "why trees win" into building the honest neural baseline.
