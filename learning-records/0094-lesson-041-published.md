# 0094 — Lesson 041 published: The Deep-Tabular Landscape & rtdl

**Date:** 2026-08-05
**Status:** Published (awaiting user completion)
**Curriculum:** Year 2 · Q1 · lecture 041 — **first lesson of Year 2** (Advanced Tabular Deep Learning).
Topic per `CURRICULUM.md`: *Deep tabular landscape & rtdl*; curriculum lab: *rtdl repo setup*.
**Primary reading:** Gorishniy, Rubachev, Khrulkov & Babenko, NeurIPS 2021 — *Revisiting Deep Learning
Models for Tabular Data* ([arXiv:2106.11959](https://arxiv.org/abs/2106.11959) ★).

## Single skill
Read the neural-tabular literature as a **map anchored by a fair protocol and two reference models** (a
strong ResNet baseline + FT-Transformer), so any "DL beats GBDT" claim is judged against a strong,
equally-tuned baseline — not a strawman — and set up the `rtdl` reference implementations to be trained
from L042.

## Why this was the ZPD
Year 1 closed on the *tree* half of the single-table bar (tuned GBDT, L040) and the written inductive-bias
account. The obvious Year-2 opener would be a flashy architecture, but the load-bearing next step is
methodological: the learner already owns the fair-comparison contract (L020), corrected tests (L023), and
the peer-review checklist (L038). Gorishniy 2021 is that exact discipline applied to a whole subfield —
the natural bridge from "honest about the tree baseline" to "honest about the neural baseline." It also
recruits prior recall (ResNet L028, embeddings L031, TabTransformer L032, Grinsztajn L024).

## What shipped
- **Lesson** `lessons/0041-deep-tabular-landscape.html` (~50 min): warm-up → the baseline problem
  ("why revisiting") → landscape map of the Y2-Q1 syllabus → the two anchor models (ResNet + FT-Transformer,
  with a Feature-Tokenizer / [CLS] schematic) → predict-before-reveal on "does FT-T beat GBDT?" →
  no-universal-winner results table → teach-back → rtdl toolkit → thesis bridge → 3 quizzes → primary
  reading → rtdl-setup lab (inline steps) → teacher/ask-teacher → nav.
- **One new reusable viz** (`file://`-safe): `assets/tabular-dl-map-viz.js` — the neural-tabular landscape
  as 10 clickable model chips across 5 families (baseline / attention / tree-inspired / feature-crosses /
  incumbent GBDT bar); each readout gives mechanism + inductive-bias stance + the Y2-Q1 lesson. Default =
  ResNet ("do these first"). CSS prefix `.tdm-`.
- **Headless check** `labs/_viz_check_l041.js` — **all pass** (10 models, 5 families, default resnet,
  FT names Feature Tokenizer + [CLS] + L046, GBDT = incumbent bar + Grinsztajn, click selection, CSS
  coupling, warm-up upTo:41, arXiv id present). Browser MCP unavailable → node verification only
  (consistent with L021–L040).
- **No lab notebook yet.** L041's curriculum lab is environment setup ("rtdl repo setup"); described as an
  actionable inline Lab section (pip install, import MLP/ResNet/FTTransformer, forward-pass smoke test,
  param-count intuition, one-sentence rationale). `labPath` left `null` (as L038/L039). A full
  `0041-*.ipynb` can follow if desired.

## Design choices
- **Methods paper, not architecture, as the Y2 opener.** Matches the curriculum table (041 = "landscape &
  rtdl") and the mission's honest-baseline thread. FT-Transformer's *full* mechanism is deferred to L046;
  here it is a well-explained preview (tokenizer + [CLS]) contrasted with TabTransformer (L032).
- **One primary viz (the map), not a tokenizer viz.** Per `lesson-visuals`, the landscape is the lesson's
  one genuinely spatial mechanism; the tokenizer is a *preview* served by a static schematic + defboxes
  (text/table is enough), with the full mechanism viz reserved for L046.
- **`quarter: "Y2Q1"`** on the three new retrieval-pool items (first Year-2 tag; the bank interleaves by
  quarter string).

## Artifacts synced
- `assets/retrieval-pool.js` +3 (`l041-revisit`, `l041-tokenizer`, `l041-nowinner` [misconception]).
- `assets/paper-deck.js` +1 card `gorishniy2021-ftt` (distinct from the existing L028 `gorishniy2021`
  baseline card — this one is FT-Transformer / landscape focused).
- `misconceptions.md` **M50** (Gorishniy 2021 shows DL finally beats trees → *no universal winner*; the
  contribution is a strong baseline + shared protocol).
- `reference/glossary.html` — new **Year 2 · Q1** section, +8 terms (strong simple baseline, shared tuning
  protocol, Feature Tokenizer, [CLS] token, FT-Transformer, ResNet (tabular), no universal winner, rtdl).
- `thesis-dossier.md` — Evidence Ledger +L041 (BAR, C3/C4) + narrative paragraph (neural-half BAR-raising).
- `RESOURCES.md` — Gorishniy 2021 entry promoted to ★ and extended as L041 primary reading (+ rtdl repo,
  + Borisov 2021 survey companion).
- `lessons/manifest.json` → **41 entries** (L041 = year 2, quarter 1, `checkpoint:false`, `labPath:null`).
- L040 nav now links forward to L041.

## Next
User runs the rtdl-setup lab and pastes the one-sentence rationale + param-count note (or says "lab done").
On completion, open **Lesson 042 — MLP & ResNet baselines (do these first)**: train a ResNet baseline with
rtdl under the shared protocol against a tuned GBDT, per `CURRICULUM.md` Y2 Q1.
