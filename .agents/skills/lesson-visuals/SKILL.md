---
name: lesson-visuals
description: Decide when a lesson needs a visualization, build or reuse assets/, and verify correctness in the browser before publishing.
---

Use when creating or editing lesson HTML in this workspace.

## Decision tree

1. **Text or a small table is enough** — skip a viz (definitions, bullet lists, short code snippets).
2. **Spatial, temporal, or mechanistic concept** — add a viz:
   - join / leakage / point-in-time
   - train/valid/test folds, grouped CV, nested CV
   - pipeline fit scope (per-fold vs full data)
   - missingness mechanism (MCAR/MAR/MNAR)
   - class imbalance threshold effects, calibration curves
   - tree splits, boosting residuals, attention patterns (Q2+)

## How many visuals per lesson? (one per distinct mechanism)

Default to **one visual per distinct mechanism, claim, or "strength" the lesson teaches** — not one visual per lesson. A lesson with three mechanistic beats should have three visuals. The one-viz-per-lesson habit is a cap to break, not a target.

- **Every mechanistic beat gets its own visual.** If a section makes a spatial/temporal/mechanistic claim (per the decision tree above) and the next section makes a *different* one, each earns its own viz. A section that only has a static results table where the mechanism is visualizable is an under-served beat — add the viz.
- **Split multi-mode widgets when the modes teach different ideas.** A single widget with a mode toggle is right when the modes are *the same mechanism under a knob* (e.g. one board, slider = tile count). It is the wrong call when each "mode" is a *separate concept* the prose treats as its own section — that reads as "one viz" and forces the reader to mentally context-switch inside one frame. Prefer two adjacent, individually-captioned viz over one overloaded toggle.
- **Place each viz next to the prose that explains it.** A viz referenced from two different sections ("the widget above", "switch the widget below") is a smell that it should have been two viz, each inline with its own paragraph.
- **Reuse still applies.** Multiple viz in one lesson can each be a separate reusable asset, or several mounts of the same parameterised asset with different config — either is fine, as long as each mechanism is *seen*, not just described.

> **Anti-pattern (L019 "When trees win").** Teaches three inductive biases (irregular targets, uninformative features, orientation) but ships one `biases-viz` widget covering biases 1 and 3 via a mode toggle, while bias 2 gets only a static table. Three mechanisms, ~1.5 visuals. The fix: a dedicated viz for the noise-feature mechanism, and splitting the toggle into two inline viz.

## Build rules

- **Reuse first:** read `./assets/` (`missingness-viz.js`, `pipeline-viz.js`, `split-viz.js`, `group-viz.js`, etc.).
- **New reusable component:** add to `./assets/<name>-viz.js`; never inline one-off JS in the lesson.
- **Document expected states** in a comment at the top of each viz file (see `missingness-viz.js`).
- **Companion CSS:** lesson-local `<style>` with class prefix matching the viz (e.g. `.miss-viz`).

## Verification checklist (mandatory before publish)

Open the lesson in a browser (`file://` or local server). Run this for **every viz in the lesson**, not just the first:

- [ ] Each viz renders without console errors (all mount IDs resolve)
- [ ] Each viz's default state matches its prose caption
- [ ] Each interactive mode/toggle shows the behaviour described in text
- [ ] Labels (column names, fold IDs, leakage arrows) match lesson terminology
- [ ] Mobile viewport: every viz remains readable (375px width)
- [ ] Multiple viz in one lesson are visually consistent (shared CSS conventions, sit inline with their own section)

Use `browser-testing-with-devtools` skill if available.

## Retrofit priority (Q1 gaps)

- L002 — join / PIT leakage (`leakage-viz.js`)
- L003/L004 — audit `split-viz.js` / `group-viz.js` against lesson claims
- L010 — `checklist.js` rubric items match leakage-spine table
