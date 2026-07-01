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

## Build rules

- **Reuse first:** read `./assets/` (`missingness-viz.js`, `pipeline-viz.js`, `split-viz.js`, `group-viz.js`, etc.).
- **New reusable component:** add to `./assets/<name>-viz.js`; never inline one-off JS in the lesson.
- **Document expected states** in a comment at the top of each viz file (see `missingness-viz.js`).
- **Companion CSS:** lesson-local `<style>` with class prefix matching the viz (e.g. `.miss-viz`).

## Verification checklist (mandatory before publish)

Open the lesson in a browser (`file://` or local server):

- [ ] Viz renders without console errors
- [ ] Default state matches the prose caption
- [ ] Each interactive mode/toggle shows the behaviour described in text
- [ ] Labels (column names, fold IDs, leakage arrows) match lesson terminology
- [ ] Mobile viewport: viz remains readable (375px width)

Use `browser-testing-with-devtools` skill if available.

## Retrofit priority (Q1 gaps)

- L002 — join / PIT leakage (`leakage-viz.js`)
- L003/L004 — audit `split-viz.js` / `group-viz.js` against lesson claims
- L010 — `checklist.js` rubric items match leakage-spine table
