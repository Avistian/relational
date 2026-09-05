---
name: lesson-visuals
description: Decide when a lesson needs a visualization, build or reuse assets/, and verify correctness in the browser before publishing.
---

Use when creating or editing lesson HTML or notebook figures in this workspace.

## Model architecture requirement (user, 2026-09-05)

Whenever a lesson introduces a new model, include a dedicated **Model architecture**
section with an end-to-end diagram, even when individual blocks already have visuals.
Show inputs and preprocessing/embeddings, the main blocks and their connections, repeated
stages, and the prediction head/output. Label important tensor dimensions and explain the
forward pass beside the diagram. Identify the pictured variant and any differences between
the paper and the lab; distinguish pretraining-only paths from prediction-time paths.
Keep it readable on mobile and provide an accessible text explanation. Carry the overview
into the companion notebook as a portable embedded figure, and keep notebook builders in
sync. Check that external image assets are included in the published site build.
This requirement is conditional on introducing a model; it does not add architecture
sections to lessons that only teach evaluation, tools, or writing. L047 and L048 are retrofits.

## Visual teaching standard — raised after L048 (user, 2026-09-05)

The user asked for better visuals in future lessons. Treat L048 as a baseline to improve,
not a finished design template. The following criteria make that preference actionable:

1. **Choose the question before the drawing.** State what the learner should be able to
   predict or trace after inspecting the figure. Use the visual form that exposes that
   relationship: matrices for coordinate mixing, connections for information access,
   geometry for projections, and aligned plots for comparisons. A sequence of named boxes
   can explain routing; it is insufficient when the question is how an operation works.
2. **Show the operation inside the architecture.** For a load-bearing transformation,
   expose the relevant input, operator, intermediate and output, with dimensions or units.
   Highlight one coordinate or path consistently through the computation. For example,
   a low-rank diagram should show the d→r→d projection and factors, not only a full matrix
   beside a matrix with columns removed. Label special illustrative cases as special cases.
3. **Keep one worked example coherent.** Carry the same values and symbols through the
   diagram, adjacent derivation and code explanation. Explain deliberate changes of example.
   Generate displayed arithmetic from the operation; disclose rounding. Use progressive
   stages when showing every detail at once would obscure the main relationship.
4. **Make interaction answer a causal question.** A control should change an input or
   assumption and visibly update the affected computation and interpretation. Where a
   comparison matters, retain a baseline beside the changed state or show the difference.
   Say what stays fixed. Put a short prediction prompt before the intervention and provide
   reset/default behavior, visible control values, and keyboard-accessible native controls.
5. **Design for the actual reading size.** Keep a clear reading order, aligned elements,
   nearby labels and unambiguous arrows. Use color consistently for semantic roles, backed
   by text or shape. On a narrow screen, reflow or disclose detail instead of shrinking a
   desktop diagram until its labels are technically present but unreadable. Inspect notebook
   images at their displayed width and the print layout; source PNG resolution is not proof
   of readability. Use whitespace to group information, not to stretch a sparse diagram.
6. **Make result comparisons legible and honest.** Align comparable scales, label units,
   show seed variability and explain interval types. Use a clearly marked detail view when
   small differences matter. Keep synthetic examples, measured runs and published targets
   visibly distinct; label protocol mismatches next to the comparison itself.
7. **Review teaching value separately from geometry.** Before delivery, identify the new
   mechanism's closest prior visual and the concrete improvement made. Check that a reader
   can trace the operation and answer the intended question without decoding a paragraph
   inside every box. Revise any figure that only restates its heading. Then perform the
   rendering checks below. A high test count or more figures does not establish visual quality.

Apply these criteria to exported notebook figures too. Choose snapshots that retain the
important intermediate or comparison without requiring the HTML controls, and caption
the exact state shown. Keep shared code and styling in `assets/`.

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
- **Companion CSS:** reusable component styles belong in `assets/`, with class names scoped
  to the component. Reuse course typography, spacing and semantic color conventions.

## Verification checklist (mandatory before publish)

Open the lesson in a browser (`file://` or local server). Run this for **every viz in the lesson**, not just the first:

- [ ] Each viz renders without console errors (all mount IDs resolve)
- [ ] Each viz's default state matches its prose caption
- [ ] Each interactive mode/toggle shows the behaviour described in text
- [ ] Labels (column names, fold IDs, leakage arrows) match lesson terminology
- [ ] Mobile viewport: every viz remains readable (375px width)
- [ ] Baseline/changed states, control values and affected quantities remain understandable at actual display size
- [ ] Keyboard controls work; labels or shapes convey distinctions independently of color
- [ ] Notebook snapshots and print output retain the explanation without interactive controls
- [ ] Multiple viz in one lesson are visually consistent (shared CSS conventions, sit inline with their own section)

Use `browser-testing-with-devtools` skill if available.

### How to actually run the browser pass (no workspace dependency)

Check available browser tools and installed executables rather than assuming the environment
matches an earlier session. L043 used `/usr/local/bin/google-chrome` with puppeteer-core;
L047/L048 did not have that browser available. When an installed browser is usable, drive it
with tooling **outside the repo** so the workspace keeps its zero-dependency, no-`package.json` posture:

```bash
mkdir -p /tmp/pptr && cd /tmp/pptr && npm install puppeteer-core
# then a throwaway script that loads file:///workspace/lessons/NNNN-*.html
```

The script should: fail on any `console` error / `pageerror` / `requestfailed`; re-mount each widget so it
holds the returned api; drive **every** stage, preset and slider extreme through that api; assert the SVG
painted (non-trivial height, a plausible label count, a non-empty readout) and that no label renders
outside the svg box; and screenshot each state at **900 px and 375 px**. Then *look at the screenshots* —
that is the part that finds the real bugs. L043's pass caught an SVG **z-order bug** (edge labels are
painted before boxes, so a label on a box edge disappears behind it), two label collisions, and a bar
scale that used a third of its vertical space in the default state. None are visible to a fake-DOM check.

If a browser is unavailable or its installation is declined, respect that boundary. Run the
available interaction/geometry checks and inspect vector-rendered desktop/mobile states,
then explicitly record browser verification as unperformed. This fallback does not establish
browser or live Colab compatibility and is not a standing reason to skip a future available
browser. L048's `_render_check_l048.py` demonstrates glyph bounds, overlap and card-containment
checks with librsvg; it caught text that anchor-only bounds checks missed.

### Assert the geometry, don't eyeball it twice

Hand-written SVG coordinates rot silently under later edits. In `labs/_viz_check_lNNN.js`, add for every
hand-laid-out diagram: **every filled box and every label inside the viewBox**, and **no two boxes
overlapping** (strict inequalities, so boxes that merely touch pass). Also assert exact arithmetic for any
operator the widget teaches, and that public setters **clamp** when the drawing uses a fixed pixel scale.
Reference implementation: `labs/_viz_check_l043.js` (`geometry()`), whose fake DOM also implements
`firstChild`/`removeChild` so redraws really clear instead of accumulating children.

## Retrofit priority (Q1 gaps)

- L002 — join / PIT leakage (`leakage-viz.js`)
- L003/L004 — audit `split-viz.js` / `group-viz.js` against lesson claims
- L010 — `checklist.js` rubric items match leakage-spine table
