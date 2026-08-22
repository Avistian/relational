# 0103 — L043 retrofitted with architecture diagrams; and the mandatory browser check is now actually runnable

**Date:** 2026-08-22
**Trigger:** user, on L043 — *"can you add architecture diagram to the lesson about tabnet, pure equations
make it hard to read and understand"*.
**Scope:** `lessons/0043-tabnet.html` + three new `assets/` components. No lesson claim, verified number,
quiz, or lab changed.

## What was wrong

L043 shipped with **one** viz (`tabnet-mask-viz`, the masks + prior scale) for **five** distinct
mechanisms, and that viz sat *after* every equation. So the reader met
`M[i] = sparsemax(P[i−1] · h_i(a[i−1]))`, `P[i] = ∏(γ − M[j])`, `L_sparse`,
`[d[i], a[i]] = f_i(M[i] · f)` and `d_out = Σ ReLU(d[i])` before knowing where any of those symbols
lived. This is the recorded L019 anti-pattern (standard #9: one visual per distinct mechanism, inline with
its prose) reappearing in a Y2 lesson — worth noting that a lesson can pass all 81 of its content checks
and still fail the *visual* standard, because nothing was asserting it.

## What shipped

Three reusable components, each mounted next to the prose that explains it:

- **`assets/tabnet-arch-viz.js`** (`.tna-`) — the encoder data flow, paper **Fig. 4a**, as a 7-stage
  stepper. Placed in a new section *before* "The mechanism, piece by piece", and each stage prints the
  equation its highlighted block computes. One step is drawn, not the unrolled stack: the two edges
  returning to the attentive transformer read as a loop, which is the point, and the mask viz already
  unrolls three steps.
- **`assets/sparsemax-viz.js`** (`.spx-`) — Algorithm 1 as a water line. Bars for the logits, `τ` as a
  dashed threshold, shading for what pokes above it. The slider moves the **separation** of the logits
  (not a threshold), which exposes two things the algebra hides: the shaded area always totals exactly 1,
  and at flat logits `τ` goes **negative** so every feature survives — sparsity is a property of the
  logits, not a promise of the operator. Softmax on the same logits for contrast, plus the mask entropy
  that `L_sparse` penalises.
- **`assets/tabnet-fblock-viz.js`** (`.tnf-`) — the feature transformer, paper **Fig. 4c**: shared vs
  step-dependent GLU layers, the three `√0.5` merge nodes, the split, and an inset opening one layer.

## The insight worth keeping: draw from the code, not from the figure

Both diagrams were laid out against `labs/relkit/tabnet.py` rather than from memory of the paper's
figures, and that surfaced a **gap in the lesson's prose**: `a[0]` comes from an extra feature transformer
run over the **unmasked** features at step 0, whose `d` half is discarded. The paper's Fig. 4a shows it;
the equations never mention it; the lesson had inherited the omission. It is the piece people leave out
when reimplementing TabNet from the equations alone, and it is now a defbox and a diagram stage.

Generalisable: **a diagram is a consistency check on the prose.** Drawing every box forces you to name
every tensor, and any tensor the prose cannot source is a hole. Do this when retrofitting visuals to any
equation-heavy lesson.

Also corrected in passing: LR-0101 cited the attentive transformer as Fig. 4b. Per arXiv:1908.07442v5,
**4a = encoder, 4b = decoder, 4c = feature transformer, 4d = attentive transformer**.

## The other durable outcome: browser verification is possible here

Every L043-era record says *"Browser MCP unavailable → node verification only (consistent with
L021–L042)"*, which quietly downgraded the `lesson-visuals` **mandatory** browser check to an aspiration
for ~22 lessons. It is not unavailable. This machine has **`/usr/local/bin/google-chrome`**, and
`npm install puppeteer-core` **outside the repo** (e.g. `/tmp/pptr`) drives it without adding a single
dependency to the workspace — which still has no `package.json`, and should keep it that way.

What that bought immediately, none of which the headless fake-DOM harness could have caught:

1. **A z-order bug** — the `P[i−1]` edge label was painted *behind* the attentive-transformer box
   (arrows are drawn before boxes so the boxes sit on top). Invisible in the default state, visible only
   in the stages where that box is dimmed.
2. **Two label collisions** — the frame caption ran through the `a[i]` edge, and the two lane labels in
   the feature-transformer diagram overlapped mid-word.
3. **A badly-chosen scale** — the sparsemax bars used ~30% of their vertical space at the default
   setting, because the fixed pixel scale was sized for a slider maximum the reader rarely visits.
   Fixed by narrowing the slider range (max 3.0 → 2.0) and enlarging the scale.

**Method to reuse:** load the lesson over `file://`, fail on any console/page error, drive every stage and
preset through the widget's own api, assert the SVG actually painted and that no label renders outside the
svg box, and screenshot each state at 900 px and 375 px for eyeballing. Keep the harness in `/tmp`.

## What is now asserted rather than eyeballed

`labs/_viz_check_l043.js`: **81 → 222 checks**. The additions worth copying to future viz checks:

- **Geometry checks on hand-laid-out diagrams**: every filled box and every label inside the viewBox, and
  **no two boxes overlapping**. Hand-written SVG coordinates are exactly the thing that rots silently
  under later edits, and this catches it in 10 ms without a browser.
- **Exact-arithmetic checks** on sparsemax: the mask equals `max(z − τ, 0)`, survivors match the logits
  above `τ`, both rows sum to 1, softmax never hits zero, sparsemax entropy stays strictly below softmax
  entropy, and the survivor count is **non-increasing** as the logits separate.
- **Out-of-range api calls**: `setSpread(99)` must clamp, not draw off-canvas. A fixed pixel scale makes a
  public setter a way to break the picture.
- **Readout ↔ equation coupling per stage**, and lesson ↔ asset coupling: every emitted CSS class is
  defined in the lesson, and the architecture map is asserted to appear *before* the first equation.

The fake DOM in that file also gained `firstChild`/`removeChild`, so `while (svg.firstChild)
svg.removeChild(...)` now really clears between redraws instead of silently accumulating children — which
is what makes the geometry checks trustworthy after a state change.

## Implications for future lessons

1. **Diagram before derivation** for any mechanism lesson: a stage-stepper whose stages print the
   equations, placed ahead of the subsections that define them. This is now the reference pattern
   (L032's `tabtransformer-arch-viz` was the ancestor; L043's is the fuller version, with equations in
   the readout and geometry under test).
2. **Add the geometry + browser checks to every new viz**, and consider back-filling the browser pass on
   the Q3/Q4 lessons whose viz were only node-verified.
3. A lesson being "published" is **not** evidence its visuals meet standard #9. Count mechanisms against
   viz when reviewing an old unit.
