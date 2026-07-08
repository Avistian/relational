---
name: lesson-pedagogy
description: Apply retrieval practice, spacing, interleaving, and prediction-before-reveal so lessons build storage strength (durable memory), not just in-the-moment fluency. Use when authoring or editing any lesson HTML.
---

Use when creating or editing a lesson. Knowledge and skills only stick if the lesson is built for
**storage strength** (long-term retention), not **fluency** (feeling fluent while re-reading). The
`teach` skill names fluency as the enemy. Three cheap, reusable mechanisms enforce this.

## 1. Open every lesson with a spaced-retrieval warm-up (`assets/retrieval-bank.js`)

Before new material, make the learner recall **older** material from memory. Spacing + interleaving +
effortful retrieval are the highest-leverage retention levers.

```html
<h2>Warm up</h2>
<div id="warmup"></div>
...
<script src="../assets/retrieval-pool.js"></script>
<script src="../assets/retrieval-bank.js"></script>
<script>
  RetrievalBank.mount(document.getElementById("warmup"), { upTo: 21, count: 3 });
</script>
```

- `upTo` = **this** lesson's number. The bank only draws from lessons *before* it (spacing), never the
  fresh material.
- It is Leitner-scheduled in `localStorage`: a missed item returns next session, a mastered one returns
  much later. Nothing to configure per lesson.
- **Feed the pool** (`assets/retrieval-pool.js`) whenever a lesson ships a durable, testable idea — add
  one item with a **stable `id`** (never renumber; Leitner state is keyed on it). Options should be
  similar length (quiz-fairness standard #2).

## 2. Add a prediction-before-reveal prompt before any result (`assets/predict.js`)

Where a lesson reveals a number or an outcome (a bake-off table, a "who wins" comparison, a surprising
result), make the learner **commit a prediction first** (the pretesting effect). Place it *above* the
reveal.

```html
<div id="predict1"></div>
<script src="../assets/predict.js"></script>
<script>
  Predict.mount(document.getElementById("predict1"), {
    prompt: "Before reading: tuned XGBoost or an untuned Random Forest — which wins on credit_g?",
    options: [ { label: "Tuned XGBoost wins", value: "xgb" }, { label: "Untuned RF wins", value: "rf" } ],
    correct: "rf",
    reveal: "RF 0.901 edged tuned XGB 0.896 — a near-tie, and the honest baseline lesson."
  });
</script>
```

Use it once or twice per lesson, on the genuinely non-obvious results — not on every table.

## 3. Keep the misconceptions ledger in sync (`misconceptions.md`)

When a lab CHECK, warm-up, or check-in exposes a wrong belief:

1. Add a row to `misconceptions.md` (wrong belief · correct belief · origin lesson · status `active`).
2. Add a matching item to `assets/retrieval-pool.js` tagged `"misconception": true`, so the wrong
   belief re-enters the spaced rotation and keeps coming back until it is gone.
3. When it is answered correctly across ≥2 spaced sessions (Leitner box ≥ 3), mark the row `retired`
   (keep it for history).

## Verification (browser MCP down → headless)

`node labs/_check_pedagogy.js` verifies pool integrity, spacing, interleaving, Leitner transitions, and
the predict commit/reveal flow. Run it after editing any of the three assets or the pool. When browser
MCP returns, also eyeball the warm-up + predict widgets in a lesson (per `lesson-visuals` checklist).

## Reference implementation

**L019 ("When trees win")** carries both a spaced-retrieval warm-up and a prediction-before-reveal prompt
(on the "clean features → MLP wins" result). Copy its pattern.
