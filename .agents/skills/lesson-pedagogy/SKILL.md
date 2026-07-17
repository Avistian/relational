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

## 3. Add one teach-back / Feynman prompt per lesson (`assets/teachback.js`)

Recognition (multiple choice) is the weakest retrieval; **explaining a concept in your own words** is the
strongest, and it trains the mission goal ("explain the full stack without hand-waving"). Once per lesson,
on the *load-bearing* idea, make the learner write an explanation before revealing a model answer + a
self-check list of the points a good answer hits.

```html
<div id="teachback1"></div>
<script src="../assets/teachback.js"></script>
<script>
  Teachback.mount(document.getElementById("teachback1"), {
    prompt: "In your own words: why is X true?",
    points: ["key point 1", "key point 2", "key point 3"],
    model: "A model answer to compare against."
  });
</script>
```

The reveal is gated until the learner writes something (forces an attempt). Static HTML can't grade prose,
so the widget nudges the learner to paste their explanation to the teacher (chat) for real feedback — when
they do, grade it and, if they can explain it cold, add the term to `GLOSSARY.md` (personal mastery log).

## 4. Spaced paper flashcards (`assets/flashcards.js` + `assets/paper-deck.js`)

The curriculum is paper-dense; each paper otherwise surfaces once. Add a one-claim flashcard to
`assets/paper-deck.js` whenever a lesson assigns a **core (★) paper**. The deck is reviewed on the standalone
`flashcards.html` page (Leitner self-rated recall). Keep one crisp claim per card; use a stable `id`.

## 5. Keep the misconceptions ledger in sync (`misconceptions.md`)

When a lab CHECK, warm-up, or check-in exposes a wrong belief:

1. Add a row to `misconceptions.md` (wrong belief · correct belief · origin lesson · status `active`).
2. Add a matching item to `assets/retrieval-pool.js` tagged `"misconception": true`, so the wrong
   belief re-enters the spaced rotation and keeps coming back until it is gone.
3. When it is answered correctly across ≥2 spaced sessions (Leitner box ≥ 3), mark the row `retired`
   (keep it for history).

## 6. Explain everything introduced (thoroughness standard, from L027)

Storage strength needs the material to be *understandable on its own*. From L027 (user directive
2026-07-17), lessons must **explain every term, symbol, formula, and mechanism they introduce**, from
first principles, before relying on it — even terms an earlier lesson already used. Do not gate a
definition behind "see Lesson 0XX"; restate it inline or in a callout. Derive results rather than
asserting them, and unpack each experiment (held fixed / varied / measured / why it isolates the
mechanism) before the numbers. **Length is not a constraint** — drop the ~35–45 min soft cap when the
content needs more; a longer self-contained lesson beats a short one that leans on unstated recall. Keep
the chunking + worked-example + check-block structure so the extra length stays digestible. Full rule:
`NOTES.md` Preferences standard #17. Reference implementation: **L027**.

## Cross-cutting artifacts to keep current

- **`reference/glossary.html`** — the authoritative ubiquitous language. Every lesson must use terms
  consistently with it; add a row when a lesson introduces a term. (`GLOSSARY.md` is the learner's separate
  *personal mastery log* — do not conflate.)
- **`thesis-dossier.md`** — after each lesson add one Evidence-Ledger line (FOR / BAR / AGAINST the thesis).
  Never delete counter-evidence. This is the skeptic-facing case the mission is ultimately for.

## Verification (browser MCP down → headless)

`node labs/_check_pedagogy.js` verifies pool integrity, spacing, interleaving, Leitner transitions, the
predict commit/reveal flow, the paper deck + flashcard scheduling, and the teach-back gate/self-check. Run
it after editing any pedagogy asset, the pool, or the deck. When browser MCP returns, also eyeball the
widgets in a lesson (per `lesson-visuals` checklist).

## Reference implementation

**L019 ("When trees win")** carries a spaced-retrieval warm-up, a prediction-before-reveal prompt (on the
"clean features → MLP wins" result), and a teach-back prompt (on rotational invariance). Copy its pattern.
The paper deck is reviewed on `flashcards.html`.
