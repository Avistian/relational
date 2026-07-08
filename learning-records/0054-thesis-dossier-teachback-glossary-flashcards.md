# Pedagogy batch 3 — thesis dossier, teach-back, authoritative glossary, paper flashcards

Continuing the post-Q2 "learn as effectively as possible" thread. From a third batch of proposals the user
accepted four: **J** (thesis dossier), **K** (teach-back / Feynman), **M** (authoritative glossary), and
**O** (paper flashcards). These extend beyond retrieval mechanics into *transfer, the mission itself, and
navigation* — the levers that matter more as the curriculum scales toward 240 units.

## What shipped

- **J — `thesis-dossier.md`.** The skeptic-facing argument, structured as claims (C1 flattening is lossy ·
  C2 loss is recoverable by learning over structure · C3 the gain is real *and fair* · C4 the field
  undervalues it), an **evidence ledger** tagged FOR / BAR (raises the honest baseline) / AGAINST, the
  assembled "honest bar" (fair-comparison contract + beat a tuned GBDT *and* a leak-free stack under
  temporal splits), skeptic objections with current answers, and a **Current Verdict: undecided, honestly**
  — Q1–Q2 mostly *raised the bar*; supporting evidence is still conceptual, the demonstration is the Y3–Y4
  burden. Rule: never delete counter-evidence.
- **K — `assets/teachback.js`.** Free-recall widget: write an explanation (reveal gated until an attempt is
  made), then reveal a model answer + a self-check list of the points a good answer hits, plus a nudge to
  paste to the teacher for real grading. Recognition is the weakest retrieval; explaining trains the
  "explain the full stack without hand-waving" criterion.
- **M — `reference/glossary.html`.** The **authoritative** ubiquitous language: one canonical definition per
  term across L001–L020, grouped by quarter, with a live filter box and forward entries for Year-3+ thesis
  vocab (RDL/REG/message passing). Kept **distinct** from `GLOSSARY.md`, which stays the learner's *personal
  mastery log* (own words, only once explained cold) — a callout in each cross-links the two.
- **O — `assets/paper-deck.js` + `assets/flashcards.js` + `flashcards.html`.** Spaced self-rated recall of
  the 15 core papers assigned so far (Fey, Cawley&Talbot, van Buuren, He&Garcia, Saito&Rehmsmeier,
  Niculescu-Mizil&Caruana, Kanter&Veeramachaneni, Breiman, Friedman, Chen&Guestrin, Ke, Prokhorenkova,
  Bergstra&Bengio, Wolpert, Grinsztajn). Front = recall prompt, back = the central claim; Leitner-scheduled
  in localStorage. Standalone review page linked from the home page.

## Conventions recorded

- `lesson-pedagogy` skill extended: teach-back (§3), flashcards (§4), and a **Cross-cutting artifacts**
  section (glossary + dossier obligations). Reference implementation note updated (L019 now carries warm-up
  + predict + teach-back).
- NOTES Preferences **standards #13–#16** (dossier, teach-back, two-glossaries, paper flashcards).
- `index.html`: glossary added to Reference; a new **Review** section links `flashcards.html`.
- L019 retrofitted with a teach-back on rotational invariance (the subtlest of the three biases).

## Verification

`node labs/_check_pedagogy.js` extended to **40 checks** (adds paper-deck integrity, flashcard due-queue +
reveal + Leitner self-rating + upTo spacing, teach-back gate + model reveal + per-point self-check). All
pass. Browser MCP still unavailable → headless only (proposal A still open and now covers six widget types).

## Still open (deferred proposals)

- **A** real browser viz/widget verification (now the highest-value infra gap — six interactive widget
  families verified only headlessly).
- **B** close the EXIT-ticket scoring loop (L017–L019 unscored).
- **D** the community / "wisdom" leg.
- **F** confidence calibration on quizzes; **I** interleaved GBDT discrimination sets; **L** "what breaks
  if…" lab mutations; **N** cumulative concept-dependency map.
