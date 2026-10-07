# Remove opening reviews — October 7, 2026

The learner asked to remove review tests at the start of lessons because they were not helpful. Removed opening warm-up widgets, written recall questions and their answer panels. Preserved prerequisite explanations, learning objectives, worked examples, model diagrams, in-lesson predictions, labs and exit exercises. Updated the lesson-pedagogy skill to reflect this explicit preference.

The change touches 207 lesson pages, 166 manuscripts, 207 notebooks and 114 notebook previews. Other lessons had no matching opening review to remove. `scripts/lesson_openings.py` performs targeted source/output cleanup; the shared publication pass applies it after legacy builders, preventing reintroduction. Missing warm-up containers are a supported no-op in `RetrievalBank.mount`, so later widgets initialize normally.

Validation:

- All 228 lesson pages loaded in Chromium at 375px without an opening retrieval widget or JavaScript errors.
- Representative mobile openings were visually inspected; screenshots are included.
- The code cells, saved outputs and execution counts in all 207 modified notebooks are byte-equivalent as parsed JSON to the prior commit. Only Markdown changed. Matching notebook-hash receipts were refreshed; no training execution is claimed.
- No new broken internal fragments were introduced in changed lesson or notebook-preview HTML.
- Five focused policy checks cover removing reviews while retaining model retrieval, prerequisites and exercises.
- Existing visual checks and L081/L082 delivery checks pass. The shared publication pass is idempotent.
- Clean Git-index publication build is run after staging.

No learner mastery records, saved quiz progress or historical training results were changed. Later exercises and optional standalone study tools remain available.
