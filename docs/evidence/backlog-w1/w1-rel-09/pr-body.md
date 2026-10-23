Flashcard integrity now checks evaluated JavaScript cards for unique IDs, six required fields, and agreement with the actual progress report. Disposable mutations exercise duplicate IDs, missing fields and count drift.

The observed 48 versus 103 difference comes from 48 unquoted and 55 quoted ID keys. Both JavaScript evaluation and the actual progress report count 103.

[Source and raw test evidence](docs/evidence/backlog-w1/w1-rel-09/evidence.md). RED: four assertion failures. GREEN: four tests pass. Full source verification, canonical replay and independent review are recorded in the task handoff; this preliminary PR body must be finalized with their actual results and committed proof before acceptance/publication. Local fixtures do not establish CI or browser results.
