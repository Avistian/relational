# Authority note (not part of grill fields): Piotr told Herder in chat 2026-10-24 ~21:00 Europe/Warsaw to handle grills till morning. Answers below are recorded as Piotr via that standing order so intake can proceed without inventing a new lead-guard overnight window. Confirm/revise in morning summary.

QUESTION_TO: Piotr
QUESTION: REL-19 real-data leakage test: should it check the existing relmini feature joins against each label's temporal cutoff using only cached data, with missing cache explicitly NOT_RUN and no downloads?
ANSWER_BY: Piotr
ANSWER: Yes — check existing relmini feature joins against each label's temporal cutoff using only cached data. Missing cache is explicitly NOT_RUN (not a pass). No downloads, no training, no fetch.
