# Glossary — personal mastery log

Terms **you've demonstrated mastery of**, with a one-line definition in *your own words*.
The tutor adds an entry when you can explain a term cold, and re-surfaces it in retrieval
when you've forgotten it. Newest at the bottom.

> This is your *personal* log. For the **authoritative** definition of every term the course uses (the
> ubiquitous language every lesson is consistent with), see
> [`reference/glossary.html`](reference/glossary.html). A term graduates to this file, in your own words,
> once you can explain it cold.

| Term | Your definition | First mastered | Lesson |
|------|-----------------|----------------|--------|
| Single-table assumption | Tabular learners need a design matrix and label vector — they don't consume raw tables, so relational data must be flattened first. | 2026-06-24 | 001 |
| Design matrix | The feature matrix **X** where each row is one training example and each column is one feature. | — | 002 |
| Label vector | The target **y** aligned row-for-row with **X** — what the model predicts. | — | 002 |

## HGT additions · Lesson093

- **Meta-relation:** one source-node-type, edge-relation-type, target-node-type triple.
- **Relative temporal encoding:** a representation of the receiver/source time gap added to the source before key/value projection. It does not impose a data cutoff.
- **HGSampling budget:** accumulated normalized neighbor scores, maintained separately per node type; squared scores determine sampling probabilities.

## L097 · Sampling contracts

- **Negative proposal q(j|u,r):** declared distribution over eligible destination nodes for a fixed source and relation; an unobserved draw is not a verified false fact.
- **Hard negative mining:** select high-scoring eligible pairs under the current model; separate sampling decisions from differentiating the selected loss.
- **Evaluation candidate set:** the items competing in a ranking metric; changing it changes the measurement even with frozen model scores.
