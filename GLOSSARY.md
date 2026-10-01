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

## Lesson157 · Contribution evidence (prepared definitions)

- **Evidence replay:** recheck frozen inputs/outputs, identities, coverage and metrics without fitting the model again. It can validate saved results but cannot stand in for a fresh training run.
- **Experiment manifest:** the pre-dispatch source/protocol hashes recorded by each run. It identifies executable bytes, not the author's identity.
- **Release manifest:** hashes of the final shipped files, including documentation and saved evidence, except the manifest itself.
- **Selected reproduction:** execution of the complete named experiment, with its declared population, seeds, epochs, preprocessing, selection and metric. Its scope need not be the whole paper.
- **PENDING_PUBLICATION:** local contribution preparation has not yet produced a verified public artifact URL. Neither local checks nor a drafted issue imply a public submission.

## L163 · row-encoder comparison (reference definitions; mastery unassessed)

- **Row serialization:** an explicit mapping from schema/value pairs to ordered text; preserving association, escaping and target exclusion are part of its contract.
- **Masked mean pooling:** averaging included token vectors while excluding padding; this experiment includes BOS/EOS.
- **Presentation intervention:** change input order or displayed field names while holding row values, target and prediction head fixed. Canonical typed field identity is retained here.

### Relational in-context learning
Prediction conditioned on labeled relational examples while task-time model weights stay fixed. Distinguish label context inside a rooted subgraph from context across subgraph representations. A historical example's input cutoff stays at its own anchor, even when its outcome becomes available to a later query. Course availability checks require an earlier anchor, completed outcome window and arrived label. These contracts do not authenticate proprietary model masks. See [Lesson165 reference](reference/relational-in-context-learning.html).

### RDB-PFN and relational synthetic priors

- **Relational prior:** a distribution over schemas, key structures, latent states and dependent cells used to generate prediction episodes. Valid foreign keys alone do not establish useful cross-table dependencies.
- **DFS linearization:** following relationships and aggregating related rows into a task feature matrix. In RDB-PFN this happens before transformer inference; the base model does not receive a raw FK graph.
- **Support-only row attention:** attention across example rows whose keys/values are restricted to labeled support rows. Feature attention is a separate operation within each row.
- **Released-checkpoint replay:** fresh evaluation using supplied pretrained weights. It may reproduce a selected score without reproducing pretraining or authenticating historical preprocessing. See [reference](reference/rdb-pfn.html).


### Tabular-to-relational transfer audit

- **Transfer map:** a layer-by-layer account of reusable mechanisms, required database machinery, counterexamples and evidence limits. Distinguish representation, pretraining prior and adaptation.
- **Label readiness:** the time at which a support target is available for prediction; it can follow the historical query time by an outcome window and reporting delay.
- **Saved-evidence audit:** a new verification of existing prediction artifacts, identities and metrics. It does not execute fresh model inference. See [Lesson167 reference](reference/tabular-relational-transfer.html).
