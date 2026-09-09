## A useful representation can still have the wrong information

Your goal is to extract representations that give a downstream classifier a fair training problem. Read [Ye et al., Section 6](https://arxiv.org/html/2502.17361v1#S6). The paper distinguishes a labeled **support/context** row from an unlabeled **query** row. Their feature values can be identical while their representations differ, because one receives its known label as input.

This matters for heterogeneous tables: an encoder can transform different feature types and distributions into a common hidden space. But a visually separable embedding plot does not prove that a new, unlabeled row will be separable in the same way. If training embeddings already contain their own labels, a linear classifier may learn to read a channel that disappears at inference. This is an information-role mismatch, not merely a scaling problem.

## Trace the label route

Imagine two rows with identical feature vector x and opposite labels. A context representation of the form `encode(x)+label_embedding(y)` can distinguish them perfectly. At prediction time, both rows have unknown targets, so that direct label difference is absent. A scatter plot of context embeddings may therefore look impressive for a reason unavailable to the deployed predictor.

The appropriate counterfactual is to remove a row's label while extracting that row's representation. In the reduced PFN, the embedding function receives labeled context and separate query features. It has no query-target argument. In the official v2 API, use the query/test representation route and inspect the exact package version; `get_embeddings(..., data_source='train')` and `data_source='test'` describe different information roles.

<!--figure:mechanism-->

## Cross-fit the representation

Partition the outer training rows into K folds. For each fold, use the other K−1 folds as labeled context and extract the held-out fold as unlabeled queries. Scatter the representations back into their original row order. Every training row receives one query-role vector. Fit the downstream head on those vectors and the ordinary training targets. Labels can train the head; they must not have entered each row's own extracted representation.

For rows `[0,1,2,3]` with folds `[0,1,0,1]`, context `[1,3]` generates representations for queries `[0,2]`, then context `[0,2]` generates `[1,3]`. Concatenating the two outputs without restoring row order would pair embeddings with the wrong labels while preserving array shape. Your CHECK tests both context exclusion and scatter alignment.

Cross-fitting the backbone is not sufficient to make the head's training score an unbiased estimate. The head still fits those targets. Select its hyperparameters on outer validation data and evaluate it on outer test data. Any feature-layer selection, context-size choice or concatenation strategy also consumes validation information.

## Model architecture: a pretrained encoder and a learned head

The end-to-end route is: fixed pretrained v2 weights → fold-specific labeled contexts → query-role hidden vectors → training-fitted standardization → logistic regression → class probabilities. A logistic head maps a hidden vector h to a logit `w·h+b`; sigmoid converts it to a binary probability. L2 regularization limits coefficient magnitude, with its strength selected on validation.

The author experiment uses three folds, final-layer vectors and a validation-selected head. Validation and test representations are averaged over the three context-specific encoders. This is a declared approximation: the head's training vectors each came from one complementary context, whereas its evaluation vectors average several contexts. The paper uses ten folds and studies intermediate-layer choices and combinations on 29 tasks. Our result does not reproduce that table.

<!--figure:architecture-->

## Inspect separability without assuming causality

The lab measures a query-role linear head against native v2 predictions on diabetes, blood transfusion and phoneme with three seeds and identical outer rows. A small gap can show that the final hidden space supports a useful linear decision rule under this extraction protocol. It does not show that v2 internally reasons linearly, that those coordinates are causally meaningful, or that the head will retain its advantage under shift.

<!--figure:results-->

For an exploratory embedding visualization, fit PCA on training representations only, then transform validation/test vectors. PCA chooses directions of maximum variance; the first two components need not preserve the directions important for classification. Keep the projection and its fitting population fixed when comparing label-role interventions, or the changing display can conceal the effect you meant to inspect.

## Exit: prove the information boundary

Submit original row IDs, fold membership, embedding shapes, head selection results and native-versus-head test losses. Demonstrate that changing a held-out row's target does not change its query-role embedding when context is fixed. Then deliberately build an own-label context representation on a tiny fixture and explain why its apparent separability is misleading.

For the relational bridge, ask the same question of a row encoder: which related labels were available when this row representation was built? Point-in-time correctness must hold inside representation construction, not only in the final train/test split. A classifier cannot undo leaked information that an encoder has already placed in its inputs.
