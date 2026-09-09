## A labeled table becomes the model's input

Your goal is to trace one TabPFN v1 prediction and run its historical checkpoint without confusing inference with pretraining. Read [Hollmann et al., Sections 2–4](https://arxiv.org/html/2207.01848v6). The paper studies small numeric classification tasks. Its principal OpenML-CC18 subset has at most 1,000 training rows, 100 numeric features and 10 classes, without missing values. These define a studied regime; a program accepting a larger array is not evidence for that regime's conclusions.

A **context row** supplies a feature vector and a known target. A **query row** supplies features but no target. The pretrained model combines them in a forward computation. The method's task prior mixes randomly generated functions and structural causal models, favoring relatively simple explanations. Lesson 63 will inspect what a causal data generator does and what it cannot guarantee.

## Model architecture: the row-token route

For a context feature vector x, a learned encoder maps the padded numeric row to a hidden vector. A target encoder maps its known label to the same hidden width, and the two vectors are added. Query rows receive feature representations without their true labels. Transformer blocks repeatedly mix the allowed row representations, with residual connections, normalization and feed-forward transformations. A prediction head maps each final query representation to class logits; softmax normalizes them into probabilities.

Our visible **RowPFN** mirrors this computation at two layers and width 32 with an explicit unknown-label embedding. It is a key-part implementation, not the exact historical checkpoint architecture: the unknown representation, normalization, width, preprocessing and prior differ. The separate pretrained runner loads the actual released v1 weights. This separation lets you inspect the mechanism and measure a real pretrained model without claiming that the reduced network generated those measured scores.

<!--figure:architecture-->

## Trace information access before tracing numbers

With three context rows and two queries, the attention-allow matrix has five rows and five columns. Every row may read the first three columns. The last two columns are blocked. Thus context representations depend only on context, while each query can read context and retain its own features through residual paths. Queries do not read one another's target or feature tokens through row attention.

This rule produces two useful invariances. Reordering context features and labels together should preserve a query's prediction in the position-free reduced model. Appending an unrelated query should not change the original query's prediction. Reordering features is different: v1's row encoder has feature-coordinate-specific weights, and preprocessing/permutation ensembling complicates full-system invariance. Do not infer a feature-order theorem from a row-order test.

<!--figure:mechanism-->

For a query with scaled attention scores `[0, ln(2), 0]`, softmax weights are `[.25,.50,.25]`. Values `[−1,2,0]` give the weighted result `.75`. The values in a trained network are hidden vectors, not necessarily labels or probabilities. This one-coordinate example exposes the weighted sum; it does not reduce TabPFN to a nearest-neighbor classifier. Your attention implementation is checked against PyTorch's scaled-dot-product operation, and the released v1 layer provides a separate reference for context/query masking.

## Run the historical model explicitly

The notebook's reference path uses `tabpfn==0.1.11` and a SHA256-verified v1 checkpoint, in a separate environment from Nature v2. The historical package needs a documented typing-import compatibility shim with the current PyTorch version; it does not alter tensor operations. The checkpoint contains Python-serialized configuration objects, so its loader is enabled only after checking the pinned artifact identity.

The author experiment uses diabetes, blood transfusion and phoneme, each capped at 300 rows with 180 context rows, 60 validation rows and 60 test rows. One ensemble view and three seeds are used. These seeds describe inference configuration variability; they are not three independently pretrained foundation models. Preprocessing and optional view choices can dominate that variation.

<!--figure:results-->

The experiment also compares a query predicted alone with the same query predicted alongside unrelated rows. This tests the whole observed inference path, including preprocessing. A source-level attention mask alone cannot prove that upstream normalization or later ensembling is independent of query batch composition. Inspect both tests and report their precise scopes.

## What “a second” leaves out

Separate checkpoint download and loading, context preparation, preprocessing, forward passes, ensemble views and any GPU synchronization. A warm prediction measurement is not a cold-start service measurement. Offline pretraining is amortized across future tasks; it is economically real even when excluded from a downstream latency comparison. The paper's speed claims use its declared hardware, tasks and competitors, so the local CPU timings are not reproductions of those numbers.

## Exit: defend the mask and the measured path

Implement the allow-mask and scaled attention, prove query isolation with an adversarial appended row, and run the reduced model on real numeric rows. Then regenerate or audit the historical checkpoint predictions. Submit model/package/checkpoint identities, context/test row IDs, log loss and timing, and a sentence identifying which code produced the score. Explain why an untrained reduced RowPFN passing invariance tests does not reproduce TabPFN's accuracy.
