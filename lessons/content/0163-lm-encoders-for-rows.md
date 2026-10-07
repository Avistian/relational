<div class="lab-access"><strong>Lesson package</strong> · <a href="../labs/0163-lm-encoders-for-rows.ipynb">Student notebook</a> · <a href="../labs/html/0163-lm-encoders-for-rows.html">Executed author preview</a> · <a href="../labs/solutions/0163-lm-encoders-for-rows.ipynb">Solution</a><br><a href="../reference/row-encoders.html">Quick reference</a> · <a href="../labs/l163-comparison-template.md">Comparison template</a> · <a href="../labs/l163-reproduction.md">Reproduction contract</a> · <a href="../labs/evidence/l163/report.md">Complete measured results</a></div>

## 1 · What does the row vector remember?

**Single win:** encode the same row as text and as typed features, then defend an encoder choice using predictions and information loss. Read the core lesson in about 10 minutes; do the notebook comparison in a separate practice session.

[Lesson 162](0162-the-relational-fm-vision.html) placed a row encoder before graph message passing. That left a gap: what information reaches the graph in the first place? [Lesson 125](0125-pytorch-frame-deep-dive.html) introduced semantic column types. Here we compare a numerical value carried as a number with that same value carried through language-model tokens. Recall also [CARTE in Lesson 74](0074-carte-cross-table-transfer.html): how you represent a row changes what later computation can use.

Our worked row describes a product: price **10 USD**, weight **2 kg**, colour **red**, condition **used**. A **schema** names and types these fields. An **encoder** turns their values into features a prediction model can use. A **head** is the final fitted mapping from features to the predicted target.

> Before asking which encoder wins, ask what information each path receives, what is fitted, and what stays fixed.

## 2 · Model architecture: two routes from one row

Vogel, Hilprecht and Binnig propose serializing rows together with schema, encoding them with BART, and adding graph context. Their prototype adapts BART on table reconstruction before GNN training. Our comparison isolates the row-input decision: it uses a frozen text-pretrained BART encoder and a typed baseline, with no GNN. The pooling and regression head below are course choices, not a reconstruction of their hidden implementation details. [Primary reading: §§2–4](https://arxiv.org/html/2305.15321v1#S2)

[[FIG:architecture]]

**Text route.** We serialize only allowed inputs:

```python
{"table":"products","price_usd":10.0,"weight_kg":2.0,
 "colour":"red","condition":"used"}
```

Serialization must preserve which value belongs to which field. JSON escaping prevents a quoted string from accidentally becoming another field. IDs and the prediction target never enter the text. A tokenizer maps the string to integer token IDs. A token may represent a word fragment, punctuation or part of a number; it is not a typed database cell.

The pinned `facebook/bart-base` encoder has six layers and hidden width 768. For a batch of B rows, padding gives L token positions; final hidden states have shape **B×L×768**. We average real token vectors into **B×768** row features. This pooling is explicit; a generic BART output is not automatically a single row vector. Only the encoder runs. BART's pretrained encoder–decoder design and attention-mask interface are documented in the [model card](https://huggingface.co/facebook/bart-base) and [version-pinned API](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/bart).

**Typed route.** Fit numeric means/scales and categorical vocabularies on training rows. For each numeric column, compute `z=(value−training_mean)/training_SD`. A category becomes a one-hot block: one position is 1, others 0. With two numeric columns and three values each for colour/condition, our data gives **8 features**. This transparent baseline illustrates type-specific treatment; it is not a trained PyTorch Frame neural encoder.

**Read the units.** SD means standard deviation, a measure of spread around the mean. Subtracting the mean centers a numeric column; dividing by its training SD expresses the value in units of that spread. Using training statistics keeps validation and test values from changing the fitted transform. MAE, used below to select the head, means mean absolute error: average distance between predictions and targets, in target units.

**Worked arithmetic.** If the training price mean is 20 and SD 10, price 10 becomes −1. If weight mean is 1 and SD 0.5, weight 2 becomes 2. With category orders `[blue,green,red]` and `[new,refurbished,used]`, the row becomes `[-1,2,0,0,1,0,0,1]`. These moments are a hand example, not the experiment's fitted statistics. An unseen category produces an all-zero block in our implementation; that policy trades an explicit unknown signal for simplicity.

Both paths then receive training-fitted feature standardization and a ridge head. **Ridge regression** minimizes squared training error plus `alpha × sum(coefficient²)`; the intercept is unpenalized. Larger alpha discourages large coefficients. We choose alpha on validation MAE, then freeze the complete pipeline before testing. Different feature dimensions and pretraining histories mean equal head-selection rules do not imply equal model capacity or compute.

## 3 · Padding, order and names are testable choices

[[PREDICT]]

A short row might be padded to match another row in its batch. Padding positions must not enter the average. If token vectors are `[1,3]`, `[3,7]` and padded `[900,900]`, the correct pooled vector is `[2,5]`, not an average of all three. We include BART's beginning/end special tokens and exclude only padding:

```python
pooled = (hidden * mask[:, :, None]).sum(axis=1)
pooled = pooled / mask.sum(axis=1)[:, None]
```

The mask has shape B×L; adding the last axis broadcasts it over 768 features. A zero denominator is an error. Padding exclusion at pooling complements the encoder's attention mask; both matter.

Here **broadcasting** applies the same keep/drop value to every coordinate of one token vector. The attention mask stops padding from being read inside the encoder; the pooling mask stops it from entering the final average. BOS/EOS in the diagram mean beginning/end-of-sequence tokens.

**Now intervene on presentation.** Reverse the field order, or rename the columns to `c0…c3`, preserving every value. A positional text model receives a different sequence. Our typed route looks up canonical field identities, so these presentation changes leave its vector unchanged. That invariance is built into its input mapping. It would not survive an actual loss of field identity.

[[TOKENS]]

This explorer shows actual tokenization saved from the pinned BART tokenizer for the first three fixture rows. It runs no model in your browser. Switch a row or schema presentation and inspect token IDs, token count, and the row's precomputed change in pooled features. Token strings use the tokenizer's vocabulary notation; `Ġ` marks a preceding space where present. Dense attention-pair count L² is only a size proxy, not measured end-to-end compute.

**What text can offer.** A pretrained LM is a plausible way to reuse language information in descriptive names and values. That is a hypothesis to test on appropriate data. Turning a number into tokens does not impose arithmetic distance: the character sequence for 20 is not defined to be twice the representation for 10. Conversely, a numeric transform preserves the specified numeric relation but cannot infer the meaning of an unfamiliar product description. Our experiment contains short categories and no free-text descriptions, so it cannot settle that semantic trade-off.

## 4 · A complete, deliberately narrow comparison

We generated **240 synthetic rows** and disclosed the target:

`target = 2×price + 5×weight + condition_offset + colour_offset + noise`

Noise has standard deviation 2. This target is designed to favour a linear numeric/one-hot representation. It tests whether the LM path makes a simple numeric relationship easier or harder to recover; it is not a neutral tournament across all tabular tasks.

For each of three fixed splits, both methods get 144 training, 48 validation and 48 test rows. Preprocessing uses training rows only. Five alphas (`0.01,0.1,1,10,100`) compete on validation MAE. We keep the selected training-only head without refitting on validation. Reordered/renamed test text goes through that **same head**. MAE is the average absolute prediction error in synthetic target units; lower is better. [Exact protocol and commands](../labs/l163-reproduction.md)

[[RESULTS]]

[[FIG:comparison]]

**Read the result causally.** The typed path can directly express the target's numeric and categorical terms. Frozen pooled BART features make that relationship less accessible to this head. Renaming also moves representations away from the distribution used to fit the head. Large coefficients or rescaling of low-variance features can amplify such a shift. Our measurements establish the shift and prediction change; they do not isolate each internal cause or prove a general failure of LM encoders.

**Trace one measured prediction shift.** In split seed 0, the first test row is `p086`. Its target is 106.8246. The frozen BART head predicts 121.6190 from the baseline text and −497.9347 after column renaming. No target, numeric value, scaler or head coefficient changed. The prediction shift is about −619.5537 synthetic target units.

For this fixed linear head, each coordinate contributes `coefficient × (renamed_feature − original_feature) / training_scale`. The two largest contributions by absolute size are:

<table class="compact-trace" style="min-width:0;border-collapse:separate;border-spacing:3px"><thead><tr><th>Feature coordinate</th><th>Change after scaling</th><th>Contribution to prediction</th></tr></thead><tbody><tr><td>326</td><td>−22.2731</td><td>+82.4473</td></tr><tr><td>650</td><td>−18.9601</td><td>−42.6215</td></tr><tr><td>Other 766 combined</td><td>—</td><td>−659.3795</td></tr><tr><td>Total</td><td>—</td><td>−619.5537</td></tr></tbody></table>

Coordinate indices are zero-based array positions, not interpretable database fields. For example, coordinate 326 has coefficient −3.7017: two negative factors produce its positive contribution. Contributions can cancel; this decomposition explains the fixed head's arithmetic, not why BART changed its representations. **Try it:** if only coordinate 326 changed, would the prediction fall? **Check:** it would rise by 82.4473 to about 204.0663. The large negative total comes from the combined changes across the representation. [Saved head parameters](../labs/evidence/l163/selection.json) · [Keyed predictions](../labs/evidence/l163/report.json).

The typed scores repeat across presentation variants because the canonical features are identical. The three test sets overlap, so their SD describes split variation and is not a confidence interval. A train-mean baseline, every split score and all 864 keyed predictions are retained in the report. The numeric result does not establish transfer to new databases, unseen categories, missing values, or meaningful free-text tasks.

## 5 · What “full reproduction” means here

**Local experiment:** all rows, three splits, both encoders and both presentation interventions are part of the executable contract. Fresh frozen BART encoding and cached head replay have separate receipts. The notebook defaults to the portable replay so you can inspect and refit the heads cheaply; an explicit cell enables fresh encoding at the pinned revision.

**Historical target:** the paper's Table 1 wikiTables BART_table versus +GNN reconstruction experiment remains **NOT_RUN**. Exact data/splits, matching code/checkpoints and enough training details remain unresolved, so historical fidelity is **NOT_ESTABLISHED**. Our regression comparison is not that experiment, and the paper does not provide a typed-versus-text result to reproduce. [Historical contract](../labs/l163-reproduction.md)

## 6 · Make the comparison yourself

In the student notebook, implement three functions whose outputs are used by the experiment:

1. **`serialize_row`** — allow-list inputs, preserve schema/value association and implement both presentation interventions. CHECK rejects target leakage and broken escaping.
2. **`masked_mean`** — use the real-token mask and reject empty rows. CHECK rejects a plain padded mean. Stored real token states for the first eight rows let your implementation affect the replayed features; the remaining rows use verified pooled features.
3. **`choose_alpha`** — choose from validation losses with a deterministic tie rule. CHECK rejects always choosing the first alpha. The runner uses your choice for all six heads.

Then fill the [comparison template](../labs/l163-comparison-template.md): inputs, fitted/frozen parts, shape, cost, three-split MAE, sensitivity, limitation and next falsifiable test. Explain one schema perturbation before looking at its measured output. A useful next test would add a real free-text task, matched target information and held-out entities; it is a proposal, not evidence supplied here.

[[TEACHBACK]]

**EXIT:** defend your encoder choice in 200–300 words. Name the synthetic bias, a leakage control, the difference between fresh encoding and cached replay, and the historical gap. Passing code checks is author/implementation evidence; your reasoning still needs review. Learner status stays **PENDING_WRITTEN_DEFENSE**.

**Spaced retrieval:** tomorrow, derive `[2,5]` from the masked token example without opening the solution. In one week, explain why a presentation-invariant typed feature map does not demonstrate unseen-schema transfer. Ask the teaching agent about any unclear step or submit your table for review.

**Bridge to Lesson 164:** a row encoder supplies local features. A relational foundation model must also decide how to exchange information across rows and tasks. Griffin introduces that next architectural question; neither text serialization nor a typed vector alone creates relational context.
