## Separate representation construction from in-context prediction

Your goal is to implement TabICL's inducing-vector computation, trace the full prediction route and measure what a larger labeled context buys. Read [Qu et al., Sections 3–4](https://arxiv.org/html/2502.05564v1). TabICL first builds distribution-aware cell and row representations, then applies a dataset-level in-context learner. This separation changes the cost structure compared with repeatedly alternating full table axes.

The paper reports experiments at much larger context sizes than early TabPFN. Those sizes are measured/model regimes, not unconditional guarantees that every table of that row count will fit any GPU. Feature count, query batching, number of ensemble views, precision and available memory still matter. The local timing intervention reaches 540 context rows; the 500,000-row experiment is explicitly unrun.

## Model architecture: column, row, dataset

Start with N rows and F scalar features. A shared column encoder processes each feature's collection of values, using labeled-training rows as the source of distribution information. It generates a hidden vector for each cell. A row Transformer mixes a row's feature vectors and learned summary tokens; concatenated summary outputs form a fixed-width row representation. A final Transformer adds encoded labels for context rows, lets queries attend to context, and feeds query outputs through a class-probability head.

In the original paper configuration, the column and row hidden width is 128; four row summary tokens produce width 512 for the dataset-level learner. The paper uses multiple induced-attention blocks, row-attention layers and a 12-layer final learner. Our visible implementation isolates the inducing-attention and distribution-conditioned affine mechanisms. The actual measured scores use the separately pinned pretrained v1.1 checkpoint, not those untrained reduced blocks.

<!--figure:architecture-->

## Compress a column through inducing vectors

Let U be the hidden vectors of a column's N scalar values and I be m learned inducing vectors. First compute `M = Attention(I, U_context, U_context)`. Each inducing vector becomes a weighted summary of the training column. Then compute `V = Attention(U, M, M)`: every input cell reads those summaries. Only context rows supply first-stage keys and values; adding an unrelated query cannot change the summaries read by existing rows. This follows the paper's Equations 4–5, with the complete multihead blocks adding learned projections, residual paths and feed-forward transformations.

<!--figure:mechanism-->

The second stage produces cell-specific information, not one scalar statistic copied to the entire column. Linear heads generate a scale W and offset B for each cell; the embedding is `W*value+B`, coordinate by coordinate. If a value is 2, W is `[.5,−1]` and B is `[.1,.3]`, the resulting vector is `[1.1,−1.7]`. W and B depend on the column context; they are not globally fixed per-feature parameters. The lab keeps the two operations separate so you can check both the information boundary and the arithmetic.

## What becomes cheaper, and what remains quadratic?

Full self-attention over N rows forms N² scores per head. Inducing attention uses roughly `m*C + N*m` scores for one column, where C is context length. With fixed m this column stage grows linearly in row count. But the final dataset-level learner still contains context attention. Calling the entire method “linear attention” would erase that remaining cost.

The cost exhibits in lessons 64 and 70 display column, row and dataset score-element counts separately, holding feature count and inducing count explicit. These are accounting identities for a simplified attention layout, not a peak-memory estimator. The paper's efficient inference also uses batching and memory-management techniques. Measure actual runtime and memory before claiming a hardware ceiling.

## Feature symmetry has a trade-off

If row attention treats equally distributed columns as indistinguishable, different rows that permute those feature values can collapse to similar representations. TabICL uses rotary positional embeddings in row attention to distinguish feature positions. RoPE rotates query/key coordinates by position-dependent angles; this introduces an ordering signal. Ensembling over column permutations can reduce sensitivity, but it is not exact feature-permutation invariance for one fixed view. [Paper Section 3.3](https://arxiv.org/html/2502.05564v1#S3.SS3).

This is a useful correction to “tables have no natural feature order, so any ordering signal is wrong.” Columns also have identities. A suitable model must represent those identities without making arbitrary serialization choices dominate prediction. Your architecture trace should explain both the symmetry goal and the symmetry-breaking device.

<!--figure:results-->

## Exit: explain the bottleneck you measured

Implement the two attention stages and conditional affine map. Change an unrelated query and prove that existing cell outputs remain fixed. Then measure nested contexts of 60, 180 and 540 labeled rows with the same query rows, checkpoint and one-view setting. Report accuracy/log loss and context-preparation versus prediction time across seeds; do not turn this one-task scale intervention into a universal accuracy comparison.

Use the five-dataset checkpoint evidence for broader comparisons. State which part of the architecture you implemented, which pretrained version produced measurements, and which large-context result remains a cited paper claim. Explain why a retrieval method and an inducing-vector method reduce different costs even though both summarize a large context.
