## Freeze the procedure

Record original data bytes, row IDs, semantic positive class, preprocessing fit scope, candidate and epoch rules, checkpoint revision/hash, wrapper version, inference view/batch policy and seed roles. Test labels enter scoring after selection. A class-1 column is not always the adverse outcome: WDBC class 1 is benign.

## Compute the actual quantities

- Binary loss: mean `-y*log(p)-(1-y)*log1p(-p)`, probabilities for class 1, float64 endpoint clipping.
- Candidate selection: first validation minimum. Neural epoch choice here: last epoch at minimum validation loss.
- Corrected TabM-mini: only first R adapter; shared backbone biases; independent heads; mean member training loss and mean probability prediction.
- Dataset rank: mean seeds within dataset, rank those means, then average five dataset ranks. `rank(mean)` differs from `mean(rank)`.
- Paired uncertainty: subtract aligned model/baseline errors; average seeds; bootstrap whole dataset means. Five datasets remain five blocks.
- Lifecycle model: preparation + batches × measured batch prediction. An illustration of 100+B versus 10+4B crosses at 30 batches.

## Preserve model identity

The primary seven-arm panel uses fresh corrected TabM plus six archived arms. The original 105-result archive is immutable; an eight-arm diagnostic retains the incorrect historical mini variant under its original identity. Original-package replay independently verifies the five pretrained arms; archive-score reconstruction alone is weaker evidence.

TabPFN-3 compresses circular feature groups to row embeddings before ICL and retrieves class mass from context labels. Its class-count-independent decoder algebra does not remove the released 160-class ceiling or identify an unseen semantic label. TabICLv2 uses mixed-radix embedding views and hierarchical ICL for many-class tasks.

## Interpret the intervention

Erase the most train-correlated feature with a training-median fallback, keeping the fitted models and all other test values fixed. Positive loss change means damage. Negative change is retained, not filtered. This is a feature-loss stress test, not causal attribution or retraining after deletion.

## Source and reproduction boundary

[TabPFN-3 §§2.1–2.4, C, E](https://arxiv.org/html/2605.13986v1), [TabICLv2 §3, A, J–K](https://arxiv.org/html/2602.11139v1), [TabM §3.3/5.1](https://arxiv.org/html/2410.24210v3). Local five-table binary log loss does not reproduce TabArena AUROC/8-fold bagging, TALENT classification accuracy, full synthetic pretraining or relational benchmarks. See the full reproduction contract for commands and exact omitted work.
