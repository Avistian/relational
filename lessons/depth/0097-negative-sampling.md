## The big picture · the sampler helps define the objective

<div class="learning-route"><p class="route-kicker">THE BIG PICTURE · 097 / LEARNING FROM COMPARISONS</p><p><strong>Bring forward:</strong> L095 fixed the fitting graph and recommendation candidates. Now learn a scorer from positive–unobserved comparisons. <strong>Follow:</strong> positive pair + conditional negative draw → shared scorer → score difference → pairwise loss → full-catalog ranking. <strong>Carry forward:</strong> L098 samples context instead of supervised comparisons; these are separate forms of sampling.</p><p><a href="0095-bipartite-graphs.html">L095: candidate eligibility</a> · <a href="0087-link-prediction.html">L087: encoder and decoder</a> · <a href="0098-hetero-mini-batching.html">L098: sampled context</a> · <a href="../reference/0091-0100-model-map.html">Sequence map</a></p></div>

[[DEPTH_MAP:097-bpr]]

### Follow the shared parameters

There are two score evaluations but one scorer. The user vector appears in both, and the positive and negative items are rows of the same item table. Write the difference as `Δ = U_u · (V_i − V_j) + b_i − b_j`. This makes the comparison explicit: the loss trains a relative ordering for one user. A user-only additive bias would cancel from this difference, which is why it would not help distinguish that user's two items.

The [BPR paper, §4 and §5.1](https://arxiv.org/abs/1205.2618) supplies the pairwise-ranking rationale and matrix-factorization handoff. The course experiment's optimizer, regularizer scaling and candidate policies are documented separately below. “Uses the BPR loss” is a narrower claim than “reproduces LearnBPR.”

### Differentiate one pair before training many

Take a scalar user vector `U=2`, positive item `V_i=1`, negative item `V_j=.5`, and zero biases. Scores are 2 and 1, so Δ=1 and the data loss is `softplus(-1)≈.3133`. The derivative with respect to Δ is approximately −.2689. Therefore the derivatives with respect to U, V_i and V_j are approximately −.1345, −.5379 and +.5379. Gradient descent raises the positive item vector and lowers the negative item vector in this example.

The general user gradient is `-sigmoid(-Δ) × (V_i−V_j)`. Its direction depends on which negative was sampled. Changing q therefore changes more than computational cost: it changes the weighting of update directions. These numbers omit the declared regularizer to isolate the data term, and use one pair rather than a batch mean.

### Make the target expectation explicit

Suppose the eligible candidates are a and b, and their pair losses are .2 and .8. Uniform sampling has expected loss .5. Sampling probabilities `.9,.1` give .26. Neither number is a biased calculation of its own declared expectation. They are different expectations. To estimate the uniform target with the latter sampler, importance weights are `.5/.9` and `.5/.1`; the weighted expectation returns .5, with potentially worse variance.

Coverage matters. If q assigns zero probability to b while the target assigns positive probability, no finite importance weight can recover b's contribution from those samples. A hard-negative selector adds another issue: its distribution changes with the current model. The lab deliberately compares uncorrected policies and labels them as different training objectives.

### Keep the evaluation question fixed

At inference, freeze the scorer and rank every eligible item under the main protocol. Training may have seen only one comparison item at a time, but that does not justify evaluating only one comparison. Conversely, evaluation candidates must not flow backward into the fitting exclusion mask unless the experiment explicitly allows that information. An unobserved item can later become a held-out positive; that uncertainty is part of implicit feedback.

<details><summary>Predict first: can an item with the smallest score still be a harmful negative?</summary><p>Yes. Its true preference is unobserved, so an easy negative can still be a false negative. “Hard” describes the current scorer's difficulty, not the correctness of the assumed comparison. A high score increases the pairwise loss but does not certify a negative label.</p></details>

**Bridge to L098.** Write three different random variables: the negative item j, the sampled message neighborhood, and the evaluation candidate subset. They affect the objective, encoder computation, and reported metric respectively. Keeping their seeds equal does not make them the same intervention. Reproduce the scalar gradients tomorrow, then explain which table rows receive updates and why the same trained weights can produce different sampled-catalog metrics.
