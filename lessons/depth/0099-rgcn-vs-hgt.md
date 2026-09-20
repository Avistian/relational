## The big picture · turn an architecture comparison into a causal question

<div class="learning-route"><p class="route-kicker">THE BIG PICTURE · 099 / CONTROLLED MODEL CONTRASTS</p><p><strong>Bring forward:</strong> L060 compared complete learning procedures; L091/L093 defined the operators. <strong>Follow:</strong> common information contract → candidate fits → validation selection → frozen test predictions → limited attribution. <strong>Carry forward:</strong> L100 repeats this discipline when sampling introduces another moving part.</p><p><a href="0060-broad-model-comparison.html">L060: fair procedures</a> · <a href="0091-r-gcn.html">L091: relation means</a> · <a href="0093-hgt.html">L093: typed attention</a> · <a href="0100-heterogeneous-gnn-checkpoint.html">L100: sampled training</a> · <a href="../reference/0091-0100-model-map.html">Sequence map</a></p></div>

[[DEPTH_MAP:099-comparison]]

### Name the contrast before reading the score

“Does HGT beat R-GCN under this recipe?” compares two complete systems. They differ in normalizers, transformations, nonlinearities, residuals, normalization layers and parameter counts. “Does the learned scorer help within this HGT design?” is narrower: compare HGT with the declared uniform-HGT control. The MLP asks a different question again: what can the paper's own words predict without graph context?

These controls do not provide perfect causal identification of an abstract mechanism. Training trajectories adapt after the intervention. They do, however, specify what was changed more precisely than a comparison of names. Read [R-GCN Eq. 2](https://arxiv.org/html/1703.06103v4) and [HGT §3](https://arxiv.org/html/2003.01332v1) beside the concrete course operators, including their declared release differences.

### Follow the same target through four routes

Every arm starts with the same paper identity and feature row. The MLP processes that row alone. The graph encoders also use author and subject connections. In two layers, another paper can influence the target through a shared author: first paper→author, then author→paper. This is the concrete additional information that a graph model may exploit under this static transductive contract.

Conference links would directly reveal the target class and are excluded. Subject links are included as declared context; their predictive strength must still be interpreted in the context of the task. A high score does not demonstrate that all relational domains have equally informative available neighbors. This is where the earlier baseline lessons remain relevant: a relational claim needs a credible flat-table comparator and an honest information-access statement.

### Separate normalization from attention

Take two author messages 2 and 8, plus one subject message −4, ignoring learned transforms and self terms. R-GCN's sum of relation means is 1. Uniform HGT's global incoming mean is 2. Attention weights proportional to `[1,3,2]` give 3. These are different operators before fitting anything.

Now duplicate **all** author messages, keeping the subject once. R-GCN still gives 1. Uniform HGT becomes `(2+8+2+8−4)/5=3.2`. If copied author edges retain their original logits, attention's numerator becomes `2+24+2+24−8=44`, denominator `1+3+1+3+2=10`, and the result becomes 4.4. Duplicating one relation changes its total competitive mass under cross-relation softmax. This is a diagnostic multigraph calculation, not a claim that the released ACM graph contains duplicate edges.

### Selection is itself part of the learning procedure

An epoch is selected using validation loss; a learning rate is selected using the declared aggregation across seeds. Only then are test predictions scored. Taking the maximum test score over epochs or rates would create a new procedure with test-label access. More search can help an arm even when its architecture stays fixed, so matching two tested rates is an explicit budget choice rather than a universal fairness guarantee.

<details><summary>Try first: if HGT beats R-GCN but ties uniform HGT, what can you conclude?</summary><p>The complete HGT procedure may help under this protocol, but the contrast does not isolate a benefit from learned attention scoring. Other retained components or optimization effects remain possible explanations. A point-estimate tie is also not proof of equivalence; report the paired results and uncertainty.</p></details>

**Bridge to L100.** Keep one graph as one statistical dataset, regardless of the number of seeds. The next checkpoint adds sampled context, target-order control and mini-batch optimization. Predict which conclusions would survive that change and which require a new run. The existing evidence and reproduction ledger below remain the authority for what was actually executed.
