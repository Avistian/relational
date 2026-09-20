## The big picture · defend the whole prediction procedure

<div class="learning-route"><p class="route-kicker">THE BIG PICTURE · 100 / INTEGRATION AND DEFENSE</p><p><strong>Bring forward:</strong> L091–L093 supply encoders; L096 supplies identity; L098 supplies batches; L099 supplies comparison discipline. <strong>Follow:</strong> legal graph → paired sampled dependencies → seed-only learning → validation selection → frozen prediction → written claim. <strong>Carry forward:</strong> temporal learning adds availability to every boundary already tested here.</p><p><a href="0096-multi-relational-data.html">L096: typed keys</a> · <a href="0098-hetero-mini-batching.html">L098: seed boundaries</a> · <a href="0099-rgcn-vs-hgt.html">L099: model controls</a> · <a href="0070-foundation-model-checkpoint.html">L070: evidence defense</a> · <a href="../reference/0091-0100-model-map.html">Sequence map</a></p></div>

[[DEPTH_MAP:100-checkpoint]]

### Build an argument whose links can fail independently

Start with the final claim: one complete procedure obtained a measured score on fixed ACM targets. To defend it, recover the target IDs, reconstruct the allowed graph, identify the sampled inputs, trace the actual encoder, show which labels entered the loss, and reproduce the selected model's predictions. A correct last metric cannot repair an incorrect earlier identity map.

This is the same standard as [L040's exit exam](0040-year-1-exit-exam.html) and L070's evidence chain, now applied to typed message passing. It is valid to conclude that graph context or attention did not help enough under this protocol. The checkpoint rewards a reconstructable claim, not a mandatory neural win.

### Trace one training seed across the interfaces

Select global paper ID 17. In its batch it occupies seed row 0. An author with global ID 2 may occupy author row 1, so a local author→paper edge `(1,0)` refers to `(author 2,paper 17)`. The author map and paper map are different functions. The encoder turns local typed features into hidden states; the classifier produces paper logits; seed row 0 contributes one training loss.

After selection, evaluate the frozen model on the full allowed graph and gather test IDs. Training batch positions are not persistent identities and must not become evaluation keys. The native [NeighborLoader contract](https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/loader/neighbor_loader.html) and the source-pinned course checks make these handoffs explicit. The course sampler is not the HGT paper's HGSampling algorithm.

### Understand why a gradient audit can pass while training differs

Use a scalar parameter θ and two losses: `L₁=.5(θ−1)²`, `L₂=.5(θ+1)²`. Their equally weighted mean has gradient θ. At θ=0, one full-mean gradient step leaves θ at zero.

Now apply separate SGD steps of size .1. The first loss moves θ from 0 to .1. The second loss then has gradient 1.1, moving θ to −.01. Both per-example gradients were correct. They were evaluated at different parameters. This tiny teaching example isolates the same distinction as the checkpoint: an accumulated-gradient audit holds weights fixed, whereas ordinary mini-batch training updates between batches. Adam adds its own state to that difference.

The actual epoch also has unequal batch sizes. With 804 seeds and batches 256/256/256/36, the final batch's correct weight in a reported per-seed mean is `36/804≈.0448`, not .25. Correct reporting does not imply that the optimizer used the same trajectory as full-batch training. Keep objective accounting, gradient parity and optimization trajectory as three separate claims.

### Specify what every audit rules out

An ID round trip rules out a particular remapping error. A context-label perturbation rules out direct use of those labels in the seed loss. A full-neighbor forward comparison checks sampled dependency completeness under the audited operators. A fixed-weight gradient comparison checks objective weighting and differentiation. None of them establishes finite-fanout unbiasedness, historical paper reproduction, or general superiority across datasets.

The architecture sources remain [R-GCN](https://arxiv.org/html/1703.06103v4) and [HGT](https://arxiv.org/html/2003.01332v1); their published experiments remain distinct from this combined course comparison. Preserve every `INCOMPARABLE`, `NOT_ESTABLISHED` and `NOT_RUN` boundary in the ledger.

<details><summary>Try first: what would make the final claim too broad?</summary><p>Claiming that HGT is generally better from three seeds on one static graph; calling a uniform-HGT contrast an isolated proof about attention without its optimization limits; or calling this native-sampler course run a reproduction of HGT's full paper protocol. State the observed comparison and its conditions instead.</p></details>

**Your defense handoff.** Submit a trace for one typed edge, a derivation of the seed-weighted objective, and one paired result whose interpretation you could defend if its sign reversed. Then name the additional temporal availability checks needed for the next curriculum stage. The prepared artifacts support that defense; learner mastery remains **PENDING_WRITTEN_DEFENSE** until you write it.
