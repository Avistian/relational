## The big picture · a route is an input assumption

<div class="learning-route"><p class="route-kicker">THE BIG PICTURE · 092 / ROUTE-CONDITIONED REPRESENTATIONS</p><p><strong>Bring forward:</strong> R-GCN preserves individual edge roles; GAT learns neighbor weights. HAN first specifies which complete routes count as neighborhoods, then makes two attention decisions. <strong>Follow:</strong> route graph → per-route heads → semantic fusion → prediction. <strong>Carry forward:</strong> HGT composes typed one-hop operations without requiring these endpoint graphs.</p><p><a href="0091-r-gcn.html">L091: typed layers</a> · <a href="0084-gat.html">L084: neighbor attention</a> · <a href="0093-hgt.html">L093: typed attention</a> · <a href="../reference/0091-0100-model-map.html">Sequence map</a></p></div>

[[DEPTH_MAP:092-han]]

### Three representations of the same route

Write `paper → author → paper` at the top of a sheet. Under it, write a concrete walk `p0 → a7 → p3`. Under that, write the endpoint edge `(p0,p3)`. The first line is a schema template, the second an instance, the third an input edge for a path-specific encoder. Confusing these levels makes it easy to claim that the neural model sees intermediate authors when it actually sees only paper endpoints.

With paper-author incidence B, `B Bᵀ` counts matching author instances. Replacing positive counts with one collapses that information. If two papers share two authors, the binary edge is the same as if they share one. This is a deliberate representation choice. An attention coefficient learned later cannot recover which discarded author supplied the route. See [HAN §3–4 and Figure 2](https://arxiv.org/html/1903.07293v2).

### Follow one paper through both softmax axes

For a fixed paper i and fixed path p, choose its eligible endpoint neighbors. Each head projects their features, scores pairs, and normalizes over those neighbors. Eight eight-coordinate outputs concatenate into 64 coordinates. This happens separately for PAP and PSP, producing two vectors for the same paper. The second softmax chooses how to combine those **two vectors**, not which raw author or subject to select.

Try a two-coordinate fusion trace. Let `z_PAP=[2,0]` and `z_PSP=[0,4]`. Semantic weights `[.75,.25]` give `z=[1.5,1]`. A teaching classifier with weight columns `[1,-1]` and `[-1,1]` gives logits `[.5,-.5]`. The first class probability is approximately .731. Neighbor attention, semantic attention and class softmax normalize over three different sets. None can substitute for another merely because each uses exponentials.

### The reduction order changes the dependency graph

The paper's semantic scorer averages over nodes before applying the path softmax. The pinned release applies the path softmax separately to each node. That difference changes whose input can affect whose prediction. Under the paper-global rule, changing an otherwise distant paper can change the shared path mixture. Under the released node-wise rule, semantic fusion alone introduces no such global dependence, though ordinary graph paths can still transmit information.

This connects to [L047 SAINT](0047-saint.html): ask which companion examples can affect a prediction, and identify the exact operation that permits it. It does not mean HAN and SAINT have the same mechanism. Here the relevant reduction is over node scores; in SAINT the interaction is row attention. Always specify the evaluation graph and the chosen semantic mode.

<details><summary>Predict first: can moving the averaging operation be harmless?</summary><p>It is harmless for certain special score patterns, such as every node having the same pair of path scores. In general softmax is nonlinear, so averaging scores before softmax differs from averaging node-wise softmax weights. Agreement on a symmetric fixture is insufficient; use unequal scores and perturb one node.</p></details>

### Separate the representation from the reported endpoint

The class head trains the encoder, but the released table-oriented evaluation extracts frozen embeddings and fits KNN probes. A good classifier score is therefore not automatically a reproduction of the KNN table. Preserve the distinction between encoder training labels, validation checkpoint selection, and downstream probe partitions. Read the existing protocol below beside the [pinned `SimpleAttLayer`](https://github.com/Jhy1993/HAN/blob/71bac29a07fb8fab908d50a806a7bc38aa6c6611/utils/layers.py).

**Bridge to L093.** HAN cannot attend through an omitted meta-path graph. HGT changes the route construction strategy by learning typed edge computations whose stacks compose walks. Before moving on, explain which author identities HAN's projection discards and which attention axis uses the endpoint features. Ask the teacher to challenge your explanation before treating familiarity with the figure as mastery.
