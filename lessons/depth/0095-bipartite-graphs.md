## The big picture · recommendation needs a path back to items

<div class="learning-route"><p class="route-kicker">THE BIG PICTURE · 095 / TYPED INTERACTIONS</p><p><strong>Bring forward:</strong> L092 separated path instances from endpoint projections. Here users and items occupy different identity spaces. <strong>Follow:</strong> fitting interactions → normalized transitions → three-hop item scores → seen-item mask → ranking metric. <strong>Carry forward:</strong> L097 replaces this fixed scorer with a learned pairwise scorer while keeping candidate eligibility explicit.</p><p><a href="0092-meta-paths.html">L092: projected neighborhoods</a> · <a href="0087-link-prediction.html">L087: hidden edge boundaries</a> · <a href="0097-negative-sampling.html">L097: learned ranking</a> · <a href="../reference/0091-0100-model-map.html">Sequence map</a></p></div>

[[DEPTH_MAP:095-walk]]

### Start with the output shape

For U users and I items, a recommender must eventually produce scores indexed by `(user,item)`. A user–user matrix is not that output, even when it captures useful similarity. This shape check forces an explicit final operation that returns from similar users to candidate items. Keep the types beside every multiplication: `[U,I] × [I,U] × [U,I] → [U,I]`.

The first transition chooses an item associated with the starting user. The second chooses another user associated with that item. The third chooses an item associated with that user. Intermediate identities remain meaningful, and degree normalization decides how much probability each step contributes. The diagram shows a deterministic baseline: its counts are fitted state, but it has no learned neural weight matrices or training loss.

### Work through the mass, then apply eligibility

Use two users and three items: u0 likes i0/i1; u1 likes i1/i2. Starting from u0 gives first-hop probabilities `[1/2,1/2,0]`. The first item returns to u0 with certainty; the shared item divides its mass between u0 and u1. After two steps, user probabilities are `[3/4,1/4]`. One more user-to-item step gives `[3/8,1/2,1/8]`.

Those scores sum to one before masking. Masking the two seen items leaves only i2 eligible, with score 1/8. Ranking does not require renormalizing the remaining mass to one. If you do renormalize for presentation, do not mistake the new number for a calibrated purchase probability. The model describes a walk on observed likes, not an exposure or causal preference model.

Now compare a two-hop user projection. `B Bᵀ` gives `[[2,1],[1,2]]`. It preserves shared-item counts, but the value 1 does not name the shared item. A high-degree item creates many user pairs and can inflate projection size. Preserving the bipartite graph keeps the normalization at each actual step inspectable.

### One interaction has several representations

A raw rating row has user, item, rating and timestamp. The fitting graph retains likes under the stated threshold; the seen mask retains every fitting-rated item, including dislikes. These structures answer different questions. A low rating may contribute no positive walk edge while still excluding an item from recommendations. Derive both from the same fitting partition, not from the full release.

The [GroupLens README](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt) defines the release and split files. It does not declare this course's like threshold, mixture-selection rule or ranking protocol. Keep released facts distinct from our experimental choices. Likewise, a successful official-split audit does not establish a valid future-purchase evaluation.

<details><summary>Predict first: what if a held-out like remains only in the reverse store?</summary><p>The graph still contains the withheld fact. A message or walk can use its reverse direction even if the forward entry was removed. Split raw interaction identities first and generate both directions from the allowed fitting rows.</p></details>

**Bridge to L096 and L097.** Database association rows may represent repeated events, quantities or timestamps; collapsing them to one binary edge loses those distinctions. A learned ranking objective then chooses which unobserved pairs to compare against positives. Before either extension, defend the current scorer's cold-user fallback, catalog mask and metric denominator. Ask the teacher to change one of those assumptions and predict which number in the trace moves.
