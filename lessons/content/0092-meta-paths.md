## The skill you will leave with

**Define a typed route, turn it into a neighborhood, and explain both attention decisions in HAN.** Your deliverable is a trained ACM model with measured PAP/PSP weights and a defensible reproduction report. This connects the course mission—learning from relationships—to a precise choice: which relational routes should supply a prediction?

Take this in three sittings if needed: retrieval and the worked trace; implementation; then the full experiment and written defense. Reading the page is preparation. Mastery remains **PENDING_WRITTEN_DEFENSE**.

## 1 · Retrieve before reading

[[WARMUP]]

Without opening earlier lessons, write three answers: what does a GAT attention coefficient normalize over? Why does R-GCN assign different matrices to different edge roles? What makes a graph experiment transductive? Then check [L084 GAT](0084-gat.html), [L091 R-GCN](0091-r-gcn.html), and [L090 checkpoint](0090-gnn-checkpoint.html).

**Bridge from L091.** R-GCN learns how each edge type changes a message. Two layers can compose those messages through an intermediate node. But that computation also mixes the intermediate node's other incoming information. Today's choice is different: specify a complete typed route first, connect its endpoints, then attend over the resulting neighbors. This does not make HAN universally better. It makes the route itself an explicit modeling assumption.

Primary reading: [Wang et al., HAN, §§3–4 and Figure 2](https://arxiv.org/html/1903.07293v2#S4). Read the mechanism first, then §§5.3–5.4 and Table 3 when auditing the lab. The [authors' release](https://github.com/Jhy1993/HAN/tree/71bac29a07fb8fab908d50a806a7bc38aa6c6611) is a second primary source; it does not agree with every paper equation.

## 2 · A meta-path is a template, not one walk

> **In plain terms.** “Papers that share an author” and “papers that share a subject” are two ways to define a paper's neighbors. The endpoint type is identical; the meaning is different.

A **heterogeneous graph** contains multiple node or edge types. A **meta-path** specifies a sequence of those types and relations. Here P means paper, A author, and S subject. **PAP** follows paper→author→paper. **PSP** follows paper→subject→paper. A **path instance** is one concrete sequence, such as paper p0→author a1→paper p2. A meta-path-based neighbor is a reachable endpoint. HAN's endpoint graph does not retain each intermediate author as a separate message.

**Worked example.** Let p0 have authors a0 and a1, p1 have a0 and a1, and p2 have a1 only. There are two PAP walks from p0 to p1, but p1 is one distinct neighbor. All three papers share a1, so the PAP endpoint graph connects every pair in this tiny example. Self is reachable too: p0→a0→p0.

Write the paper-author incidence matrix B with paper rows and author columns:

```text
       a0 a1
p0      1  1
p1      1  1
p2      0  1

B Bᵀ = [[2,2,1],       A_PAP = 1[B Bᵀ > 0] = [[1,1,1],
        [2,2,1],                              [1,1,1],
        [1,1,1]]                              [1,1,1]]
```

Matrix multiplication counts matching intermediates. Thresholding makes the result a reachability relation. Keeping counts would introduce a different message weighting. In database terms, this resembles a self-join through author IDs followed by `DISTINCT` on endpoint pairs. The shared-author route is specified by you; HAN learns weights within and between the routes you supply.

[[PATH_FIG]]

**CHECK before proceeding.** If p0 and p1 acquire a third shared author, does their binary PAP edge change? No. If p2 loses a1, does its neighborhood change? Yes. These interventions distinguish multiplicity from reachability. The lab makes you implement this distinction.

> **Scope check.** ACM's distributed artifact already contains PAP and a matrix named PLP. The authors' README says PLP represents PSP. The lab audits these endpoint matrices and practices typed multiplication separately; it cannot recover the discarded author/subject incidences from an endpoint adjacency matrix.

## 3 · First attention: which neighbor within this route?

> **In plain terms.** First choose the route. Then ask how much each reachable paper should contribute to this paper's representation.

A **feature vector** xᵢ lists the input measurements for node i. For ACM these are keyword indicators. A head projects that vector into eight learned coordinates, hᵢ = W xᵢ. A **head** is one independently parameterized attention calculation. Different heads can learn different notions of usefulness.

For receiver i and sender j, compute a score eᵢⱼ = LeakyReLU(a_leftᵀhᵢ + a_rightᵀhⱼ + c_left + c_right). LeakyReLU leaves positive values unchanged and multiplies negative values by 0.2. The affine offsets c are present in the released implementation. Only j in the chosen route's neighborhood is eligible.

Normalize with a **softmax**: exponentiate scores and divide by their sum over eligible senders. Thus αᵢⱼ = exp(eᵢⱼ)/Σₖexp(eᵢₖ). Before dropout, each receiver's coefficients sum to one. The head returns ELU(Σⱼαᵢⱼhⱼ + bias). ELU leaves positive inputs unchanged and maps a negative input t to exp(t)−1. This is the GAT operation from L084, applied to one meta-path graph.

**Worked example.** Suppose p0's three eligible messages have scalar values [1,3,5] and pre-softmax scores [0,ln 2,0]. The exponentials are [1,2,1], so α=[0.25,0.50,0.25]. The weighted message is 0.25×1 + 0.50×3 + 0.25×5 = 3. Increase the last score to ln 4: α=[1/7,2/7,4/7], and the message becomes 27/7≈3.857. Features and reachability stayed fixed; only the neighbor score changed.

[[NODE_FIG]]

The runnable implementation uses sorted `(receiver, sender)` indices, a numerically stable per-receiver softmax, then sparse matrix multiplication. A separate dense masked-softmax oracle checks its outputs **and gradients**. An empty row would return zero, but the ACM loader includes one self edge for every paper.

**Release detail.** Dropout randomly zeros values and rescales survivors during training. The released head applies dropout at three sites: input features, attention coefficients, and projected values. The rate is 0.6; inference disables all three. Individual dropped coefficient rows need not sum to one. Each path/head has its own parameters, shared across that head's node/edge computations.

## 4 · Model architecture: preserve the two meanings until fusion

[[ARCH_FIG]]

Follow p0 through the diagram. Its same-author and same-subject neighborhoods are different supports. Eight heads on PAP concatenate into zᵖᴬᴾ₀ with 64 coordinates. Eight other heads on PSP produce zᵖˢᴾ₀. **Concatenation** places head outputs next to each other; it does not average them. The stacked tensor Z has shape [3,025 nodes, 2 paths, 64 coordinates].

The semantic scorer has parameters Wₛ [64,128], bₛ [128], and q [128]. For each node/path vector it computes sᵢᵖ = qᵀ tanh(zᵢᵖ Wₛ + bₛ). The scorer parameters are shared across paths and nodes, so its two scores are comparable within the learned representation. A weighted sum across the two path embeddings returns one 64-vector per paper. A linear 64→3 classifier produces three **logits**, unnormalized class scores.

The training objective is cross-entropy on the 600 training labels plus the released L2 penalty. Cross-entropy penalizes low predicted probability for the correct class. Validation labels choose the checkpoint; they do not enter gradient updates. The complete graph and all keyword features are visible during training, so this is **transductive** learning. That is the declared protocol, not an inductive test on unseen graphs.

Two inference handoffs must remain distinct: classifier argmax gives direct test accuracy; the frozen 64-dimensional embeddings feed the Table 3 KNN evaluation. The KNN stage fits a separate classifier and uses additional labeled probe examples. Its F1 score is not the neural head's accuracy.

## 5 · Second attention: which route—and whose weights?

> **In plain terms.** Once each route has produced a representation, combine those representations. The paper and original code disagree about whether that route mixture must be identical for every node.

**Paper equation (7).** Average the semantic scores over nodes first: wᵖ = (1/N)Σᵢsᵢᵖ. **Equation (8)** applies softmax across paths, yielding βᵖ. **Equation (9)** computes zᵢ = Σₚβᵖ zᵢᵖ. Every node shares the same β, although each node's embeddings differ. A remote feature change can therefore affect another node's mixture through that average.

**Released `SimpleAttLayer`.** The [pinned code](https://github.com/Jhy1993/HAN/blob/71bac29a07fb8fab908d50a806a7bc38aa6c6611/utils/layers.py) applies softmax directly to each node's path scores. It computes βᵢᵖ, without averaging over nodes. The mean of these per-node weights is a useful descriptive summary, but it is not the globally shared coefficient from equation (8).

[[PREDICT]]

**Worked example.** Two nodes have path scores [[2,0],[0,0]]. Paper-global averaging gives [1,0] and shared weights [0.731,0.269]. Released per-node softmax gives [0.881,0.119] for the first node and [0.5,0.5] for the second. Averaging these weights yields [0.690,0.310]. **Softmax of the mean and mean of softmax are different operations.**

[[SEMANTIC_FIG]]

[[SEMANTIC_WIDGET]]

The lab supports explicit modes `paper_global` and `release_node`; no silent toggle hides this distinction. Task C implements both, and the actual model calls that live function. The full reproduction lane follows `release_node` because its contract is the released training program. A small paired intervention shows the effect of changing this mode on the same weights; it is not a trained ablation or evidence that one variant wins.

**Interpretation boundary.** If PAP has the highest average weight, report that fact for this model, split, and seed. It does not establish that shared authors cause a paper's research area. Correlated routes can substitute for each other, embeddings have learned scales, and high weight alone is not a performance ablation. To test predictive reliance, remove a path, refit under the same selection protocol, and compare held-out scores across declared seeds.

## 6 · Reproduce the named experiment, not a familiar-looking score

The target is **HAN Table 3, ACM, all four KNN training fractions**. A 20% fraction here refers to the downstream probe subset, not “train HAN on 20% of all graph nodes.” The released main program trains on its fixed 600 labels, restores a checkpoint, extracts embeddings for the 2,125 test IDs, and passes those embeddings to `my_KNN`. That routine shuffles the available embeddings, fits 5-nearest-neighbor classification on the first fraction, and scores the remainder. A **nearest-neighbor classifier** predicts a class from the five closest stored feature vectors, using Euclidean distance and a majority vote here.

**Micro-F1** pools class counts before computing F1; for single-label multiclass prediction it equals accuracy. **Macro-F1** averages each class's F1 equally. Both are preserved, along with every probe split and prediction. Ten repeated probe splits reuse the same nodes and frozen embedding; they are not ten independent graph datasets or ten independent encoder fits.

| Protocol item | Executable lesson contract |
|---|---|
| Source | Jhy1993/HAN revision `71bac29a07fb8fab908d50a806a7bc38aa6c6611` |
| Data | Hash-verified DGL ACM3025 mirror; raw 1,870-dimensional features; binary PAP/PSP including self |
| Split | Released 600 train / 300 validation / 2,125 test IDs |
| Model | 2 paths × 8 heads × 8 coordinates; semantic width 128; three-class linear head |
| Optimizer | Adam 0.005; 0.001×½Σθ² over all parameters; dropout 0.6 |
| Schedule | 200-epoch ceiling; patience 100; release OR-reset / AND-save checkpoint rule |
| Evaluation | Frozen test embeddings; KNN k=5; fractions 0.2/0.4/0.6/0.8; ten cumulative shuffles per fraction |
| Randomness | Declared encoder seed and separate probe seed; historical RNG states unavailable |
| Acceptance | Descriptive ±2 percentage-point tolerance fixed before execution; protocol verdict remains separate |

**Why checkpoint detail matters.** The release resets patience when validation accuracy improves **or** validation loss improves. It saves weights only when both conditions hold at the same epoch, with equality allowed. That differs from selecting minimum validation loss alone. The code records the selected epoch and the entire validation trace; test labels never select an epoch.

**Why regularization detail matters.** The source excludes a list of short variable names, but TensorFlow variable names carry scopes and suffixes. The port explicitly regularizes all learned parameters, including biases. It uses Glorot-uniform projection/readout initialization and normal semantic parameters with standard deviation 0.1, following the released initializers. PyTorch's optimizer/random-number implementation still differs from TensorFlow 1.

[[RESULTS]]

> **Scope check.** This is a full-size released-protocol reconstruction, not established full-paper reproduction. The original MAT file's byte identity with the DGL mirror is unverified. The release's semantic reduction and stopping rule differ from the paper. The historical seeds, exact paper checkpoint, and table-to-release mapping are unavailable. DBLP, IMDB, all baselines, clustering, and paper-wide ablations remain **NOT_RUN**. Historical parity is **INCOMPARABLE** and full-paper parity **NOT_ESTABLISHED**, even if numerical scores are close.

**Independent replay.** The complete inline notebook implementation also ran all 200 epochs and forty probes from a fresh directory. Predictions, checkpoint choice and KNN results matched exactly; the largest validation-loss difference was 0.0000066. Bitwise loss equality failed. [Replay audit](../labs/_replay_l092_results.json) records that difference. The downloadable solution has separately regenerated diagnostic outputs with the full switch off; its saved outputs do not pretend to contain the earlier full run.

[Protocol/deviation ledger](../labs/l092-reproduction.md) · [Source and data hashes](../labs/_sources_l092.json) · [Raw results](../labs/_paper_l092_results.json) · [Behavior checks](../labs/_verify_l092_results.json)

## 7 · Lab: make the trained computation depend on your code

Open the [student notebook](../labs/0092-meta-paths.ipynb) or [prepared HTML](../labs/html/0092-meta-paths.html). The notebook includes this explanation and portable figures. All model, loader, trainer, and evaluation code is visible.

1. **TODO A — reachability.** Compose typed incidence matrices and collapse positive walk counts to distinct endpoints. CHECK distinguishes duplicate walks from new neighbors.
2. **TODO B — node attention.** Normalize scores within receiver rows and aggregate the correct sender values. CHECK compares exact arithmetic and gradients against an independent dense implementation.
3. **TODO C — semantic fusion.** Implement both reduction orders. CHECK exposes the difference and verifies that parameters receive gradients.
4. **RUN — smoke, then full.** The small diagnostic establishes wiring. Enable the separate full-run switch for the 200-epoch target and all forty KNN evaluations. A disabled switch prints `NOT_RUN`.
5. **EXIT — defend the claim.** Submit the JSON plus a 150–250 word explanation of reachability, both softmax axes, which path received greater weight, and the remaining reproduction gaps.

Command from the course root:

```bash
.venv/bin/python labs/_verify_l092.py
.venv/bin/python labs/_run_l092.py --seeds 0 --output labs/_paper_l092_results.json
```

The CPU run uses the full graph and model. Runtime and exact environment are recorded in the artifacts. Repeating the command retrains; it does not quietly load an earlier score. Extra `--seeds 0 1 2` runs measure initialization variability; the primary one-fit result has no encoder-seed error bar.

## 8 · Defend, then connect forward

[[TEACHBACK]]

**Exit challenge.** Explain why averaging two R-GCN layers is not generally the same computation as constructing PAP then applying one GAT layer. Name what the intermediate node can mix and what the endpoint projection discards. Then explain why a learned route weight does not by itself justify choosing that route for a production database.

**Next: L093 HGT.** HAN starts from a hand-specified set of multi-hop routes. HGT instead makes attention depend on source type, edge type, and target type. Later, [L141 in the curriculum](../reference/curriculum.html) revisits direct routes through relational databases. Before then, preserve the distinction between the schema template, its instances, the endpoint graph, and the learned weight.

Return tomorrow and derive both attention axes without notes. In a week, implement the three tasks again from the contracts. Ask the teacher follow-up questions or paste your written defense for a strict review; an executed notebook alone does not pass the lesson.
