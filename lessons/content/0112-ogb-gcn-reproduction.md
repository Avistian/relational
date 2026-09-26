<p class="stream-kicker">One claim · one frozen protocol · ten fresh fits</p>

Your tangible win is to **reproduce and defend one published OGB result**: a full-batch GCN on `ogbn-arxiv`. By the end you should be able to trace one prediction, explain which information entered training, and decide whether the resulting ten-run mean supports a bounded reproduction claim.

Read sections 1–5 in about 25 minutes; then work through the three live notebook tasks. Full training is a separate compute session. Author execution does not establish your mastery: **PENDING_WRITTEN_DEFENSE** until you submit the EXIT explanation.

[Student notebook](../labs/0112-ogb-gcn-reproduction.ipynb) · [Executed solution](../labs/solutions/0112-ogb-gcn-reproduction.ipynb) · [Readable lab](../labs/html/0112-ogb-gcn-reproduction.html) · [Reference](../reference/ogb-reproduction.html) · [Exact reproduction protocol](../labs/l112-reproduction.md)

## 1 · Start from the question, before the score

Imagine that a colleague hands you a test accuracy of 71.9%. Is this a reproduction? You cannot tell yet. The number needs an identity: dataset bytes, split indices, model, optimization schedule, selection rule, and aggregation across runs. Two equal scores can come from different experiments.

The named target is **Hu et al., OGB, Table 6, GCN on ogbn-arxiv**: test accuracy **71.74 ± 0.29%**, validation **73.00 ± 0.17%**. The original graph is made undirected for this baseline. Read [§4.3 and Table 6](https://arxiv.org/html/2005.00687v6#S4.SS3) first; consult the [released implementation](https://github.com/snap-stanford/ogb/blob/61e9784ca76edeaa6e259ba0f836099608ff0586/examples/nodeproppred/arxiv/gnn.py) for the operational settings.

We freeze a **0.5 percentage-point absolute gap in the mean** as our descriptive closeness threshold before the runs. A 72.10% mean would be CLOSE because the gap is 0.36 pp. A 72.40% mean would be OUTSIDE_TOLERANCE even though it is higher. This is a reproduction criterion, not a significance or equivalence test. We compare variability separately rather than forcing the new standard deviation to equal the historical one.

**Retrieve before continuing:** what went wrong in lesson 110 if weights were restored without all the relevant state? For this static GCN, there is no event memory, but batch-normalization running means and variances are still part of the selected state. Reusing only the selected linear weights would define a different predictor.

## 2 · Time splits and information visibility are separate decisions

The release contains 169,343 papers, 128 features per paper, and 40 target classes. Its official year split has **90,941 train**, **29,799 validation**, and **48,603 test** nodes. Training labels come from papers through 2017, validation from 2018, and test from 2019 onward. Use the official index files rather than creating another split. See the [OGB dataset specification](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv).

This is a transductive full-graph baseline. All nodes' features and the complete released citation structure enter propagation. Only training labels enter the objective. Validation labels select an epoch; test labels score that selected predictor. Batch normalization also sees hidden features from all nodes during fitting. The released `gnn.py` runs the model on `data.x` and the full symmetric adjacency before indexing the training loss.

That distinction connects directly to [lesson 109's database timestamp contract](0109-database-timestamp-contracts.html) and [lesson 110's temporal checkpoint](0110-temporal-gnn-checkpoint.html). A time-based **label split** does not imply a time-filtered **feature graph**. This baseline answers the benchmark's transductive question. To answer “what could a deployed model have known in 2017?”, we would need another visibility contract, features available at that date, and a separately reported experiment.

[[FIG:visibility]]

**Predict:** if you change a test label, should the next training gradient change? No. If you change a test node's features, can a training node's representation change? Yes: there may be a message-passing path, and batch normalization couples node activations. Label masking protects the objective; it does not remove held-out nodes from the computation.

## 3 · Model architecture: follow a paper to its prediction

[[FIG:architecture]]

Start with the feature matrix **X ∈ R^(169343×128)**. Turn the directed citations into a binary undirected adjacency, remove duplicate relations, and add one self-loop per node. This operation changes the information flow: a cited paper and a citing paper can now both send messages. It is part of the released baseline's definition.

Let Ã = A + I after deduplication, dᵢ = Σⱼ Ãᵢⱼ, and Sᵢⱼ = Ãᵢⱼ / √(dᵢdⱼ). One convolution is:

**Z = S H Wᵀ + b**.

Our weight is stored as `[output channels, input channels]`, hence the transpose. Each receiver i adds transformed messages from neighbors j, including itself, scaled by their degrees. The bias is added **after** aggregation. An isolated node has dᵢ = 1 and sends its own features unchanged through S; it still receives the learned linear transformation.

The first two convolutions are followed by batch normalization, ReLU, and 50% dropout. The third maps to 40 class logits, followed by log-softmax. Training uses negative log-likelihood on training nodes, with one full-graph Adam step per epoch. Evaluation turns dropout off and uses the stored batch-normalization statistics. The final predicted class is the largest log-probability.

The complete path is **128 → 256 → 256 → 40**, with **110,120 trainable parameters**. These details follow the pinned [GCN class and training defaults](https://github.com/snap-stanford/ogb/blob/61e9784ca76edeaa6e259ba0f836099608ff0586/examples/nodeproppred/arxiv/gnn.py#L13). The visible lab exposes the sparse multiplication, normalization, model, optimizer loop and checkpoint selection; there is no hidden model service.

### Work one propagation step by hand

Take a chain 0—1—2 with scalar features `[2, 4, 8]`. After self-loops, degrees are `[2, 3, 2]`. Before a learned transform, the middle node becomes:

**h′₁ = 2/√6 + 4/3 + 8/√6 ≈ 5.416**.

This is not a simple neighbor mean. The central coefficient is 1/3 and endpoint coefficients are 1/√6. If both 0→1 and 1→0 occur in the raw citations, treating the concatenated edge list as a multigraph would alter degrees and messages. Task 1 catches that error with duplicate, reciprocal, self-loop and isolated-node fixtures.

[[CODE:normalized_adjacency]]

**Task 1 · TODO → CHECK:** implement `normalized_adjacency`. The independent dense oracle checks both its values and gradients through a feature matrix. Explain why reversing every raw edge and merely summing duplicates is insufficient.

## 4 · Mask the objective, then select one whole predictor

Once messages reach the output, we need to decide which answers may teach the model. The model produces a score for every node, but the objective indexes only `train_idx`:

[[CODE:training_loss]]

**Task 2 · TODO → CHECK:** implement `training_loss`. The check changes every held-out label and demands the same loss; held-out output rows must receive zero loss gradient. Upstream features can still receive gradients through graph connectivity. Do not confuse these two statements.

Every run lasts **500 epochs**. Adam uses learning rate **0.01**, with the library defaults for its other settings. After each update we evaluate validation accuracy. When it strictly improves, save the whole `state_dict`, including the two batch-normalization layers' running buffers. If epochs tie, retain the **first** maximum, matching the released logger's `argmax`. At the end, restore that state and evaluate its predictions.

[[CODE:selected_epoch]]

**Task 3 · TODO → CHECK:** implement `selected_epoch`. Try validation `[0.60, 0.80, 0.70, 0.80]`: the zero-based answer is 1. The function takes no test score, so it cannot legitimately choose by test performance. The full runner checks that restored predictions reproduce the selected epoch's metrics.

[[WIDGET]]

<details open><summary>Worked selection example and no-JavaScript fallback</summary><p>Suppose validation accuracies are 72.0, 73.0, 72.8, 73.0, 72.9% and test accuracies are 71.0, 71.6, 72.2, 71.8, 72.0%. Select epoch 2, the first validation maximum, and report its test score 71.6%. Selecting epoch 3 by test gives 72.2% but uses the test set to make a training decision. The apparent 0.6 pp improvement does not measure better generalization.</p></details>

The released training script logs test scores every epoch, but its logger selects by validation. Our runner preserves that behavior and freezes all choices before the full run. Test traces are retained as audit evidence, not as a reason to adjust the model or rerun favorable seeds. A safer new research implementation could defer test evaluation until selection; that would need its own recorded protocol.

## 5 · Reproduce the distribution, not a lucky seed

Fix seeds 0–9 and run all ten complete schedules. Each seed starts with fresh parameters and optimizer state. We retain every run, even one with disappointing accuracy. Seed 100 is reserved for the timed pilot and is excluded from the published-target comparison.

For the ten test accuracies a₁,…,a₁₀ in percent, report mean **ā = Σaᵢ/10** and sample seed standard deviation **s = √[Σ(aᵢ−ā)²/9]**. The ± bar describes variability among seeds on this fixed dataset. It is not automatically a confidence interval over new datasets, and ten seeds do not test ten independent graphs.

[[RESULTS]]

[[FIG:results]]

Read the dots before the mean. Then compare the mean to the predeclared target and tolerance. Finally inspect the protocol and deviations. A close number plus a different visibility policy is not a faithful reproduction; an outside-tolerance result with a careful audit is still useful scientific evidence.

## 6 · What the evidence establishes

The [protocol ledger](../labs/l112-reproduction.md) names the data hash, pinned source revision, exact environment and commands. The author checks independently reconstruct accuracy from saved node-level predictions and compare the visible implementation with the original model using identical parameters. Forward/gradient agreement and full checkpoint replay test implementation fidelity. They do not establish that the original historical seeds or hardware have been recovered.

Our explicit deviations are a native PyTorch sparse-CSR operator instead of the release's `torch_sparse` representation; separately seeded runs instead of an unseeded sequence of resets; a current pinned runtime; and a custom raw reader whose arrays are checked against the official OGB loader. All affect what “reproduction” can honestly mean. Historical bitwise identity and reproduction of the entire OGB paper remain **NOT_ESTABLISHED**.

There is no hyperparameter search in this lesson: it replays the selected published baseline. Other OGB datasets, link tasks, alternative GNNs, and the paper's original tuning procedure are outside this selected experiment. The short notebook exercise is labeled TEACHING_ONLY; enabling its full gate runs the complete ten-seed schedule and uses your live task implementations.

## 7 · EXIT: defend the result to a skeptical collaborator

Write a short experimental defense that answers all four prompts:

1. Trace the shapes from one feature row through the complete GCN to a class prediction. Explain which normalization coefficients depend on its neighbors.
2. Explain why changing held-out labels must not change training loss, while changing held-out features can. Would this be valid for a 2017 production forecast?
3. Use the saved training history to justify the selected epoch, its tie policy and its batch-normalization state. Explain why the best test epoch is not the answer.
4. Report the ten-seed mean, seed SD, target gap and numerical verdict. Name two deviations and one claim the result cannot establish.

Tomorrow, reconstruct the propagation equation and selection rule without opening the notebook. A week later, inspect another OGB entry and list which parts of this protocol transfer. Planned lesson 113 asks how sampling changes the memory/computation budget; lesson 114 asks where this fitted model fails. Neither a faster run nor a higher aggregate score substitutes for those analyses.

This work supports the course mission by practicing a defensible graph baseline before making relational-learning claims. An arXiv citation result alone does not show that relational learning beats strong tabular models on business databases. Bring your EXIT answer or any confusing computation back to the agent for feedback; we can trace it together.
