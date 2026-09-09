## The prior is a distribution over learning problems

Your goal is to build a small structural causal model, trace its generated observations, and identify a concrete mismatch between that prior and a possible deployment task. In a PFN, the prior determines which tasks the model practices before seeing your table. It is part of the model's inductive bias, alongside architecture, loss and optimization. Read [TabPFN v1, Section 4 and the prior appendix](https://arxiv.org/html/2207.01848v6), then compare the [TabICL prior description](https://arxiv.org/html/2502.05564v1#S4.SS1).

An **SCM** assigns every variable a function of its parents and an exogenous noise variable: `V_j = f_j(parents_j, epsilon_j)`. A directed acyclic graph records which variables can enter each function. **Acyclic** means that a variable cannot ultimately depend on itself through a directed path, so one can evaluate nodes in topological order. Exogenous noise comes from outside the modeled graph; its independence assumptions are part of the generator, not facts inferred from an arbitrary observed table.

## Compute a graph, not just draw one

Our small generator stores `weights[parent, child]` in a strictly upper-triangular matrix. For node j, multiply already computed parent values by their edge weights, apply tanh and add that node's sampled noise. A zero entry means that edge is absent. Tanh is a bounded nonlinear function; it makes descendants nonlinear even with linear weighted parent combinations.

Start with a linear activation to check arithmetic. For noises `[1,.2,−.1]` and edges `0→1` of weight 2 and `1→2` of weight −1, node values are `[1,2.2,−2.3]`. Node 2 uses the computed value 2.2, not the original noise .2. A matrix multiplication of the raw noise by the adjacency matrix is therefore not a general SCM sampler: it skips the recursive dependence.

<!--figure:mechanism-->

The lab first verifies this linear fixture, then runs nonlinear four-node models with fixed noise. Reject a lower-triangular entry or a diagonal edge in this representation; silently evaluating it as if it were a DAG produces an incorrectly specified experiment. Other graph orderings are valid, but they must first be topologically reordered or evaluated by a graph algorithm.

## A paired intervention isolates one mechanism

The author run changes the edge from node 1 to node 2 while holding all exogenous noise draws fixed. Ancestors remain numerically identical; descendants may change. This pairing makes the effect of that edge visible without confounding it with different random samples. Inspect the full path from the changed parent input through the nonlinear operation to the target.

This is an **edge-mechanism intervention**. It differs from conditioning on a variable taking some observed value. It also differs from a node intervention such as `do(V_2=c)`, which replaces the entire structural assignment of node 2 with a constant. Name the operation exactly: causal vocabulary should describe the computation performed, not decorate a correlation plot.

<!--figure:results-->

## Turn latent variables into a supervised task

An SCM alone does not specify a tabular benchmark. You must choose which generated nodes are observed features, which node becomes the target, which remain hidden, how continuous targets become classes, and which rows enter context versus query. Those observation choices can create confounding, redundant features or missing information. The same latent graph can produce very different supervised tasks when you change what the learner sees.

For classification, thresholding a generated target using statistics from all rows would let the query distribution influence task construction. Synthetic pretraining may deliberately define tasks jointly, but that is a generator contract to state, not a preprocessing recipe to copy into deployment evaluation. In the lab, record the threshold rule and use the same sampled task mechanism for context and query unless you are explicitly testing shift.

The four-node tanh generator does not reconstruct the released TabPFN prior. It omits the mixture of function families, graph/width distributions, hidden-node selections, noise distributions and preprocessing choices. The lesson mirrors topological sampling and mechanism intervention, which are independently testable load-bearing parts. A plot that looks “tabular” is not a fidelity certificate.

## Prior mismatch is a hypothesis, not an automatic diagnosis

If deployment has abrupt thresholds but pretraining favors smooth functions, a threshold-rich prior might help. If deployment changes its target mechanism over time while synthetic tasks remain stationary, temporal extrapolation may fail. If a necessary parent variable is unobserved, no clever prior guarantees that the target is identifiable from the supplied features.

To test a prior explanation, hold architecture, optimization steps, training-task count and evaluation data fixed, then change the prior component. Otherwise a gain could come from more compute or a different model. Compare in-prior held-out tasks, controlled out-of-prior tasks and real tasks separately. A causal generator does not mean the fitted PFN has identified the true causal graph of your database.

## Exit: produce a prior card

Submit the graph, weight matrix, topological trace, paired edge intervention and unchanged-ancestor check. Your prior card must name task-function families, noise assumptions, observed/hidden variables, class-construction rules and context/query sampling. Propose one alternative prior and one controlled evaluation that could falsify the claim that it helps. Connect it to the relational mission: which database dependency would a single-table synthetic prior fail to represent?
