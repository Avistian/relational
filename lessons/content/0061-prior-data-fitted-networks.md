## Learn an inference procedure

A conventional classifier learns one mapping from features to labels on one training dataset. A **prior-data fitted network (PFN)** learns across many sampled supervised-learning tasks. Its input includes a new task's labeled context and an unlabeled query. Its output approximates the label distribution implied by the prior and that context. Read [Müller et al., Sections 2–3](https://arxiv.org/html/2112.10510v7).

Separate three objects: the prior over tasks, the distribution of examples within a task, and the neural weights shared across tasks. During pretraining, gradients update those weights. At ordinary in-context inference, the weights are fixed; context examples alter the computation and therefore the answer. Calling the latter “no training” is shorthand for no downstream parameter optimization, not zero learning cost or no use of labels.

## Derive a posterior you can check exactly

Let a task be an unknown coin probability theta. Draw theta from a Beta distribution with positive shape parameters alpha and beta. Conditional on theta, each observed label is a Bernoulli draw: 1 with probability theta, otherwise 0. For a context with s successes and f failures, multiplying the prior density by its likelihood gives exponents `alpha+s−1` and `beta+f−1`. The posterior is therefore `Beta(alpha+s, beta+f)`.

To predict the next label, integrate over this posterior instead of plugging in one fitted theta. The next-success probability is `(alpha+s)/(alpha+beta+s+f)`. For a uniform `Beta(1,1)` prior and labels `[1,0,1]`, the prediction is `3/5=.6`. The empirical success rate is `2/3`; the Bayesian answer shrinks it toward the prior mean `.5`. With no observations, it returns the prior mean. More observations reduce the prior's relative contribution.

The output `.5` alone does not distinguish a well-known fair coin from uncertainty between nearly deterministic coins. A posterior predictive probability mixes uncertainty about the task with randomness within it. If you need a posterior distribution over theta, that is a different output contract.

<!--figure:mechanism-->

## Why query-label loss learns the Bayesian answer

Sample a task, draw context labels and draw an independent query label from the same task. Train a network to predict the query from the context using binary cross-entropy, `−y log(q)−(1−y)log(1−q)`. Conditional on a particular context, let p be the true posterior predictive probability. Expected query loss is `−p log(q)−(1−p)log(1−q)`.

Differentiating with respect to q gives `−p/q+(1−p)/(1−q)`, which vanishes at `q=p`. Equivalently, expected log loss is the true predictive entropy plus a nonnegative KL divergence between the true prediction and the model's prediction. Finite capacity, optimization and the sampled training distribution determine how closely a neural network reaches that optimum. The theorem does not say an arbitrary trained Transformer has become an exact Bayesian engine for every real table.

The context and query must share the same sampled theta. If you redraw theta for the query, context labels contain no information about it; the best predictor collapses to the prior mean. The lab asks you to construct this broken generator as a negative control.

## Model architecture: isolate the PFN objective

The local **CountPFN** uses the exact sufficient statistics `[successes, failures]`, divides them by 32, and passes them through `2→32→32→1` affine layers with GELU nonlinearities. A sigmoid turns the final logit into a next-success probability. A sufficient statistic retains all information about theta that these exchangeable Bernoulli observations provide. Using it removes representation learning from this first experiment.

This is a complete PFN-objective demonstration, not a Transformer or TabPFN implementation. The paper's general architecture embeds examples and learns set-conditioned prediction using attention. Lesson 62 introduces that row-token route; the count encoder here is an intentional decomposition so you can independently verify the learning objective before adding attention and richer priors.

<!--figure:architecture-->

## Fit on sampled tasks, evaluate against analysis

Each of three author runs takes 1,500 optimizer steps, with 256 independently sampled coin tasks per step and context lengths from 1 through 32. Training targets are sampled query labels, not analytic posterior values. The evaluation grid checks every success count at context lengths 4, 16 and 32, then challenges extrapolation at length 64. The analytic formula appears only in the evaluation path.

Predict whether a small average approximation error inside the training range guarantees accurate length-64 predictions. It does not: the network has learned a function over a sampled range of counts, while the analytic rule holds at every valid length. Plot predicted versus exact probabilities and inspect errors near extreme counts; an overall mean can hide those failures.

<!--figure:results-->

## Exit: distinguish three kinds of success

Implement the posterior formula and the task generator, train your predictor and report in-range and extrapolation errors. Compare a constant `.5`, the empirical success rate and the learned predictor against the analytic posterior. Explain why the analytic oracle is an evaluation target, not a fair competitor whose “training time” should be compared with neural pretraining.

Your verdict must separate the correctness of the conjugate calculation, the measured approximation quality of this trained CountPFN, and the broader paper claim about attention-based posterior approximation. Then name what changes for a tabular classification task: the latent task is now a function or generating process, and the labeled context contains features as well as targets.
