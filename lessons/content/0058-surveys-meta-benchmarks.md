## A survey is a map; a benchmark is an experiment

**Outcome:** turn a model-family claim into a traceable evidence card, then implement and test a small benchmark whose selection cannot read the methods reserved for evaluation. You will parse actual published results, orient metrics, rank a fixed model pool, bootstrap paired task differences, and implement a random-search subset selector. No new predictive model is introduced. The object we learn to build is an **evaluation procedure**.

Retrieve before reading: what does an outer test set protect in lesson 057? Why can averaging folds as independent datasets distort lesson 056's leaderboard? Can an accurately implemented method still fail to reproduce its paper? Write answers now. In this lesson the protected information is sometimes an entire **method column**, not a row's target.

A **survey** organizes concepts and references. A **benchmark** evaluates specified procedures on specified tasks. A **meta-analysis** aggregates evidence across tasks or studies. These roles can coexist in one paper. [Borisov et al., arXiv v3, Sections III–IV and VII](https://arxiv.org/html/2110.01889v3) both propose a taxonomy and conduct their own five-dataset comparison. Treating that paper as only a literature map discards its experimental contribution; treating its landscape discussion as another independent experiment double-counts evidence.

The relational mission needs credible single-table rivals. A favorable neural result on an outcome-selected collection cannot establish that relational modeling adds useful information. First choose strong fitted procedures under the intended temporal and information constraints; then ask whether relationships improve them.

## Classify the operation, not the marketing name

Borisov's taxonomy separates **data transformations**, **specialized architectures**, and **regularization**. These answer different questions: what representation reaches a model, how it combines inputs, and what training behavior is encouraged. They are useful lenses even when a modern method occupies several categories. The [2024 survey, Sections 2–3](https://arxiv.org/html/2410.12034v1) provides a broader reading map; use its references to locate original mechanisms. Do not transfer a broad survey statement into a current performance claim without its experiment.

| Familiar procedure | Representation and interaction | What changes on the target task? | What one prediction needs |
|---|---|---|---|
| XGBoost / CatBoost | Successive tree partitions and additive corrections | Tree structures, leaves and selected settings | Features plus fitted trees |
| RealMLP | Numerical preprocessing, affine layers and nonlinearities; a full training recipe | Weights, target-task preprocessing statistics | Features and fitted parameters |
| FT-Transformer | Feature tokens exchange information through attention | Tokenizer, attention and head parameters | One row's feature tokens |
| TabR | A learned encoder retrieves and combines labeled examples | Encoder/predictor parameters and reference store | Query plus selected training examples |
| Pretrained PFN | Across-task training; context conditions a prediction | Context can change while weights remain frozen | Features, labeled context and checkpoint |

This table revisits earlier mechanisms; it does not introduce or implement these models. A PFN's synthetic pretraining is not the same operation as learning feature representations from a target table's unlabeled rows. Likewise, TabR's target-task learned retrieval is not automatically cross-task transfer. Ask which parameters update, which labels are available, and whether the cost grows with context size. These questions will distinguish the foundation-model lessons beginning at 061.

**Worked classification:** suppose a numerical encoding feeds an MLP and the training loss adds a penalty. Calling it an “MLP” describes the backbone but omits two potentially decisive choices. An ablation that replaces the encoding while also retuning the optimizer cannot isolate architecture alone. A benchmark row should name the complete procedure being compared.

## Read the original experiment before its conclusion

Borisov's Section VII compares HELOC, Adult, HIGGS, Covertype and California Housing. It describes 100 Optuna iterations with five-fold evaluation, numerical normalization, ordinal categorical encoding, and model-specific handling. Its table reports accuracy/AUC for classification and MSE for regression. The authors' strongest neural result is on the large HIGGS case; the five-task experiment is not an estimate over every future business table. Their subsequent discussion explicitly recognizes dataset and encoding sensitivity. We do not rerun that benchmark here.

[TALENT v3, Section 4.2](https://arxiv.org/html/2407.00956v3#S4.SS2) describes a different procedure: random 64/16/20 train/validation/test partitions, validation tuning and stopping, 100 Optuna trials, then 15 training seeds with selected hyperparameters. Accuracy is the principal classification criterion; regression uses RMSE. These are paper-reported settings, not information recoverable from one rounded result cell. An equal trial count does not imply equal wall time or equally effective search spaces.

Its task curation aims at standard same-distribution prediction and removes timestamps as attributes. That is a deliberate scope choice. It does not answer lesson 055's future-period deployment question. Related source variants are sometimes retained; benchmark size is not identical to the number of independent domains. A broad random-split benchmark and a narrower deployment-aligned temporal experiment provide different evidence.

The paper itself needs an audit: Section 4 reports 120 binary, 80 multiclass and 100 regression tasks, while the Figure 3 caption reverses binary/regression counts. The checked local tables have **120/80/100** rows. Resolve such contradictions against the exact artifact being analyzed, and retain a note rather than silently combining versions.

## Reconstruct the denominator

The lab uses three checksum-pinned Markdown tables at revision `1b973adffa4203c3f62c4949957b1ba3efbe60d3`. The [source record](../labs/_sources_l058.json) names each URL and SHA256. Raw training arrays are not downloaded in this audit; exact Markdown hashes do not validate the underlying dataset identities. Their release note points to a corrected benchmark, mentioning duplicate removal and task-type corrections. A repository revision containing a notice about replacement data does not certify that every older table has been regenerated with those corrections. We call this a **frozen historical release**, not today's leaderboard.

A cell `0.7645+0.0110` contains a mean and reported dispersion, not individual observations. The new parser preserves both in separate aligned frames, recognizes explicit missing markers including `nan+nan`, and rejects malformed nonmissing values and repeated row identifiers. It does not manufacture seeds. The Pima/XGBoost cell above is a real trace: its parsed mean enters ranking, while `0.0110` stays in the dispersion frame and is never treated as a standard error.

The main comparison fixes six arms: XGBoost, CatBoost, MLP, RealMLP, TabR and FT-Transformer. All six have entries on all 300 rows. The separate tiny-benchmark track uses the paper's named 13-method family roster for classification and 12 for regression, which omits TabPFN. These are separate rank pools. `LR` is an explicit alias for the release's `LogReg`/`LinearRegression`; original dataset labels remain intact with a task prefix in the audit output.

## Turn incomparable units into comparable positions

Let S[d,m] be the published mean for dataset d and method m. Convert to a lower-is-better quantity E[d,m]: use −S for accuracy and S for RMSE. Rank each row **within the declared method pool**, assigning average occupied positions to ties. Write that position as R[d,m]. The dataset-balanced mean is r[m] = (1/D) Σd R[d,m], where D counts datasets. Never pool raw accuracy and RMSE.

For errors A/B/C on two datasets, `[.10,.11,.90]` and `[100,10,11]`, the row ranks are `[1,2,3]` and `[3,1,2]`. Their average is `[2,1.5,2.5]`. B leads this summary. The second task's large units cannot dominate after ranking, but a tiny win and a large win now count equally. Inspect effect sizes on your actual deployment task before choosing a system.

If three rounded values occupy positions 2, 3 and 4, each receives rank 3. This states what the released precision supports. It does not establish exact population ties. Removing a competitor can change remaining mean ranks without changing predictions. Ranks also depend on task weighting: the 120/80/100 collection weights task types 40%/26.7%/33.3%, whereas giving each task type one third is a different estimand, meaning a different quantity we intend to summarize.

<!--figure:mechanism-->

## Make selection bias visible

The historical lab deliberately selects the 45 rows with the largest XGBoost-versus-MLP rank advantage. Its archived summary changes XGBoost's mean rank from 3.593 to 1.300. That is an outcome-selection demonstration, **not TALENT's Tiny Benchmark 1 or 2**. The old result and its original operator hash remain unchanged. Tie handling in that archived selection follows its original NumPy sort; the new experiment below defines its own first-candidate tie rule.

TALENT Section 8 has two distinct goals. Tiny Benchmark 1 is a diagnostic collection with tree-friendly, DNN-friendly and near-tie examples across size groups, with feature-type refinements. Its 15 binary/12 multiclass/18 regression tasks are not simply the 45 most tree-friendly rows. An encoding intervention on that collection asks where it helps under those selected conditions. Tiny Benchmark 2 instead seeks a subset whose average method ranks resemble the full collection. These objectives can produce different tasks and support different claims.

**Reading check:** suppose a new encoding helps on tree-friendly tasks but hurts on DNN-friendly tasks. This identifies an interaction worth studying. It does not make the selection invalid, and it does not establish that the encoding should be applied on every future table. Prespecify the deployment population and verify the mechanism in a controlled experiment.

## Build a tiny benchmark that can fail honestly

This is the paper contribution the repaired lab reconstructs. Let A be the methods allowed to guide selection and B the full task collection. For each candidate subset U with k tasks, calculate

**L(U; A) = mean over m in A of |mean over d in U R_A[d,m] − mean over d in B R_A[d,m]|.**

Here R_A ranks methods within A only. The absolute value prevents positive and negative discrepancies canceling. Average ranks rather than rank sums keep a 15-task subset comparable to a 100-task collection. Random search proposes unique task subsets, computes their losses and retains the first smallest loss. The candidate count is an optimization budget; “random selection” in this setting means selecting the best among random proposals, not using one unselected random subset.

The source is inconsistent in notation: Equation 5 displays sums while its prose discusses averages; the detailed selection descriptions and Table 7 caption use MAE, while an applications paragraph says MSE. We implement the **explicit mean-rank MAE objective above**, not a claim of literal equation or Table 7 parity. The upstream visualization script also mean-imputes missing ranks; this lab uses a complete panel and exposes missingness. No imputation is needed for the chosen pools.

For a numerical trace, six synthetic tasks have A/B ranks `[1,2], [1,2], [2,1], [2,1], [1,2], [2,1]`, so the full mean is `[1.5,1.5]`. Candidate rows `[0,1]` give `[1,2]`, hence MAE .5. Candidate `[0,2]` gives `[1.5,1.5]`, hence MAE 0. More proposals cannot worsen the best **seen-method objective** when the proposal sequence is nested. Nothing in that statement guarantees improvement on methods selection never saw.

<!--figure:protocol-->

The notebook first reserves method columns, then computes ranks among the remaining methods. This order matters. Computing all-method ranks and only afterward hiding some columns leaks information: the hidden methods' scores already influenced the visible positions. The selector accepts only a seen-method rank matrix. After it freezes task IDs, the evaluator reads held-out columns and ranks them within their own pool. Changing held-out scores must not change selected IDs; a CHECK tests that boundary directly.

The new real-table track uses five declared method rotations, three search seeds (58–60), and separate complete task-type subsets: 17/116 binary, 9/60 multiclass, 15/100 regression (15% rounded to the nearest integer). The 13-method classifier panel loses four binary and twenty multiclass tasks because the historical TabPFN column is missing; the 12-method regression panel is complete. This yields 276 eligible tasks and 41 selected tasks, not the paper's 300 and 45. Missingness is a substantive change in the target population; the result artifact lists every exclusion. It tests 1,000 proposals and then 10,000, the latter matching the paper's random-search count. Within each task type and fold, the baseline is the first identical random proposal. The original paper's exact method splits, corrected table version, original tiny membership and selector implementation are not recovered. Independent seen/unseen rank pools also differ in scale, so compare selected-versus-random **within** each group; do not treat their raw MAEs as directly interchangeable.

<!--figure:tiny-->

**Measured outcome:** with 1,000 candidates, the selected subset has worse held-out fidelity than the first random proposal in **15 of 45** declared cases; 26 improve and four tie within 1e−12. With 10,000 candidates, 20 worsen, 24 improve and one ties. Increasing the budget from 1,000 to 10,000 improves the seen objective in aggregate but worsens the held-out objective in 23 cases, improves it in 13 and leaves nine unchanged. These are overlapping method rotations and search seeds, not independent success-rate estimates. The result exposes the intended failure mode: a more accurately optimized proxy need not be a more transferable benchmark.

At 1,000 proposals, mean held-out MAE changes (selected minus first random) are −0.0618 for binary, +0.0065 for multiclass and −0.0387 for regression. The small positive multiclass average is a failed improvement, even while its seen-method objective improves. Compare these changes within task type and held-out pool; differences between task types also reflect different rank scales and available-task populations. Every case and selected task ID is available in the [fresh analysis](../labs/_verify_l058_v2_results.json), with the [10,000-candidate comparison](../labs/_verify_l058_closer_results.json) separately saved.

## What uncertainty can you recover?

A dataset bootstrap samples D dataset indices with replacement and uses that same index list for all model columns. Each draw recomputes a mean. The 2.5% and 97.5% quantiles form a **percentile interval**, not a t interval. Related tasks and retained source variants limit the exchangeable-dataset interpretation. The result measures sensitivity to this historical task collection conditional on its published means; it cannot restore training-seed uncertainty.

For an actual pairwise claim, compute the paired row difference first. Here G[d] = R[d,CatBoost] − R[d,RealMLP]. Negative means CatBoost has the better rank. Bootstrap G directly. Comparing whether two marginal intervals overlap is not the same calculation. An interval crossing zero does not prove equivalence, and a narrow interval over this collection does not cover future temporal shift.

<!--figure:results-->

**Measured outcome:** CatBoost has the smallest six-method mean rank, **2.908**, followed by RealMLP **3.022** and TabR **3.028**. The paired CatBoost-minus-RealMLP rank gap is **−0.113**, with dataset percentile interval **[−0.408, 0.192]**; CatBoost-minus-TabR is **−0.120**, interval **[−0.450, 0.197]**. Negative favors CatBoost. Both intervals cross zero, so this audit does not resolve that ordering under its resampling assumptions; it also does not establish equivalence. All three remain reasonable candidates for a deployment-aligned comparison. The small rank differences cannot justify discarding an otherwise feasible baseline.

The tiny-benchmark comparisons reuse tasks and overlapping method pools. Fifteen fold/seed records per task type are sensitivity cases, not fifteen independent new benchmarks. Report every paired change, not a p-value that pretends the rotations are independent samples. Distinguish dataset bootstrap randomness, subset-search randomness and training randomness; only the first two are recomputed here.

## From meta-features to a discriminating experiment

TALENT Sections 6–7 also examine learning-curve prediction and dataset meta-features. A **meta-feature** is a task-level summary such as size or a distribution statistic. A relationship between that statistic and model rank is observational: preprocessing, source domain, sample size and other properties can vary together. Feature importance in a curve predictor is not a causal explanation of which architecture wins. The lab does not reproduce those sections' predictive models, feature selection or reported associations.

Connect this distinction to lesson 051. A controlled rotation or uninformative-feature intervention holds a task and protocol fixed while changing a specified property. A cross-dataset correlation compares different tasks that may differ in many ways. Use the latter to propose an intervention; use the former to test a mechanism. Neither alone establishes transfer to a new relational deployment.

For example, if a method seems weak on tables with uneven feature distributions, compare alternative preprocessing on the same rows and split, with matched selection budgets. Separately verify that a temporal holdout keeps its advantage. Finally add relational information while holding the strong single-table procedure fixed. That chain creates evidence for the mission instead of interpreting a taxonomy label as an experimental result.

## Exit: choose the next experiment

Submit five implemented functions, the coverage and source audit, complete and task-specific ranks, paired percentile intervals, all tiny-subset method holdouts, and the saved EXIT JSON. Add a mechanism map for four candidate baseline procedures and one limitation per evidence source. Predict whether 10,000 candidates must improve unseen-method fidelity before running that required final track.

Your lesson 060 recommendation must name a strong tree procedure, a strong neural procedure, the deployment split and compute constraint, and a result that would change the shortlist. Explain one case where optimizing the tiny benchmark improved seen-method fidelity but failed on unseen methods, or report honestly if the observed run did not show such a case. Ask the tutor about any CHECK or inference you cannot defend; revisit the opening questions before rereading tomorrow.
