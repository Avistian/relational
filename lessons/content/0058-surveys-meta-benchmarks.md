## A survey is a map; a benchmark is an experiment

Your task is to decide what evidence a model-family claim actually rests on. A **survey** organizes ideas and references. A **benchmark** specifies datasets, splits, preprocessing, model selection, metrics and aggregation, then measures a comparison. A **meta-analysis** combines results across tasks; its unit of evidence matters as much as the number of fits. Neither a long bibliography nor a large run count automatically establishes broad generalization. The relational thesis needs a baseline chosen from trustworthy evidence, so this distinction determines what you should reproduce next.

Read [Borisov et al.'s survey](https://arxiv.org/abs/2110.01889) for the landscape and [TALENT, Sections 4–5 and 8](https://arxiv.org/html/2407.00956v3) for an actual evaluation. The [2024 deep-tabular survey](https://arxiv.org/abs/2410.12034) supplies another taxonomy. Add an independent distinction to your map: fitting a model per dataset versus using a model pretrained across tasks. These are organizing lenses, not interchangeable experimental evidence.

## Classify the operation, not the marketing name

An MLP composes learned affine maps and nonlinearities. A boosted tree ensemble adds partition-based corrections. Attention models compute input-dependent weighted combinations. Retrieval models condition predictions on selected stored examples. A prior-data fitted network learns across generated tasks and then conditions on a new labeled context. These categories overlap: TabM is an MLP ensemble, TabR combines a parametric encoder and retrieval, and TabICL uses attention for both representation construction and task conditioning.

Build a map with four independent columns: **what learns across tasks**, **what changes at deployment**, **what information a prediction can access**, and **what determines inference cost**. In a pretrained PFN, adding a labeled context can change predictions without changing weights. In a fitted MLP, learning a newly supplied target generally requires a parameter update. That is a concrete difference you can test; calling both models “deep” conceals it.

## Reconstruct the denominator

The lab parses the authors' released binary, multiclass and regression tables. A cell such as `0.7645+0.0110` contains a reported mean and dispersion. It is not a list of seed measurements. Strip presentation emphasis, parse the mean, preserve absent results as missing, and record table direction: larger classification scores are better; smaller regression errors are better. Do not invent raw repeats from the published dispersion.

The pinned release contains 300 datasets. Its accompanying [release note](https://github.com/qile2000/LAMDA-TALENT/tree/1b973adffa4203c3f62c4949957b1ba3efbe60d3/results) points to a newer corrected benchmark, mentioning duplicates and task-type corrections. This makes the frozen tables useful historical evidence, not an automatically current leaderboard. The notebook reports coverage for each task type and uses a declared six-model pool: XGBoost, CatBoost, MLP, RealMLP, TabR and FT-Transformer. Changing that pool can change ranks even when no model prediction changes.

## Turn incomparable units into comparable positions

For dataset d, rank its model scores after orienting them so smaller is better. Assign average ranks to ties. Then average each model's ranks across datasets. Rank 1 means best within that declared pool on that task; it says nothing about the size or practical value of the gain.

Consider lower-is-better errors for A/B/C: dataset one has `.10/.11/.90`; dataset two has `100/10/11`. Averaging raw errors lets the second dataset's units dominate. Within-dataset ranks are `[1,2,3]` and `[3,1,2]`, giving mean ranks `[2,1.5,2.5]`. B leads this rank summary. You still need the original errors to decide whether a difference matters. Keep separate classification and regression views alongside an overall rank summary, because task composition determines the latter.

<!--figure:mechanism-->

## Make selection bias visible

Before viewing the result, predict what happens if we choose the 45 datasets where XGBoost has the largest rank advantage over the MLP. We have changed the evaluation population using the outcomes. The resulting rank table answers “how do these methods compare on the selected tasks?” It does not independently confirm that those tasks represent future deployment.

The lab performs this deliberately biased selection next to the full 300-dataset analysis. TALENT's compact benchmark construction is useful precisely when its selection objective is understood; reusing an outcome-selected subset to validate the same favored property requires care. For a new model, preregister an additional evaluation population or validate that the subset preserves conclusions outside the methods that selected it. [TALENT Section 8](https://arxiv.org/html/2407.00956v3#S8).

<!--figure:results-->

## What uncertainty can you recover?

A paired dataset bootstrap samples dataset rows with replacement and recomputes mean ranks for every model on the same sampled rows. Its interval describes sensitivity to this empirical task collection under an exchangeable-dataset approximation. Related tasks, duplicate datasets, and shared source populations weaken that approximation. The procedure cannot reconstruct model-seed uncertainty from rounded published means, and it cannot resolve changes between benchmark versions.

Your evidence card must therefore include source revision, task coverage, score direction, model pool, selection regime, missingness handling, aggregation unit, and one excluded claim. “Six models on 300 historical tasks, complete-case rank reanalysis” is defensible. “We reproduced the paper's full leaderboard” is not the same claim.

## Exit: choose the next experiment

Submit the coverage audit, full and outcome-selected rank tables, paired dataset intervals, and your family map. Write a four-sentence recommendation for lesson 60: which baselines to retain, which deployment regime to test, which evidence is stale or incomplete, and what result would change your choice. Do not choose a relational baseline solely because its single-table rival did poorly on a selected subset.
