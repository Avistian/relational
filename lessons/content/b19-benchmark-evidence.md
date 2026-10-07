# B19 · BeyondArena, contamination and moving benchmarks

<p class="subtitle">Research bridge · Core ★ · One win: decide whether two results answer the same question</p>

[B18a](b18a-context-state.html) showed that fixed weights do not fix a predictor: support, preprocessing and inference settings also matter. Now imagine a leaderboard improves next month. Did the model improve, did the test change, or did the evaluation replace missing runs? Before interpreting the score, identify the experiment.

This serves our mission directly. A relational model must beat a strong baseline under the information and split rules of its intended deployment. A high rank on unrelated public tables does not settle that comparison.

[Student notebook](../labs/b19-benchmark-evidence.ipynb) · [Executed solution](../labs/html/b19-benchmark-evidence.html) · [Download solution](../labs/solutions/b19-benchmark-evidence.ipynb) · [Printable reference](../reference/b19-benchmark-evidence.html) · [Reproduction contract](../labs/b19-reproduction.md) · [Source audit](../labs/evidence/b19/source-audit.json)

## 1 · Start with the question, then choose the split

Recall [grouped validation](0004-grouped-nested-cv.html) and [B18a's predictor identity](b18a-context-state.html). Without looking back: what may be fitted using test labels? Does changing the support change the predictor? Are two rows from the same person independent evidence about performance on a new person?



**In plain terms.** A test split is a promise about what will be unfamiliar at deployment. A random split with familiar customers tests a different promise from holding out whole customers.

An **IID** assumption treats observations as independent draws from the same distribution. A **grouped split** keeps every row of each group together, so test groups are absent from training. A **temporal split** puts the test period after training. Choose according to the intended prediction, not according to which split produces the best score. Grouped and temporal structure may coexist: a future-customer task may need both boundaries. All features and labels must also have been available at the query's cutoff. [BeyondArena §2](https://arxiv.org/html/2606.30410v1#S2)

**Worked example.** Customer A has eight visits and one customer-level label. If six visits enter training and two enter test, remembering A's training label can succeed without learning how to predict for a new customer. That may be useful for a familiar-customer task if those labels are legally available. It is the wrong evaluation for unseen customers.

{{PIPELINE}}

**Outer versus inner.** The outer split protects the final test. An inner split uses only outer-training data to select settings. Holding out customers outside while randomly splitting their visits inside can still choose the wrong settings for new-customer deployment. Preprocessing must be fitted within the appropriate training partition too. Our diagnostic has fixed predictors and no inner selection, so it isolates only the outer boundary.

## 2 · Trace a complete split diagnostic

We generated 24 groups with eight rows each. Each group has a binary label, and each row has a noisy binary signal. A **group-memory predictor** looks up the average training label for a known group; for an unseen group it uses overall training prevalence. A **signal predictor** emits probability 0.8 for signal 1 and 0.2 for signal 0. These are deliberately simple rules, not trained foundation models.

Held fixed: the generated table, both rules, and their accessible feature columns. Varied: test selection. Random testing holds out two rows per group; grouped testing holds out eight entire groups. We repeat the construction with seeds 0, 1 and 2 and retain every outcome.

The **Brier loss** is the mean squared probability error: `mean((p − y)²)`. Here `p` is the predicted positive probability and `y` is 0 or 1. Lower is better. A probability 0.8 for a positive row contributes `(0.8 − 1)² = 0.04`; for a negative row it contributes `0.64`.

<div id="b19-predict"></div>

{{SPLIT_WIDGET}}

{{SPLIT_FIG}}

{{SPLIT_RESULTS}}

The group-memory rule is perfect under the random split. Under grouped testing it falls back to prevalence. The signal rule wins in two seeds and loses in one. This supports the mechanism—familiar groups can make a task easier—not a universal superiority claim about either predictor.

> **Scope check.** Test rows differ across regimes: 48 random-test rows versus 64 grouped-test rows. Comparing their losses changes the target population; it is not a paired same-row treatment estimate. Three synthetic seeds are not three independent real datasets. This diagnostic does not reproduce BeyondArena's models or datasets.

**Read the load-bearing code.** The notebook makes you implement the split audit; the experiment then calls it before scoring.

```python
train_groups = {by_id[i]["group"] for i in train_ids}
test_groups = {by_id[i]["group"] for i in test_ids}
overlap = train_groups & test_groups
```

Zero row overlap is insufficient. The group intersection answers the new-group question. Duplicate or unknown row IDs must fail before this calculation.

## 3 · A number needs an origin

BeyondArena's original report substitutes default Random Forest performance for certain unrun TabPFN-2.6 and TabDPT cases. That permits a complete comparison under a declared fallback policy, but it is not a measurement of the missing model. The current evaluator also exposes this policy in code. [Paper §5](https://arxiv.org/html/2606.30410v1#S5) · [Pinned evaluator context](https://github.com/autogluon/tabarena/blob/1ce4cae6c12972227dea6247bac83234a2915748/packages/tabarena/src/tabarena/contexts/beyondarena/context.py)

Separate three states: **measured**, **missing**, and **imputed** (filled by a declared rule). A fourth distinction is **replayed**: recalculating an aggregate from saved measurements creates no fresh model run.

**Worked example.** In our constructed four-dataset table, A has losses 0.10, 0.10, 0.10 and one missing result. B has 0.20 everywhere. The RF fallback for the missing dataset is 0.90. On the three jointly measured datasets A wins, 0.10 versus 0.20. After filling A's missing result, its mean is `(0.10 + 0.10 + 0.10 + 0.90) / 4 = 0.30`, so B wins.

{{MISSING_WIDGET}}

{{MISSING_FIG}}

Neither policy discovers A's actual fourth score. Common support changes which datasets the claim covers; substitution changes which system produced part of the score. Keep the original missing cell and label the added value `imputed:RF`. Never silently replace it with a value marked `measured`.

These constructed losses share units. Real benchmark datasets can use different metrics and scales: do not average raw RMSE and AUROC error together. Ranking or normalized-error protocols need their own declared handling of ties, missing values and dataset weighting. Here “winner” means lowest mean constructed loss, not BeyondArena Elo.

## 4 · Nine seeds are not nine datasets

Suppose A−B loss differences are 0.10, 0.11, 0.09 on dataset a; −0.04, −0.05, −0.03 on b; and 0.02, 0.03, 0.01 on c. Positive means A has higher loss. Average seeds inside each dataset: the three means are 0.10, −0.04 and 0.02. Their equal-dataset mean is about 0.0267.

A **standard error** describes variability of an estimated mean under sampling assumptions. For independent dataset means, the familiar estimate is their sample standard deviation divided by the square root of the number of datasets. Seed runs share a dataset, so treating all nine rows as independent changes the assumption.

The fixture gives dataset-level SE ≈0.04055 versus naive seed-row SE ≈0.02048. Duplicating each seed result with a new seed ID makes the naive estimate smaller again, without adding a single dataset. Neither SE is a calibrated generalization guarantee: real datasets can share sources or tasks too.

{{UNCERTAINTY_WIDGET}}

{{UNCERTAINTY_FIG}}

A **bootstrap** samples observed units with replacement. The lab enumerates all 27 ordered ways to resample three dataset means. As a teaching contrast, it also enumerates 729 ways to draw three of the nine seed rows. Both draws contain three units; the latter is deliberately not the usual nine-row bootstrap. The operation makes the choice of sampling unit visible. It does not turn this tiny constructed fixture into population evidence.

**CHECK.** If one dataset has 30 seeds and another has three, averaging all seed rows weights the first dataset ten times as heavily. Average within dataset first if the declared estimand—the quantity you want to estimate—is mean performance across equally weighted datasets. Report paired differences on common datasets; do not inflate the dataset count with folds, variants or repeated tasks.

## 5 · Contamination is a separate question from splitting

A correct test split cannot erase exposure during pretraining. **Contamination** means evaluation information has entered model development or training in a way that invalidates the claimed holdout. Distinguish these cases:

| Exposure | What to check | What the check does not establish |
|---|---|---|
| Test labels used in fitting/selection | Fit scopes, timestamps, trial logs | Pretraining cleanliness |
| Dataset used in pretraining | Training manifest, hashes, dataset lineage | Absence of renamed derivatives |
| Near-duplicate or transformed table | Row/column matching and feature/target summaries | Perfect detection under arbitrary transforms |
| Repeated benchmark-driven development | Versioned decisions and untouched holdout | Independence merely because weights were frozen for the last run |

TabDPT's contamination appendix describes several identity/statistics checks across training and evaluation tables. Hash equality detects identical bytes; unequal hashes alone cannot rule out reordered, renamed or transformed copies. Statistical similarity is a screening signal requiring investigation, not proof of leakage. [TabDPT Appendix B.1](https://arxiv.org/html/2410.18164v3#A2.SS1)

**Worked trace.** Reorder a training table and rename its columns. The file hash changes while its information can remain identical. Conversely, two independently collected binary tables can have equal row counts and similar feature means. A good audit preserves both kinds of uncertainty. Record `UNKNOWN` exposure when the pretraining manifest is unavailable; do not substitute “clean.”

## 6 · Version the claim, not just the model name

{{VERSION_FLOW}}

A later model can solve an older failure without contradicting the older measurement. Freeze the paper version, model/checkpoint, dataset bytes, split IDs, support policy, preprocessing, candidate search, missing-result policy and metric. A version label alone does not authenticate the weights or test population.

The pinned current BeyondArena registry retains the original suite but replaces its default Linear entry with a September rerun. Its quickstart uses a reduced core set of splits. These are observable changes, not evidence that either version is invalid. Using today's defaults cannot automatically reproduce yesterday's figure. [Pinned registry](https://github.com/autogluon/tabarena/blob/1ce4cae6c12972227dea6247bac83234a2915748/packages/tabarena/src/tabarena/contexts/beyondarena/methods.py) · [Benchmark log](https://github.com/autogluon/tabarena/blob/1ce4cae6c12972227dea6247bac83234a2915748/packages/tabflow_slurm/BENCHMARK_LOG.md)

{{MATRIX}}

The enterprise study's internal tasks were filtered for nontriviality and predictive signal. Its task composition and restricted data access must travel with any ranking claim. A vendor-affiliated study is not automatically wrong; affiliation and independent reproduction are separate fields in the matrix. Record categorical fraction, semantic types, imbalance, missingness, task mix and selection rules before transporting a ranking to your business problem. [Enterprise study §§3–5](https://arxiv.org/html/2606.30452v1#S3)

**Representativeness checklist.** Do not collapse strings, categories and identifiers into one feature type. For a business transfer claim, attach these fields to each compared suite:

| Field | BeyondArena evidence to inspect | Enterprise-study boundary |
|---|---|---|
| Task mix | 142 curated datasets spanning IID, grouped and temporal tasks | 557 selected internal tasks: 491 classification, 66 regression |
| Categorical fraction / semantic types | Per-task type metadata and actual preprocessing; one fraction cannot summarize the suite | String prevalence is not identical to categorical fraction; inspect semantic typing |
| Imbalance / missingness | Per-task minority fraction, imbalance ratio and missing-value fraction in the archived metadata | Aggregate distributions do not identify every task; internal row-level audit is not available here |
| Provider involvement | Authors include Prior Labs and academic researchers | SAP-affiliated internal-data study |
| Independent verification | B19 audits released grouped score tables, not fresh model outputs | B19 has not reproduced the internal benchmark |

[BeyondArena paper and affiliations](https://arxiv.org/html/2606.30410v1) · [Enterprise task counts and affiliations](https://arxiv.org/html/2606.30452v1)

Use [B03](b03-pfn-tabpfn-generations.html) and [B06](b06-mitra-prior-mixtures.html) as checks on historical model identity. Their diagnostic successes and original-paper source gaps remain separate. [TabBench](https://huggingface.co/spaces/Neuralk-AI/tabbench) is another benchmark to audit; B19 does not authenticate or reproduce its current ranking.

## 7 · Our full-reproduction ledger

**Named target:** BeyondArena v1 Figure F.2: two datasets, grouped and IID splits, six model families, with the original displayed variants and full protocol. The published rank correlations are targets to check, not newly measured B19 values.

{{PAPER_RESULTS}}

We fetched all six original-suite model-result tables and corresponding HPO tables, plus metadata. The complete selected grouped extraction contains **9,540 score records**. It covers 60 musk folds and 30 SAT11 folds, 26 configurations for each of four classical models and one for each TFM. The audit authenticates bytes, checks complete keys, compares every selected record against its original table and recomputes twelve default means. It does not rescore original predictions.

**The missing half matters.** The fetched tables have no corresponding IID-ablation scores. Both IID curation notebooks exist, but their existence does not supply the missing results. UUID-based renaming in source notebooks also means rerunning them is not proof of identical historical input bytes. Our bounded artifact search does not prove that unpublished or differently hosted artifacts do not exist.

> **Scope check.** Full Figure F.2: `INCOMPLETE_SOURCE_PROTOCOL_GATE / NOT_RUN`. Fresh paper-model fits and the whole benchmark: `NOT_RUN`. The complete grouped score-table audit is a narrower success. The explicit paper command refuses rather than calling the grouped half a reproduced figure. $0 paid compute; all local attempts share a 3,600-second cap.

## Lab · Build and defend the comparison

The portable notebook contains all load-bearing course code and three live tasks: audit splits, aggregate paired scores while preserving provenance, and summarize paired seed differences at dataset level. The final experiment uses your functions; each task has immediate checks. Blank functions cannot pass. The saved-paper lane embeds the selected records; the repository command additionally authenticates the full original tables.

**Preregister an untouched comparison.** Before examining its test outcomes, choose one accessible dataset that has not guided your decisions. Complete the [preregistration template](../labs/b19-preregistration.md): deployment population; full row/entity/time keys; legal inputs and label-availability times; dataset and model exposure status; outer and inner splits; strong baseline and challenger; preprocessing and support access; exact candidate/seed budget; one primary metric; missing-run rule; paired analysis; cost cutoff; and a falsification criterion. Hash and date the contract before execution. This is learner work, not an author claim that an untouched benchmark was run.

**EXIT.** Defend why the split matches deployment, identify which results were actually measured, explain the independent unit, and say which source gaps prevent historical reproduction. Then state what evidence would change your architecture choice. Ask your teacher about any unclear step; revisit the split, missing-score and uncertainty traces after 1, 7 and 30 days.

<div id="b19-teachback"></div>

Next: [B19a's planned probability-quality lesson](../plan/year-5-6-bridge.md#b19a). Correct evaluation identity is necessary; it does not guarantee calibrated probabilities. Carry the contract into [Year 6](../reference/curriculum.html), particularly the proposal and experiment stages. Learner status remains `PENDING_WRITTEN_DEFENSE`.
