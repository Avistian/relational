# L193 — RDBLearn full 21-task reproduction ledger

**Model reproduction: INCOMPLETE_SOURCE_PREPROCESSING_GATE.** All567validation and63selected-test slots are NOT_RUN. No model scores were fabricated or imported as fresh measurements. Cloud/APIUSD0. Fresh synthetic original-source diagnostic COMPLETE; complete task/reference/schedule audit PASS. Learner PENDING_WRITTEN_DEFENSE.

## Named target and frozen protocol

RDBLearn toolkit paper arXiv2602.18495v1, Tables1–3, RDBLearn column: 8 RelBench classification tasks, 8 RelBench regression tasks, 5 4DBInfer classification tasks. Exact IDs and published scalar targets: `evidence/l193/packet/tasks.json`. RDBLearn0.1.2 git b5b03ebf8091547285a6e06cba53d2d1a40cb171; FastDFS0.2.1. Eighteen frozen source files were authenticated directly against git objects. Sources and licenses reside in `sources/l193/`.

Per task/seed: depths2/3/4 × TabPFNv2/TabPFNv2.5/LimiX16M; official splits; uniform training-support subsampling at10000; complete validation candidate grid; fixed listed candidate order for ties; selected full test evaluated once. Seeds0/1/2 are course repeatability seeds, not recovered historical identities. Full plan:21×3×9=567validation candidates and21×3=63selected test evaluations. No new backend pretraining; other paper method columns are references, not newly reproduced baselines. The paper's best-config table cannot replace validation search.

Expected named checkpoints are tabpfn-v2-classifier-finetuned-zk73skhh.ckpt, tabpfn-v2-regressor.ckpt, tabpfn-v2.5-classifier-v2.5_default.ckpt, tabpfn-v2.5-regressor-v2.5_default.ckpt and LimiX-16M. Their bytes/hashes and access are NOT_ESTABLISHED in L193 because the earlier admission gate failed. Raw full task databases, feature construction and per-query temporal availability have NOT_RUN audits. No historical data/environment identity claimed.

## Measured stopping condition

`_preflight_l193.py` freshly imports and executes the original full `TabularPreprocessor`, including AutoGluon1.5.0. It authenticates source files first. Synthetic categories b/c/d are fitted; unseen a or0 changes known codes0/1/2 into1/2/3. Unseen z or e leaves codes unchanged. All four numeric controls are unchanged. The known query b gets code0 alone but code1 alongside a or0. Full observations, environment versions and original source hash are in `packet/preprocessing.json`.

This is a preprocessing semantic failure in the recorded modern environment. Its occurrence and AUROC/MAE effect on any real benchmark are NOT_ESTABLISHED. It does not prove all21tasks are affected. The declared shared-pipeline admission policy stops the suite before any backend call. We did not repair or switch versions silently. A repair requires a separately named, reviewed protocol and its own validation budget.

## Aggregation and evidence boundaries

All63seed outcomes retain NOT_RUN and null scores. Full-seed mean/SD and all fresh group means are null. Completed tasks0/21, model evaluations0/630. Missing runs are never converted to zeros or silently dropped. Test key checking and independent AUROC/MAE scoring are prerequisites for future COMPLETE records, not checks executed on nonexistent predictions.

A separate arithmetic audit extracts all21published target values from original HTML. RelBench classification mean over8rounded scalars=.7452875; 4DBInfer classification mean over5=.7879. These are table arithmetic, not checkpoint inference, prediction replay, a reconstruction of seed variance, or whole-paper reproduction. A separate stdlib HTMLParser/Decimal verifier checks the BeautifulSoup extraction.

Raw Table2MAEs remain per-task targets. Section5's normalized-MAE reference baseline lacks an unambiguous supplied predictor/denominator contract. Do not substitute AutoGluon-without-RDB scores or target standard deviation. Regression aggregate parity NOT_ESTABLISHED. When verified denominators exist, normalize task MAE first, then give each task equal weight. AUROC and MAE never share an aggregate. Sample SD across3support seeds describes within-task repeatability, not database-level uncertainty.

## Exact local commands

Run from repository root. All numerical attempts, expected failures and delivery checks pass through the3600second budget wrapper; it records failed attempts and kills subprocess groups at cutoff.

```bash
python3 labs/_budget_l193.py .venv/bin/python labs/_test_l193.py
python3 labs/_budget_l193.py .venv/bin/python labs/_replay_l193.py
python3 labs/_budget_l193.py .venv/bin/python labs/_verify_l193.py
.venv/bin/python labs/_reproduce_l193.py --show-plan
python3 labs/_budget_l193.py .venv/bin/python labs/_reproduce_l193.py
```

The last command intentionally exits2 with the source gate and dispatches nothing. This is an executable complete admission/ledger operator. The downstream GPU executor is NOT_IMPLEMENTED_SOURCE_GATE, not an end-to-end model runner. No unsafe override exists. Preparation/freezing commands are author-time operators; they intentionally require the pinned upstream git checkout and original source files. The portable audit notebook requires only Python standard library and embeds its exact packet.

Fresh source diagnostic as executed:

```bash
python3 labs/_budget_l193.py /tmp/l192-env/bin/python labs/_preflight_l193.py
```

That temporary environment is local, not a portable environment path. Recreating it requires the recorded dependencies: Python3.12, pandas2.3.3, NumPy2.5.0, scikit-learn1.9.0, pydantic2.11.7, AutoGluon features/common1.5.0 plus the frozen FastDFS/RDBLearn packages and their dependencies. Its dependency snapshot is provided; clean-room reconstruction of this upstream environment is NOT_CHECKED. The standalone audit solution was separately executed from an empty working directory; it replays observations without these dependencies.

## Budget and continuation

USD10total, planned stop8/reserve2, covering preparation/retries/validation/overhead. No paid pilot was justified after source failure; full-suite cost forecast NOT_ESTABLISHED, not a claim that the suite fitsUSD10. Local safety cap3600execution seconds. `local-budget.json` reports all wrapped attempts and a conservative allowance for initial unwrapped preparation. No API/cloud billing.

Continuation needs: (1) source resolution and version/protocol decision; (2) full data, cutoff, labels and feature/preprocessing audit; (3) checkpoint authentication; (4) exact regression normalizers; (5) implemented/validated backend executor; (6) measured cost of the entire grid including retries under cap. Stop rather than quietly reduce tasks, seeds, depth choices or backend candidates. Any changed experiment is explicitly separated from the original paper target.

## Delivery and mastery

Lesson/reference,3portable figures, student and executed solution, full packet, source, manifests, tests, independent verifier and admission operator are part of the package. `_execution_l193_results.json`, `_delivery_l193_results.json` and `_checkout_l193_results.json` record actual checks when present. Notebook replay, browser delivery, local Pages build, live Colab and deployment are distinct. Live Colab and deployment NOT_CHECKED. Learner completion requires their own TODO implementations and EXIT defense; no mastery inferred from author execution.
