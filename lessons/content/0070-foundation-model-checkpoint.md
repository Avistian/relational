## Select a baseline whose evidence you can defend

Your goal is to produce an independently reviewable comparison of pretrained in-context learners, a trained TabM and XGBoost, then identify what relational information could add beyond them. A passing report can recommend a tree, an in-context learner or an ensemble under a stated budget. It must not manufacture a foundation-model win.

The core references are [Nature TabPFN v2](https://www.nature.com/articles/s41586-024-08328-6), [TabICL](https://arxiv.org/abs/2502.05564), [TabM](https://arxiv.org/abs/2410.24210) and the version-specific [TabPFN-3 technical report](https://arxiv.org/abs/2605.13986). The curriculum's newer TabPFN-2.5/3 and TabICLv2 extension arms are explicitly tracked. Availability of their packages is not proof of checkpoint access or a completed run.

## Keep the requested roster visible

The measured historical panel contains XGBoost, TabM-mini, Nature-v2 TabPFN and TabICL v1.1. It uses five real binary tasks: diabetes, blood transfusion, kc1, phoneme and Wisconsin diagnostic breast cancer. Each task is capped at 600 rows, split 60/20/20 with split seed 70, and evaluated with model/inference seeds 0, 1 and 2. The pretrained arms use one view; fitted arms use two validation-selected small recipes. The historical four-arm summary is retained separately so adding methods does not silently rewrite its rank definition.

The extended panel **measures all seven arms**: the four historical arms plus TabPFN-2.5-synthetic, TabPFN-3 and TabICLv2. The current extension uses the exact same data identities, row partitions and seeds; the assembler rejects a mismatched or incomplete panel. It adds 45 runs, producing 105 selected task/seed/model results in total. TabPFN package 8.5.0 and TabICL package 2.2.0 load explicitly selected, revision-pinned checkpoints. Read the checkpoint filename and SHA-256, not just the package version.

The 2.5 arm intentionally names its synthetic-pretrained checkpoint (`v2.5_default-2`), which is distinct from the later real-data-tuned default. It is not a claim about every 2.5 checkpoint. Newer models receive one inference view, just like the historical pretrained arms. These local runs do not reproduce the technical reports' larger benchmark suites, ensembles or compute allocations. The complete roster is measured; the **full paper benchmark protocols remain unreproduced**.
This distinction is the checkpoint's first test of judgment. Rank only a declared complete panel. Report missing requested arms beside that panel; never fill their scores with a nearby version, silently drop them from a claimed full comparison or treat an access error as poor model accuracy.

## Freeze a shared prediction contract

Each arm predicts the probability of the same positive class on the same test row IDs. Training-derived preprocessing, context membership and validation decisions are recorded. Test labels enter scoring after selection. Each saved result includes predictions, targets, validation candidate errors, the selected index and measured durations. A CHECK reconstructs log loss from these artifacts and verifies that the winner was selected on validation.

Inference seeds for a pretrained model vary its downstream configuration, not its pretraining history. Zero variation across three such seeds does not mean there is no uncertainty. Fixed splits also hide split uncertainty. For the five-dataset panel, dataset means—not individual seeds—form the blocks in the rank analysis.

<!--figure:mechanism-->

## Compare quality and cost without mixing ledgers

For each task, show all seed points, mean log loss, sample standard deviation and paired differences from XGBoost. Then show mean ranks and an exploratory Friedman/Nemenyi summary over the five task means. Raw gaps preserve effect magnitude; ranks provide a scale-independent cross-task view. With five small datasets and a restricted method pool, neither is a universal ordering of tabular learning.

Report context preparation/fitting and prediction time separately. The measured inference path uses local CPU and the recorded batch policy. Download and historical offline pretraining are separate costs, not zero. Timings were observed on a shared CPU host; other processes can affect them, so they are a practical run ledger rather than a controlled hardware benchmark. Pretraining-data overlap with these public tasks has not been independently audited. A fixed pretrained model may need little downstream fitting yet expensive context-conditioned prediction. TabM pays optimization cost up front, while its later prediction does not attend to all training rows. Those cost structures matter for deployment volume and context refreshes.

<!--figure:results-->

<!--figure:comparison-->

The paired intervals condition on the fixed split and only three downstream seeds; they do not cover split uncertainty or pretraining variation. A negative loss gap favors the named model over XGBoost. Inspect each task in its own metric units.

<!--figure:ranks-->

The critical-distance bar belongs to the complete declared method pool. It is an exploratory multiple-comparison threshold over dataset-level ranks, not a confidence interval around a model score.

## Make a transfer test, not a victory slide

Choose one mechanism from lessons 61–69 and predict an intervention outcome before running it: reduce context, change label availability, remove a feature, add an unsupported class or switch from random to temporal evaluation. Keep the unperturbed baseline and identify what changed. Carry that prediction and its actual outcome into your report, including a failed prediction.

Use the evidence from lesson 65 to discuss representation access, lesson 67 to discuss local context and adaptation, and lesson 68 to discuss prior mismatch. Do not pool incompatible split seeds, row caps or metrics across those lessons into a new leaderboard. They form a connected argument because the mechanisms and contracts are explicit, not because every number belongs in one table.

## Write a relational research handoff

Name the entity, target time and prediction horizon for a future relational task. List information available to a strong single-table baseline and additional relations you expect to matter. Propose a relational intervention that preserves the same prediction contract and evaluates whether those relations add value. Beat the best deployable baseline selected using validation, not the model that happened to lose one of these small historical comparisons.

State one result that would count against your relational thesis. For example, if a leakage-safe aggregate table plus a strong pretrained learner matches the relational model at lower cost on the target regime, that is substantive counter-evidence. It should remain in the thesis dossier even if a different dataset favors relational learning.

## Exit: submit an executable argument

Submit the frozen protocol, version/checkpoint and data identities, complete measured panel, explicit unrun-arm ledger, paired uncertainty/ranks, cost report and one prespecified transfer test. Add a short deployment recommendation and a research handoff with a falsifiable relational hypothesis. Distinguish which code you implemented, which checkpoints you ran, which saved results you reanalyzed and which original paper results remain unreproduced.

The learning sequence is ready when these tasks are runnable and reviewable. Your personal checkpoint mastery is assessed only after you submit the evidence and explain it without relying on the lesson text. Ask the tutor to challenge the weakest link in your argument before moving into self-supervised encoders.
