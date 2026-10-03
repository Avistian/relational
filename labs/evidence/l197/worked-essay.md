# A relational opportunity with unfinished proof

*Author reference essay, dated 2 October 2026. This is not the learner's submission or a claim of mastery. Its numerical claims refer to the frozen L197 packet, not a continuously updated leaderboard.*

## Position

My Year 5 position is that relational prediction remains a promising research direction, but the stronger claim that learned relational systems are broadly superior and economically undervalued is not established by our evidence. This distinction changes what I would do next. I would invest in a fair, falsifiable comparison, rather than present selected score advantages as a completed argument. The useful conclusion is a research commitment with a stop condition: pursue the opportunity while retaining the possibility that well-designed relational summaries offer the better practical system.

The mission's four steps must remain separate. Useful information in linked records does not automatically require a graph-native predictor. A good graph predictor does not automatically outperform a strong, equally informed alternative. A repeatable advantage on selected tasks does not establish robust advantage across databases. Finally, predictive performance does not by itself show that customers, engineers or investors underestimate the economic value. Each step requires additional evidence.

## A map of the design choices

The landscape is clearer when organized by representation, prior experience and adaptation. A representation specifies what the predictor receives. A prior here means the patterns learned before the target task. Adaptation specifies how the new task influences predictions. These axes can vary independently; the three course strategies are useful organizing examples rather than mutually exclusive classes.

In graph-native learning, rows are represented as linked objects and the predictor learns to combine information across those links. Griffin illustrates pretraining followed by target-task fine-tuning; graph-native therefore does not necessarily mean frozen in-context inference. In a synthetic relational-prior strategy, generated relational tasks supply prior experience. RDB-PFN shows that such a prior can be paired with DFS-linearized inputs. A reuse strategy such as RDBLearn aggregates related records and supplies the resulting table to an existing tabular foundation model. Its lack of new relational pretraining does not remove the backend's earlier training or the cost of constructing features. [Griffin](https://arxiv.org/html/2505.05568v1), [RDB-PFN](https://arxiv.org/html/2603.03805v5), [RDBLearn](https://arxiv.org/html/2602.18495v1).

Consider a customer at a fixed prediction cutoff. Linked orders can enter as individual row representations or as summary columns. Both routes use relational data. If two systems consume the same summaries but have different pretrained predictors, their comparison concerns those systems under that representation. It cannot establish that learned graph processing is necessary. Conversely, a summary's inability to distinguish two histories only matters predictively if that distinction bears on the target. Expressivity and useful accuracy are related questions, but one is not a substitute for measuring the other.

## The open/proprietary gap depends on the contract

“Best open model” is incomplete until it identifies what is open and which methods are eligible. Implementation code, weights, training data and protocol access are different resources. A prediction API can be useful without making historical training reproducible. Open code can be inspectable while leaving a local rerun blocked. I therefore separate reported performance from reproduction access.

The frozen L191 comparison uses KumoRFM-2 v1 Tables 3, 4, 7 and 8. On Table 3, including only the declared open-code foundation methods yields a 4.0533-point AUROC gap to Kumo. Including the declared supervised methods reduces it to 1.5417 points. The winner in that broader single-method pool is RelGNN. This does not make either calculation dishonest; it makes the eligible pool essential to interpreting the sentence. Both are retrospective comparisons of published test results. Neither supplies a validation-based model-selection procedure. [KumoRFM-2 v1](https://arxiv.org/html/2604.12596v1), [L197 report](report.json).

The audit preserves all 491 numeric cells, including summaries. Five aggregates exceed the derived rounding bounds, and 34 displayed mean ranks differ from their published counterparts. These differences limit the precision of a polished headline. They are retained as discrepancies, not interpreted as evidence of misconduct. Regression gaps are computed from taskwise ratios to the same LightGBM baseline. They cannot be pooled with AUROC or raw errors from other targets.

## The strongest objection survives the replay

The strongest objection to a sweeping learned-architecture thesis is that a well-informed simpler pipeline may already recover much of the useful signal. In the complete saved F1 comparison, the basic GNN's test MAE averages 4.1231 across five seeds, versus 3.9489 for relational engineered features. The conditional driver-bootstrap interval for GNN advantage is −0.4496 to +0.0859. This crosses zero, so it establishes neither superiority nor equivalence. Both arms use relational information; the result does not show that such information lacks value. The bootstrap fixes the models and split and does not capture all race/time or new-database uncertainty. [L195 replay](../l195/falsification-brief.md).

The favorable evidence also deserves its full scope. Saved RDB-PFN predictions exceed TabICL by 0.004369 mean AUROC on the selected task, with positive differences on six of ten support draws. This makes the pretrained predictor worth investigating. The draws reuse a test population, however; they are not independent databases or proof of universal transfer. Re-executing these calculations does not create a new experimental replication.

The 21-task RDBLearn report is a different kind of evidence. Its published reference signs are 17 favorable, three unfavorable and one equal against AutoGluon+DFS. All fresh scores remain missing behind the preprocessing gate. A blocked reproduction should reduce confidence in what we have independently established, but it is not a measured performance loss. Counting it as one would turn an implementation obstacle into a fabricated empirical result.

## What would change my mind

I would strengthen the practical thesis after a preregistered comparison on untouched tasks that separates target-only inputs, legal relational summaries and learned relational processing. The study must fix temporal availability, validation search budgets, comparators and the unit of uncertainty before accessing test outcomes. A practically useful margin should be justified in the application's units, not chosen after seeing results. Total engineering, training and serving costs belong beside predictive benefit.

If strong relational summaries meet that margin at lower total cost across the declared task population, I would narrow the claim that learned graph processing offers the best practical route. If learned systems retain a robust advantage under those controls, I would strengthen that claim. I would still need a separate measurement of utility and adoption or investment expectations to argue undervaluation. This is a testable Year 6 direction, not a completed economic conclusion.
