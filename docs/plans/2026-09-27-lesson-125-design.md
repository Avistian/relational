# Lesson 125: PyTorch Frame deep dive

Approved by the user on 2026-09-27. Scope: L125 only; L125b remains separate.

## Learning contract
Trace raw typed columns through materialization, per-type encoders, column interaction,
row embeddings, and the GNN/prediction/loss boundary. Deepen L075 and connect L122–124.
Learners implement meaningful encoder/alignment functions, predict failure modes,
and defend train-only preprocessing and row identity in their written exit artifact.
Author execution does not establish learner mastery.

## Deliverables
HTML lesson with computation-specific figures and an interactive intervention;
student and solution notebooks with portable images and visible load-bearing code;
reference sheet; pinned source/protocol/deviation ledger; exact reproduction commands;
behavioral, source, notebook, desktop/mobile/no-JS/print and copied-Pages checks.

## Selected reproduction
Hu et al., PyTorch Frame arXiv:2404.00776v2 §5.3/Table 2:
rel-stackex-engage, Frame ResNet + PyG HeteroSAGE, reported ROC-AUC .854.
Audit historical dataset, task, split, preprocessing, model, training, selection,
seeds and metric before any numerical comparison. A modern RelBench task or F1 fit
cannot silently replace this target. If artifacts/protocol cannot be recovered,
record the precise gaps and full target NOT_RUN; execute useful mechanism evidence
without claiming paper parity. Whole-paper reproduction remains a distinct scope.

## Compute
Aggregate USD10 limit including pilot, seeds, retries and validation. Current Modal
T4 +2 physical CPU cores +16GiB = USD .00022572/s. At most eight one-hour worker
reservations = USD6.500736 plus USD3.499264 overhead reserve. Pilot gates subsequent
work, no automatic retries, stop before exceeding reservations. Historical
unavailability is not repaired by spending on a different dataset.

## Implementation plan
1. Audit upstream history and data availability; pin evidence and report status.
2. Test encoder/identity/leakage contracts; implement visible teaching components.
3. Execute complete available-data feature path and the target pilot if feasible.
4. Write causal lesson, figures, reference, notebooks and meaningful exercises.
5. Update manifest/resources/curriculum; verify execution and copied-site delivery.
6. Report exact completed checks and remaining reproduction limitations.

Writing-plans skill is not installed; this explicit plan supplies the transition
from approved design to implementation. No publication was requested.
