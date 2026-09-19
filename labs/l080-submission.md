# Year 2 exit exam — my submission

Name/date:
Run file and SHA256:
Preset / dataset hashes / model source hashes / checkpoint hash / environment:

## Before running (freeze this)
Prediction time and available inputs:
Random and temporal split definitions; label availability assumptions:
Metric / candidate grids / stopping / tie rule / seeds:
My predicted outcome and a result that would change my recommendation:

## A. Cold teach-back
1. Irregular targets: mechanism, an intervention, and when the advantage disappears.
2. Privileged feature orientation: mechanism, a rotation intervention, and a limitation.
3. Uninformative features: mechanism, a nuisance-feature intervention, and a limitation.
4. TabM: shared weights, diversity, training loss, inference aggregation.
5. TabPFN v2: synthetic pretraining, labeled context, query mask, cost, prior mismatch.
6. TabICL 2025: column embedding, row interaction, in-context prediction; what this exam did not measure.

## B. Evidence
Paste per-dataset/per-regime mean ± sample SD over seeds, selected recipes and total selection costs.
Explain seed uncertainty versus dataset uncertainty.
State why random-to-temporal changes are not a causal drift estimate here.
Describe one failed audit and how you repaired it.

## C. Decision (250–400 words)
Recommendation for future-time deployment:
Why the single-table input can plateau:
Where ICL helps and when its prior/context/cost hurts:
First point-in-time aggregate to test before a GNN:
What would falsify my recommendation:

## D. Reproducibility ledger
Fresh training / frozen pretrained inference / saved-prediction replay:
Published-paper reproduction status and deviations:
Exact commands and any environment failures:

## E. Self-assessment
Score each rubric row 0–2 and cite an artifact. List uncertainties.
Submit this document, result JSON, completed notebook and cold explanations to the tutor.
A passing automatic audit is not a mastery pass.
