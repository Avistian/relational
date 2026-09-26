# Lesson 113: approved design and execution plan

User approved the combined scope on 2026-09-26: a controlled L112 GCN mini-batch bridge, followed by OGB Table 4 ClusterGCN on ogbn-products. The named experiment uses GraphSAGE aggregation; never attribute its accuracy to the L112 GCN.

## Frozen target

Ten runs, 50 epochs, 15000 METIS partitions, 32 partitions/batch, 3 layers, width256, dropout0.5, Adam0.001. Evaluate epochs20,25,...,50; select first best validation. Targets valid92.12±0.09%, test78.97±0.33%; predeclared descriptive mean tolerance0.5pp. Original seed list and METIS partition unavailable: record these historical gaps. No test-based tuning.

## Work sequence

1. Pin publication-era release and runtime; audit model/data/sampler/selection.
2. Independent dense and original-model tests for aggregation, gradients, induced graph boundaries, exact layerwise inference, masking, and first-max selection.
3. Visible canonical model/trainer and bounded cloud data preparation/pilot. Measure partitioning, epoch and inference times separately. Budget is USD10 aggregate, including failures, builds, checks, retries and storage; automatic retries disabled.
4. Start the ten-run schedule only when a pilot-based estimate with reserve fits. Otherwise execute affordable complete runs and mark ten-run experiment INCOMPLETE. Do not shorten the schedule and call it full reproduction.
5. Detailed lesson, portable figures, reference card, live student TODO/CHECK/EXIT notebook and executed solution. Include L112 mini-batch GCN bridge and a sampling/memory intervention.
6. Reconstruct metrics/checkpoints independently. Test desktop/mobile/no-JS/print, notebook figures, manifest navigation and copied Pages assets.

Complete source-pinned runner ships regardless of paid execution scope. Historical identity and whole-paper parity stay NOT_ESTABLISHED. Learner status stays PENDING_WRITTEN_DEFENSE. Live Colab and deployment are separately reported.

The brainstorming skill's suggested writing-plans skill is unavailable in the installed skill directories; this explicit checklist supplies the implementation plan.
