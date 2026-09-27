# Lesson 138 — E-commerce task deep dive

Approved by user 2026-09-27. Target rel-amazon/user-churn, RelBench v1 Table 6 basic RDL five runs: validation70.45±.06/test70.42±.05 AUROC percentage points. Full released source/data/preprocessing/training/evaluation protocol; no silent row/feature/epoch reductions. USD10 aggregate including preparation, seeds, retries and checks; USD3 overhead reserve. No publication requested.

## Teaching

Carry L137 error diagnosis into a new domain: what event does the target actually observe? Reconstruct eligible customer queries and future-review absence; trace customer-review-product messages under one query cutoff; distinguish ranking quality from an intervention decision. Three live tasks: window contract, temporal neighborhood, tied-score AUROC. Include architecture, worked boundary examples, prediction-before-reveal, reference and portable student/solution notebooks. No learner completion inferred.

## Execution

1. Pin source commit9aa346267c2e1c560bd92da07d6f4ad1ca2f0639, database/task archive checksums and GloVe revision. Record historical/current differences.
2. Measure full archive dimensions, independently audit task labels, and time actual text embedding/preparation. Project complete workload including five seeds and validation. Reject unaffordable dispatch before spending; retain all attempts.
3. Supply complete visible model/trainer and executable local/Modal/Colab lanes. If preparation or training exceeds cap, preserve exact full protocol as NOT_RUN/INCOMPLETE and complete bounded task audit separately.
4. Independently verify metrics and identity; execute standalone notebook, meaningful boundary checks, browser desktop/mobile/keyboard/reset/noJS/print and copied Pages assets. Update course manifests and evidence ledger.

## Evidence

Declared descriptive closeness tolerance:1.0 AUROC percentage point, not equivalence. Pilot timings are projections, not completed runs. Any course baseline is separate from paper RDL. Missing full runs remain NOT_RUN; whole paper and historical run identity NOT_ESTABLISHED; live Colab/deployment NOT_CHECKED unless actually verified.
