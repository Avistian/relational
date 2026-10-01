# L165 KumoRFM-v1 Context and Reproduction Audit

Complete synthetic contract audit. No KumoRFM inference, fitted weights or benchmark predictions.

Counts: {'context': 144, 'graph': 96, 'scoring': 24, 'query_label_interventions': 2}.
Changing the isolated held-out label leaves query/context inputs identical. This checks this course packet, not proprietary internal masking.

| Scoring case | AUROC |
|---|---:|
| permutation-00 | 0.875000 |
| permutation-01 | 0.875000 |
| permutation-02 | 0.875000 |
| permutation-03 | 0.875000 |
| permutation-04 | 0.875000 |
| permutation-05 | 0.875000 |
| permutation-06 | 0.875000 |
| permutation-07 | 0.875000 |
| permutation-08 | 0.875000 |
| permutation-09 | 0.875000 |
| permutation-10 | 0.875000 |
| permutation-11 | 0.875000 |
| permutation-12 | 0.875000 |
| permutation-13 | 0.875000 |
| permutation-14 | 0.875000 |
| permutation-15 | 0.875000 |
| permutation-16 | 0.875000 |
| permutation-17 | 0.875000 |
| permutation-18 | 0.875000 |
| permutation-19 | 0.875000 |
| permutation-20 | 0.875000 |
| permutation-21 | 0.875000 |
| permutation-22 | 0.875000 |
| permutation-23 | 0.875000 |

The identical scoring values come from permutations of four invented records; they are not independent seeds or measured performance.

All context and graph interventions are recorded in report.json. Historical v1 driver-dnf target: 0.8241 AUROC (82.41 on the report scale), NOT_RUN; fidelity NOT_ESTABLISHED. Learner PENDING_WRITTEN_DEFENSE.
