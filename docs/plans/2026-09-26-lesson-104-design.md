# Lesson 104: information leakage in time

Approved by the user on 2026-09-26 after review of the lesson, lab and ten-checkpoint evaluation scope.

## Learning outcome
Find a forbidden dependency in a temporal GNN prediction, implement a point-in-time boundary, and defend a corrected metric. Follow L103 without introducing another architecture. Distinguish event time, availability time, prediction time, label maturity and model-selection time. Connect the audit to Kapoor and Narayanan's taxonomy and Fey et al.'s temporal relational graph.

## Implementation plan
1. Freeze local L103 implementation, raw and processed data, all ten checkpoints, predictions and identities in a SHA-256 manifest. Preserve pre-existing user edits.
2. Test strict eligibility, equal-time exclusions, delayed availability, label-maturity selection and paired metric alignment before implementing the audit functions. Use the complete visible L103 TGAT model/trainer.
3. Replay the complete released all/new-node test populations from saved pre-evaluation RNG states. Check event IDs, negatives and scores against archived predictions, then independently reconstruct metrics.
4. In a separate extension, compare corrected strict-past, same-time-inclusive and +86400-second lookahead histories. Fix weights, examples, negative destinations and uniform variates across arms. Record illegal sampled records, prediction arrays and seed-paired AP changes. An illegal arm is invalid even if its score decreases.
5. Ship full from-scratch training commands and a gated notebook lane. Existing training is reused, not represented as fresh L104 training; release quirks and full-paper gaps stay explicit.
6. Build retrieval-first HTML, timeline/recursive-neighborhood visuals, portable notebook figures, three meaningful TODO/CHECK functions, reference, provenance and reproduction ledger. Register via manifest. Learner status stays PENDING_WRITTEN_DEFENSE.
7. Execute the solution, semantic/mutation checks, independent metric reconstruction, browser desktop/mobile/print/no-JS, deterministic rebuild and copied Pages checks.

## Budget and stopping rule
User limit: about USD10 total, including pilot/retries/checks. Current https://modal.com/pricing (2026-09-26): T4 USD0.000164/s, physical CPU USD0.0000131/core/s, RAM USD0.00000222/GiB/s. Conservative 2-core/8-GiB ceiling = USD0.00020796/s (USD0.748656/hour). One <=300-second pilot; ten <=3000-second seed calls; reserve at most two <=3000-second retries: USD7.55 resource-time ceiling, leaving USD2.45 for startup/build/check overhead. Record timed pilot before full launch. No automatic unbounded retries; no paid retraining in this lesson. Abort if projected aggregate exceeds the remaining budget; report incomplete coverage.

## Evidence boundaries
Checkpoint replay is fresh inference with reused training. Leakage arms are course interventions, not published TGAT or Kapoor result reproductions. Published target: Xu et al. Wikipedia Tables 1/2, retained from L103 with documented release/population differences. Historical identity INCOMPARABLE; full-paper parity NOT_ESTABLISHED. Live Colab and publication require separate evidence.
