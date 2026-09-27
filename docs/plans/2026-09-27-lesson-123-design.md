# Approved Lesson 123: Temporal heterogeneous graphs

User approved 2026-09-27. Build from L122 REG construction toward L124 query tables. Teach root-time, multi-hop heterogeneous neighborhoods, explicit static assumptions, boundary equality, availability and reverse-link semantics. Include visible sampler/eligibility/audit tasks, an interactive two-query trace, independent SQL and PyG checks, full real F1 temporal graph exercise, standalone notebooks and reference.

## Execution plan
1. Implement and test temporal eligibility, incoming multi-hop sampling and audits. Use synthetic adversarial cases for unavailable ingestion histories; do not manufacture real F1 availability times.
2. Preserve L117 released model/trainer. Instrument every training/evaluation loader batch without consuming randomness. Audit timestamps, global edge identities and query isolation. Pin source and data.
3. Reserve eight one-hour T4/2-core/16GiB worker slots at USD .00022572/second = USD6.500736 plus USD3.499264 reserve under USD10 aggregate. Run one pilot, then five fresh ten-epoch full F1 fits only after the pilot passes. No automatic retries.
4. Audit every prediction/checkpoint and released query identity; report RelBench v1 Table7 driver-position MAE 3.193/4.022 with predeclared descriptive .2 tolerance. Distinguish selected release replay, whole-paper/historical parity, availability guarantees and learner mastery.
5. Build and execute notebooks, inspect figures and desktop/mobile/no-JS/print, verify copied Pages links and deterministic regeneration. Update manifest/curriculum/resources/evidence. Publishing not requested.

The named writing-plans skill was not available in the installed skill catalog or filesystem search; this approved execution sequence serves as the implementation plan. Existing unrelated work is preserved.
