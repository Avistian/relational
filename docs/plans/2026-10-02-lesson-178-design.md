# Lesson 178: FM vs tuned GNN vs RDBLearn

Approved by the user on 2026-10-02. Named experiment: L178 Matched-Information F1 Comparison. Companion lane: complete L166 published-result replay, RDB-PFN v5 Table 9, three arms, seeds 0–9, 512 support rows, all 702 test queries.

## Frozen scope

Fresh course comparison: rel-f1/driver-dnf, complete 702-query released test population, independently audited raw labels, outcome windows and full (driverId, cutoff) keys. Same 1024 training examples per seed, seeds 0–2. Released RDB-PFN, freshly trained RelGNN and actual RDBLearn with TabICL. No extra target-history labels. Regenerate relational features from the same database under the same temporal contract. No opaque cached feature matrix accepted as an all-feature temporal proof.

RelGNN learning rates {0.001,0.005}, ten complete epochs per fit, validation-only checkpoint/configuration selection. Freeze ICL configurations before model execution. All task-level validation and test queries retained. Report AUROC and paired seed differences only after complete selection/identity checks. Equal information and equal spending ceilings are not identical computation. This is a course comparison, not historical paper replication. Preserve L175 temporal failure and L176 repeatability limitations.

Source/hash, raw-label, temporal and label-access gates precede paid model runs. Any source/provenance/scientific failure stops the fresh track as INCOMPLETE, without changing the task, dropping rows or silently shrinking the protocol. Retain evidence of the failure and an executable gate. Do not present unvalidated downstream code as a runnable established reproduction.

## Budget

USD10 total maximum; USD8 planned stop. Up to USD1.50 per model, USD3 shared overhead, protected remaining margin. Preparation, failed attempts, validation and retries are included. Timed pilot must project completion inside allocations before remaining work. No automatic retries. New cloud jobs require immutable identities, reservations and explicit timeouts. No paid dispatch before the scientific gates pass. Conservative local numerical runtime cap 3600 seconds includes preparation, tests, audits and notebook executions.

## Delivery

1. Pin original paper/source inputs and authenticate complete L166 replay evidence. Implement and test live comparison gates; independently reconstruct task semantics before model runs.
2. If admitted, execute complete fresh grid within reservations and independently audit predictions; otherwise preserve exact stopped state and diagnostic evidence.
3. Connected HTML lesson, quick reference, three-path architecture visual, interactive comparison gate, portable student and solution notebooks with meaningful live learner functions, visible code, source/protocol/deviation ledger and saved reports.
4. Execute portable solution from an empty directory; validate desktop/mobile, keyboard/reset, print/no-JS, local links, deterministic generation, source parity and actual figures. Build real Pages from Git index and authenticate copied evidence.
5. Update manifest, curriculum, resources and author preparation notes. Preserve existing staged work. No push/deployment requested; learner PENDING_WRITTEN_DEFENSE.

The historical paper-mirror skill was recovered from /home/avist/.codex/memories/skills/relational-paper-mirror-lesson/SKILL.md. The brainstorming-referenced writing-plans skill is unavailable; this approved document records the implementation sequence directly.
