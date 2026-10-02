# Lesson 171 — Corpus of databases

User approved the concrete scope on 2026-10-02. One skill: construct an auditable database corpus and exclude held-out source families before pretraining. Bridge L170's design defense to L172's schema tokenization.

## Frozen execution and limits

L171 RelBench Corpus and Holdout Audit: inventory all seven original databases from pinned installed RelBench 1.1.0 source definitions; extract declared table/key/time schemas, registry archive hashes and temporal boundaries. Audit every row/table/foreign-key relationship in the complete cached rel-f1 archive, whose SHA256 matches the installed registry. Exercise all seven leave-one-database-out manifest splits and deliberately contaminated source-family/duplicate-archive cases. Other six databases receive SOURCE_INVENTORY_ONLY, not row-level certification. No sampling or silent reduction.

USD0 cloud/API; 1800 seconds aggregate local data-audit, checks and notebook execution including failed attempts. Initial read-only parquet/provenance probe took 1.41537022 seconds and is included. Stop with INCOMPLETE at cutoff. Source fetching, authoring, rendering and browser/publication validation are delivery work. No model training or publication. Whole-paper/foundation-model pretraining NOT_RUN. Historical availability, undisclosed lineage and learned-model transfer NOT_ESTABLISHED. Learner PENDING_WRITTEN_DEFENSE.

## Implementation and delivery

1. Use test-first contracts for manifest validation, transitive source-family/duplicate-archive holdout exclusion and full PK/FK/time audit. Independent membership/join oracle, source-extraction checks and corrupt-evidence cases.
2. Pin seven source modules, registry hashes, loader semantics, primary reading, complete F1 zip and extracted parquet bytes. Record source-family declarations separately from proof of provenance; data usage rights remain review fields, not automatic clearance.
3. Create a concise core HTML lesson plus complete portable student/solution notebooks, reference card and corpus manifest. Visible canonical code and live learner functions drive results. Synthetic contamination cases clearly separated from real records. Notebook embeds evidence and figures and runs in an empty directory.
4. Visualize database source-family exclusion, full actual F1 FK coverage and the distinction between snapshot time windows and availability. Interactive holdout selector and contamination control preserve clean baseline; figures legible in notebook and mobile.
5. Execute independent checks, solution parity, desktop/mobile/keyboard/reset/no-JS/print, source/links/build determinism and real Git-index Pages build. Update manifest, resources, notes, reference vocabulary and thesis ledger without mastery claims.
