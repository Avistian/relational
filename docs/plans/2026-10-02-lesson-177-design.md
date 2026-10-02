# Lesson 177: Compute budget realism

Approved by the user on 2026-10-02. Named experiment: L177 Compute Feasibility Ledger.

## Frozen scope

Fully replay the accounting of all 300 L176 inference receipts, all six L173 pretraining fits, all twelve L174 adaptation fits and all three L175 cloud attempt reservations. Authenticate original input bytes. Independently reconstruct completeness, elapsed totals, reservations and pilot admission decisions. Preserve failed attempts, repeated validations and missing measurements. No fresh training, model inference, cloud submission or invoice access.

Separate repeated per-run metadata from distinct events: L176 load_seconds is repeated per task/model/phase; count it once per group. Worker-body time encloses the timed evaluations and model loading; do not add these nested totals to the enclosing clock. CUDA allocator peaks are not total device memory requirements. Source inspection clarified that L173 lacks individual fit timers while L174 has them; both CPU lessons lack peak-memory measurements. Human time was not recorded.

Compare what each workload achieves, not cross-model speed or accuracy. Pin current provider prices and RT-v1's reported eight-A100 durations as explicitly hypothetical cost calculations. A GPU-only lower-cost scenario is sufficient to reject an over-budget plan, but never sufficient to admit a complete plan. Existing run completion is not future memory feasibility or an invoice.

## Budget and delivery

USD0 new cloud/API spend. 1800-second aggregate local numerical limit including failures, preparation, tests, audits and standalone notebook execution. Stop/report INCOMPLETE at cutoff. Preserve earlier lessons and their staged changes. Whole-paper reproduction and fresh training remain NOT_RUN; learner PENDING_WRITTEN_DEFENSE. No push/deployment requested.

1. Write failing behavioral tests for reservation accounting, conservative forecasting, unknown measurements and scientific stops. Implement visible functions and independent adversarial checks.
2. Freeze source snapshots, complete receipts and accounting provenance; execute full audit and independent reconstruction. Save machine-readable ledger and uncertainties.
3. Build a connected HTML lesson, reference, reusable calculator, three portable figures and student/solution notebooks. Three learner functions must directly control the full audit and scenario decisions.
4. Execute solution in an empty directory; inspect visuals, desktop/mobile/keyboard/reset/print/no-JS behavior, complete links, inline source and deterministic generation.
5. Update manifest, curriculum, resources and author preparation notes. Stage intended files, build the real Pages workflow from Git index and verify copied evidence hashes.

The brainstorming-referenced writing-plans skill and historical paper-mirror skill are unavailable in the current skill locations. This approved plan records the implementation sequence directly; teach and test-driven-development provide the applicable workflow.
