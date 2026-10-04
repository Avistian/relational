# B20 author review

Approved scope: `docs/plans/2026-10-04-b20-design.md`; design commit44f95c01. Course experiment B20-ORDER-2x2-3 is separate from the historical Curriculum Matters Table1 A/B target. No subagent review or learner mastery is claimed.

## Scientific evidence

- Complete12 fits,6 matched ordering pairs and4608 held-out predictions. Save every pool, full exposure schedule, optimization loss trace, initial/final identity and weights. Fixed192 updates/96 tasks twice, no result-driven tuning. Models are weak near chance; no consistent staged advantage. Final metric validation uses positive-negative pair enumeration independent of sklearn's original scorer; original saved model predictions reproduce exactly.
- Model output/gradient parity is against B13's inherited visible RDB-PFN implementation at identical compact dimensions. This is not Curriculum Matters author-code parity. Query labels rejected, query-batch and support-permutation checks pass; support-only normalization stays unchanged under a query intervention.
- All322 scores across14 rows and23 tasks reconstructed from archived Tables5–6. All12 Table1 averages match at3 decimals. The best-ours row averages to0.729652 rather than0.715 and equals the per-task maxima across stages; a per-task oracle differs from one selected checkpoint. FinalA beatsB on21 tasks; B wins Diabetes130US and pol. AppendixB exact4-decimal claims do not all match rounded input arithmetic, though the selected differences are compatible with rounded inputs.
- Historical corpus/generator/schedule/architecture/optimizer/seed/split/selection/support identities remain unresolved. The primary paper supplies no experiment code link; bounded search found no authenticated author implementation. TF11 omission, duplicateG and C/E initialization confound retained. Historical reproduction INCOMPLETE_SOURCE_PROTOCOL_GATE; fresh paper inference/training and whole-paper reproduction NOT_RUN.

## Teaching and visuals

One skill: identify an ordering effect at fixed exposure. Worked scalar SGD trace reveals path dependence, then a full input-to-query-head PFN diagram shows where the scheduler enters training. All12 seed scores and the AUROC-denominator figure remain separate. Compared with the suggested `growth-viz.js` (tree expansion), a dedicated schedule component directly computes weight updates from the same four examples. The existing tree widget does not represent this mechanism and was not repurposed.

Student notebook has three live definitions, with surrounding source visible in coherent chunks. Frozen artifact bytes are a collapsed bootstrap cell; they are hidden in prepared HTML so the model and trainer remain easy to find. Portable figures are embedded. Default lane authenticates/replays the complete saved evidence; optional fresh course training exposes the full12-fit loop. Paper operator explicitly fails closed instead of claiming an invented historical trainer.

## Verification receipts

- `labs/_verify_b20_results.json`: independent scorer,6 pairs,8 rejected corruptions and model invariance/source checks.
- `labs/_build_b20_results.json`: deterministic lesson/reference, notebooks, four figures and packet.
- `labs/_execution_b20_results.json`:18-code-cell solution executed from an empty directory; exact full-report parity.
- `labs/_delivery_b20_results.json`: learner blank/wrong functions, corrupt source, paper gate, real browser desktop/375px, keyboard/reset/prediction/teach-back, noJS/print and portable images.
- `labs/_pages_b20_results.json`: real workflow executed from isolated Git index, copied bytes/links and real-index preservation.

Live Colab NOT_CHECKED; deployment NOT_REQUESTED. Full package stays local. Only approved design committed. Author preparation is not learner completion; PENDING_WRITTEN_DEFENSE.

Final copied-site check: actual workflow PASS;68 B20 files byte-identical and45 local links valid; real Git index preserved. The final seal and bookkeeping receipts were finalized after this copy check.

Aggregate local numerical accounting: 237.186/3600 seconds, including failed attempts, repeated final notebook builds/checks,60-second preparation and30-second final allowances. Paid compute USD0. Actual PNG figures and desktop/mobile/notebook renders inspected; scrollable figures preserve label size.
