# Course review from Lesson 50

The user moved the systematic review start to L050 after L001–002 corrections had already been made. Those corrections are included in this batch; L003–049 are outside the current pass. QUEUED ledger entries are inventory, not completed audits.

L050: corrected a claim that pairing automatically cancels seed variation; added two aligned numerical traces with the same model means but different difference uncertainty. Removed the notebook generator's opening recall prompt. Existing benchmark code and results are unchanged.

L051: added the full three-row smoothing calculation at bandwidths 0, 0.5 and 1, showing unchanged hard labels versus class collapse. Removed the notebook generator's opening recall prompt and clarified that invertibility establishes information preservation.

Verification: fresh mathematical checks and copied-weight logit/input-gradient parity for L050; fresh mathematical and pinned-upstream smoothing/rotation checks for L051; independent arithmetic for new examples; code/output preservation in student notebooks; delivery checks over stored predictions; 375px and 1200px browser checks. Inspected the changed mobile trace and split an unreadable six-column table into two readable three-column tables. Training was not rerun; historical scores retain their original scope. Live Colab is not verified.

L001: fixed a wrong RelBench source URL, replaced multiplicative raw child joins with independently aggregated CTEs, and checked displayed SQL against independent SQLite fixtures. L002: explained event time versus availability and added a delayed-arrival intervention with real feature arithmetic; browser toggling and eight numerical states checked.

Usage after the first two lessons from the requested start: weekly remaining 72%, compared with 73% at this iteration's baseline. The rounded account-wide allowance is not an exact task charge. Read usage.jsonl for actual session telemetry. User cutoff is 5% remaining; new work stops at 7% to reserve delivery.

L052–056 batch: L052 explains the key map's selection, weighting and directed-value roles with a scale/translation oracle; official forward/gradient checks pass. L053 clarifies that RealMLP combines architecture choices and meta-tuned defaults, removes generator recall, and repairs its obsolete partial publication staging. L054 fixes a stale notebook depth anchor and derives actual layer storage counts, including a small case where sharing costs more. L055 corrects its opening claim that random splitting is intrinsically leaky and temporal evaluation necessarily harder; 54 stored prediction/selection cases reconciled. Its upstream report checkout is absent, so the git-blob recheck was not completed in this pass.

L056 rejects null dataset/split/method/metric identities before aggregation. The missing-metric regression failed before the fix. The revised solution executed all 23 code cells in a fresh process. Frozen-input rank summaries and bootstrap statistics are exactly unchanged from the pre-fix committed JSON; pinned rank-to-win parity remains within 1.8e-15. This is fresh score reanalysis, with no model training. Existing numerical and source checks, mobile/desktop browser checks, and delivery verification accompany each lesson in the ledger.
