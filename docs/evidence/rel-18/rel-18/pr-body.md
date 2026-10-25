# REL-18: cached constructor-feature ablation

Add strictly prior constructor history to the local relmini driver-DNF baseline,
with a paired constructor-feature ablation that preserves the original driver
features, split rows, seed and model. Latest conflicting team assignments become
unknown; missing history uses zero constructor features. Constructor record count
counts records, not distinct races. Historical team and missing classified position
are proxies for label-day team and DNF respectively.

Accepted package: `16b1274a36abf727809e0b04aa64a0ff5f7769c2`.
Parent separate verifier: ACCEPT; **risk_flag=YES**. Acceptance covers the local
software/scientific evidence package. This Markdown descendant requires a separate
verifier before publication; package acceptance does not accept this new document.

**LOCAL reproducibility evidence only; not RelBench-comparable.**

## Actual cached paired results

Existing pinned local cache: rel-f1 / driver-dnf; train=11411, validation=566,
test=702. Both arms use seed 0, StandardScaler and
LogisticRegression(max_iter=1000, random_state=0), fitted on training rows only.
Six original driver features are preserved; three constructor fields add prior
record count, mean points per record and a history indicator. Driver assignment
and constructor aggregation both use dates strictly before the label date.

| Arm | Validation AUROC | Test AUROC |
|---|---:|---:|
| Original baseline / without constructor | 0.7251156462585033 | 0.7885912262723856 |
| With constructor | 0.740607709750567 | 0.7915385741472696 |

Values above are verbatim from the committed
[full cached pair receipt](https://github.com/Avistian/relational/blob/16b1274a36abf727809e0b04aa64a0ff5f7769c2/docs/evidence/rel-18/rel-18/source-full/receipt.json).
[Raw paired predictions](https://github.com/Avistian/relational/blob/16b1274a36abf727809e0b04aa64a0ff5f7769c2/docs/evidence/rel-18/rel-18/source-full/predictions.csv)
retain 2,536 records: both arms on the same 1,268 evaluation labels, with exactly
equal original/repeated probabilities. Duplicate-fit equality shows local
repeatability only. The separate verifier recomputed AUROC from these predictions;
last-decimal differences from the receipt are floating-point rounding.

One dataset/task, one seed, unofficial features and no official evaluation harness
or uncertainty estimates: these descriptive numbers support no significance,
causal benefit, general improvement or multi-seed robustness claim. The verifier
checked predictions/features/provenance without independently refitting real models.

## Retained proof and verification scope

- Scientific execution / canonical GREEN: `557fa7ba846588ec182ec2fd4b51fc98ae3a1a57`.
  The receipt pins source, input, split and artifact hashes. The
  [artifact manifest](https://github.com/Avistian/relational/blob/16b1274a36abf727809e0b04aa64a0ff5f7769c2/docs/evidence/rel-18/rel-18/artifact-manifest.json)
  retains the evidence inventory.
- [Canonical replay receipt](https://github.com/Avistian/relational/blob/16b1274a36abf727809e0b04aa64a0ff5f7769c2/docs/evidence/rel-18/rel-18/tdd-receipt.json):
  missing-cutoff case only; RED run=1 failed=1 errors=0 skipped=0,
  GREEN run=1 failed=0 errors=0 skipped=0. This is historical proof, not a new replay.
- Historical builder `/verify-project`:
  [status](https://github.com/Avistian/relational/blob/16b1274a36abf727809e0b04aa64a0ff5f7769c2/docs/evidence/rel-18/rel-18/configured-verification/status.json)
  and [raw stderr](https://github.com/Avistian/relational/blob/16b1274a36abf727809e0b04aa64a0ff5f7769c2/docs/evidence/rel-18/rel-18/configured-verification/stderr.txt):
  PASS, exit=0, timed_out=false, dirty=false, 95 tests on
  `054b684641ce30fbdc0d3e85abad3bf34ddc1b0b`. The parent acceptance report also
  records a separate historical 95-test PASS on delivery `16b1274`.
  Neither run verifies this Markdown descendant.
- [Builder handoff](https://github.com/Avistian/relational/blob/16b1274a36abf727809e0b04aa64a0ff5f7769c2/docs/evidence/rel-18/rel-18/REPORT.md)
  preserves original failures and source lineage. Its SOURCE_READY wording predates
  the parent's separate ACCEPT in task `rel-18/evidence/final-verifier/accept/REPORT.md`.
  Ancestry is implementation `557fa7b` → verification evidence `054b684` → accepted
  delivery `16b1274`; the later commits contain evidence changes only.

This change adds only this PR body. `/tdd` skipped: documentation only. Current
verification is static link/numeric/claim consistency checking; no new full suite,
model fits, canonical replay or scientific runs were performed for this document.

Hosted CI: **PENDING** for the eventual published head. Supported delivery receipts,
any required fresh source-bound environment admission, separate descendant review,
PR publication and human merge remain parent-owned gates. Existing readiness is
source-bound and expires; historical proof must retain its original SHA and actor.
No push, merge or merge approval is claimed here.
