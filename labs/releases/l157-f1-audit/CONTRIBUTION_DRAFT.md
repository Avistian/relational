# Draft: Reproduce F1 preprocessing policies with explicit fit-horizon audits

Status: local draft, NOT_SENT. Public artifact URL: PENDING_PUBLICATION. Intended route: standalone reproducibility repository; any upstream issue must first verify applicability to current upstream.

## Problem and resulting behavior

In the pinned RelBench implementation9aa346267c2e1c560bd92da07d6f4ad1ca2f0639, F1 processors use dated database rows through2010-01-01 before query-specific temporal sampling. A deployment contract requiring processors fixed at2005-01-01 has a different fit population. This package reproduces both policies and tests the difference explicitly. It does not claim that released transductive preprocessing violates the benchmark's intended rules.

## Concrete check

Run `python cli.py replay` to validate all archived evidence. Use the complete environment and prepare/train/freeze-evidence sequence in README.md for a fresh reproduction. See protocol.json for all seeds, query populations, clipping, selection and the precise paper target. The README command block is the canonical executable recipe.

A dated result after2005 can change a processor fitted through2010 even when the neighbor sampler excludes that result for an earlier query. Fitting dated type discovery, vocabularies and statistics by2005 removes that specific fit-population dependence; it does not recover absent arrival histories. A small upstream regression test should vary a post-horizon row while keeping the allowed fit rows fixed, then check processor statistics. Before writing that patch, establish the expected policy with current upstream code and maintainers.

## Measured evidence

All ten fresh package-only fits completed, five per policy and ten full epochs per fit. Released test MAE **4.015191 ± 0.150212**; fixed2005 **4.073480 ± 0.250688**; corrected−released **+0.058289**. Uncertainty shown is sample seed SD. All8712labels and443552SQLvalues independently rebuilt; all12590held-out predictions independently rescored. Source-model parity and root/edge/cutoff checks accompany each run. Complete checkpoint selections, source hashes and package-manifest identity are retained.

## Limits and review question

This is one task and a course policy intervention. Original nonfinite first-backward gradients are preserved. Historical data identity, static attribute versions, schedule publication and ingestion histories remain NOT_ESTABLISHED. Prior test exposure is disclosed; no new test-based tuning. No claim of universal leak freedom, modern upstream bug, statistically established superiority or whole-paper reproduction follows.

Would an explicit processor-fit population be useful as a documented option for deployment studies? A contribution should clarify that policy without silently changing published baseline semantics.

## Before public submission

Review the exact package and license notices, select the repository destination, publish the versioned artifact, and verify its URL and bytes. Public sharing is not maintainer endorsement. This draft records no maintainer response and was not sent automatically.
