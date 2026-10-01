# Lessons 161–170 publication verification

Published commit: `c66c81af664e6e33c0bb59f7d5790b6bf88fe3a0`.

[GitHub Pages workflow 36923060015](https://github.com/Avistian/relational/actions/runs/36923060015) completed successfully for both build and deployment.

- All ten lessons were reviewed individually; [review.md](review.md) records findings, fixes and retained evidence boundaries.
- Ten contract suites and ten browser-delivery suites passed, covering 5,600 interaction states. Shared pedagogy checks passed.
- Notebook prose and portable diagrams were rebuilt; executable solution cells match their saved execution hashes exactly and successful outputs were preserved. No new training, benchmark inference, cloud jobs or paid API calls ran during this review.
- The complete Pages build succeeded from the Git index. Across lessons, prepared notebooks and reference pages, 809 local links resolved.
- All 197 checked public files matched the tested build by SHA256, including notebooks, diagrams, linked assets and the review.
- Every live lesson passed at 1200px and 375px: all figures loaded, no page overflow, no desktop figure clipping, and no JavaScript exceptions, failed site requests or HTTP errors. All 26 figures were included. Narrow-screen figures scroll locally rather than shrinking their text.
- Both manifest-driven galleries expose all ten entries. Live print/no-JavaScript views and the L170 evidence-gate intervention/reset passed.

[Start at Lesson 161](https://avistian.github.io/relational/lessons/0161-what-is-a-foundation-model.html).

The machine-readable records are `checks.json`, `notebooks.json`, `pages.json`, `browser.json`, `deployment.json`, `site-hashes.json`, `live.json` and `live-browser.json`. The three `*_check.py` scripts reproduce the build/link, browser and live-byte checks. Screenshots were inspected locally; the browser record records their paths.

## Evidence limits

Historical authoring-time `deployment=NOT_CHECKED` fields remain historical receipts; this record establishes the later successful publication. Live Colab and personal written defenses remain unverified. Griffin's budget stop, KumoRFM's historical-artifact gaps, checkpoint-replay versus pretraining distinctions, temporal/provenance gaps and L169's exact-repeatability diagnostic are retained. Successful delivery does not upgrade those scientific or mastery claims.

## Existing site-size follow-up

The tested complete site contains 10,975 files totaling 1,538,363,933 bytes before later review-record additions. Most of that size predates these ten lessons (about 168 MB belongs to their new lesson packages). This exceeds [GitHub's documented 1 GB published-site limit](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits). The actual workflow and live checks succeeded for this release; that observation is not a guarantee of future capacity. Reducing the inherited publication footprint needs a separate pass that preserves existing lesson download URLs and byte-pinned evidence. No archived evidence was silently removed or rewritten to reduce size.
