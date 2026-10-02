# Lessons 181–190 publication verification

Published site commit: `543985d1d9a9e62f3b6ae2e363f127dd6a24fab2`.

[Individual review](review.md) records findings and changes for every lesson number. The user confirmed L190 authoring complete during review; it is included in this final release. L189 remains explicitly labeled as a draft. Publication does not complete the learner's written defenses or upgrade scientific evidence.

## Reviewed changes

Added explicit adjacent-lesson bridges and prerequisite explanations. Fixed desktop diagram clipping in GelGT and serving architecture, enlarged applicable diagrams, and preserved readable mobile scrolling. L190 now explains its different research shortlist and composite contrast relative to L189, defines its statistical vocabulary, and includes an updated five-page worked document. Notebook prose follows the lessons.

## Verification

- Ten contract suites and ten delivery suites passed, covering 1,116 interaction states plus keyboard/reset, print, no-JavaScript and deterministic-build checks.
- All ten solution notebooks preserve their recorded executed code hashes and saved outputs. No fresh model training, inference or paid runs were performed for this review.
- 1,275 sealed files remained unchanged; only 61 reviewed teaching/delivery artifacts changed. Pinned inputs and numerical evidence were preserved.
- The actual Pages workflow built from a clean Git index. All 1,596 checked local links/anchors resolved; the ignored L190 solution notebook was explicitly included.
- All ten local lesson pages passed at 1200px and 375px, with 31 loaded figures, no document overflow, no desktop figure clipping, no JavaScript exceptions and no failed site requests.
- Both manifest-driven galleries expose all ten lesson/lab entries. Portable notebook figures load; key architecture, mobile, notebook and print screenshots were visually inspected. L190's document exports to exactly five PDF pages with all page footers retained.

## Live publication

[GitHub Pages workflow 37041953015](https://github.com/Avistian/relational/actions/runs/37041953015) completed successfully for build and deployment. All **267 checked live files** match the tested site by SHA256, including lessons, notebooks, figures, shared assets, the review and the five-page PDF. All **ten live lessons** passed at 1200px and 375px, with no JavaScript errors, HTTP errors, failed site requests, document overflow or desktop figure clipping. Print, no-JavaScript and both gallery checks also passed on the public site.

[Start at Lesson 181](https://avistian.github.io/relational/lessons/0181-relbench-v2-autocomplete.html) · [Lesson 190 checkpoint](https://avistian.github.io/relational/lessons/0190-research-gap-checkpoint.html).

## Boundaries and workspace

L189 remains a reviewable draft. The user-confirmed completion of L190 authoring is distinct from its INCOMPLETE research checkpoint and PENDING_WRITTEN_DEFENSE learner status. Live Colab, new scientific experiments, historical reproduction identity and learner mastery are outside this publication claim. Historical authoring-time deployment receipts remain historical; this record verifies the later publication.

Publication used the isolated `review/lessons-181-190` worktree to preserve concurrent authoring. Reviewed artifact changes were copied back only where the active workspace still matched its pre-review bytes; concurrent later-lesson work was preserved. The active authoring branch/index was not reset or restaged. Remote `main` contains the release.

Machine-readable records and scripts accompany this report: `checks.json`, `notebooks.json`, `portable-figures.json`, `sealed-evidence.json`, `pages.json`, `browser.json`, `deployment.json`, `site-hashes.json`, `live.json`, and `live-browser.json`. Post-deployment receipts are committed separately without changing the tested lesson bytes.
