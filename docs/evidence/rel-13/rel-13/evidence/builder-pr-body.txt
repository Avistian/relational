Adds a visible **Course progress** link to the existing index Reference navigation, targeting relative `progress.html` with existing styling. The production diff is one HTML list item.

Validation: frozen semantic HTML test failed on the missing link in test-only RED `71026606d87663721c9a3e16d5703542bd3ff0ec`, then passed unchanged on source `72002f56cf08cba06fdfde6fa8a22174e020173b`. Configured worktree verification passed 130 tests (exit 0, receipt `dirty:true` due to untracked evidence), and `scripts/search_index.py --check` passed.

[Readable evidence](docs/evidence/rel-13/rel-13/evidence.md)

Pending: independent verifier /blast-radius, four actual inspected index/progress desktop/mobile captures, supported parent acceptance receipts, publication/PR, hosted final-head CI and human merge. No browser screenshots, independent review or CI success are claimed by this builder handoff.
