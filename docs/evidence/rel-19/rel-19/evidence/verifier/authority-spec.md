# REL-19
FEATURE_KIND: non-ui
FEATURE_ESTIMATE: rel-19 | 45

## Scope
Yes — check existing relmini feature joins against each label's temporal cutoff using only cached data. Missing cache is explicitly NOT_RUN (not a pass). No downloads, no training, no fetch.
AREA: labtask
OWNS: tests/relmini/test_real_leakage.py, docs/evidence/rel-19/rel-19

## Done when
Existing relmini joins respect label temporal cutoff on cached real data; no downloads or training; fixture proves future feature rejected; missing cache explicit NOT_RUN not completion.

TDD genuine tests-only RED, frozen GREEN, separate verifier and local proof. No push/merge, installs/spend, browser/model/protocol executions or permission/profile changes.

## Done when
Real cached-data cutoff assertions and a deliberately future-dated fixture prove leakage rejection; missing cache is NOT_RUN.
