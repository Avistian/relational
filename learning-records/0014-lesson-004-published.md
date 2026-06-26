# Lesson 004 published — grouped & nested CV

Published `lessons/0004-grouped-nested-cv.html`, reference `reference/grouped-nested-cv.html`, and reusable `assets/group-viz.js` (group-leakage grid: random K-fold vs GroupKFold atomicity). Two skills: (1) pick `GroupKFold` / `StratifiedGroupKFold` for non-i.i.d. grouped rows; (2) use nested CV to get an unbiased tuned estimate (report outer mean, not `GridSearchCV.best_score_`).

Primary reading: Cawley & Talbot (JMLR 2010) on selection bias / nested CV, plus sklearn §3.1.2.4 grouped iterators. Added both to RESOURCES.md (Year 1).

**Verified against sklearn 1.9:** KFold mean > GroupKFold mean (optimism confirmed); nested mean ≤ grid `best_score_`. Corrected the grouped+nested code — metadata routing requires `sklearn.set_config(enable_metadata_routing=True)` first; lesson now gives that plus a portable manual-loop alternative that slices `groups[tr]`.

**Implications:** User already uses grouped CV — lesson deepens to StratifiedGroupKFold + nested CV + groups-plumbing pitfall rather than re-teaching basics. Exit when lab prints four numbers (KFold mean, GroupKFold mean, grid best_score_, nested mean). Lesson 005 (pipelines & preprocessing) unlocks after. Temporal splits still deferred to Lesson 021.
