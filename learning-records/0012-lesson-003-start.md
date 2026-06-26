# Lesson 003 started — train / valid / test & CV

User began Lesson 003 after completing Lesson 002 (5/6 on join-sketch classification; country/static-dimension distinction reinforced). Prior signal: already uses grouped CV in homework pipelines — do not re-teach i.i.d. basics from scratch; focus on train/val/test roles, stratification, pipeline-inside-CV, and where standard K-fold stops being valid.

**Focus this session:** three-set contract, `stratify=y`, `cross_val_score` on train only, split-viz widget, sklearn primary reading §3.1. Exit when lab prints matching class rates and CV mean ± std (or user says "lab done").

**Implications:** Lesson 004 (grouped & nested CV) unlocks after exit. Defer temporal splits to Lesson 021; user can optionally request a split-strategy audit on `report.md` before L004.
