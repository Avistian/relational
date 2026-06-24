# Teaching Notes

## Session 0 — 2025-06-24

- User invoked `/teach` with a contrarian thesis: relational deep learning and foundational relational models are undervalued and may outperform current approaches.
- Mission established: multi-year arc, ~1 hour/day, from basic tabular through cutting-edge RDL research.
- Goal includes demonstrating value to the world (benchmarks, publication, or shipped result).
- Prior signal from workspace: user has done serious tabular pipeline work (grouped CV, structural missingness, sklearn baselines in `homework/report.md`) — do not re-teach median imputation from scratch; do validate leakage and evaluation discipline early.

## Session 1 — 2025-06-24

- User requested **detailed expert curriculum**: lectures + readings year-by-year, then novel research framing.
- Created `CURRICULUM.md` (240 lectures, 6 years, agent-facing) and `reference/curriculum.html` (student-facing).
- Extended roadmap from 5 to 6 phases (Year 6 = novel research & thesis validation).
- RESOURCES.md reorganized by curriculum year.

## Session 2 — 2026-06-24

- User asked to critically revise `CURRICULUM.md` for completeness on tabular/relational mastery; ensure TabPFN + newer approaches included. Used arXiv MCP (`user-arxiv-mcp-server`) to verify papers/IDs.
- **Year 2 fully restructured** into 4 coherent quarters: (Q1) classic neural architectures, (Q2) modern DL + honest baselines, (Q3) tabular foundation models / ICL, (Q4) SSL + encoders + bridge.
- **Added (verified arXiv IDs):** TabR `2307.14338`, RealMLP `2407.04491`, TabM `2410.24210` (best DL baseline), TabReD `2406.19380`, TabArena `2506.16791`, TabPFN v2 (Nature 2025), TabICL `2502.05564`, LoCalPFN `2406.05207`, Drift-Resilient TabPFN `2411.10634`, open-env critique `2505.16226`, ExcelFormer `2301.02819`, Trompt `2305.18446`.
- **Relational frontier added:** ContextGNN `2411.19513`, RelGNN `2502.06784`, RelGT `2505.10960`, Griffin `2505.05568`, RDB-PFN `2603.03805` (open synthetic-prior relational FM), GelGT `2605.15575`, RelGT-AC `2606.03040`.
- Added a **Verified paper index** appendix to CURRICULUM.md and a **currency rule** (re-search arXiv each quarter; only add SOTA / failure-mode / baseline papers).
- Critical framing added up top: GBDTs + RealMLP/TabM still win single-table under temporal splits; thesis lives at the *relational* frontier.

## Session 3 — 2026-06-24

- User started Lesson 001; completed comprehension check-in.
- Strong on temporal leakage; baseline-fairness intuition present.
- Retrieval gap closed: design matrix + label vector (structural forcing mechanism).
- **Lesson 001 complete** (2026-06-24).
- Lesson 002 published: design matrix & leakage join sketch.

## Session 4 — 2026-06-24

- User asked to add **optional papers** ("follow when time allows") and update the curriculum; scan arXiv + Christoph Molnar's blog for candidates.
- arXiv MCP (`user-arxiv-mcp-server`) was **not available** in this (cloud) environment — verified all arXiv IDs via web search/fetch instead.
- Introduced a **★ core / ◆ optional** paper tier in `CURRICULUM.md` (framing note + per-year "Optional / time-permitting" blocks + an "Optional / extension reading" index with verified IDs).
- **Added optional papers (verified IDs):** Y1 SHAP `1705.07874`, Molnar *Interpretable ML* 3rd ed.; Y2 CARTE `2402.16785`, Interpretable-ML-for-TabPFN `2403.10923`, TabLLM `2210.10723` (as the LLM boundary the thesis rejects), Molnar *Tabular Foundation Models* book + Mindful Modeler; Y3 over-smoothing `1801.07606`, DropEdge `1907.10903`, PNA `2004.05718`, over-squashing `2006.05205`, curvature `2111.14522`, Graphormer `2106.05234`, GraphGPS `2205.12454`, plus HAN `1903.07293` / Cluster-GCN `1905.07953` / TGAT `2002.07962` / RE-Net `1904.05530` indexed for lessons that already name them; Y4 4DBInfer `2404.18209`; Y5 deferred to the currency rule.
- Synced `RESOURCES.md` (optional groups per year + Mindful Modeler community) and `reference/curriculum.html` (hero note, TOC link, new Optional Reading section).
- **Christoph Molnar** (note spelling: Christoph, not Christopher) is now the interpretability spine; he's currently writing a *Tabular Foundation Models* book — directly on-thesis.

## Paper progress

Track with ✓ as completed:

- [ ] Y1: Chen 2016 XGBoost
- [ ] Y1: Grinsztajn 2022
- [ ] Y1: Fey 2024 §1
- [x] Y1: Fey 2024 §1 assigned in Lesson 001

## Preferences

- **Lesson length:** prefer longer, richer lesson HTML (~35–45 min reading + quizzes) over short skims; chunk with worked examples and multiple check-your-understanding blocks
- **Pace:** year-by-year phases; do not skip tabular foundations even though the thesis is relational
- **Time:** ~1 hour/day **baseline** (minimum on typical days); may study longer when energy and schedule allow — curriculum pacing assumes 360 h/year at baseline
- **Extra time:** prefer deeper labs, paper appendices, checkpoint work, or spaced retrieval — not skipping year exit exams
- **Technical:** lessons use plain `<script>` tags, not ES modules (works on `file://` and GitHub Pages)
- **GitHub Pages:** site at `https://avistian.github.io/relational/` once pushed and Actions enabled (Settings → Pages → Source: GitHub Actions)
- **Mobile:** bookmark the home page; lesson list grows as new lessons are published
- **Out of scope:** see [[MISSION.md]]
