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

## Paper progress

Track with ✓ as completed:

- [ ] Y1: Chen 2016 XGBoost
- [ ] Y1: Grinsztajn 2022
- [ ] Y1: Fey 2024 §1
- [x] Y1: Fey 2024 §1 assigned in Lesson 001

## Preferences

- **Pace:** year-by-year phases; do not skip tabular foundations even though the thesis is relational
- **Time:** ~1 hour/day **baseline** (minimum on typical days); may study longer when energy and schedule allow — curriculum pacing assumes 360 h/year at baseline
- **Extra time:** prefer deeper labs, paper appendices, checkpoint work, or spaced retrieval — not skipping year exit exams
- **Technical:** lessons use plain `<script>` tags, not ES modules (works on `file://` and GitHub Pages)
- **GitHub Pages:** site at `https://avistian.github.io/relational/` once pushed and Actions enabled (Settings → Pages → Source: GitHub Actions)
- **Mobile:** bookmark the home page; lesson list grows as new lessons are published
- **Out of scope:** see [[MISSION.md]]
