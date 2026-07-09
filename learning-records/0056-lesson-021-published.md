# Lesson 021 published — Data splits in the wild (Q3 opener)

**Lesson 021** (curriculum lec 021, Year 1 Q3 · "Evaluation rigor & benchmark literacy"). The first Q3
unit and the on-ramp to the whole quarter: the *split* leg of L020's fair-comparison contract.

**Single skill:** recognize when a random (i.i.d.) split leaks the future and inflates the reported score,
and build a **temporal split** (train past → test future) — plus its CV analogue `TimeSeriesSplit` — that
measures what deployment actually faces. Default posture flipped: **assume temporal until you can argue the
rows are truly exchangeable.**

**Primary reading.** Accessible: Huyen, *Designing ML Systems* (train/test splits + data distribution
shifts, Ch. 4 & 8). Empirical anchor + core-paper **preview**: TabReD (Rubachev et al. 2024,
[arXiv:2406.19380](https://arxiv.org/abs/2406.19380)) — abstract + §1 + §5.4. Verified the abstract by
fetching arXiv directly (arxiv MCP unavailable; OpenML blocked by egress — arxiv.org reachable). Direct
quotes used in the lesson: random splits *"could present overly optimistic performance estimates"* and
time-based splits *"lead to significant differences in rankings… XGBoost's performance margin diminishes."*
TabReD also flags `electricity` as a leaky dataset — which is why the lab is synthetic, not electricity.

**Dataset / verification.** No network for OpenML (egress allowlist: arxiv yes, openml no) and no local
cache, so — appropriately, since this is a *mechanism* lesson — a controlled **Tier-C synthetic stream**
with a *rotating* label rule (pure concept drift; feature distribution stationary). `labs/_verify_l021.py`
+ the executed solution reproduce identical numbers:
- **logistic:** random-CV **0.846** · TimeSeriesSplit **0.819** · temporal-holdout **0.758** → optimism gap **+0.088**
- **hist_gbdt:** random-CV **0.832** · TimeSeriesSplit **0.809** · temporal-holdout **0.757** → gap **+0.076**
- **drift made visible:** corr(x0,y) **+0.72 → +0.12**, corr(x1,y) **+0.10 → +0.71**, prevalence flat
  (0.497–0.530) → the *rule* drifted, not the base rate.
- ordering (random-CV > TimeSeriesSplit > temporal-HO) is the whole lesson in one row: random CV
  contaminates training with the deployment period; the last-block holdout is the harshest/most honest.

**Two new reusable viz assets (Q2-retrospective standard #9 — one viz per distinct mechanism):**
- `assets/temporal-split-viz.js` — the *same* time-ordered stream split two ways, stacked side by side
  (not a toggle): random scatters red test rows across all time; temporal cuts once (dashed line, test =
  future). Reshuffle button. Mechanism = split geometry / "future bleeds into training."
- `assets/drift-viz.js` — grouped bars of corr(x_j, y) across 5 time buckets (x0 fades, x1 rises, x2
  flat) + a rotating "rule direction" dial driven by a bucket slider. Mechanism = concept drift / *why*
  temporal matters.
Both verified headless in Node (`labs/_viz_check_l021.js`, 14/14). **Browser MCP still unavailable** (empty
tools folder) → headless only, per the standing gap.

**Pedagogy (all Q2-retrospective standards applied):** spaced-retrieval warm-up (`upTo: 21`);
prediction-before-reveal on the optimism gap (random vs temporal, who's higher); teach-back on *why* random
CV is optimistic under drift; 3 quizzes. Fed the artifacts: retrieval-pool +2 (`l021-temporal`
[misconception], `l021-timeseriessplit`); paper-deck +1 (`rubachev2024`); misconceptions **M18** (Q3
section: "a random split is default-safe" → temporal/`TimeSeriesSplit` under drift); thesis-dossier +1
(BAR+FOR, C3/C1); `reference/glossary.html` +Q3 section (i.i.d./exchangeable, temporal split, concept
drift, covariate shift, TimeSeriesSplit, optimism gap). `node labs/_check_pedagogy.js` clean.

**Lab:** `labs/0021-data-splits-in-the-wild.ipynb` — Tier-C, self-contained (no relkit; incremental rule is
for reproduction labs, not concept labs — same call as L019). 3 TODO (crucial fragment each) + stretch:
T1 = the temporal cut (`X[:cut]`/`X[cut:]`), T2 = `TimeSeriesSplit` + `cross_val_score`, T3 = per-bucket
`np.corrcoef(...)[0,1]` drift diagnostic. Student blank (9 `____`, 0 outputs); solution executed clean (all
CHECK + EXIT) and gitignored. Manifest → 21 entries (quarter 3); all labs re-rendered to `labs/html/`.

**Thesis bridge:** temporal evaluation is load-bearing for the RDL thesis — RelBench/RDL score with strict
time cutoffs by construction (Fey 2024), and TabReD shows time-based splits are where single-table
rankings actually shake out. This is both the split leg of the fair-comparison contract and the reason
claim **C3** (the RDL gain must survive honest temporal evaluation) is where the thesis is won or lost.

Next: Lesson 022 (label leakage patterns — Kapoor & Narayanan 2022, `2207.07048`; spotting leakage in
feature engineering), continuing Q3.
