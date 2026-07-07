# Year 6 — Novel Research & Thesis Validation · Lesson Decomposition

**Year goal:** an original contribution that **supports or falsifies** the contrarian thesis, with fair
baselines and honest limitations, released as a public artifact.

**These are not concept lessons.** Each numbered row is a **research milestone** spanning weeks; the
calendar is set by experiments and writing, not readings. The schema is adapted: **Skill** → the
competency the milestone exercises, **Work** → the concrete deliverable (replaces Lab), **Viz** →
report/figure artifacts where relevant, **Bridge** → dependency + thesis tie. Almost no new papers —
the reading was front-loaded in Years 1–5; the currency rule still runs each quarter for baselines.

The collapsed ranges `211–218 · Ablation matrix` and `231–234 · External review` are decomposed into
individual milestones below.

---

## Q1 · Hypothesis & experiment design (201–210)

### 201 · Sharpen hypothesis — *— (framing milestone)*
- **Skill** — state the research question as a single falsifiable claim with a measurable success/failure
  criterion.
- **Work** — a one-paragraph hypothesis + the exact metric/threshold that would confirm or refute it.
- **Viz** — reuse `checklist.js` (falsifiability rubric).
- **Bridge** — starts from Y5 L199's selected direction; grounds everything downstream.

### 202 · Related work deep dive — *— (positioning milestone)*
- **Skill** — position the hypothesis precisely against RelGNN, RelGT, and the FM paradigms — what's new,
  what's borrowed.
- **Work** — a related-work section (2–3 pages) with an explicit "closest prior work + delta" table.
- **Viz** — reuse `arch-family-viz.js` (where the method sits).
- **Bridge** — callback Y4 L141/L145, Y5 L170b; prevents reinventing; forward to L203.

### 203 · Experiment protocol — *pre-registration style*
- **Skill** — pre-register the experiment (datasets, splits, metrics, baselines, stopping rules) *before*
  running it to prevent post-hoc rationalization.
- **Work** — a pre-registration doc: exactly what will be run and how it will be judged.
- **Viz** — reuse `checklist.js`.
- **Bridge** — the honesty backbone of the project; callback Y2 L059/Y4 L148; forward to L204.

### 204 · Baseline suite locked — *— (baseline milestone)*
- **Skill** — lock the fair baseline suite: **trees + manual FE + RelGNN + TabICLv2 + RDBLearn + best open
  FM**, all under identical budget/splits.
- **Work** — the frozen baseline configs + a smoke-test run of each.
- **Viz** — reuse `checklist.js` (baseline parity).
- **Bridge** — the bar the contribution must clear; callback Y5 L178/L192–194, Y2 L066b; forward to L205.

### 205 · Implementation sprint 1 — core method — *— (build milestone)*
- **Skill** — implement the core of the novel method end to end on one task.
- **Work** — a running core method producing a first (possibly weak) number on one task.
- **Viz** — reuse `rdl-stack-viz.js` / `atomic-route-viz.js` as fits the method.
- **Bridge** — first evidence the idea runs; callback the relevant Y4/Y5 architecture; forward to L206.

### 206 · Implementation sprint 2 — ablations scaffolding — *— (build milestone)*
- **Skill** — refactor the method so each component can be toggled for clean ablation.
- **Work** — a config-driven implementation with per-component switches + logging.
- **Viz** — reuse `rdl-stack-viz.js` (toggle-able blocks).
- **Bridge** — enables the Q2 ablation matrix; callback Y4 L148 ablation discipline; forward to L211.

### 207 · Negative results log — *— (record milestone)*
- **Skill** — record what didn't work (and why) as first-class evidence, not noise.
- **Work** — a running negative-results log with hypotheses/outcomes.
- **Viz** — none (log artifact).
- **Bridge** — feeds the limitations section (L228); guards against selective reporting.

### 208 · Mid-point review — *continue / pivot decision*
- **Skill** — make an explicit continue/pivot decision from the evidence so far.
- **Work** — a go/no-go memo with the decision criteria and the call.
- **Viz** — reuse `checklist.js`.
- **Bridge** — callback L201 criteria; can loop back to Y5 L198's alternatives if pivoting.

### 209 · Statistical analysis — *Demšar 2006*
- **Skill** — apply significance testing across tasks/seeds (paired tests, multiple-comparison correction)
  and report effect sizes, not just point wins.
- **Work** — the statistical-analysis plan + a dry run on preliminary numbers.
- **Viz** — reuse `checklist.js` + a results table with CIs.
- **Bridge** — callback Y1 L023 Demšar; the rigor the claim needs; forward to L220.

### 210 · **Q1 checkpoint** — *Deliverable-based*
- **Deliverable** — a **working prototype** producing a real (baseline-comparable) number on ≥1 task under
  the pre-registered protocol.
- **Bridge** — proves feasibility before the full ablation grind; callback L203–L205.

---

## Q2 · Execution & ablations (211–220)

The `CURRICULUM.md` range `211–218 · Ablation matrix` is decomposed into eight ablation milestones.

### 211 · Ablation — core component — *— (ablation milestone)*
- **Skill** — isolate the contribution of the method's central novel component (on vs off).
- **Work** — the core-component ablation row across tasks.
- **Viz** — reuse `checklist.js` + results table.
- **Bridge** — the "does the new idea matter?" test; callback L206.

### 212 · Ablation — encoder vs message passing — *— (ablation milestone)*
- **Skill** — attribute gains between the row encoder and the graph/attention component (callback Y4 L148).
- **Work** — the encoder-vs-MP attribution rows.
- **Viz** — reuse `rdl-stack-viz.js`.
- **Bridge** — prevents mis-attributing a data/encoder gain to the method; callback Y4 L125b/L148.

### 213 · Ablation — graph construction — *— (ablation milestone)*
- **Skill** — measure sensitivity to REG construction (filtering/injection, callback Y4 L157b).
- **Work** — the graph-construction ablation rows.
- **Viz** — reuse `hetero-graph-viz.js` (variants).
- **Bridge** — callback Y4 L157b Desired-graph; isolates data-structure effects.

### 214 · Ablation — data/pretraining source — *— (ablation milestone)*
- **Skill** — test synthetic vs real vs mixed pretraining/data effects (callback Y5 L165b/L166).
- **Work** — the data-source ablation rows.
- **Viz** — reuse `pfn-prior-viz.js` if synthetic priors are involved.
- **Bridge** — callback Y5 L170b paradigms; sources gains to data, not just architecture.

### 215 · Ablation — scale — *— (ablation milestone)*
- **Skill** — characterize performance vs data/model scale + compute (callback Y4 L134, Y5 L177).
- **Work** — the scale curves.
- **Viz** — reuse `arch-family-viz.js` (annotated with scale) + a scaling curve.
- **Bridge** — callback Y5 L169 scaling; tests whether the effect survives scale.

### 216 · Ablation — temporal / split sensitivity — *— (ablation milestone)*
- **Skill** — verify the result holds under temporal splits and doesn't depend on a lucky split (callback
  Y2 L055, Y4 L156).
- **Work** — random-vs-temporal-split rows + a leak re-audit.
- **Viz** — reuse `split-viz.js` + `leakage-viz.js`.
- **Bridge** — callback Y2 L055/Y4 L156; the single most credibility-critical ablation for the thesis.

### 217 · Ablation — hyperparameter sensitivity — *— (ablation milestone)*
- **Skill** — show the result isn't a hyperparameter artifact (sensitivity around the chosen config).
- **Work** — the sensitivity sweep.
- **Viz** — reuse `search-viz.js`.
- **Bridge** — callback Y1 L017/Y4 L135 fair tuning; robustness of the claim.

### 218 · Ablation — cross-domain / cross-DB — *— (ablation milestone)*
- **Skill** — test whether the contribution generalizes across domains/databases (callback Y5 L168).
- **Work** — the cross-domain rows.
- **Viz** — reuse `hetero-graph-viz.js` (multiple schemas).
- **Bridge** — callback Y4 L139/Y5 L168; single-benchmark claims are insufficient (forward to L223).

### 219 · Robustness checks — *seeds, splits, domains*
- **Skill** — run multi-seed, multi-split, multi-domain repeats and report variance for every headline number.
- **Work** — variance-annotated versions of all headline results.
- **Viz** — reuse `checklist.js` + results table with error bars.
- **Bridge** — callback L209 stats; makes numbers defensible; forward to L220.

### 220 · **Q2 checkpoint** — *Deliverable-based*
- **Deliverable** — a **complete experiment table**: method vs the full locked baseline suite, across tasks,
  with ablations and variance, under temporal splits.
- **Bridge** — the empirical heart of the thesis; callback L211–L219.

---

## Q3 · Communication (221–230)

### 221 · Write technical report — *— (writing milestone)*
- **Skill** — write an 8–12 page report (problem, method, experiments, honest analysis) at publication clarity.
- **Work** — the full report draft.
- **Viz** — reuse the L220 tables + method figures.
- **Bridge** — the primary artifact; callback L220; forward to L225 critique.

### 222 · Open-source release — *— (release milestone)*
- **Skill** — release reproducible code + docs + fixed env (seeds, data access, one-command run).
- **Work** — a public repo passing a clean-clone reproduction.
- **Viz** — none (repo artifact).
- **Bridge** — callback Y4 L157 first contribution; the reproducibility standard the field expects.

### 223 · RelBench leaderboard submission — *RelBench v1 + v2 + 4DBInfer subset*
- **Skill** — submit to the RelBench leaderboard(s) **if results merit**, across v1 + v2 (+ a 4DBInfer
  subset) so the claim isn't single-benchmark.
- **Work** — the submission(s) + the multi-benchmark results.
- **Viz** — reuse `checklist.js` (submission protocol).
- **Bridge** — callback Y4 L136/Y5 L181; single-benchmark claims are insufficient (callback L218).

### 224 · Blog / talk outline — *— (communication milestone)*
- **Skill** — translate the thesis + result for a general technical audience.
- **Work** — a blog draft or talk outline with the one-slide thesis.
- **Viz** — reuse `flatten-loss-viz.js` / `arch-family-viz.js` as explainer figures.
- **Bridge** — legibility to skeptics (a mission success criterion); forward to L238 teach-back.

### 225 · Respond to critique — *steel-man opposing view*
- **Skill** — steel-man the strongest opposing view and address it with evidence, not rhetoric.
- **Work** — a rebuttal section answering the best anti-thesis argument.
- **Viz** — reuse `ott-viz.js` / `checklist.js`.
- **Bridge** — callback Y5 L195 stress-test; forward to L226 revision.

### 226 · Revise based on gaps — *second experiment round*
- **Skill** — run the targeted follow-up experiments the critique exposed.
- **Work** — the second-round results closing the identified gaps.
- **Viz** — reuse the L220 table (updated).
- **Bridge** — callback L225; forward to L227 final numbers.

### 227 · Final results — *lock the numbers*
- **Skill** — freeze the final, variance-reported results and stop tuning.
- **Work** — the locked results table (no further changes).
- **Viz** — reuse the final results table.
- **Bridge** — callback L219/L226; the numbers that go in the artifact; forward to L228.

### 228 · Limitations section — *honest scope*
- **Skill** — write honest limitations (where the thesis is weak, what didn't work, external validity).
- **Work** — the limitations section drawing on the negative-results log.
- **Viz** — reuse `checklist.js`.
- **Bridge** — callback L207/Y5 L179/L195; the intellectual honesty the mission demands.

### 229 · Future work — *next 3 experiments*
- **Skill** — propose the next three experiments the result opens up.
- **Work** — a ranked future-work list.
- **Viz** — reuse `checklist.js`.
- **Bridge** — callback Y5 L198; forward to L239 next-5-year map.

### 230 · **Q3 checkpoint** — *Deliverable-based*
- **Deliverable** — a **submission-ready draft**: report + released code + (if merited) leaderboard
  submission + limitations.
- **Bridge** — the artifact is complete pending external review; callback L221–L228.

---

## Q4 · Launch (231–240)

The `CURRICULUM.md` range `231–234 · External review` is decomposed into four milestones.

### 231 · External review — solicit — *— (review milestone)*
- **Skill** — solicit substantive external review from the community (callback Y5 L196).
- **Work** — review requests sent to ≥2 qualified reviewers/venues.
- **Viz** — none.
- **Bridge** — callback Y5 L196; the wisdom leg of the teach philosophy; forward to L232.

### 232 · External review — incorporate — *— (review milestone)*
- **Skill** — triage and incorporate external feedback (accept/reject each point with reasoning, per the
  `receiving-code-review` discipline).
- **Work** — a feedback-disposition log + the revised artifact.
- **Viz** — reuse `checklist.js`.
- **Bridge** — callback L231; verification, not performative agreement; forward to L233.

### 233 · External review — validate reproduction — *— (review milestone)*
- **Skill** — have someone else reproduce the headline result from the public repo.
- **Work** — an independent reproduction confirmation (or fixes if it fails).
- **Viz** — none (reproduction artifact).
- **Bridge** — callback L222; the strongest credibility signal; forward to L234.

### 234 · External review — final polish — *— (review milestone)*
- **Skill** — final editorial + reproducibility polish before release.
- **Work** — the camera-ready-quality artifact.
- **Viz** — final figures pass.
- **Bridge** — callback L230–L233; forward to L235 release.

### 235 · Camera-ready / release — *publish or ship*
- **Skill** — publish the artifact (paper, leaderboard entry, or shipped result).
- **Work** — the public release (DOI/preprint/leaderboard/deployment link).
- **Viz** — none (release).
- **Bridge** — the mission's "public artifact" success criterion; callback L234.

### 236 · Production pilot (optional) — *real deployment*
- **Skill** — pilot the model in a real setting with production constraints (callback Y5 L186).
- **Work** — a pilot deployment + a monitoring/latency/freshness report.
- **Viz** — reuse `checklist.js` (production readiness).
- **Bridge** — callback Y5 L186; optional evidence the value is real, not just benchmark.

### 237 · Year 6 retrospective — *— (reflection milestone)*
- **Skill** — assess what the thesis got right and wrong from the evidence.
- **Work** — the retrospective (thesis verdict with evidence).
- **Viz** — reuse the final results + `checklist.js`.
- **Bridge** — callback L227/L228; the honest verdict; forward to L240.

### 238 · Expert teach-back — *— (mastery milestone)*
- **Skill** — explain the full stack live — from XGBoost on a flat table to a relational FM — without
  hand-waving (a mission success criterion).
- **Work** — a recorded/live teach-back covering the whole arc.
- **Viz** — reuse `arch-family-viz.js` / `flatten-loss-viz.js` / `message-passing-viz.js` as the stack story.
- **Bridge** — demonstrates the expertise the 6-year arc was for; callback every year's synthesis essay.

### 239 · Next 5-year map — *— (planning milestone)*
- **Skill** — map the adjacent research directions the work opens for the next phase.
- **Work** — a next-5-year research map.
- **Viz** — reuse `arch-family-viz.js`.
- **Bridge** — callback L229; sustains the trajectory past graduation.

### 240 · **Graduation** — *Deliverable-based*
- **Deliverable** — a **public artifact** (paper, leaderboard result, or production deployment) with **fair
  baselines and honest limitations**, plus the thesis verdict.
- **Exit criterion (from CURRICULUM)** — public artifact with fair baselines and honest limitations.
- **Bridge** — the terminal milestone. The contrarian bet has been tested in public with evidence — success
  = the thesis is now *legible to skeptics*, whether it was confirmed or falsified.
