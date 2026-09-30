# Sequential review: Lessons 136–150

Reviewed 2026-09-30. Lessons 136–149 are publication candidates. Lesson 150 is under construction and is excluded from this publication review's completion claim. This is a teaching and delivery review, not fresh model training or a new paper-fidelity audit.

The sequence has a coherent progression: audit a score → diagnose errors → transfer the task to two domains → defend reproduction → inspect new architectures → compare them fairly → formulate questions → ablate components → record counter-evidence. Existing numerical results and reproduction limitations are retained.

| Lesson | Individual finding | Revision / visual assessment |
|---|---|---|
| 136 · Leaderboard literacy | Strong keyed example and normalization derivation; next link went to the year plan. | Added an accessible reading route and cutoff/checkpoint/seed reminders; linked directly to L137. Alignment, normalization and task-coverage figures fit an evaluation lesson; no new model diagram required. |
| 137 · Error analysis | Strong paired-loss and whole-driver bootstrap examples; needs a clearer handoff from error analysis to target validity. | Added support/uncertainty reminders and an explicit Amazon handoff. Paired error, snapshot and cluster figures explain the mechanisms; no new model introduced. |
| 138 · Amazon | Strong eligibility endpoints and two-hop explanation; the ending did not tell the learner what to carry into trials. | Added row/message/positive-label reminders and a missing-observation handoff. Retained the domain-specific book → review → customer architecture, future-row exclusion and prediction head. |
| 139 · Trial | Good pipeline-transfer contrast, but clinical vocabulary and tensor/training abbreviations arrive quickly. | Defined primary outcome, p-value and modifier before using the label rule; refreshed fanout, B, BCE and AUROC. Retained the facility → association → study map with dimensions, relation aggregation and training path. Linked to L140. |
| 140 · Reproduction checkpoint | Good separation of execution, score and protocol. L141 was still called planned. | Refreshed hyperparameters, percentage points and seed SD; updated the L141 handoff. Its two-domain architecture correctly compares recipes without inventing a new model. |
| 141 · Composite messages | Strong scalar derivation, but the architecture overview was a text-heavy stack of boxes. | Added tensor/channel/linear-map reminders. Replaced the overview with colored source/fact/destination roles, explicit fusion and weighted-message arithmetic inside the complete sampling-to-head path. Portable notebook image updated. |
| 142 · Edge pathology | Strong distinction between graph information and a lossy update; prevents overclaiming its scalar collision. | Added synchronous-layer/collision reminders and a direct L143 handoff. Retained walk-count and collision visuals plus the asymmetric one-layer/two-layer architecture comparison. |
| 143 · RelGNN reproduction | Clear protocol but assumes gradient, L1 and differential-oracle vocabulary. Ending lacks the transition to recommendation. | Added definitions and the scalar-prediction → candidate-ranking handoff. Uses the improved L141 architecture because this is the same model, not a new architecture. |
| 144 · ContextGNN | Strong worked two-tower example and owner-specific score replacement. | Refreshed bipartite roles, dot product, score-matrix dimensions and local membership; linked the return to scalar prediction in L145. Retained the two-branch architecture and local/distant ranking figures. |
| 145 · RelGT | Model-specific five-element and two-branch diagrams are informative. Sweep wording ambiguously referred to “both dropout settings.” | Made the nine configurations explicit: three depths × three dropout values, each value applied to both dropout sites. Added token/projection/local/global reminders and comparison handoff. |
| 146 · GNN vs transformer | Strong sampling/propagation/readout separation and honest comparison limitations. | Refreshed readout, MLP, centroids and seed matching; linked the unresolved question to L147. Retained side-by-side complete course models, clearly labeled as reduced variants. |
| 147 · Research questions | Good synthesis rather than a fabricated new model. | Added mechanism/falsifier/transfer reminders and a concrete intervention handoff to L148. Its information map and feasibility ranking serve the lesson better than another architecture diagram. |
| 148 · Ablations | Good intervention map, but interprets the measured interaction before deriving it. | Moved that interpretation after the formula and worked example; refreshed paired effects. Retained the component intervention map and history/interaction controls. |
| 149 · Weakness catalog | Useful counter-evidence but overlaps L137, and “normalized MAE” can be confused with L136. | Explained the new cross-task catalog versus the repeated within-task diagnosis pattern. Distinguished normalization by RDL error from normalization by training-target SD. Added provenance reminders; explicitly marked L150 as under construction. |
| 150 · Q3 checkpoint | Draft connects L143 reproduction, L145 validity and L149 counter-evidence. It still contains unexpanded markers and references to unfinished deliverables. | Reviewed the intended handoff only. No finished-lesson verdict, regenerated artifacts or publication of its unfinished files in this change. |

Every published lesson now has a short reading route, a keyboard-accessible prerequisite reminder and a concrete carry-forward question. Detailed evidence, visible code and existing learner exercises remain. Notebook code is compared exactly with its pre-review version; original outputs are preserved and revised prose/figures are re-exported. No new cloud experiments are needed for these edits.

Delivery evidence is recorded separately in `notebooks.json`, `checks.json`, `pages.json`, `browser.json`, and the deployment record when available. Browser geometry checks support rendering; they do not establish learner comprehension, historical paper identity or live Colab execution.

## Validation completed before publication

- All 14 existing lesson contract checks and all 14 interaction/delivery suites passed.
- Clean Git-index build passed; 799 local links across lessons, prepared notebooks and reference pages resolved.
- All 14 lessons passed 1200px and 375px browser checks, keyboard prerequisite access, print and no-JavaScript checks. No JavaScript or local HTTP errors.
- Original code and saved execution outputs were preserved in all 14 solution notebooks. Revised prose and portable diagrams were re-exported.
- RelGNN label sizes were increased after screenshot inspection; the diagram stays horizontally scrollable on narrow screens. New SVG labels fit its viewBox.
- The release manifest ends at 149. Lesson 150 authoring files and its pending workflow block remain in the shared workspace.

The complete existing site is approximately 1.30 GB uncompressed. GitHub documents a [1 GB published-site limit](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits). Deployment status and live verification are separate evidence; this review does not delete older reproduction artifacts or alter their hashes to reduce size.
