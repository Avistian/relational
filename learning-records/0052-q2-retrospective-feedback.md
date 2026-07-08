# Q2 retrospective feedback — multiple visuals per lesson

After the **Q2 checkpoint (L020)**, the user gave a checkpoint improvement: **lessons should use more than one visual when a single one is not enough**. Their observation — "it looks like there is only 1 viz per lesson." This now governs all future lessons (and viz retrofits), alongside the Q1 retrospective standards.

**The gap.** Published Q1–Q2 lessons settled into a one-viz-per-lesson habit even when a lesson teaches several distinct mechanisms. The clearest case is **L019 (When trees win)**: three inductive biases (irregular targets, uninformative features, orientation), but one `biases-viz` widget covers biases 1 and 3 via a mode toggle while bias 2 gets only a static table — three mechanisms, ~1.5 visuals. Each mechanism is a separate "strength" the prose treats as its own section, so each deserved its own visual.

**Standard #9 (NOTES Preferences).** Default to **one visual per distinct mechanism / claim / strength**, not one per lesson:
- Every mechanistic beat that passes the `lesson-visuals` decision tree gets its own inline viz, placed next to the prose that explains it.
- Split a mode-toggle widget into separate adjacent viz when the modes are *different concepts*; keep a toggle only when the modes are the *same mechanism under a knob* (e.g. one board, slider = tile count).
- A section with only a static results table, where the mechanism is visualizable, is an under-served beat — add the viz.
- A viz referenced from two different sections ("the widget above / below") is a smell it should have been two viz.
- Reuse still applies: multiple viz can be separate reusable assets or several mounts of one parameterised asset.

**Implications for Q3+.** Invoke `lesson-visuals` before publishing and count mechanisms → count viz. Verify **every** viz in the browser (not just the first). The skill's decision tree, a new "How many visuals per lesson?" section, and the verification checklist were updated to match. L019 is recorded in the skill as the anti-pattern; retrofitting it (dedicated noise-feature viz + splitting the toggle) is a candidate future task, not yet done.

**Open thread.** Browser-based viz verification has been unavailable for ~8 sessions (empty MCP tools folder) → viz have been verified headlessly in Node only. Adding more viz per lesson raises the cost of that gap; addressed separately in the post-checkpoint improvement proposals.
