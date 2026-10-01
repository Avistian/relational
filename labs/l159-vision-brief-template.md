# Lesson 159 — one-page foundation-model vision brief

Learner status: PENDING_WRITTEN_DEFENSE. Aim for400–600words. Author code execution does not complete this brief.

1. **The remaining problem.** State one concrete limitation supported by L158's evidence. Explain why reusable pre-training is a plausible response, not an already proven solution.
2. **Objective and information.** Choose cell value, column name or table name. Specify the prediction unit, all semantic copies that must be masked, permitted context, train/validation/test unit and one cheap shortcut that would invalidate your claim.
3. **Computation and learning.** Trace row serialization→encoder→graph→decoder→loss. State which weights move in each stage and explain how a frozen decoder can pass gradients into the graph.
4. **Evidence and falsifier.** Separate the paper's Table1, local synthetic experiment, and your desired multi-database transfer claim. Propose a held-out-database experiment with scratch baseline, allowed adaptation data/budget, selection rule and a disconfirming outcome. Do not invent a successful result or choose a threshold from test results.
5. **Feasibility.** List missing source/protocol artifacts and the budget gate. Explain why more epochs of the small lab would not reproduce the published table.

Review rubric,0–2each: objective clarity; corruption/information discipline; architecture and gradient explanation; evidence/transfer boundary; falsifiable feasible next experiment. Target8/10with no zero after teacher review. Ask the teaching agent for review and follow-up explanations.

Tomorrow, retrieve the three mask levels and the difference between frozen and detached. In a week, diagnose a fresh schema-copy leak without looking at this lesson.
