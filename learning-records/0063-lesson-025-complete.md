# Lesson 025 complete — Inductive Bias: Smoothness

**Date:** 2026-07-16
**Status:** Complete (marked done on the user's word)

User signalled completion: **"lab/lesson 25 done"** (in the same message that asked to build L026).
No EXIT-ticket text was pasted, so — following the L017/L018/L020/L021/L022/L024 precedent — **no
numeric rubric score** is recorded; the unit is marked done on the user's word.

**What L025 covered.** Grinsztajn §5.2 (Finding 1): the **smoothness / spectral bias**. An MLP trained
by gradient descent fits low-frequency structure easily and high-frequency (jagged) detail slowly or
not at all (Rahaman 2019), while typical tabular targets are irregular — so a piecewise-constant tree
follows the jags and an MLP over-smooths. Proof by ablation: Gaussian-smoothing the *target* collapses
the GBT-vs-MLP gap in lockstep with the variance removed (repro: +0.33 R² → ~0, MLP even edges ahead
once smooth). Lesson + lab: [[0062-lesson-025-published.md]].

**Grinsztajn arc progress.** L024 built the measuring instrument; L025 dissected the first of the three
mechanism biases. Remaining:
- **L026 — rotation** (§5.4, this session): trees are not rotationally invariant; MLPs are — and on
  tables that invariance is a liability.
- L027 — uninformative features (§5.3), linked to rotation by Ng's (2004) sample-complexity theorem.

**Citation correction noted.** L025's forward-reference parenthetical had the two remaining findings'
section numbers swapped (it read "§5.3 (rotation) and §5.4 (uninformative features)"). The actual paper
is the reverse — **rotation is §5.4 (Finding 3)** and **uninformative features is §5.3 (Finding 2)**
(verified against the arXiv HTML, HAL, NeurIPS, and OpenReview copies). Fixed the one line in
`lessons/0025-inductive-bias-smoothness.html` for consistency; the *lesson order* (rotation before
uninformative) is unchanged, as it is baked into L025's quiz 3 ("that is the L026 bias") and the arc
records. See [[0064-lesson-026-published.md]].

**Next.** L026 (Inductive bias: rotation — Grinsztajn 2022 §5.4) published this session; see
[[0064-lesson-026-published.md]].
