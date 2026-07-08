# Lesson 019 complete — When trees win

User signalled "lesson/lab 19 done" at the start of the L020 session. As with L017/L018, **no
EXIT TICKET was pasted**, so no rubric score is recorded; the lab
(`labs/0019-when-trees-win.ipynb`) had shipped with a verified solution (all CHECK + EXIT clean,
gitignored).

Retained skill assumed solid from the lab structure: the **three inductive biases** that make trees
win on tabular data, each isolated by one knob — irregular/non-smooth targets (checkerboard
frequency), robustness to uninformative features (append noise columns), and orientation (random
rotation collapses the tree, not the ~rotation-invariant MLP). Watch on a future check-in that the
**honest half** stuck: with clean all-informative features the MLP *won* (row k=0), so "trees always
win" is too strong — the real claim is that typical tabular data has all three properties at once,
and the edge narrows with large data, perceptual inputs, and the Year-2 tabular nets.

Next: Lesson 020 published (Q2 checkpoint) — the gate out of Q2.
