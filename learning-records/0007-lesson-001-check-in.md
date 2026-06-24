# Lesson 001 comprehension check-in

User completed Lesson 001 quiz and primary-reading block. Demonstrated solid grasp of **temporal leakage** (aggregates computed with future rows smuggle information into past labels). Articulated that relational claims need **comparison baselines** and that tree/GBDT-style baselines are strong in practice.

**Gap to close before Lesson 002:** the *structural* reason tabular ML forces feature engineering — learners consume one flat design matrix `(X, y)`, so multi-table databases must be joined and aggregated upstream. User's first answer described why FE is *hard to maintain* (drift, obsolescence), which is adjacent but not the forcing mechanism named in the lesson.

**Implications:** Lesson 002 (design matrix & leakage) is the right next step; reinforce matrix contract before REG vocabulary.
