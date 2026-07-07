# Lesson 018 complete — Ensembling & stacking

User signalled "lesson/lab 18 done" at the start of the L019 session. As with L017, **no EXIT
TICKET was pasted**, so no rubric score is recorded; the lab
(`labs/0018-ensembling-stacking.ipynb`) had shipped with a verified solution (all CHECK + EXIT
clean, gitignored).

Retained skill assumed solid from the lab structure: build **out-of-fold** meta-features by hand
(`cross_val_predict`), diversity is the fuel (GBDTs ≈0.89 correlated → redundant; logistic ≈0.68
carries diversity), and the in-sample leak crowns the memorizer (1-NN weight +3.00, held-out gap
+0.035). Watch on a future check-in that the **honest framing** stuck — on a tiny single table
ensembling barely moves the number (+0.001); the durable lessons are the OOF mechanism, the
diversity requirement, and the leak, not the headline. The thesis bar is a *leak-free stacked
ensemble*, not a single default.

Next: Lesson 019 published (When trees win — Grinsztajn 2022 preview), then L020 = Q2 checkpoint.
