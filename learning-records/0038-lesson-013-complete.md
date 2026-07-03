# Lesson 013 complete — boosting intuition

User: "lesson/lab 13 done." Fourth Q2 unit (ensemble arc) closed.

**Lab score ≈ 9/10** (lab-authoring rubric, 0–2 × 5):
- Correctness 2 — all CHECK pass, EXIT ticket prints (toy MSE collapse, lr sweep, credit_g tree/RF/GBDT).
- Leakage discipline 2 — scoring via `relkit.cv_pr_auc` (StratifiedKFold), no fold leakage.
- Conceptual takeaway 1 — the by-hand residual loop was built correctly; the EXIT `takeaway:` string was left as `____` (deliverable sentence not written). One improvement: state in one line that boosting cuts **bias** by fitting residuals and why RF tied it on tiny noisy `credit_g`.
- Mid-zone effort 2 — implemented the stagewise loop + lr sweep by hand (crucial fragment).
- Reproduction n/a for a concept lab.

**Retained skills:** residual = negative gradient (squared error); shrinkage as regularization (small lr → more rounds). Callback still open: L011 ΔG = children − parent sign — carried into L014 where the split-gain formula returns.

Next: Lesson 014 (XGBoost).
