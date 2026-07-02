# Lesson 013 published — boosting intuition

Third Q2 unit (ensemble arc). Builds on L012: bagging cuts **variance** by averaging independent trees; boosting cuts **bias** by fitting residuals sequentially. Friedman (2001) framing: stagewise additive modelling = gradient descent in function space (residual = negative gradient for squared error).

**New reusable asset:** `assets/residual-viz.js` — 1-D regression viz: model starts as a flat mean, each round fits a depth-1 stump to the current residuals and adds a shrunken step. Red bars = residuals; bold blue = running fit; dashed green = truth. Slider = boosting rounds M; MSE-vs-truth readout collapses as M grows. Reusable for any stagewise/residual lesson. Implements a real SSE-minimizing stump in JS (not precomputed), so the picture is authentic.

**Verified live (sklearn 1.9, `relkit` harness):**
- Toy residual loop (Tier C, `f = sin(2πx)+0.5x`): MSE vs truth **0.359 → 0.004** over 60 rounds (lr 0.3). Monotone fall.
- Learning-rate sweep (40 rounds fixed): lr 0.05 → 0.084, lr 0.3 → 0.008, lr 1.0 → 0.007 — small lr under-fits at a fixed budget.
- **credit_g** (n=1000, prevalence 0.700): single tree 0.757, RF(300) **0.901**, GBDT(300, lr0.05, depth3) 0.879 — RF edges untuned boosting on tiny noisy data.
- **adult** (n=48,842, prevalence 0.239): RF(200) 0.785, GBDT(200, lr0.1, depth3) **0.824** — boosting wins with real signal + more data.

**Honest contrast is the point:** on 1000 noisy rows bagging's variance reduction wins; on 48k rows with structure boosting's bias reduction pulls ahead, and the gap widens with tuning (→ XGBoost L014). Keeps the mission's "honest baselines" discipline rather than over-claiming "boosting always wins."

**Lab (incremental rule active):** `labs/0013-boosting-intuition.ipynb`.
- **Crucial fragment (Task 1):** student implements the stagewise residual loop by hand (`resid = y − pred`; fit stump to resid; `pred += lr·h`) — this *is* gradient boosting, so XGBoost/LightGBM lose their mystery.
- Task 2: learning-rate trade-off. Task 3: single tree vs RF vs `GradientBoostingClassifier` on `credit_g` via `relkit`. Stretch: swap to `adult` to see boosting win.
- Solved copy executed end-to-end (all CHECK + EXIT clean); solution in `labs/solutions/` (gitignored).

**Callback (from L011):** ΔG-vs-weighted-children sign confusion — watch it resurface in L014 (XGBoost regularized gain), where the split-gain formula returns.

**Verification note:** interactive browser MCP was unavailable this session; `residual-viz.js` was checked headlessly in Node (mount + slider handler, MSE 0.082 flat-mean → 0.005 at M=40, no runtime errors). Numbers in the lesson HTML come from the executed lab/solution, not parametric memory.

**Primary reading:** Friedman 2001, *Greedy Function Approximation* (§1–3 + Algorithm 1).

Next: Lesson 014 (XGBoost — Chen & Guestrin 2016; tune XGB on one task, regularized split gain).
