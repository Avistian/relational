# Lesson 019 published — When trees win (Grinsztajn 2022 preview)

Tenth Q2 unit (curriculum lec 019), the conceptual capstone before the L020 Q2 checkpoint.
**Preview** of Grinsztajn, Oyallon & Varoquaux 2022, *Why do tree-based models still outperform
deep learning on typical tabular data?* — NeurIPS 2022 D&B, [arXiv:2207.08815](https://arxiv.org/abs/2207.08815)
(verified via user-arxiv MCP). Read abstract + §1 only; the full 45-dataset benchmark + §5 experiments
are Y1 Q3 (lec 024–027). No new dependency — sklearn only (RandomForest vs MLPClassifier); no boosters.

**Concept / single skill:** name and recognize the **three inductive biases** that make trees win on
tabular data — the mirror of Grinsztajn's three challenges for NN designers — and know the honest
conditions under which each advantage flips:
- **Irregular / non-smooth targets.** NNs are biased toward smooth functions; trees place axis-aligned
  steps anywhere. Rougher target → wider tree lead.
- **Uninformative features.** Trees do implicit feature selection (split on the best column, ignore junk);
  an MLP feeds every column into every neuron. More junk → tree gets relatively better.
- **Orientation (rotation).** Tabular columns are individually meaningful; trees split one axis at a time
  (not rotation-invariant) and exploit that; an ~rotation-invariant MLP can't tell a real column from a
  scrambled mixture. Random rotation is the sharpest single piece of evidence in the paper.

**New reusable asset:** `assets/biases-viz.js` — a checkerboard visualizer with two modes. *Irregular*:
tile buttons 2/4/8 draw a sharp checkerboard (tree) beside the SAME board through an SVG `feGaussianBlur`
(the NN smoothness bias) — finer tiles wash the blur to gray. *Rotation*: axis-aligned XOR with straight
green tree splits (fit) vs a red **staircase** polyline after a 30° rotation (misfit), NN boundary rotates
with the data (invariant). CSS `.bv-*` in the lesson `<style>` (mirrors `.stk-*`). Headless Node mount
check clean (`labs/_viz_check_l019.js`, 7/7). **Browser MCP unavailable again** (empty tools folder; only
user-arxiv authed) → headless verification only.

**Verified live (`labs/_verify_l019.py` + executed `solutions/0019-when-trees-win.ipynb`, synthetic Tier-C,
RF-300 vs MLP-(256,256), test accuracy):**
- **Irregular (checkerboard 2→16 tiles/side):** tree 1.000·0.999·0.969·0.807 vs NN 0.994·0.975·0.837·**0.503**;
  gap widens +0.006 → **+0.304**.
- **Uninformative (+0→+100 noise cols):** k=0 tree 0.936 vs **NN 0.965 (NN wins clean!)**; k=100 tree
  **0.876** vs NN 0.826. Gap swings **−0.029 → +0.049**; MLP loses 0.139, tree only 0.061.
- **Rotation (axis-aligned XOR + 6 noise, random orthogonal Q):** tree 0.998→**0.831** (drop 0.167),
  NN 0.979→0.976 (invariant). (Lab numbers; standalone verify gives 0.996→0.817 / 0.964→0.962 — the lab's
  Task-2 CHECK re-draws from `rng`, shifting state. Lesson + viz quote the **lab-executed** numbers since
  that is what the student reproduces.)

**Honest framing (the L019 myth-buster):** "trees always win" is too strong — with clean, all-informative
features the MLP won. The real claim is that *typical tabular data has all three properties at once*
(irregular, junky, oriented), which trees are built to exploit. The edge is a small/medium-data, single-table
phenomenon; large data, perceptual inputs, and the Year-2 tabular nets (FT-T/RealMLP/TabM/TabPFN) all narrow
or flip it. Kept the curriculum's "GBDTs are not dead" guardrail explicit.

**Thesis bridge:** this is the intellectual core of the Y1 exit criterion. The RDL bet is *not* "DL is bad at
tables" — it is "the single-table representation discards relational structure." Grinsztajn explains why the
flat-table incumbent is strong (its biases fit flat data), which sets the honest bar: beat a learner whose
biases genuinely match tabular data, don't strawman it. Sets up Q3 lec 024–027 (benchmark reproduction of
each bias) and the Y1 exit exam.

**Lab:** `labs/0019-when-trees-win.ipynb` — 3 TODO (one crucial fragment each) + stretch. Task 1 = write the
checkerboard parity label; Task 2 = `np.hstack` a block of `rng.normal` noise features; Task 3 = random
orthogonal `Q` via `np.linalg.qr` + `X @ Q`. Standalone **concept lab** (Tier C), self-contained (no relkit —
the incremental-relkit rule is for reproduction labs). Student blank (7 `____`, 0 outputs); solution executed
clean (all CHECK + EXIT) and gitignored. Committed `_verify_l019.py` + `_build_l019.py` + `_viz_check_l019.js`.
Manifest → 19 entries (kept `version: 2`, hand-appended to preserve key order); all labs re-rendered to
`labs/html/`.

Next: Lesson 020 = **Q2 checkpoint** (Chen 2016 + Ke 2017; match or beat a published tree baseline). No
advancing past it without the checkpoint passing.
