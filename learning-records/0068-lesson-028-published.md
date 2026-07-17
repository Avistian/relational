# 0068 — Lesson 028 published: MLP & ResNet Tabular Baselines

**Date:** 2026-07-17
**Status:** Published (awaiting user completion)
**Curriculum:** Year 1 · Q3 · lecture 028 — pivots the quarter from *why trees win* (the Grinsztajn
arc, 024–027) into *building the honest neural baseline*. First **PyTorch** lesson/lab.
**Core paper:** Gorishniy, Rubachev, Khrulkov & Babenko 2021, *Revisiting Deep Learning Models for
Tabular Data* (NeurIPS 2021, arXiv:2106.11959), **§3.2 (the MLP and ResNet baselines)** + §5 (the
"no universal winner" comparison with GBDT). Residual idea: **He et al. 2015** (*Deep Residual
Learning*, arXiv:1512.03385), the degradation-problem argument.

## Single skill
Build the Gorishniy tabular **ResNet** — a linear embedding, a stack of pre-activation residual blocks
(`BatchNorm → Linear → ReLU → Dropout → Linear → Dropout`, plus a **skip**), and a prediction head;
explain why the **skip connection** (output `x + f(x)`) is what lets a deep tabular net train at all
(makes the identity map free → fixes the degradation problem); and know that a *properly-tuned*
MLP/ResNet — not the latest transformer — is the honest neural baseline a GBDT must be compared
against.

## Why this was the ZPD
The Grinsztajn arc established that trees win on typical tabular data. To make (or contest) the
relational bet we now need the *neural* contestant built correctly — and the field's honest baseline
is precisely Gorishniy's MLP/ResNet. It is also the natural place to introduce the neural machinery
(residual block, BatchNorm, dropout, PyTorch) that Year 2 (FT-Transformer/TabR/TabM) and Year 4 (row
encoder → GNN → head) are assembled from. Matches the CURRICULUM Q3 row (028: MLP & ResNet baselines,
Gorishniy §3.2, "train ResNet baseline").

## What shipped
- **Lesson** `lessons/0028-mlp-resnet-baselines.html` (~50 min, standard #17 thoroughness): a full
  vocabulary section defining *everything introduced* from first principles (baseline, linear layer,
  ReLU, MLP/hidden dim, epoch/minibatch/SGD/Adam, BatchNorm, dropout, residual/skip connection, the
  degradation problem) → the MLP baseline → the degradation problem (incl. an explicit "why it is NOT
  vanishing gradients — BatchNorm" callout) → the residual block + block viz → why identity-is-free
  fixes it → predict → depth-trainability viz + table (train *and* test) → the full architecture +
  head/logit → honest bake-off viz + table → teach-back → thesis bridge → subtleties → 3 quizzes →
  primary reading (Gorishniy §3.2/§5 + He 2015) → lab.
- **Three reusable viz** (standard #9, one per mechanistic beat), all driven by REAL verify numbers:
  - `assets/resnet-block-viz.js` — block anatomy; toggle skip ON (draws the `+` node and skip path,
    "output = x + f(x)") vs OFF (plain block, warns of degradation).
  - `assets/depth-trainability-viz.js` — test-accuracy vs depth for plain vs ResNet; "Show training
    accuracy" toggle adds dashed train curves exposing the degradation (plain train 1.000 → 0.927).
  - `assets/baseline-bakeoff-viz.js` — MLP/ResNet/GBDT on credit_g with ±sd whiskers; AUC/accuracy
    toggle; readout = MLP≈ResNet tie, GBDT ahead, "no universal winner".
  - Headless check `labs/_viz_check_l028.js` — 18/18 pass. **Browser MCP unavailable** (headless env)
    → node verification only, consistent with L021–L027.
- **Lab** `labs/0028-mlp-resnet-baselines.ipynb` — **first PyTorch lab**; **Tier A** (real credit_g via
  `relkit`) for the bake-off + **Tier C** (synthetic) for the depth mechanism (1000 rows can't show a
  depth effect). Crucial fragment (Task 1) = `ResNetBlock.forward` (`return x + f(x)` vs `f(x)`);
  Task 2 = depth sweep skip on/off with train+test acc; Task 3 = MLP/ResNet/GBDT bake-off. Student
  blank (7 `____`, 0 outputs); solution executed clean & gitignored. Built via `labs/_build_l028.py`.
- **Verify** `labs/_verify_l028.py` — depth sweep + bake-off; ran clean.

## Verified live
- **Depth trainability (synthetic n=8000, 32 feat, BatchNorm on both, AdamW 60 epochs, same arch skip
  on/off), test accuracy:** depths [1,2,4,8,16,32] → plain [0.917, 0.905, 0.900, 0.885, 0.884, 0.866]
  vs ResNet [0.912, 0.897, 0.909, 0.902, 0.897, 0.899]. **Train** accuracy: plain [1.000, 0.998,
  0.994, 0.985, 0.981, **0.927**] vs ResNet [1.000, 0.998, 0.999, 1.000, 0.997, 0.997]. The plain
  net's *training* accuracy falls with depth ⇒ degradation (optimization), **not** overfitting; the
  ResNet holds. (Both use BatchNorm, so it is not vanishing gradients — the skip's free identity is the
  mechanism.)
- **Honest bake-off (credit_g, Tier A, 5 seeds), ROC-AUC:** MLP **0.752 ± 0.011** ≈ ResNet
  **0.743 ± 0.010** (bands overlap — a tie), GBDT **0.793 ± 0.025** ahead. Accuracy: 0.734 / 0.740 /
  0.754. No universal winner; on small categorical data the tuned tree wins (consistent with
  L024–L027). The ResNet is an honest baseline, not a headline winner.
- **Lab reproduction (40 epochs, depths [2,8,16], 3 seeds)** is deliberately more extreme and honest:
  the plain net's TRAIN accuracy collapses **0.998 → 0.496** by depth 16 (near-random) while the
  ResNet holds ~0.90 — a starker version of the same degradation. Bake-off: MLP 0.758 ≈ ResNet 0.750 <
  GBDT 0.793. All CHECK + EXIT clean. (Lesson/viz quote the gentler 60-epoch verify numbers; the lab's
  shorter budget makes the degradation unmistakable — same mechanism.)

## Honest framings kept
- **Not vanishing gradients:** with BatchNorm the failure is the *degradation problem* (He et al.),
  measured via the training-accuracy drop. The lesson states this explicitly and uses it to justify
  measuring train accuracy.
- **DL does not win here:** the ResNet ties the MLP and both trail the GBDT on credit_g; the lesson's
  claim is that Gorishniy's contribution is *methodological* (honest baselines), not a new winner.
- **ResNet ≠ FT-Transformer:** the attention baseline (which starts to break the L026/L027 biases via
  per-feature embeddings) is deferred to Year 2 Q1 on purpose.

## Artifacts synced
- `assets/retrieval-pool.js` +2 (`l028-skip` [misconception], `l028-baseline`).
- `assets/paper-deck.js` +1 (`gorishniy2021`).
- `misconceptions.md` **M25** ("more layers always help / a deep net doing worse must be overfitting"
  → degradation problem, skip fixes via free identity, not vanishing gradients).
- `thesis-dossier.md` +1 ledger line (BAR + FOR, C3/C1 — the bar is now a tuned GBDT *and* a tuned
  ResNet; the neural machinery is what the RDL stack is built from).
- `reference/glossary.html` +8 Q3/DL terms (MLP, residual/skip connection, degradation problem,
  tabular ResNet, BatchNorm, dropout, honest neural baseline, no universal winner).
- `requirements-labs.txt` +`torch>=2.2` (first PyTorch lab).
- `lessons/manifest.json` → 28 entries; all labs re-rendered to `labs/html/` (incl. `0028-*.html`).
- `node labs/_check_pedagogy.js` clean; `node labs/_viz_check_l028.js` 18/18 clean.

## Env note
No `python3-venv`/uv preinstalled again → installed `uv` via curl and built `.venv` (scikit-learn
1.9.0, numpy 2.5.1, scipy, pandas, jupyter, **torch 2.13.0+cpu**; Node v22). **torch is now a lab
dependency.** The env-setup agent should preinstall the lab venv *including CPU torch* so future
sessions (and all of Year 2+) skip the ~1-2 min torch install and the uv bootstrap (recurring note
since Sessions 22–23; now larger because of torch).

## Next
Lesson 029 (Manual FE vs AutoML — Feurer et al. 2015, Auto-sklearn, skim; compare a tuned XGB to
AutoML), continuing Q3 toward the L030 Q3 checkpoint (1-page benchmark report).
