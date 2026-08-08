# 0101 — Lesson 043 published: TabNet (sequential attention)

**Date:** 2026-08-08
**Status:** Published **with lab** (awaiting user completion).
**Curriculum:** Year 2 · Q1 · lecture 043 (Advanced Tabular Deep Learning).
Topic per `CURRICULUM.md`: *TabNet (sequential attention)*; curriculum lab: *train + read masks*.
**Primary reading:** Arik & Pfister, AAAI 2021 — *TabNet: Attentive Interpretable Tabular Learning*
([arXiv:1908.07442](https://arxiv.org/abs/1908.07442) ★), architecture §3.2 / Fig. 4a–4d + Appendix E
(ablations) + Appendix F (HP guidance) + **Appendix A (KDD)**.
**Companions:** Martins & Astudillo 2016 ([arXiv:1602.02068](https://arxiv.org/abs/1602.02068), sparsemax
Algorithm 1); Chen et al. 2018 ([arXiv:1802.07814](https://arxiv.org/abs/1802.07814), the L2X synthetic
generators used for the mask claims).

## Single skill
Implement TabNet's **sequential attention from scratch** — sparsemax, the attentive transformer, and the
**prior scale** — then *read its masks on data whose relevant features are known*, and hold the whole model
to **L042's baseline-first rule** against a tuned MLP/ResNet under one shared frame.

## Why this was the ZPD
L041 named the neural bar; L042 made it *trainable and fair* (from-scratch MLP/ResNet, shared protocol,
multi-dataset ranks). L043 is the first time that bar is actually **used on a novel architecture** — and
the first architecture in the curriculum whose selling point is *interpretability*, which forces the
separate skill of testing an attribution claim rather than accepting it. It also recruits heavy prior
recall: the tree-like inductive bias TabNet is imitating (L025–L027), fair-budget search (L017), the L020
split contract, rank-based comparison + under-powered tests (L023/L030), and HP-budget parity (L038).

## Standard #24 (paper-mirror) — how each axis was met
- **(A) Implementation.** `labs/relkit/tabnet.py` implements the paper's load-bearing pieces from scratch,
  annotated with figure/section refs: `sparsemax` (Martins Alg. 1), `GhostBatchNorm` (Hoffer 2017),
  `GLUBlock`, `FeatureTransformer` (Fig. 4c — 2 shared + 2 step-dependent layers, `sqrt(0.5)` residuals),
  `AttentiveTransformer` (Fig. 4b), `TabNetEncoder` (Fig. 4a — prior scale, sparsity entropy loss,
  `d_out = ∑ ReLU(d[i])`, `M_agg` weighted by `η[i] = ∑_c ReLU(d_c[i])`), plus `train_tabnet` with the
  paper's geometric LR decay (App. F). The library is a **checker only**.
- **(B) Datasets.** The interpretability claim is run on the **paper's own evaluation data**:
  `labs/relkit/synth.py` implements **Syn2** and **Syn4** from the L2X/INVASE generators the paper cites
  (Tier C — the only tier where a mask claim is falsifiable, because relevance is known by construction).
  Gap statement shipped in the lesson and the lab: we do **not** run the paper's Forest Cover / Poker /
  Sarcos / Higgs tables, so the paper's **Table 3–6 AUC/accuracy values are not reproduced here** — only
  the mask-reading claim is. The bake-off substitutes four small Tier-A OpenML tables under #23.
- **(C) Reproducibility.** Fixed seeds (splitter seeds included), one shared frame across arms,
  `labs/_verify_l043.py` → committed `labs/_verify_l043_results.json` (858 s), plus two committed
  hypothesis-test harnesses (`_verify_l043_refcheck.py`, `_verify_l043_refsched.py`) for the discrepancy
  below. Environment of record: **torch 2.13.0+cpu, pytorch-tabnet 4.1.0, rtdl_revisiting_models 0.0.2**
  (`pytorch-tabnet>=4.1` added to `requirements-labs.txt`). Note the L037 lockfile predates torch 2.13 and
  these two packages; it remains the L037-era snapshot, so the versions above are the L043 record.

## Verified numbers (evidence of record — `labs/_verify_l043.py`, budget 6, seeds 0/1/2, 120 epochs)
**Mask reading (the paper's own generators).**
- **Syn2** (relevance is **global**: X3–X6 for every row): test AUC **0.804**, top-4 masked features
  **exactly** [X3, X4, X5, X6], **76.8%** of `M_agg` mass on the truth. A clean success.
- **Syn4** (relevance is **instance-wise**, switched by X11): test AUC **0.666**, switch feature X11 gets
  mask weight **0.118**, and mass moves the right way (X1–X2 mass **0.211** on X11<0 rows vs **0.089** on
  the rest) — but only **15.6%** of X11<0 rows actually favour their own group, against **97.9%** on the
  other side. **Partial** recovery, reported as such. The paper used **10M** rather than 10k samples for
  its sharp Fig. 5 masks, and XOR (X1·X2) is the hardest case for a mask because neither feature is
  informative alone.

**Bake-off (4 small tables, one shared frame, equal budget).**

| dataset | TabNet | MLP | ResNet | GBDT |
|---|---|---|---|---|
| credit_g | 0.748 ± 0.021 | 0.796 ± 0.028 | **0.810 ± 0.030** | 0.780 ± 0.016 |
| diabetes | 0.824 ± 0.019 | **0.833 ± 0.026** | 0.820 ± 0.021 | 0.805 ± 0.019 |
| blood_transfusion | 0.755 ± 0.025 | **0.760 ± 0.043** | 0.754 ± 0.046 | 0.727 ± 0.018 |
| kc1 | 0.794 ± 0.003 | 0.794 ± 0.003 | **0.805 ± 0.012** | 0.789 ± 0.024 |
| **mean rank** | **2.50** | **1.75** | **2.00** | **3.75** |

Friedman **chi2 = 5.700, p = 0.127** (k = 4, N = 4).

**The disciplined reading (now M55).** Two *separate* statements, and the lesson keeps them apart:
(1) *statistically*, p = 0.127 on four datasets licenses only "**cannot distinguish on this sample**" —
neither "significantly worse" nor "equivalent" (failing to reject is not evidence of equality, L023);
(2) *evidentially*, the baseline-first rule puts the burden of proof on the **new** model, so ranking
behind the baselines it was designed to beat means the burden was **not met**. The paper's own
**Appendix A (KDD)**, where TabNet ties or slightly trails XGBoost and CatBoost, says the same thing more
quietly — and the lesson asks why that table is in an appendix (L038).

**Correctness of the from-scratch implementation** — `labs/_check_l043.py`: **22/22 pass**, including
sparsemax vs a brute-force simplex projection (max |Δ| **1.8e-15**) and vs `pytorch_tabnet.sparsemax`
(max |Δ| **2.3e-07**), gradient support, the γ = 1 non-reuse property (overlap **0**), the
`lambda_sparse` → mask-entropy monotonicity (2.063 → 0.842), ghost-BN per-chunk normalisation, and a
learning sanity check on a sparse signal (AUC 0.998, mask puts 83% of mass on the 2 informative features).

## Honesty note — an UNEXPLAINED discrepancy, kept rather than tidied
End-to-end, the **from-scratch TabNet outscored the reference** `pytorch_tabnet` under the same protocol:
credit_g **0.748 vs 0.694**, diabetes **0.824 vs 0.766** (|Δ| ≈ 0.05–0.06, tolerance 0.04 → **out of
tolerance**). Two hypotheses were tested and **refuted**:
1. *Training length* (`_verify_l043_refcheck.py`) — gave the reference a longer leash (patience 25 /
   200 epochs). On **credit_g** it did not help at all: 0.694 → **0.686**, |Δ| *widening* to 0.062. On
   **diabetes** it did help partially: 0.766 → **0.785**, |Δ| **0.039**, which lands just inside the 0.04
   tolerance. So under-training explains *some* of the diabetes gap but **none** of the credit_g gap.
2. *Learning-rate schedule* (`_verify_l043_refsched.py`) — our loop implements the paper's "large initial
   rate, gradually decayed" (App. F) and the library's default `fit()` does not, so we added the same
   StepLR decay to the reference: credit_g **0.680**, no better.

So the **credit_g** gap stands unexplained, and the lesson states exactly that scope — *mechanism
validated exactly; end-to-end agreement holds on one table and fails on another, cause not identified* —
rather than the looser "the gap did not close everywhere". Note the
direction is the *safe* one for standard #22 (our implementation is not weaker than the library), but it is
still a discrepancy, and it does **not** license "our TabNet is better than the library's" — the two
differ in defaults we did not fully control. The lesson names the definitive next test (stretch task 4):
build both with matched shapes, copy weights across, and assert `torch.allclose` on a forward pass, which
would isolate architecture from training entirely. Recorded as an open item.

## What shipped
- **Lesson** `lessons/0043-tabnet.html`: warm-up (upTo:43) → decision-step framing → sparsemax
  (Algorithm 1, derived, with the softmax contrast) → attentive transformer → **prior scale + γ** →
  feature transformer / GLU / ghost BN → sparsity penalty → decision aggregation → `M_agg` →
  predict-before-reveal on **Syn4** → mask-reading results (Syn2 clean, Syn4 partial) →
  predict-before-reveal on the **bake-off** → rank table + the "unmet burden ≠ significantly worse"
  separation → reference-validation failure reported openly with both refuted hypotheses → teach-back on
  *what makes the attention sequential* → thesis bridge → 5 quizzes → primary reading (incl. Appendix A) →
  inline lab → teacher/ask-teacher → nav.
- **One new viz** (`file://`-safe): `assets/tabnet-mask-viz.js` — step-by-step sequential attention with
  the **prior scale** made visible (γ slider; watch a spent feature's prior fall and, at γ = 1, a
  fully-used feature get banned). CSS prefix `.tnm-`.
- **Headless checks:** `labs/_viz_check_l043.js` — **all 81 pass** (viz behaviour + every verified number
  present in the lesson + each honesty statement: the reference failure, both refuted hypotheses, the
  "unexplained" admission, the bar-vs-significance separation, the four-datasets-is-a-demonstration flag,
  the large-benchmark citation, the 10M-samples caveat, the KDD-appendix tie). `labs/_check_pedagogy.js` —
  **all 40 pass** (pool integrity, spacing/interleaving, Leitner transitions, predict, deck, teach-back).
  Browser MCP unavailable → node verification only (consistent with L021–L042).
- **Lab notebook** `labs/0043-tabnet.ipynb` (+ solution, gitignored; + rendered
  `labs/html/0043-tabnet.html`), built by `labs/_build_l043.py`. Four tasks: (1) implement **sparsemax**
  from Algorithm 1, validated against `pytorch_tabnet`; (2) implement the **attentive transformer + prior
  scale**, verifying the γ = 1 ban; (3) **read the masks** on Syn2/Syn4; (4) the **baseline-first**
  bake-off with mean ranks + Friedman and an honest verdict. **Solution executed end-to-end with nbconvert
  — all 25 CHECK assertions pass** in **126 s** on CPU; its mask numbers reproduce the verify run exactly
  (Syn2 0.804 / 76.8%, Syn4 0.666 / 0.118 / 15.6% / 97.9%) and its reduced-budget bake-off reproduces the
  *direction* (TabNet 2.67 vs MLP 1.33 / ResNet 2.33, p = 0.172).

## Design choices
- **Tasks 1–2 are the load-bearing pieces, and only those.** The feature transformer is provided in
  `relkit.tabnet` (built from the paper, readable) while the student implements sparsemax and the
  attentive transformer + prior scale — the two mechanisms that are *specifically TabNet* and that the
  misconceptions target. Implementing four blocks would have made the lab a typing exercise.
- **Sparsity and sequentiality are taught as separate properties** (M54). This is the single most
  compressible-but-wrong summary of TabNet, so the lesson, the viz (γ slider), the lab check, and a
  retrieval item all attack it from different angles.
- **The interpretability claim is *tested*, not asserted** (M53). Hence Tier C and the paper's own
  generators: on real data no attribution is falsifiable. Reporting Syn4's partial recovery is the point
  of the exercise, not a blemish on it.
- **Lab budget deliberately smaller than the lesson's** (3 datasets / budget 3 / 2 seeds vs 4 / 6 / 3), with
  the notebook stating that a downscale is a *different experiment* and that only the direction should
  reproduce — standard #20 + #23.
- **`quarter: "Y2Q1"`** on the four new retrieval-pool items.

## Artifacts synced
- `assets/retrieval-pool.js` **+4**: `l043-sparsemax`, `l043-prior`, `l043-masks` (misconception, mirrors
  M53), `l043-bar-vs-sig` (misconception, mirrors M55).
- `assets/paper-deck.js` **+1** card `arik2019` (the four mechanisms, what makes the attention sequential,
  the verified mask readings, and the bake-off verdict incl. the KDD appendix).
- `misconceptions.md` **M53** (masks gate the input ⇒ faithful free explanation → validate on
  known-answer data; Syn2 vs Syn4 numbers), **M54** (sparse ⇒ sequential → it is the prior scale), **M55**
  (p = 0.127 ⇒ the bar was cleared → unmet burden vs significance).
- `reference/glossary.html` — Year 2 · Q1 **+9 terms**: sparsemax, decision step, attentive transformer,
  prior scale (γ), feature transformer, Ghost BatchNorm, aggregate mask (`M_agg`), instance-wise feature
  selection, unmet burden vs significantly worse.
- `thesis-dossier.md` — Evidence Ledger **L043** (BAR, C3/C4) + narrative paragraph. Indirect FOR:
  *instance-wise* selection is the single-table shadow of the relational claim (different rows need
  different context) — and TabNet shows how expensive that is to buy by masking columns of an already
  flattened table.
- `lessons/manifest.json` — L043 entry with `labPath: labs/0043-tabnet.ipynb` (regenerated by
  `scripts/update-manifest.py`; 43 entries). `notebooks.html` picks it up automatically.
- `requirements-labs.txt` **+`pytorch-tabnet>=4.1`** (validation only).
- **Generator bug fixed in passing:** `scripts/update-manifest.py` recomputed `checkpoint` from the slug
  (`"checkpoint" in slug`), so every regeneration silently cleared the hand-set `checkpoint: true` on
  **L040** (the Year 1 exit exam, whose slug contains no such word). It now preserves a hand-set flag, and
  the regeneration is idempotent apart from the L043 addition.
- L042 nav now links forward to L043.
- **New files:** `labs/relkit/tabnet.py`, `labs/relkit/synth.py`, `labs/_check_l043.py`,
  `labs/_verify_l043.py` (+ results JSON), `labs/_verify_l043_refcheck.py` (+ JSON),
  `labs/_verify_l043_refsched.py` (+ JSON), `labs/_build_l043.py`, `labs/0043-tabnet.ipynb`,
  `labs/solutions/0043-tabnet.ipynb` (executed, gitignored), `labs/html/0043-tabnet.html`,
  `labs/_viz_check_l043.js`, `assets/tabnet-mask-viz.js`.

## Open items
- The **unexplained** from-scratch-vs-reference end-to-end gap (above). Definitive test = weight-level
  transplant + `torch.allclose`; offered as lab stretch task 4.
- TabNet's **self-supervised pre-training** (the paper's other contribution: mask feature columns and
  reconstruct them, then fine-tune — Table 7) is *not* implemented; offered as lab stretch task 5. If the
  user wants it, it deserves its own unit rather than a bolt-on.

## Next
User runs the lab and pastes the EXIT ticket (sparsemax validation, the γ = 1 ban, the Syn2/Syn4 mask
readings, the rank table + verdict) or says "lab done". On completion, open **Lesson 044 — NODE**
(Popov et al. 2019, differentiable oblivious decision trees): the *other* 2019 attempt at giving a neural
net a tree's inductive bias, with the same bar waiting for it.
