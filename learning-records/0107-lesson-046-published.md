# 0107 — Lesson 046 published: FT-Transformer (feature tokenizer + [CLS] readout)

**Date:** 2026-09-04
**Status:** Published **with lab** (awaiting user completion).
**Curriculum:** Year 2 · Q1 · lecture 046 (Advanced Tabular Deep Learning).
Topic per `CURRICULUM.md` / `plan/year-2.md`: *FT-Transformer — implement and train the Feature Tokenizer,
reproduce its status as the strongest single neural baseline, and remove the numeric-bypass limitation L045
named.* Curriculum lab: *build the tokenizer + [CLS] readout from scratch, race FT-T vs TabTransformer vs
MLP vs CatBoost, and prove numerics now attend.*

**Primary reading:** Gorishniy, Rubachev, Khrulkov & Babenko, 2021 — *Revisiting Deep Learning Models for
Tabular Data* ([arXiv:2106.11959](https://arxiv.org/abs/2106.11959) ★), §3.3 (Feature Tokenizer, [CLS],
Transformer), §5 (the fair-comparison result: FT-Transformer is the top deep model and ~ties tuned GBDTs).
**Companion:** Vaswani et al. 2017 — *Attention Is All You Need*
([arXiv:1706.03762](https://arxiv.org/abs/1706.03762), the self-attention kernel reused from L045).

## Single skill
Build FT-Transformer's two load-bearing pieces **from scratch** — (1) the **numeric tokenizer**
`T_j = b_j + x_j·W_j` (the affine embedding that lets a *number* become an attention token) and (2) the
learned **[CLS]** readout — then hold the model to **L042's baseline-first rule**: show it beats
TabTransformer *most where numerics carry the signal*, is the **best single neural model**, and — honestly —
still loses the flat-table metric to CatBoost.

## Why this was the ZPD
L045 built and trained TabTransformer and *named* its ceiling: only categoricals are contextualised; numeric
features are LayerNorm'd and concatenated, never attending. L046 is the surgical fix that ceiling motivates —
one clean edit (tokenise numerics too) inside the *same* Transformer machinery the student already validated.
It reuses, without rebuilding, the L045 attention kernel (matched to torch at |Δ| ≈ 1e-16), so all new
effort lands on the genuinely new idea: turning a scalar into an order-preserving token and reading the row
out through [CLS]. Heavy prior recall: self-attention (L032/L045), entity embeddings (L031), baseline-first +
shared frame + mean-rank/Friedman (L042/L023/L030), CatBoost native categoricals (L016). It closes the Q1
classic-neural cascade (MLP/ResNet → TabNet → NODE → TabTransformer → FT-Transformer).

## Standard #24 (paper-mirror) — how each axis was met
- **(A) Implementation.** `labs/relkit/ft_transformer.py` implements the load-bearing pieces **from
  scratch**, each annotated with its paper element: `affine_numeric_tokens` (numeric half of the tokenizer),
  `prepend_cls`, `FeatureTokenizer` (numeric affine + categorical embed+bias), `FTTransformer` (tokenizer →
  [CLS] → PreNorm Transformer blocks → [CLS] readout → linear head), `train_ft_transformer`/
  `ft_transformer_auc`. The attention kernel (`MultiHeadSelfAttention`, `scaled_dot_product_attention`) is
  **reused** from `relkit.tabtransformer` (validated to machine precision in L045, standard #22) — not a
  library model. torch is a **checker only**.
- **(B) Datasets.** Bake-off on four Tier-A OpenML tables spanning the numeric fraction — `credit_g`
  (num-frac 0.35), `adult` (0.43, subsampled to 4000), `churn` (0.80), `phoneme` (1.0) — via
  `relkit.data.load_tier_a`. Gap statement shipped in lesson + lab: a **down-scaled demonstration**, *not*
  the paper's 11-dataset tuned benchmark; only the *mechanism* (validated exactly) and the *direction* of
  the tokenizer's advantage + the "best neural, ~ties GBDT" claim are asserted here.
- **(C) Reproducibility.** Fixed seeds (splitter seeds included), one shared frame across arms,
  `labs/_verify_l046.py` → committed `labs/_verify_l046_results.json` (**788.2 s**). No new dependencies —
  torch + catboost already in `requirements-labs.txt`.
- **(Paper-results track, #25).** `labs/_paper_repro_l046.py` (smoke/closer/paper presets on Adult +
  Higgs-small) + `modal/l046_paper_repro.py` (T4 GPU) for an unattended scale-up closer to the paper's
  protocol; the notebook's post-EXIT NEXT STEP cell prints the Colab/Modal commands and the MATCH / CLOSE /
  FAIL / INCOMPARABLE / DIRECTION_* ledger doctrine.

## Verified numbers (evidence of record — `labs/_verify_l046.py`, seeds 0/1/2, 60 epochs, 788.2 s)
**Mechanism validated to machine precision** (reused from-scratch attention vs torch): SDPA max |Δ|
**7.8e-16**. **The numeric-bypass fix, made measurable:** a change to one numeric feature moves
FT-Transformer's **[CLS] readout by L2 = 0.438** on adult, moves the FT-Transformer *flat* (pre-readout)
path **0.0**, and moves **TabTransformer's representation 0.0**. (`labs/_check_l046.py` — **13/13 pass**:
SDPA + MHSA reuse; tokenizer shape; the **affine** property — bump `x_j` by Δ → token moves by exactly
`Δ·W_j`, no other token moves; categorical token construction; [CLS] prepended and identical across rows
pre-attention; the core FT-T-vs-TabTransformer difference; `n_layers=0` ablation; a NUMERIC 2-feature
interaction learned to AUC 1.000.)

**Bake-off — FT-Transformer vs TabTransformer (numeric-bypass, L045) vs MLP vs CatBoost (roc_auc).**

| dataset | num-frac | FT-Transformer | TabTransformer | MLP | CatBoost |
|---|---|---|---|---|---|
| credit_g | 0.35 | 0.785 ± 0.025 | 0.787 ± 0.016 | 0.804 ± 0.023 | **0.804 ± 0.024** |
| adult (4000) | 0.43 | 0.900 ± 0.004 | 0.866 ± 0.009 | 0.898 ± 0.006 | **0.903 ± 0.006** |
| churn | 0.80 | 0.919 ± 0.011 | 0.886 ± 0.010 | 0.901 ± 0.012 | **0.922 ± 0.014** |
| phoneme | 1.00 | 0.927 ± 0.008 | 0.881 ± 0.005 | 0.927 ± 0.006 | **0.946 ± 0.005** |
| **mean rank** | | **2.50** | **3.75** | **2.75** | **1.00** |

Friedman **chi2 = 9.3, p = 0.026** (k = 4, N = 4). **FT-T beats TabTransformer on 3/4** (all but the
most-categorical `credit_g`, where it loses by 0.002 — exactly where numerics matter least), **beats MLP on
3/4**, **is the best NEURAL model**, and **beats CatBoost on 0/4**. The gap over TabTransformer widens with
the numeric fraction (churn/phoneme), which is the mechanism, plotted.

## The honest takeaway — the bypass is fixed, the tree still wins (M61)
Two claims kept apart. **(1) The numeric-bypass fix is real and measurable** — the tokenizer makes numerics
first-class tokens, so a numeric change reshapes the [CLS] readout (L2 0.438) where TabTransformer's moves 0.
That is why FT-T beats TabTransformer most on numeric-heavy tables and is the **strongest single neural
baseline**. **(2) "Fixes the bypass" ≠ "beats trees"** — CatBoost still wins all 4 tables. FT-Transformer is
the best *deep* model and still a notch below a tuned tree on flat data — Gorishniy 2021's exact finding
(paper's ~ties-GBDT claim stays **cited, not reproduced** at this scale).

## What shipped
- **Lesson** `lessons/0046-ft-transformer.html`: warm-up (upTo:46) → TabTransformer-bypass recap (L045
  callback) → the Feature Tokenizer (numeric affine + categorical embed+bias) and [CLS] readout with the new
  `tokenizer-viz` stage-stepper → reused `tabtransformer-arch-viz` for direct architectural contrast →
  predict-before-reveal on the 4-model bake-off → rank table + the "best neural / trees still win /
  gap grows with num-frac" reading → the numerics-attend probe box (0.438 vs 0.0) → validated-mechanism box →
  teach-back (the tokenizer) → thesis bridge (within-row attention over *all* features still adds no
  cross-table structure) → quizzes → primary reading → inline lab → teacher/ask-teacher → nav.
- **One new viz** (`file://`-safe, headless-testable pure functions): `assets/tokenizer-viz.js` (`.tok-`) —
  a stage-stepper (tokenize → prepend [CLS] → attend → read [CLS]) with the affine `numericToken(x,W,b)` and
  a numeric-fraction readout, showing numerics becoming tokens. Reused: `tabtransformer-arch-viz`.
- **Headless checks:** `labs/_viz_check_l046.js` — **all pass** (tokenizer pure functions incl. the affine
  property; lesson↔asset CSS coupling; every verified number present; each honesty statement).
  `labs/_check_pedagogy.js` — **all pass** (incl. the new `gorishniy2021-l046` deck card). Browser
  verification **skipped** (headless-only, consistent with L021–L045).
- **Lab notebook** `labs/0046-ft-transformer.ipynb` (+ solution; + rendered
  `labs/html/0046-ft-transformer.html`), built by `labs/_build_l046.py`. Four tasks: (1) implement
  **`affine_numeric_tokens`** (`T_j = b_j + x_j·W_j`), validated vs reference + affine property; (2) implement
  **`prepend_cls`**, validated vs reference; (3) the **baseline-first** 4-model bake-off with mean ranks +
  Friedman; (4) the **numerics-attend probe** (FT-T [CLS] moves, TabTransformer 0). Both Task-1/2 functions
  are **inlined-and-KEPT** (the notebook copies `relkit/ft_transformer.py` but SKIPS those two names, so the
  model the student trains calls THEIR code — standard #25). **Solution executed end-to-end with nbconvert —
  0 FAIL across all CHECK cells** in **631 s** on CPU; down-scaled lab (2 tables × 2 seeds) reproduces the
  DIRECTION: FT-T beats TabTransformer **2/2**, best neural, CatBoost wins, probe FT-T [CLS] move **0.957**
  vs TabTransformer 0.

## Design choices
- **Reuse the L045 attention kernel, don't rebuild it.** FT-Transformer's contribution is the *tokenizer*,
  not attention. Reusing `relkit.tabtransformer.MultiHeadSelfAttention` (already machine-precision validated)
  keeps the lab's new effort on the one new idea and models standard #22 (a paper-built kernel promoted to
  relkit, not a library import).
- **The two student functions are module-level so they can be inlined-and-kept.** `affine_numeric_tokens`
  and `prepend_cls` are top-level in `ft_transformer.py`; the notebook inlines the rest of the file but skips
  those two, so the student's Task-1/2 code is load-bearing for the inlined `FTTransformer` (#25 — the
  paper's code is visible AND the student's code is the one that runs).
- **Four datasets chosen to span the numeric fraction** (0.35 → 1.0) so the tokenizer's advantage over
  TabTransformer is visible *as a function of num-frac*, and the one table where FT-T loses (`credit_g`, most
  categorical) teaches the mechanism rather than contradicting it.
- **The lab reproduces DIRECTION honestly, not the paper's numbers.** Task-3's CHECK explicitly holds the
  down-scaled 2-table ranks apart from the fuller `_verify_l046.py` run and the paper's benchmark (#23/#25).

## Artifacts synced
- `assets/retrieval-pool.js` **+3**: `l046-feature-tokenizer` (the affine numeric token), `l046-cls-readout`
  (the learned [CLS] pool), `l046-numeric-bypass-fixed` (misconception, mirrors M61 — fixes the bypass yet
  loses to trees). (Kept the pre-existing L041 preview items `l041-tokenizer` / `l041-nowinner`.)
- `assets/paper-deck.js` **+1** card `gorishniy2021-l046` (tokenizer, [CLS], the machine-precision mechanism,
  the probe, and the honest "best neural / ~ties GBDT" verdict). Coexists with the L041 landscape card
  `gorishniy2021-ftt` and the L042 protocol card `gorishniy2021-protocol`.
- `misconceptions.md` **M61** (tokenising numerics fixes the bypass ⇒ neural finally beats trees → no:
  best neural, still 0/4 vs CatBoost).
- `reference/glossary.html` — **Feature Tokenizer**, **[CLS] token**, **FT-Transformer** entries updated with
  the affine detail, the verified probe/rank numbers, and re-tagged **041·046** (L041 preview → L046 full).
- `thesis-dossier.md` — narrative **L046** paragraph. BAR (the strongest classic single-table neural model,
  built honestly, is still a notch below a tree on flat data) + FOR (perfecting attention *within* a
  flattened row buys the best neural rank yet without overtaking the tree — the untapped signal is across
  the join).
- `lessons/manifest.json` — L046 entry with `labPath: labs/0046-ft-transformer.ipynb` (46 entries); the
  notebooks/home galleries pick it up dynamically.
- `lessons/0045-tabtransformer.html` nav + body now link **forward** to L046.
- **New files:** `labs/relkit/ft_transformer.py`, `labs/_check_l046.py`, `labs/_verify_l046.py` (+ results
  JSON), `labs/_paper_repro_l046.py`, `modal/l046_paper_repro.py`, `labs/_build_l046.py`,
  `labs/0046-ft-transformer.ipynb`, `labs/solutions/0046-ft-transformer.ipynb` (executed),
  `labs/html/0046-ft-transformer.html`, `labs/_viz_check_l046.js`, `assets/tokenizer-viz.js`.

## Open items
- **Browser verification not run** this session (consistent with prior lessons). The new `tokenizer-viz` is
  node-verified only; a `file://` pass with the puppeteer method is owed if any layout doubt arises.
- **Paper-results track is wired but not yet run at scale.** `_paper_repro_l046.py` / `modal/l046_paper_repro.py`
  exist; the paper's "strongest deep model, ~ties tuned GBDT" claim stays **cited, not reproduced** until a
  Modal/Colab scale-up (with Optuna-scale tuning) is executed and its ledger pasted back.
- **The thesis FOR is still asserted, not demonstrated across tables** — the Year 3–4 burden (GNNs →
  RelBench) noted for L044/L045 carries forward. Q1's single-table neural repertoire is now exhausted.

## Next
User runs the lab and pastes the EXIT ticket (tokenizer + [CLS] validation, the bake-off ranks, the
FT-T-vs-TabTransformer count, the numerics-attend probe, and the one-sentence "what tokenising numerics
buys / does it beat a tree" verdict) or says "lab done". On completion, open **Lesson 047** per
`CURRICULUM.md` / `plan/year-2.md` — continuing the Y2Q1 deep-tabular track now that the classic neural
architectures (through FT-Transformer) have all been built honestly and held to the same baseline-first bar.
