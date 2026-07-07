# Year 2 — Advanced Tabular Deep Learning · Lesson Decomposition

**Year goal:** know every major neural tabular architecture and tabular foundation model, when each
wins, and *why strong trees/MLPs still often win on single tables*. This is the year that makes the
student **honest about baselines** — the precondition for a credible relational thesis.

Reproduction labs are **incremental from here** (`labs/relkit/`): every training/eval lab imports the
harness built in Y1, adding the fair-comparison protocol (same splits, same tuning budget, same metric).

---

## Q1 · Neural tabular architectures (041–050)

**Papers (chronological):** Popov 2019 NODE `1909.06312` · Arik 2019 TabNet `1908.07442` · Huang 2020
TabTransformer `2012.06678` · Gorishniy 2021 FT-Transformer/ResNet `2106.11959` ★ · Somepalli 2021
SAINT `2106.01342` · Wang 2021 DCNv2 `2008.13535` · Chen 2023 ExcelFormer `2301.02819` · Chen 2023
Trompt `2305.18446`.

### 041 · Deep tabular landscape & rtdl — *Gorishniy 2021 full, ★ `2106.11959`*
- **Skill** — set up the `rtdl` library and state the paper's thesis: with a fair protocol, most fancy
  tabular nets don't beat a well-tuned MLP/ResNet, and FT-Transformer is the honest attention baseline.
- **Teach** — the "revisiting" methodology (unified tuning + eval), why prior architecture claims didn't
  replicate, the taxonomy (MLP-like vs attention-like vs tree-like nets); read as a *protocol* lesson.
- **Lab** — Tier A · crucial fragment: install `rtdl`, load one dataset, run the provided MLP with the
  paper's default protocol → record baseline. Deliverable: working rtdl env + baseline number.
- **Viz** — new `arch-family-viz.js`: a taxonomy tree of tabular DL families (reused across 041–058).
- **Bridge** — callback Y1 fair-comparison discipline (L038); sets the protocol for the whole quarter.

### 042 · MLP & ResNet baselines (do these first) — *Gorishniy 2021 §3.2*
- **Skill** — train a tabular ResNet and reproduce that a tuned MLP/ResNet is a *strong* baseline, not a
  strawman.
- **Teach** — tabular ResNet block (linear→BN→ReLU→dropout + skip); why depth helps less than on images;
  the "tune the baseline before the fancy model" rule.
- **Lab** — Tier A · crucial fragment: implement the ResNet block, train vs the L041 MLP → compare to Y1
  tuned XGB. Deliverable: three-way table (MLP/ResNet/XGB), same protocol.
- **Viz** — reuse `arch-family-viz.js` (highlight MLP/ResNet branch).
- **Bridge** — the DL baseline the rest of Y2 must beat; callback Y1 XGB harness.

### 043 · TabNet (sequential attention) — *Arik & Pfister 2019, ◆ `1908.07442`*
- **Skill** — explain TabNet's sequential feature-masking (attentive transformer + feature transformer)
  and read the learned masks as instance-wise feature selection.
- **Teach** — decision steps, sparsemax masks, the interpretability claim; honest note — often *loses* to
  GBDTs and is unstable to tune.
- **Lab** — Tier A · crucial fragment: train TabNet on one task, extract + visualize the step masks.
  Deliverable: mask heatmap + "which features it selected."
- **Viz** — new `mask-viz.js`: per-step feature-mask heatmap (reused for TabNet/SAINT attention).
- **Bridge** — first "attention on tabular" idea; contrast with L046 FT-T; failure-mode framing.

### 044 · NODE (differentiable trees) — *Popov 2019, ◆ `1909.06312`*
- **Skill** — describe oblivious differentiable decision trees and when a differentiable tree ensemble
  helps over a GBDT (irregular but learnable-end-to-end targets).
- **Teach** — entmax feature selection + soft oblivious splits; the "make trees differentiable so they
  compose with DL" motivation; honest note — marginal wins, heavy compute.
- **Lab** — Tier A · crucial fragment: train NODE, compare to CatBoost (both "tree-like") on the same task.
  Deliverable: comparison + one sentence on when NODE's differentiability would matter.
- **Viz** — reuse `growth-viz.js`/`tree-viz.js` (soft vs hard split).
- **Bridge** — callback Y1 L016 (oblivious trees = CatBoost symmetric trees); the differentiable-tree lineage.

### 045 · TabTransformer (contextual embeddings) — *Huang 2020, ◆ `2012.06678`*
- **Skill** — train TabTransformer and show contextual categorical embeddings help on categorical-rich
  data, and name its limitation (numeric features bypass attention).
- **Teach** — column embeddings → transformer over categoricals → concat numeric → MLP; contrast with the
  static embedding of Y1 L031/L032.
- **Lab** — Tier A (categorical-rich) · crucial fragment: run TabTransformer vs the L031 static-embedding
  MLP. Deliverable: table + "what context bought / where it didn't."
- **Viz** — reuse `embedding-viz.js` (static vs contextual) + `mask-viz.js`.
- **Bridge** — closes the L032 preview; motivates L046 (FT-T tokenizes numeric too).

### 046 · FT-Transformer — *Gorishniy 2021 §3.3, ★ `2106.11959`*
- **Skill** — implement/train the FT-Transformer feature tokenizer (numeric + categorical → tokens + [CLS])
  and reproduce that it is the strongest single neural baseline under a fair protocol.
- **Teach** — feature tokenizer (linear per numeric feature + embedding per categorical), [CLS] readout,
  why tokenizing *numeric* features is the key upgrade over TabTransformer.
- **Lab** — Tier A · crucial fragment (paper-repro): implement the feature tokenizer; train FT-T vs
  ResNet/XGB same protocol → within tolerance of paper ranking. Deliverable: reproduction table.
- **Viz** — new `tokenizer-viz.js`: a row → per-feature tokens → [CLS] attention (reused Y4/Y5 tokenization).
- **Bridge** — the canonical attention baseline reused all year and as an RDL row-encoder ancestor; callback
  L042 protocol.

### 047 · SAINT (row + column attention) — *Somepalli 2021, ◆ `2106.01342`*
- **Skill** — explain inter-sample (row) attention and why attending across *rows in a batch* is a
  retrieval-like mechanism.
- **Teach** — column attention + inter-sample attention + contrastive pretraining; row attention previews
  retrieval (L052 TabR) and ICL (L066 TabICL).
- **Lab** — Tier A · crucial fragment: train SAINT; ablate inter-sample attention on/off. Deliverable:
  ablation table + "what row attention adds."
- **Viz** — reuse `mask-viz.js` (row-vs-column attention panels).
- **Bridge** — direct conceptual parent of TabR (L052) and TabICL's column-then-row attention (L066).

### 048 · DCNv2 & explicit feature crosses — *Wang 2021, ◆ `2008.13535`*
- **Skill** — build a cross network and explain explicit bounded-degree feature crossing vs implicit MLP
  crossing.
- **Teach** — the cross layer `x_{l+1}=x_0 ⊙ (W x_l)+x_l`, why explicit crosses matter for
  recommendation/CTR, parallel vs stacked structure.
- **Lab** — Tier A · crucial fragment: implement one cross layer; DCNv2 vs MLP on a cross-heavy task.
  Deliverable: table + "when explicit crosses win."
- **Viz** — new `cross-viz.js`: feature-cross lattice (degree-2, degree-3) — reusable for recsys in Y4.
- **Bridge** — recsys lineage → Y4 L138 (e-commerce), L144 (ContextGNN recsys).

### 049 · ExcelFormer & Trompt (GBDT-surpassing claims) — *Chen 2023 ◆ `2301.02819` · Chen 2023 ◆ `2305.18446`*
- **Skill** — read two "we beat GBDTs" papers *critically*: separate the architectural idea from the
  evaluation protocol that produced the claim.
- **Teach** — ExcelFormer (semi-permeable attention, data-specific), Trompt (prompt-style per-sample
  weights); then the protocol audit — dataset selection, tuning budget, split type (callback L038, L055).
- **Lab** — Tier A · crucial fragment: reproduce one claim on the paper's dataset *and* on a temporal-split
  dataset; document where the claim holds/breaks. Deliverable: claim-audit note.
- **Viz** — reuse `arch-family-viz.js` + `checklist.js` (claim-audit variant).
- **Bridge** — trains the skepticism the thesis needs; sets up Q2's honest-baseline reckoning.

### 050 · **Q1 checkpoint** — *Gorishniy 2021 · Deliverable-based*
- **Deliverable** — FT-Transformer vs tuned XGB on **3 datasets, identical protocol** (splits, tuning
  budget, metric, variance). Written verdict on where FT-T ties/wins/loses.
- **Bridge** — locks the fair-comparison harness for TFMs in Q2–Q3; callback L041/L046.

---

## Q2 · Modern tabular DL & the honest baseline (051–060)

**Papers:** Grinsztajn 2022 (revisit `2207.08815`) · TabR `2307.14338` ★ · RealMLP `2407.04491` ★ ·
TabReD `2406.19380` ★ · TabM `2410.24210` ★ · Cai & Ye 2025 `2502.20260` ★ · TabArena `2506.16791` ★ ·
BeyondArena `2606.30410` ★ · TALENT `2407.00956` · survey `2410.12034`.

### 051 · Why trees still win (revisit) — *Grinsztajn 2022 §5, ★ `2207.08815`*
- **Skill** — re-derive the three inductive biases (irregular targets, uninformative features, rotation)
  now with the DL toolkit in hand, and predict which Y2 architecture attacks which bias.
- **Teach** — deep recap of §5 experiments (this time reproducing, not previewing as in L019); map each
  bias → architectural remedy (embeddings/tokenizer for orientation, retrieval for irregularity, etc.).
- **Lab** — Tier A/C · crucial fragment: reproduce one Grinsztajn §5 bias experiment with an FT-T *and* a
  tree. Deliverable: the reproduced gap + which bias it isolates.
- **Viz** — reuse `biases-viz.js` (from L019).
- **Bridge** — the analytic backbone of Q2; callback L019, forward to every "does DL beat trees?" checkpoint.

### 052 · TabR — retrieval-augmented DL — *Gorishniy 2023, ★ `2307.14338`*
- **Skill** — explain TabR's kNN-attention retrieval component and why retrieving similar training rows
  attacks the irregular-target bias.
- **Teach** — the retrieval module (query row attends to nearest neighbors' labels/features), freeze-vs-learn
  the encoder, when retrieval beats parametric memorization.
- **Lab** — Tier A · crucial fragment (paper-repro): implement the kNN-attention step; TabR vs FT-T vs XGB.
  Deliverable: table + "when retrieval helped."
- **Viz** — new `retrieval-viz.js`: query row → nearest neighbors highlighted → weighted label pooling
  (reused for L066 TabICL, L165b/L178 relational ICL).
- **Bridge** — retrieval lineage L047 SAINT → here → L066 TabICL → Y5 relational ICL. Core mechanism of the
  ICL frontier.

### 053 · RealMLP & strong defaults — *Holzmüller 2024, ★ `2407.04491`*
- **Skill** — reproduce that a well-designed *default* MLP (RealMLP) rivals tuned GBDTs with little tuning —
  the "better by default" result.
- **Teach** — the bundle of defaults (preprocessing, scaling, architecture, LR schedule); the argument that
  much prior "DL loses" was under-tuned DL; default-vs-tuned honesty.
- **Lab** — Tier A · crucial fragment: run RealMLP defaults vs tuned XGB (equal *human* budget).
  Deliverable: table + "tuning budget spent on each."
- **Viz** — reuse `arch-family-viz.js`; optional defaults-checklist via `checklist.js`.
- **Bridge** — sharpens the honest baseline: the thesis must beat *strong defaults*, not lazy DL. Callback L042.

### 054 · TabM — parameter-efficient ensembling — *Gorishniy 2024, ★ `2410.24210`*
- **Skill** — build TabM's efficient implicit ensemble (BatchEnsemble-style) and reproduce that it is the
  current best *trained* DL tabular baseline.
- **Teach** — shared weights + per-member rank-1 adapters → k models for ~1 model's cost; why ensembling is
  the reliable DL win on tabular; TabM as the DL baseline for the rest of the curriculum.
- **Lab** — Tier A · crucial fragment (paper-repro): implement the rank-1 multi-head; TabM vs RealMLP vs XGB.
  Deliverable: reproduction within tolerance + variance bars.
- **Viz** — reuse `ensemble-viz.js` (from L012) adapted to weight-sharing.
- **Bridge** — the "best trained DL baseline" cited through Y2–Y6; callback L018 ensembling, L012 bagging.

### 055 · The temporal-split reality — *TabReD, ★ `2406.19380`*
- **Skill** — demonstrate that switching from random to temporal splits **flips model rankings**, and adopt
  time-split-by-default when timestamps exist.
- **Teach** — TabReD's industrial datasets, why random splits leak future info, how rankings change; the
  single most important evaluation upgrade for the thesis (RDL lives in temporal data).
- **Lab** — Tier A (timestamped, TabReD-style) · crucial fragment: run the same models under random vs
  temporal split; show the ranking flip. Deliverable: the two ranking tables side by side.
- **Viz** — reuse `split-viz.js` (random vs temporal) with a ranking-flip readout.
- **Bridge** — callback Y1 L021 temporal split; forward to Y3 temporal graphs, Y4 temporal REG audit. Central.

### 055b · Temporal shift & DL limits — *Cai & Ye 2025, ★ `2502.20260` (ICML 2025)*
- **Skill** — apply the corrected validation protocol (minimize train↔test time lag; a *random* val split
  can beat a temporal one) and add a Fourier temporal embedding to capture trend/periodicity.
- **Teach** — why deep tabular models miss periodic/trend structure; the model-selection bug in temporal
  val splits; plug-and-play Fourier-series temporal features.
- **Lab** — Tier A · crucial fragment: implement the Fourier temporal embedding + the val-split protocol;
  measure the gain on a temporal task. Deliverable: before/after + protocol note.
- **Viz** — new `temporal-embed-viz.js`: raw time → Fourier basis features (reused Y3 temporal, Y4).
- **Bridge** — pairs with L055; the "how to *fix* temporal DL" counterpart; forward to Y3 L101–L104.

### 056 · Living benchmark literacy — *TabArena, ★ `2506.16791`*
- **Skill** — read the TabArena leaderboard methodology and cite results correctly (ensembling protocol,
  model families, dataset scope).
- **Teach** — living-benchmark concept, the cross-model ensembling protocol, how to avoid cherry-picking;
  the "2026 fair stack" (tuned GBDT + RealMLP/TabM + TabICLv2 if n≲50K, time splits).
- **Lab** — Tier A · crucial fragment: reproduce one TabArena entry with its protocol. Deliverable: matched
  number + methodology summary.
- **Viz** — reuse `checklist.js` (leaderboard-reading rubric).
- **Bridge** — the benchmark literacy needed to make thesis claims legible (callback Y1 L024 Grinsztajn
  benchmark); forward to Y4 RelBench, Y5 RelBench v2 leaderboards.

### 056b · Beyond IID / enterprise gap — *BeyondArena, ★ `2606.30410`*
- **Skill** — state precisely where tabular foundation models fail — non-IID (temporal/grouped), large,
  high-dimensional, high-cardinality/text — and why trees/DL still dominate there.
- **Teach** — the IID-benchmark selection bias, Data Foundry curation, the result: TFMs win tiny–medium IID,
  trees/DL win the hard regimes; guardrail against "TFMs dominate."
- **Lab** — Tier A · crucial fragment: evaluate a TFM vs tree on an IID *and* a non-IID/large slice; show
  the crossover. Deliverable: the crossover table.
- **Viz** — reuse `arch-family-viz.js` + a regime grid (IID/temporal/grouped × size).
- **Bridge** — the honest ceiling for single-table TFMs; motivates *relational* structure as the next axis.
  Callback L055; forward to Y5 FM claims.

### 057 · Ensembling across model families — *TabArena §results*
- **Skill** — build a leak-free cross-family ensemble (tree + RealMLP/TabM + TFM) and show it beats any
  single family.
- **Teach** — diversity across families > within family (callback L018); OOF stacking at scale; the real
  single-table baseline is an *ensemble*, not one model.
- **Lab** — Tier A · crucial fragment: OOF-stack XGB + TabM + TabICL. Deliverable: ensemble vs best-single.
- **Viz** — reuse `stacking-viz.js` (from L018).
- **Bridge** — the true bar the RDL thesis must clear (callback L018 thesis bridge); forward to Y6 baseline suite.

### 058 · Surveys & meta-benchmarks — *Borisov 2021 `2110.01889` · TALENT `2407.00956` · survey `2410.12034`*
- **Skill** — map the tabular DL family tree and use TALENT's 300+-dataset meta-analysis to make
  population-level (not cherry-picked) claims.
- **Teach** — survey taxonomy; TALENT toolbox + meta-results; "on average vs on my data" reasoning.
- **Lab** — Tier A · crucial fragment: use TALENT to pull aggregate rankings; locate your task's regime.
  Deliverable: where your data sits in the meta-analysis.
- **Viz** — reuse `arch-family-viz.js` (annotated with meta-ranking).
- **Bridge** — population-level literacy for honest claims; callback L056; ID fixes noted (TALENT `2407.00956`).

### 059 · Validation-set overfitting — *TabArena critique*
- **Skill** — diagnose an ensemble/model that overfits the validation set and fix the selection protocol.
- **Teach** — repeated model selection on one val split leaks (callback L004 nested CV); TabArena's
  guardrails; the optimism gap.
- **Lab** — Tier A · crucial fragment: induce val overfitting via many selections, then fix with nested CV;
  measure the optimism gap. Deliverable: gap before/after.
- **Viz** — reuse `checklist.js` + `group-viz.js` (nested CV).
- **Bridge** — callback L004/L038; the discipline that keeps Y4–Y6 reproductions honest.

### 060 · **Q2 checkpoint** — *TabM + RealMLP · Deliverable-based*
- **Deliverable** — beat the Y1 tuned XGB with a tuned DL model (TabM/RealMLP) under **both random and
  temporal splits**, or document with evidence why you can't. Report variance.
- **Bridge** — the honest-baseline milestone; the "2026 fair stack" is now the standing baseline for Q3/Q4
  and the thesis.

---

## Q3 · Tabular foundation models & in-context learning (061–070)

**Papers (chronological):** Müller 2022 PFN `2112.10510` · Hollmann 2022 TabPFN v1 `2207.01848` · LoCalPFN
`2406.05207` · Drift-Resilient TabPFN `2411.10634` · Hollmann 2025 TabPFN v2 (Nature) ★ · TabICL `2502.05564`
★ · TabICLv2 `2602.11139` ★ · Ye 2025 `2502.17361` · Cheng 2025 `2505.16226` ★ · Klein & Hoffart 2026
`2606.29091` ★ · (extend at runtime: TabPFN-2.5 `2511.08667`, TabPFN-3 `2605.13986`).

### 061 · Prior-Data Fitted Networks — *Müller 2022, ★ `2112.10510`*
- **Skill** — explain how a transformer trained on synthetic datasets learns to *approximate Bayesian
  posterior prediction* in a single forward pass (no per-task training).
- **Teach** — the PFN training loop (sample a prior → sample a dataset → predict held-out point);
  in-context learning = amortized inference; the prior *is* the inductive bias.
- **Lab** — Tier C · crucial fragment: implement the PFN meta-training loop on a toy prior; show it does
  posterior-like prediction. Deliverable: PFN posterior vs analytic posterior on a toy Gaussian.
- **Viz** — new `pfn-prior-viz.js`: sample datasets from a prior → the net's predictive posterior (reused
  L063, Y5 RDB-PFN).
- **Bridge** — the conceptual root of the entire FM frontier (tabular *and* relational); forward to Y5 L166.

### 062 · TabPFN v1 (≤1K rows) — *Hollmann 2022, `2207.01848`*
- **Skill** — run TabPFN v1 and reproduce that a single forward pass rivals tuned baselines on small data.
- **Teach** — the ≤1K-row / ≤100-feature constraints, no training at inference, the SCM prior; where it
  breaks (size, categoricals).
- **Lab** — Tier A (small) · crucial fragment: run TabPFN v1 vs tuned XGB on ≤1K rows. Deliverable: table +
  wall-clock (no training!).
- **Viz** — reuse `retrieval-viz.js` (context set → prediction).
- **Bridge** — first hands-on ICL; callback L061; forward to L064 v2.

### 063 · What the synthetic SCM prior encodes — *Hollmann 2022 §4*
- **Skill** — inspect samples from TabPFN's structural causal model prior and explain how prior design
  determines generalization.
- **Teach** — SCMs → synthetic (X,y); why a *causal* prior yields tabular-appropriate functions; the link
  "prior = inductive bias" (callback L061); implications for OOD (L069).
- **Lab** — Tier C · crucial fragment: sample from a simple SCM prior; show the induced function class.
  Deliverable: prior-sample gallery + one sentence on the encoded bias.
- **Viz** — reuse `pfn-prior-viz.js`.
- **Bridge** — sets up the "prior mismatch → failure" argument in L069 and the relational prior in Y5 L166.

### 064 · TabPFN v2 (Nature) — *Hollmann 2025, ★ Nature `s41586-024-08328-6`*
- **Skill** — run TabPFN v2 and reproduce it matching/beating a 4-hour-tuned GBDT on small–medium data in
  seconds; state its size ceiling.
- **Teach** — v2 upgrades (mixed types, larger, better prior), the ≤10K-row regime, the "seconds vs hours"
  result; extend at runtime to TabPFN-2.5 (~100K) / TabPFN-3 (~1M).
- **Lab** — Tier A · crucial fragment (paper-repro): TabPFN v2 vs 4h-tuned GBDT on a small task, wall-clock
  matched. Deliverable: accuracy + time table.
- **Viz** — reuse `retrieval-viz.js`.
- **Bridge** — the headline TFM result; callback L056b ceiling; forward to L070 checkpoint and Y5 transfer (L167).

### 065 · How TabPFN v2 handles heterogeneity — *Ye 2025, `2502.17361`*
- **Skill** — use TabPFN v2 as a *feature extractor* and explain mechanistically how it copes with
  heterogeneous columns.
- **Teach** — the "closer look" findings (what the embeddings encode, feature-extractor mode, failure
  patterns); mechanistic, not just benchmark.
- **Lab** — Tier A · crucial fragment: extract TabPFN v2 embeddings, feed a linear head. Deliverable:
  embedding-head vs end-to-end.
- **Viz** — reuse `embedding-viz.js`.
- **Bridge** — mechanistic grounding for L069's failure analysis; forward to relational encoders (Y4 L125b).

### 066 · TabICL — scaling ICL to 500K rows — *Qu 2025, ★ `2502.05564` (ICML 2025)*
- **Skill** — explain column-then-row attention and how it scales ICL far past TabPFN's row ceiling.
- **Teach** — the two-stage attention (per-column embedding → per-row set attention), memory tricks, the
  500K-row result; open weights.
- **Lab** — Tier A (medium/large) · crucial fragment (paper-repro): run TabICL on a medium dataset vs
  TabPFN v2 (which OOMs / caps). Deliverable: scale-vs-accuracy table.
- **Viz** — reuse `retrieval-viz.js` + `mask-viz.js` (column-then-row panels).
- **Bridge** — callback L047 SAINT row attention, L052 TabR retrieval; forward to L066b and Y5 relational ICL.

### 066b · TabICLv2 — open SOTA ICL — *Qu 2026, ★ `2602.11139`*
- **Skill** — reproduce that TabICLv2 (untuned) surpasses tuned+ensembled RealTabPFN-2.5, and name its
  three pillars.
- **Teach** — QASSMax scalable softmax (generalize to larger n without long-sequence pretraining), Muon
  optimizer, high-diversity synthetic engine; million-scale under 50GB; open weights/code.
- **Lab** — Tier A · crucial fragment: TabICLv2 vs TabICL v1 vs RealTabPFN-2.5 on TALENT/TabArena slice.
  Deliverable: untuned-v2 vs tuned-baseline table.
- **Viz** — reuse `arch-family-viz.js` (ICL branch) + `retrieval-viz.js`.
- **Bridge** — the current open TFM SOTA and the single-table baseline for the thesis (n≲50K); callback
  L056 fair stack; forward to Y5 RDBLearn (which *pairs* a relational encoder with a single-table ICL model).

### 067 · Retrieval + fine-tuning PFNs — *LoCalPFN, ★ `2406.05207`*
- **Skill** — apply local retrieval + light fine-tuning to a PFN and explain when local calibration beats
  a single global forward pass.
- **Teach** — retrieve a local context per query, fine-tune locally; bridges pure-ICL and trained models.
- **Lab** — Tier A · crucial fragment: add kNN retrieval to a PFN context; measure local vs global.
  Deliverable: local-calibration gain.
- **Viz** — reuse `retrieval-viz.js`.
- **Bridge** — callback L052 TabR; the retrieval+ICL hybrid that reappears in Y5 relational ICL.

### 068 · Distribution shift & PFNs — *Drift-Resilient TabPFN, `2411.10634`*
- **Skill** — test a PFN under temporal/distribution shift and explain why a fixed prior degrades OOD.
- **Teach** — drift-resilient prior/mechanism, the temporal-shift failure of vanilla TabPFN (callback L055);
  what a shift-aware prior buys.
- **Lab** — Tier A (timestamped) · crucial fragment: evaluate vanilla vs drift-resilient TabPFN under a
  time split. Deliverable: shift-robustness table.
- **Viz** — reuse `split-viz.js` (temporal) + `temporal-embed-viz.js`.
- **Bridge** — callback L055/L055b; forward to L069 open-world failure and Y5 FM robustness.

### 069 · **Critical:** where TabPFN v2 breaks — *Cheng 2025, ★ `2505.16226`*
- **Skill** — enumerate open-environment failure cases of TabPFN v2 (novel classes, shift, scale, leakage
  in eval) and design an honest open-world evaluation.
- **Teach** — the realistic-eval critique; closed- vs open-environment benchmarking; why headline numbers
  overstate real-world readiness.
- **Lab** — Tier A · crucial fragment: construct an open-environment split that breaks TabPFN v2.
  Deliverable: the failure case + why.
- **Viz** — reuse `checklist.js` (open-world eval rubric).
- **Bridge** — the honesty guardrail for the FM frontier; callback L063 prior mismatch; forward to L069b, Y5.

### 069b · **Critical:** operational identifiability — *Klein & Hoffart 2026, ★ `2606.29091`*
- **Skill** — state the Operational Turing Test barrier: values-only models are information-theoretically
  bounded on rule-governed databases without executable rule-derived features.
- **Teach** — legal vs rule-violating states with matched marginals → Le Cam bound → any values-only
  classifier ≥0.49 error; only rule-derived audits reach 1.0; "identifiability, not capacity."
- **Lab** — Tier C · crucial fragment: build a tiny two-state OTT pair (matched marginals, one rule
  violation); show XGB/TabPFN at chance, a rule-feature at 1.0. Deliverable: the access-ladder table.
- **Viz** — new `ott-viz.js`: two DB states with identical marginals but one broken constraint (reused Y5
  L179 failure modes).
- **Bridge** — the deepest "single-table values are not enough" result — the thesis's strongest theoretical
  ally (relational structure/rules carry the missing signal). Callback L035 flatten loss; forward to Y5 FM limits.

### 070 · **Q3 checkpoint** — *Hollmann 2025 + TabICL + TabICLv2 (+ TabPFN-3) · Deliverable-based*
- **Deliverable** — TabPFN-2.5/3 vs TabM vs XGB on **5 datasets** (mixed sizes, incl. one temporal), fair
  protocol; written verdict on where ICL helps and where it breaks.
- **Bridge** — completes the single-table SOTA picture; callback L060 fair stack; the launchpad for "add
  relational structure" in Q4 and beyond.

---

## Q4 · Self-supervision, encoders & the bridge to relational (071–080)

**Papers:** Yoon 2020 VIME (NeurIPS) · Bahri 2021 SCARF `2106.15147` · Ucar 2021 SubTab `2110.04361` · Hu
2024 PyTorch Frame `2404.00776` ★ · CARTE `2402.16785` (◆ skim) · Gilmer 2017 `1704.01212` · Kipf 2017 §2.

### 071 · VIME — masked tabular SSL — *Yoon 2020*
- **Skill** — pretrain a tabular encoder with masked feature reconstruction + mask-vector estimation, then
  fine-tune, and measure the label-efficiency gain.
- **Teach** — the two VIME pretext tasks (feature imputation + mask estimation), self- vs semi-supervised
  variants; when unlabeled tabular data helps.
- **Lab** — Tier A · crucial fragment (paper-repro): implement masked-reconstruction pretraining; fine-tune
  with few labels vs from-scratch. Deliverable: label-efficiency curve.
- **Viz** — reuse `missingness-viz.js` (masking) or new `mask-pretrain-viz.js`.
- **Bridge** — masking pretext → Y3 graph SSL, Y4/Y5 relational value-masking pretraining (ReDeLEx, RT, RDB-PFN).

### 072 · SCARF / SubTab — contrastive & multi-view — *Bahri 2021 `2106.15147` · Ucar 2021 `2110.04361`*
- **Skill** — build contrastive (SCARF, feature-corruption views) and multi-view (SubTab) tabular SSL and
  contrast with reconstruction SSL.
- **Teach** — corruption-based positive pairs, InfoNCE on rows, subsetting features into views; what
  contrastive learns that reconstruction doesn't.
- **Lab** — Tier A · crucial fragment: implement SCARF corruption + contrastive loss. Deliverable: linear-probe
  accuracy of the learned representation.
- **Viz** — reuse `retrieval-viz.js` (positive/negative pairs) or `embedding-viz.js`.
- **Bridge** — contrastive lineage → graph contrastive (Y3), cross-table transfer (L074).

### 073 · When SSL actually helps — *SSL survey `2402.01204` + your homework VIME notes*
- **Skill** — design an SSL ablation that isolates *when* pretraining helps (much unlabeled data, few
  labels, distribution match) and when it's a wash.
- **Teach** — the honest SSL picture on tabular (often marginal); the conditions that make it pay; ablation
  design (callback L038/L059).
- **Lab** — Tier A · crucial fragment: sweep labeled-fraction; find the crossover where SSL stops helping.
  Deliverable: crossover plot + rule of thumb.
- **Viz** — reuse `checklist.js` (SSL-benefit rubric).
- **Bridge** — the discipline carried into relational pretraining (Y5); avoids "pretraining always helps" hype.

### 074 · Cross-table transfer (CARTE etc.) — *literature skim, CARTE ◆ `2402.16785`*
- **Skill** — explain schema-agnostic transfer: string-aware, graph-attention pretraining that needs no
  entity/schema matching — and why it's an early bridge to the relational graph view.
- **Teach** — CARTE's table-as-graph (cells as nodes, columns as typed edges), string embeddings, transfer
  across tables with different schemas.
- **Lab** — Tier A · crucial fragment: use CARTE embeddings for transfer to a small target table.
  Deliverable: transfer vs from-scratch.
- **Viz** — new `cell-graph-viz.js`: a single table row rendered as a small graph (reused Y4 REG, Y5 RT cells).
- **Bridge** — first "a table *is* a graph" framing; direct on-ramp to Y3/Y4; callback L072 contrastive.

### 075 · PyTorch Frame — the row encoder — *Hu 2024, ★ `2404.00776`*
- **Skill** — encode a mixed-type schema (numeric, categorical, text, timestamp, embedding) into per-row
  representations with PyTorch Frame's stype system.
- **Teach** — semantic types (stypes) vs data types, materialization, the encoder→model split; PyTorch
  Frame as the *row encoder* every RDL model plugs a GNN on top of.
- **Lab** — Tier A/B · crucial fragment (paper-repro): define a `TensorFrame`, configure stype encoders,
  encode a heterogeneous table. Deliverable: encoded row tensor + stype config.
- **Viz** — reuse `tokenizer-viz.js` (mixed-type → tokens) + `cell-graph-viz.js`.
- **Bridge** — THE bridge component — reappears in Y4 L125 (deep dive) and every RDL/FM model; callback L046
  tokenizer; forward to L076.

### 076 · Encoder → predictor stack — *RDL preview*
- **Skill** — assemble the generic RDL forward pattern: row encoder (per table) → message passing (across
  PK/FK) → task head, and trace shapes end to end.
- **Teach** — the encoder→MP→head decomposition (the pattern Y4 builds); why decoupling encoder from GNN
  matters (foreshadows Y4 L125b Universal Row Encoder).
- **Lab** — Tier C/B · crucial fragment: wire PyTorch Frame encoder → a one-hop manual aggregate → head on
  a toy 2-table graph. Deliverable: end-to-end forward pass with shapes.
- **Viz** — new `rdl-stack-viz.js`: encoder → MP → head block diagram (reused throughout Y4).
- **Bridge** — the architectural skeleton of the whole relational thesis; callback L075; forward to Y3 L078, Y4 L131.

### 077 · Single-table ceiling (synthesis) — *— (writing/experiment unit)*
- **Skill** — articulate precisely what a rows-only model *cannot* represent (cross-row identity,
  multi-hop relations, cardinality, rule constraints) with evidence from Q1–Q3.
- **Teach** — synthesis: L035 flatten loss + L056b non-IID gap + L069b operational barrier → the ceiling;
  the thesis stated as a falsifiable claim.
- **Lab** — Tier C · crucial fragment: construct a task solvable with relational structure but provably not
  from the flattened row (aggregation collision from L035, scaled up). Deliverable: the ceiling demonstration.
- **Viz** — reuse `flatten-loss-viz.js` (from Y1 L035) + `ott-viz.js`.
- **Bridge** — the Year-2 thesis capstone; the exact gap Years 3–5 fill; callback L035/L056b/L069b.

### 078 · Bridge: message passing preview — *Gilmer 2017 `1704.01212` · Kipf 2017 §2 `1609.02907`*
- **Skill** — hand-compute one round of message passing (aggregate neighbors → update) and connect it to
  the leak-free relational aggregate from Y1 L034.
- **Teach** — the MPNN message/aggregate/update abstraction; a GNN layer = a *learned* PIT aggregation;
  why this generalizes manual FE (callback L009/L034).
- **Lab** — Tier C · crucial fragment: implement one manual one-hop mean-aggregation on the toy graph.
  Deliverable: node embeddings after one hop, by hand.
- **Viz** — new `message-passing-viz.js`: neighbors → aggregate → update animation (the anchor asset for all
  of Y3).
- **Bridge** — the literal hand-off to Year 3; callback L076 stack; forward to Y3 L081 (MPNN proper).

### 079 · Year 2 essay — *— (writing unit)*
- **Skill** — write a neural-tabular decision guide: given a dataset's regime (size, split, cardinality,
  text), which model family to reach for and why.
- **Teach** — synthesis of the whole year (trees vs RealMLP/TabM vs FT-T vs TabPFN/TabICL vs ensemble) keyed
  to regime; the 2026 fair stack.
- **Lab** — writing deliverable: a one-page decision table (regime → model).
- **Viz** — reuse `arch-family-viz.js` as the essay figure.
- **Bridge** — exit-exam rehearsal; the decision guide is reused whenever a baseline is needed in Y4–Y6.

### 080 · **Year 2 exit exam** — *all Y2 papers · Deliverable-based*
- **Deliverable** — train & fairly compare FT-Transformer, TabM, and TabPFN v2 against a tuned GBDT under
  **both random and temporal splits**; teach-back the three biases + TabM + TabPFN v2 + TabICL; write why
  single-table models plateau and where ICL helps.
- **Exit criterion (from CURRICULUM)** — the above, articulated in writing. Score against it.
- **Bridge** — gate to Year 3. Artifacts carried: the fair-comparison harness, the row encoder (L075), and
  the single-table-ceiling argument (L077).

**Optional (◆, after exit):** CARTE `2402.16785`; Interpretable ML for TabPFN `2403.10923`; TabLLM
`2210.10723` (the LLM boundary the thesis *rejects*); ModernNCA `2407.03257`; TabZilla `2305.02997`;
Molnar *Tabular Foundation Models* book + Mindful Modeler.
