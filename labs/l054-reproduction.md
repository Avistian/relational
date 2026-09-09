# L054 reproducibility contract

Scope: the complete **numeric** forward path of TabM-mini (a BatchEnsemble of MLPs), plus a four-arm comparison — single MLP, deep ensemble MLP×k, TabM-mini, tuned XGBoost — and the collective-versus-individual submodel measurement. Feature embeddings (the paper's "†" variants), categorical inputs, domain-aware splits, and per-model tuning are excluded. This is **not** the TabM 46-dataset benchmark.

## Primary sources

- Paper: https://arxiv.org/abs/2410.24210 (v3; §3 model, §5 analysis). Read §3.2–3.3 and §5.1–5.4.
- Reference implementation (not copied here): https://github.com/yandex-research/tabm
- BatchEnsemble: Wen, Tran & Ba, *BatchEnsemble* (ICLR 2020).

The teaching implementation in `relkit/tabm.py` is written independently from the paper equations, not adapted from the authors' code. It reproduces the load-bearing identity `Wᵢ = W ⊙ (sᵢ rᵢᵀ)` and the mean-over-members objective; it does **not** reproduce the authors' benchmark, tuning, or reported scores.

## Claims verified against the paper (v3)

- Benchmark size "46 datasets" — §5 (Figure 6 caption; dead-neuron and pruning averages).
- Dead-neuron fractions 0.29 ± 0.17 (k=1) vs 0.14 ± 0.09 (k=32) — §5.4.
- Greedy pruning keeps 8.8 ± 6.6 of 32 submodels on average — §5.2.
- "Weak individually, powerful collectively"; even the best submodel ≈ a plain MLP — abstract and §5.1.
- k too high can be detrimental; narrow (d=64) or shallow (n=1) TabM is suboptimal — §5.3.

## Data and local protocol

Reuse the verified TabR-release archive (California Housing, House 16H, Higgs Small) via `_fetch_l052.fetch`, hashes in `_data_l052.json`. Real arrays, but a practical substitute for the TabM benchmark: shared dataset names do **not** establish split equivalence. Released splits retained. Label-blind row caps 1200 train / 600 validation / 600 test; selection seeds 53/54/55; model seeds 0/1/2. z-score statistics fit on training rows only.

All neural arms: width 64, depth 3, 64 epochs, batch 256, dropout 0.1, Adam at one shared learning rate (2e-3), last-best-validation-epoch selection. TabM-mini uses k=32 with the first ±1 adapter only; the deep ensemble trains k=32 independently seeded MLPs and averages them. XGB-tuned selects among six candidates (depths {3,6} × rates {.03,.1,.2}) on validation, 150-tree cap. Same raw imputed values and splits for every tree candidate; no neural transform for trees.

Errors: original-unit RMSE (regression) or classification error (fraction of wrong argmax, **not** AUROC). Three-seed means, sample SD, and 95% Student-t intervals conditional on fixed splits. Across three tasks: mean ranks, an exploratory approximate Friedman test, and a Nemenyi critical difference. Three seeds are not three datasets; three datasets are low power. No cross-unit mean RMSE is reported.

## Measured result (this lab, `_verify_l054_results.json`)

- Diversity effect (collective < mean-individual submodel) reproduces on **all three** tasks.
- Deep ensemble MLP×32 beats the single MLP on **all three** tasks.
- Head-to-head is mixed: TabM-mini beats the single MLP on House and Higgs, loses on California; it beats MLP×32 only on Higgs. XGB-tuned wins California and Higgs but is last on House.
- Mean ranks: MLP×32 = XGB-tuned = 2.0, TabM-mini = 2.67, MLP = 3.33. Friedman p = 0.53.
- k-sweep on California (seed 0): the individual-vs-collective gap widens with k (0.000→0.008), yet the absolute collective error rises with k — averaging cannot rescue jointly-biased narrow submodels on a task that does not suit TabM-mini.
- Verdict **INCOMPARABLE** to the paper benchmark: width, depth, epochs, dataset sizes, splits and feature embeddings all differ.

Local runtime ≈ 12 CPU minutes for the full 3-seed suite plus k-sweep; the notebook's live cell is lighter (1 seed, k=16, 48 epochs).

## Rebuild and checks

From `labs/`: `python _verify_l054.py` (writes `_verify_l054_results.json`), `python _figures_l054.py` (writes `figures/l054/*.png`), `python _build_l054.py` (writes student + solution notebooks). Execute the solution notebook, then `python _render_l054.py` to refresh prose and render the prepared student HTML under `html/`. The notebook CHECK cells verify the BatchEnsemble identity, the no-op initialisation of non-first adapters, member diversity, packed heads, the mean-over-members loss, and collective ≤ mean-individual on the fitted model.

Browser render, live Colab UI, and any remote publication are separate from local generation and are not claimed here.
