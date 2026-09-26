# Lesson 114 — complete OGB MLP reproduction and graph error analysis

## Result and evidence boundaries

Ten fresh full-data MLP fits ×500epochs completed. Validation57.9110±0.1050%, test55.8036±0.0652% (sample seed SD in percentage points). OGB v6 Table6 targets57.65±0.12% and55.50±0.23%; both means CLOSE under the predeclared0.5pp descriptive tolerance. Variance was measured rather than required to match. Ten GCN checkpoints trained in L112 were freshly replayed; GCN validation73.1041±0.0977%, test71.8480±0.3119%. No fresh GCN training in L114.

Every final prediction of both models (3,386,860 combined) was replayed through canonical and original model code with zero class mismatches and zero current-CPU log-probability differences. This establishes inference parity in the checked environment, not historical optimization identity. Historical identity and whole-paper parity remain NOT_ESTABLISHED.

The validation-nominated failure rule is true-label homophily<.25, selected among fixed degree/homophily/class bins with at least200 validation nodes. On the frozen test rule: n8585, GCN17.2801±0.7297%, MLP27.2231±0.2481%; mean gap−9.9429pp. This is a new course analysis, not a published slice reproduction. All-label homophily and true class are retrospective diagnostics and cannot serve as inputs to a future unlabeled-node routing policy.

[Full results and every slice](evidence/l114/summary.json) · [Bundle](evidence/l114/analysis-inputs.npz) · [Sources](_sources_l114.json) · [Budget](_budget_l114.json) · [Independent SQL/count audit](_audit_l114_results.json).

## Frozen protocol ledger

| Component | Definition |
|---|---|
| Publication | Hu et al., OGB v6 §4.3 Table6, https://arxiv.org/html/2005.00687v6#S4.SS3 |
| Source | snap-stanford/ogb commit61e9784ca76edeaa6e259ba0f836099608ff0586; archived unmodified mlp.py, logger.py, MIT license; same commit as L112 |
| Dataset | Complete release-v1 ogbn-arxiv;169343nodes,1166243directed citation records,128features,40classes |
| Archive SHA256 |49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276|
| Split | Exact released time IDs:90941train through2017;29799valid2018;48603test2019–2020. No random re-split or sampling |
| MLP | Three Linear layers128→256→256→40; BN/ReLU/dropout.5 after each hidden layer; log-softmax;110120parameters |
| Initialization | PyTorch Linear reset distribution and BatchNorm defaults, reset once after construction; explicit independent seeds0–9 |
| Training population | Forward only training feature rows; BN statistics use training nodes only; mean NLL on their labels |
| Optimizer | Adam lr.01, betas(.9,.999),eps1e-8,weight_decay0,amsgradFalse; no scheduler |
| Schedule |500full-training-row epochs per seed; every epoch evaluates all official populations; no early stopping |
| Selection | First maximum validation accuracy; save/restore all parameters and BN running buffers; no test-based selection |
| GCN | All ten L112 selected states reused. Full binary-undirected graph plus one self-loop, symmetric normalization; full-node BN population; see l112-reproduction.md |
| Metric | Correct count/exact official count; independently checked with OGB evaluator. Ten-run mean and sample SD (ddof1) |
| Runtime | Author cloud: Python3.12,torch2.8.0,PyG2.6.1,OGB1.3.6,NumPy2.2.6,pandas2.3.2,scikit-learn1.7.1; T4,2CPUcores,8GiB; runtime identities in every seed directory |
| Diagnostic graph | Unique undirected non-self neighbors,2315598directed entries; multiplicity/reciprocity do not increase degree |
| Homophily | Same true class / degree, all neighbor labels allowed for retrospective analysis; isolate NaN is separate |
| Other graph diagnostic | Train-neighbor fraction uses split membership, not equality with focal label |
| Slice rules | Degree0,1–2,3–5,6–10,11–20,21–50,51+; homophilyundefined,[0,.25),[.25,.5),[.5,.75),[.75,1]; classes0–39; every observed year |
| Nomination | Minimum validation mean GCN-minus-MLP gap among degree/homophily/class slices with n≥200. Ties: family,slice lexical order. Year excluded because validation and test years differ |
| Budget | One10epoch pilot and ten500epoch workers, each600s timeout, retries0. Verified rate.00020796USD/s; maximum allocated resource estimate1.372536USD; remaining8.627464reserved for overhead within10USD |

## Deviations and interpretation limits

1. The source repeatedly resets one model without specifying an original seed list. Separate seeded processes0–9 preserve distributions but do not reconstruct original random-number trajectories. Original seeds are unavailable.
2. Modern libraries and T4 hardware replace historical runtime. The GCN's native CSR backend and full original-class adapter are inherited from L112; see that ledger. The MLP source parity audit checks output, gradients, BN state and Adam updates exactly in a controlled case; it does not prove identical historical training.
3. The raw gzip reader replaces the OGB data wrapper. Archive/array identities match L112's independently verified official loader; L114 checks the same archive and all prediction/label/split arrays. Independent SQL recomputes complete diagnostic graph properties from raw edges.
4. Model initialization and BN populations differ between published GCN and MLP. Their gap compares complete baselines, not an isolated causal effect of adjacency. No architecture retuning or new feature-only ablation is smuggled into the reported paper result.
5. Test metrics are logged at every epoch as in the release, but only validation determines checkpoints. The analysis plan froze slice definitions before examining MLP test scores. The test slice census is descriptive; if used to design a better model, another protected evaluation is needed.
6. Seed SD measures variation on this fixed graph. Nodes are not independent replicates. Index-pairing seed0 across architectures is not shared initialization; delta SD has no matched-treatment or node-IID confidence interpretation.
7. Year slices are distribution comparisons, not guaranteed causality of time. Full transductive graph context is allowed, so this is not strict historical forecasting. Classes and homophily require labels; they cannot directly route unlabeled predictions.
8. Checkpoints permit inference replay, not optimizer/RNG-exact mid-training resume. The MLP runner rejects any nonempty output directory. The cloud source hash gives a separate namespace for a changed implementation; this is not a resumable run.
9. Original tuning, other models/datasets, node2vec augmentation, full OGB paper, deployment, and live Colab are not established by this lesson. Learner status remains PENDING_WRITTEN_DEFENSE.

## Exact commands

Run from the repository root, in a new environment if needed:

```bash
python3 -m venv /tmp/l114-runtime
/tmp/l114-runtime/bin/pip install -r labs/requirements-l114-runtime.txt
OMP_NUM_THREADS=1 /tmp/l114-runtime/bin/python labs/_check_l114.py
OMP_NUM_THREADS=1 /tmp/l114-runtime/bin/python labs/_source_check_l114.py
# Teaching only: two full-data epochs, seed100. Fresh directory required.
OMP_NUM_THREADS=1 /tmp/l114-runtime/bin/python labs/_run_l114.py --preset smoke --output labs/results/l114/new-smoke
# Full selected published experiment: ten500epoch fits, GPU recommended.
OMP_NUM_THREADS=1 /tmp/l114-runtime/bin/python labs/_run_l114.py --preset paper --device cuda --output labs/results/l114/new-paper
```

`closer` is an explicitly non-paper50epoch teaching preset. Never compare its scores as a completed500epoch experiment. The notebook includes the full visible MLP and GCN implementations, raw reader, trainer, and gated full schedules. The default notebook recomputes the complete slice census from author predictions; it does not train new models.

The bounded author cloud commands were:

```bash
.venv/bin/modal run --detach modal/l114_repro.py --mode pilot
.venv/bin/python labs/_collect_l114.py --partial
# Inspect pilot parity/timing; budget gate must match canonical source hash.
.venv/bin/modal run --detach modal/l114_repro.py --mode paper
.venv/bin/python labs/_collect_l114.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_replay_l114.py --model gcn
OMP_NUM_THREADS=1 .venv/bin/python labs/_replay_l114.py --model mlp
OMP_NUM_THREADS=1 .venv/bin/python labs/_analyze_l114.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_audit_l114.py
.venv/bin/python labs/_figures_l114.py
.venv/bin/python labs/_build_l114.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_execute_l114.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_mutation_l114.py
.venv/bin/python labs/_delivery_l114.py
```

Cloud source-hash namespace: `630b760511da9204c384c8c84b6d713568f4b84116eedca7f3c184a4de8047d5` in volume `l114-ogb-evidence`. Already-completed cloud jobs reject existing output. A new training attempt needs a fresh namespace and a new budget accounting decision; no automatic retries.

Raw archive and checkpoint files are excluded from Git. Original source, arrays/predictions, histories and hashes are distributed; the exact local full-training command regenerates checkpoints. With the authorized cloud account, collectors recover selected checkpoints from volumes: `_collect_l112.py` for GCN and `_collect_l114.py` for MLP. To replay without that account, retrain to fresh directories and pass those directories explicitly. The separate output records the fresh training origin:

```bash
OMP_NUM_THREADS=1 /tmp/l114-runtime/bin/python labs/_replay_l114.py --model mlp --evidence-root labs/results/l114/new-paper --checkpoint-root labs/results/l114/new-paper --output labs/results/l114/new-paper/replay.json
```

This does not overwrite the archived author replay record. The full runner also prints the mean, seed SD and frozen-target verdict after all ten runs.

## Verification and delivery

- Tiny-graph semantics, denominator/discordance and minimum-support checks.
- Actual trainer invariant: changing held-out features/labels cannot affect a one-epoch MLP optimization state; existing outputs cannot be overwritten.
- Full original-class replay of10GCN+10MLP selected states; all169343nodes each.
- Official evaluator and independent correct counts for all per-seed aggregate scores; first-maximum selection reconstructed from500epoch histories.
- Full SQLite graph oracle:169343nodes;2315598unique directed non-self edges; every degree/homophily/training-neighbor fraction equal. Independent integer-count checks for1030nonempty slice-seed cases.
- Student/solution notebook execution, mutation and real browser/copied-Pages results are recorded separately in `_execution_l114_results.json`, `_mutation_l114_results.json`, `_delivery_l114_results.json`. Live Colab and deployment remain NOT_CHECKED.

The measured pilot+training resource estimate is USD0.062947; startup/build/storage overhead is unitemized and covered by the reserve. This is not an invoice. All replay, SQL and notebook validation here ran locally and added no Modal calls.
