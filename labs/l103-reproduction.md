# L103 — TGAT Wikipedia release replay and matched course comparison

## Named target and scope

Xu et al., ICLR 2020, arXiv:2002.07962v1, Tables 1/2: Wikipedia TGAT AP **95.34 (SD 0.1)%** transductive and **93.99 (SD 0.3)%** inductive, ten runs. Numerical CLOSE requires an absolute mean gap <= **0.5 percentage point**, declared before training. This is a course tolerance, not an equivalence test. The released experiment has documented differences from the literal paper populations; a close number cannot establish protocol parity. Full-paper parity **NOT_ESTABLISHED**. Reddit, industrial, node classification, baselines and ablations are outside this approved slice.

## Reproduction commands

```bash
OMP_NUM_THREADS=1 .venv/bin/python labs/_fetch_l103.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_check_l103.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_source_check_l103.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_audit_l103.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_selection_check_l103.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_padding_check_l103.py
OMP_NUM_THREADS=1 .venv/bin/python labs/_verify_l103.py --preset smoke
OMP_NUM_THREADS=1 .venv/bin/python labs/_verify_l103.py --preset paper --seeds 0,1,2,3,4,5,6,7,8,9
# Same canonical implementation, independently pinned GPU runtime:
.venv/bin/modal run --detach modal/l103_paper_repro.py::main --preset paper
# Separate matched course comparison, never compared against the paper table:
OMP_NUM_THREADS=1 .venv/bin/python labs/_compare_l103.py
```

The full CPU replay can be slow. GPU results persist under implementation SHA / preset / seed in `l103-tgat-evidence`. Record the exact SHA emitted by the run. After all ten runs and the source replay finish, download final artifacts with `.venv/bin/python labs/_download_l103.py`, then run `.venv/bin/python labs/_collect_l103.py labs/results/l103/gpu`. To also retrieve every intermediate epoch checkpoint, use `modal volume get l103-tgat-evidence /SHA/paper labs/results/l103/all-epochs --force`. Completed seeds may be reused only with identical implementation/data/runtime identity. Interrupted seeds restart; this runner does **not** claim mid-epoch resume. `selected.pt` contains the pre-evaluation NumPy RNG state for original-source prediction checks.

Source model replay after seed 0 completes:

```bash
.venv/bin/modal run modal/l103_paper_repro.py::source_check
```

## Protocol and deviation ledger

| Dimension | Released replay |
|---|---|
| Source | Publication-era `9293d10d1943c4bd4a186337cf38ba98e4c8bb99`, original authors' repository |
| License | No license file supplied. Fetch original files locally for parity checks; do not distribute them in the lesson package. SHA-256 manifest is distributed |
| Paper data | Full SNAP/JODIE Wikipedia, 157474 interactions, 9227 nodes; SHA-256 `a6b73e09c0d1e5b9db11e7e7aa416f2e87838a745273e0446a79952cf4cfae09` |
| Features | 172-dimensional edge features; all-zero node features; disjoint user/page IDs; zero reserved for padding |
| Split | 70% and 85% timestamp quantiles; Python seed 2020 holds out 922 post-cutoff nodes from training |
| Python compatibility | `random.sample(set, k)` was removed: explicit tuple conversion preserves this machine's set iteration; historical set order is unverified |
| Architecture | Two recursive temporal attention layers, two heads; each query/key width 516, each head 258; output node width 172 |
| Dimension flags | Original `node_dim`/`time_dim` CLI values are inert in this variant; actual widths come from the 172-dimensional features |
| Time | Learned cosine frequencies/phases, 172 dimensions, frequencies initialized geometrically from 1 to 1e-9; differs from paired sine/cosine Eq. 5 |
| Time precision | All released Wikipedia timestamps are integers exactly representable in float32 (maximum 2678373); child-time conversion introduces no rounding on this dataset |
| Sampling | Uniform with replacement, 20 historical interactions per query per layer; recursively use each interaction time for the child's cutoff |
| Neighbor bug | Original binary search returns a prefix omitting one eligible interaction; sorted by edge ID; preserved in replay |
| Residual/merge | Release projects concatenated head outputs, adds the root query, applies LayerNorm, then merges with the root previous-layer representation. The paper §3.2 FFN writes raw root features; the diagram explicitly teaches the released variant |
| Padding | Finite -1e10 logits; an all-padding row gets uniform softmax and can retain time-dependent padding contributions; breaks full-model clock-shift invariance on an empty-history fixture; preserved |
| Initialization | Same constructor order including unused merge module; exact source check on current runtimes |
| Decoder/objective | Shared pair MLP, sigmoid; mean positive BCE + mean negative BCE; one destination negative per positive |
| Negative pools | Train: training endpoints. Validation/all-test: full graph endpoints. New-test: new-test subset endpoints. Collisions are allowed. Global NumPy RNG, including unused sampled sources |
| Temporal order | Chronological minibatches; release shuffles an index array but never uses it. RNG consumption is preserved |
| Batch bug | End is `min(n-1,start+batch_size)`; final event omitted for training and each evaluation branch; preserved. Empty-tail crash guard affects only other sizes |
| Hyperparameters | Batch 200 train, 30 evaluation, Adam lr 1e-4, dropout .1, maximum 50 epochs, patience 3, relative improvement .001 |
| Architecture selection | Fixed released recommended configuration; the paper's layer/head/neighborhood-dropout search is not rerun. The 20-neighbor release setting is recorded explicitly rather than equated with a sampled-dropout fraction |
| Selection bug | Early-stop monitor stores an improved epoch's index + 1; replay restores that later checkpoint. First best remains index 0. If max epochs reached, final model is used |
| Metrics | Unweighted mean of per-batch AP/AUC/accuracy. A short last batch gets equal weight; not pooled AP |
| Populations | Release “old nodes” is actually all test events. “New” means at least one endpoint unseen in training; negatives need not have unseen endpoints. These are not the literal disjoint paper population descriptions |
| Release executable defect | `args.new_node` is referenced without a declared argument; inert variable omitted in our port. The exact unmodified training entrypoint cannot start |
| Shape compatibility | Use `squeeze(1)` in attention to support batch size 1; released `squeeze()` drops the batch axis. No effect on actual full-data batch sizes |
| Seeds | Independent explicit model/NumPy seeds 0–9, not recoverable historical RNG states. Split fixed across runs |
| Environment | CPU Python 3.12 / torch 2.13; GPU Python 3.12 / torch 2.8, NumPy 2.2.6, pandas 2.3.2, sklearn 1.7.1. Historical torch 1.1 / numpy 1.16.4 not reconstructed |
| Model optimization | No AMP, TF32 enablement, or architectural downscale in the full replay |
| Uncertainty | Sample SD across initializations on one split, not ten independent datasets |

## Matched course comparison

Fresh TGAT/TGN fits, three seeds, 2000 train + 400 validation + 400 test positives, chronological, one layer, latest ten strict-past neighbors, batch 200, Adam 1e-4, dropout .1, three epochs. Both arms receive exactly the same retained event graph and per-seed precomputed destination negatives. Best validation **pooled AP** selects each arm independently. Test uses pooled AP/AUC and fixed test negatives. TGN restores its selected validation memory, clocks and pending messages, not only weights. TGAT has no persistent node memory. Widths/parameter counts differ; this does not isolate “memory alone.” It is a bounded implementation comparison and cannot establish general superiority.

This corrected comparison uses all retained events and the strict lower-bound sampler. It is distinct from the original-release lane. L102's published ten-run scores are context only, never treated as a matched control or fresh L103 training.

## Evidence required before a completion claim

Source initializer/output/gradient parity; independent preprocessing/split audit; temporal boundary counterexamples; complete per-seed traces/checkpoints/predictions; independent metric reconstruction; original-model prediction replay; standalone notebook execution; browser/portable-figure and copied-Pages checks. Live Colab and deployment remain NOT_CHECKED unless directly verified. Learner status remains PENDING_WRITTEN_DEFENSE.

## Authoring and delivery

```bash
.venv/bin/python labs/_plot_l103_results.py
.venv/bin/python labs/_figures_l103.py
.venv/bin/python labs/_build_l103.py
.venv/bin/python labs/_execute_l103.py
.venv/bin/python labs/_comparison_check_l103.py
.venv/bin/python labs/_delivery_l103.py
.venv/bin/python labs/_record_l103_evidence.py
```

Author tooling is listed separately in `labs/requirements-l103-authoring.txt`. It is not required by the standalone student notebook. The builder retains executed outputs only when every ordered code cell is unchanged. The execution report fingerprints those code cells; prose/figure regeneration does not assert a new training run. The author full-run selection is fixed in `_run_selection_l103.json`. Compact predictions and result traces are distributed in `labs/evidence/l103/`; large trained checkpoints remain on the named Modal volume and have recorded hashes.
