# B04a reproduction contract

Approved target **B04A-TABFLEX-FIG9-ATTENTION-SCALING**, TabFlex v1 Appendix C.1 Figure9. Status **INCOMPLETE_SOURCE_PROTOCOL**, zero paper benchmark runs, zero cloud/API spend. Complete course diagnostic: **108 configurations / 6,912 predictions**, not pretrained TabFlex inference. Source operator parity and class-preprocessing checks are separate mechanism evidence. Learner **PENDING_WRITTEN_DEFENSE**.

## Published target and blocked dispatch

Three operators: FlashAttention-2, causal FlashLinearAttention and the noncausal linear core. Widths32/64/128/256; heads2/4/8/16; lengths2^4…2^15; batch10; five repetitions. Full grid2,880 operator repetitions;120 slots correspond to reported FLA failures for width256 with8/16heads. No reduction or substitute sweep is labeled this experiment. Listing1 times Q(KᵀV), distinct from full ELU+1/normalization in the model.

Repository revision `3c36d9c1785844d5e983b0baf0ef4116670aa809`. Source manifest authenticates paper HTML/text, original Git tree, compact executable source/config/license archive, and selected readable modules. Full upstream archive identity is retained; the 41MB original archive is available locally but excluded from the portable/publication packet. The compact archive contains all114 Python/config/license/readme files selected by the preparation script; research notebooks and other ancillary assets are excluded. The bounded audit also searched original notebook cell sources for `flash_attn`, `fla.ops`, `do_bench` and `illegal memory access`, with no hits. Compact archive search can be independently replayed; original full-tree scan is recorded as author evidence.

Unresolved: original input distribution/seeds/QKV sharing, exact benchmark hardware, precision and kernel versions, timing warmups/synchronization/forward-backward scope, peak-memory reset/aggregation, authenticated original dispatch script and raw numeric reference. Training A100 hardware is not benchmark-hardware identity. Current availability is not historical identity.

```bash
.venv/bin/python labs/_reproduce_b04a.py
.venv/bin/python labs/_reproduce_b04a.py --run
```

The first command verifies source hashes and reconstructs the gate. The second refuses dispatch. This is an executable preflight, **not a complete Figure9 runner**. Unblocking requires authenticated missing inputs, original benchmark implementation, complete-run forecast at verified current rates, and explicit deviation/tolerance accounting. Do not change a status string to bypass it. Full pretraining, whole-paper benchmarks, pretrained TabFlex inference, Wide training, BETA and current-checkpoint comparisons remain NOT_RUN.

## Frozen course protocol

`protocol.json` precedes execution. Three NumPy default_rng seeds0/1/2 generate1088rows:8standard-normal latent columns,248independent noise columns, and binary target from x0+.7x1−.4x2+.25normal_noise>0. Original rows0…1023 form nested support prefixes64/256/1024; rows1024…1087 are the fixed64queries. All original arrays/IDs/labels are frozen in `inputs.json` with SHA256. No validation, test-driven selection or model training.

For F8/64/256, either append noise or repeat latent columns in order. Support-fitted mean and population std (constant scale1), fixed Gaussian projection F×16 from default_rng(4000+F)/sqrt(F), then either scaled softmax(qkᵀ/4) or normalized ELU+1 readout with ε1e−6. Values are binary one-hot support labels. Linear outputs are renormalized across classes, cancelling the shared ε denominator from that final distribution. Every configuration uses one view, CPUfloat64, one BLAS thread. No checkpoint, learned encoder or feed-forward transformer blocks. Changing F changes projection and geometry as well as width; no pure architecture attribution follows.

Fresh process per setting; first complete preprocessing/projection/readout is cold pipeline time, followed by3warm full-pipeline repeats. Warm median is a within-process statistic. Materialization and process elapsed are stored separately. Process elapsed excludes process spawning and result serialization; parent-run ledger includes them. Peak Linux RSS includes imports, frozen JSON parsing and arrays. No OS-cache flushing, CUDA measurements or equivalent-hardware paper runtime claim. Minimum/maximum seed envelopes are descriptive, not confidence intervals. Runs occur in deterministic order; no randomized timing design or statistical speedup claim.

An independent explicit query×support kernel reconstructs all6,912 predictions at atol1e−12/rtol1e−11, then scalar scoring authenticates exact IDs/classes and accuracy/log loss (probability clip1e−15). The F8 noise/copy controls match exactly. Ten deliberate record corruptions are rejected. Performance metadata is structurally checked; raw timing itself cannot be reconstructed from predictions.

## Mechanism and class checks

`_mechanism_b04a.py` extracts and executes unmodified class/method ASTs from the authenticated compact source, avoiding unrelated imports or checkpoint downloads. Released `LinearAttention` agrees with the course readout on random q/k/v for supports1/7/64/1024 in CPUfloat64 within1e−12. This checks the operator only, not learned weights or gradient/training parity.

Released `TabPFNClassifier.preprocess` with max_num_classes=10 retains2/10classes but maps11original classes into10effective labels. Original and transformed labels are saved. This is a configured-method boundary, not the capacity of every release checkpoint. Course `class_values` rejects the overflow and preserves arbitrary explicit column order. No prediction quality is inferred from this preprocessing test.

## Fresh execution, portability and budget

```bash
.venv/bin/python labs/_budget_b04a.py .venv/bin/python labs/_test_b04a.py
.venv/bin/python labs/_budget_b04a.py .venv/bin/python labs/_verify_b04a.py
.venv/bin/python labs/_budget_b04a.py .venv/bin/python labs/_run_b04a.py
```

The runner authenticates source/input/protocol byte hashes and refuses changed-code resumes. Existing results are immutable; the third command skips completed runs. For a fresh run, extract `reproducer.zip` into a new directory, keep the frozen inputs/protocol/source, and remove `labs/evidence/b04a/runs` **in that new copy only**. Use a Python/NumPy runtime matching `environment.json`; install NumPy if needed. Then run `_budget_b04a.py python _run_b04a.py` with the corresponding paths, and `_audit_b04a.py` to write a new report. Original author timing receipts and hashes are not claims about the new environment. Mechanism parity additionally needs the recorded PyTorch/scikit-learn versions; pure replay and student mechanisms need only NumPy.

Offline notebooks embed the source/evidence packet, show the actual course implementation and complete relevant model modules, and execute from an empty directory. Saved-prediction replay is not new inference. Student TODOs must drive the operator/preprocessing/class exercises. Browser, clean-index publication build, liveColab and deployment are separate statuses.

Approved capUSD10 includes setup, retries and verification;USD2reserve and no paid reservations aboveUSD8. No paid dispatch was admitted because the source gate remains closed. Local numerical ceiling3600seconds; complete course matrix reservation900seconds, actual150.772seconds including child processes. The cumulative local ledger includes failed tests, preparation, checks and a conservative60second allowance for the separately executed source mechanism check. Unfinished work is reported explicitly; no paid rate estimate is represented as verified because no paid run was planned after the gate failed.
