# L104 — temporal leakage audit and fresh TGAT evaluation

## What is being reproduced

Named published target: Xu et al., ICLR 2020, Wikipedia TGAT AP, Tables 1 and 2: **95.34% (SD 0.1 pp)** and **93.99% (SD 0.3 pp)** across ten runs. L104 reuses the complete L103-trained checkpoints and freshly replays their released evaluation. Numerical closeness uses the existing predeclared ±0.5 pp tolerance; it is not an equivalence test.

The separate strict / same-time / one-day-lookahead experiment is a **course inference intervention**, not a result published by Xu, Kapoor or Fey. Kapoor supplies the leakage taxonomy; Fey supplies the relational temporal-graph motivation. No civil-war or full RelBench experiment is claimed.

**Fresh training: NOT_RUN in L104. Historical identity: INCOMPARABLE. Full-paper parity: NOT_ESTABLISHED.** Full released Wikipedia training remains runnable with the visible model and trainer. Reddit, industrial data, node classification, historical HPO, all baselines and ablations are not reproduced here.

## Exact replay commands

From the repository root, use Python 3.12 and the pinned runtime in `labs/requirements-l104-runtime.txt`. Existing `.venv` is the locally recorded CPU environment. Set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`.

```bash
.venv/bin/python labs/_check_l104.py
.venv/bin/python labs/_causality_l104.py
.venv/bin/python labs/_resume_l104.py
.venv/bin/python labs/_witness_l104.py
.venv/bin/python labs/_freeze_l104.py
# Small timed pilot, including raw-data loading and checkpoint verification:
.venv/bin/modal run modal/l104_replay.py --pilot
# Full fresh evaluation only; trained inputs live in the existing L103 volume:
.venv/bin/modal run --detach modal/l104_replay.py
# Or one local checkpoint; full two-layer model and populations:
.venv/bin/python labs/_run_l104.py --seed 0 --device cpu --max-seconds 2850
# Download completed result artifacts and independently reconstruct their metrics:
.venv/bin/python labs/_download_l104.py
.venv/bin/python labs/_analyze_l104.py labs/evidence/l104/full
.venv/bin/python labs/_figures_l104.py
.venv/bin/python labs/_build_l104.py
.venv/bin/python labs/_execute_l104.py
.venv/bin/python labs/_delivery_l104.py
```

Before running locally, retrieve the selected checkpoint, predictions, identity and result for each seed. The recorded manifest verifies all four files before deserialization. The author's Modal volume is account-specific; users without access can regenerate checkpoints with the complete L103 recipe below. The distributed notebook includes seed 0's checksum-pinned checkpoint and question archive, so its default lab does not require that account.

```bash
.venv/bin/modal volume get l103-tgat-evidence /6a9dc0868fd047a68b3fbc547eb250898fb741a0792a8610b45c8dfdea48ae4f/paper/seed-0/selected.pt labs/results/l103/gpu/seed-0/selected.pt --force
# Repeat for predictions.npz, identity.json and result.json and seeds 0–9.
```

The L104 output path is the SHA-256 of concatenated `labs/relkit/leakage_l104.py` and `labs/_run_l104.py` bytes, followed by `/full/seed-N` in `l104-leakage-evidence`. The runner prints its exact path. Download it with:

```bash
.venv/bin/modal volume get l104-leakage-evidence /RUN_SHA/full labs/evidence/l104/ --force
```

## Full model training from raw data

The student and solution notebooks contain the full TGAT encoder, decoder, optimizer loop, data loader, released checkpoint selection and evaluation inline. Their `RUN_FRESH_TRAINING` switch defaults off. It trains ten seeds on the full graph with no layer, width, neighbor or epoch downscale. This expensive optional lane was not launched for L104.

```bash
# Complete released-protocol training recipe, independently maintained in L103:
.venv/bin/python labs/_fetch_l103.py
.venv/bin/python labs/_source_check_l103.py
.venv/bin/python labs/_verify_l103.py --preset paper --seeds 0,1,2,3,4,5,6,7,8,9
```

See [L103's full protocol/deviation ledger](l103-reproduction.md). New training produces new artifacts and requires an independently declared input manifest; `_freeze_l104.py` deliberately refuses to overwrite already-pinned evidence. Use this isolated path to audit a new training run without replacing the author artifacts:

```bash
.venv/bin/python labs/_verify_l103.py --preset paper --seeds 0,1,2,3,4,5,6,7,8,9 --output /tmp/tgat-retrained
.venv/bin/python labs/_prepare_l104_rerun.py --checkpoint-root /tmp/tgat-retrained/paper --data-dir labs/l103-cache --workspace /tmp/l104-independent
.venv/bin/python /tmp/l104-independent/_run_l104.py --checkpoint-root /tmp/l104-independent/checkpoints --data-dir labs/l103-cache --output /tmp/l104-independent/results --seed 0 --device cpu --max-seconds 7200
# Repeat the last command for seeds 1–9; choose GPU if available.
```

The preparation command validates full-data protocol metadata, copies the visible audit/model code, and hashes the new checkpoint set. It links to the original trained files, which must remain available. The original immutable author manifest stays intact. This optional fresh-training lane was not run for L104 and must be budgeted separately. Do not present changed-runtime retraining as the same archived checkpoint replay.

## Protocol audit

| Dimension | L104 behavior and boundary |
|---|---|
| Model | Complete two-layer TGAT, two heads, width 172, 20 neighbors; frozen L103-trained weights |
| Source | L103 independent reconstruction of released commit `9293d10d1943c4bd4a186337cf38ba98e4c8bb99`; unchanged implementation SHA `6a9dc0868fd047a68b3fbc547eb250898fb741a0792a8610b45c8dfdea48ae4f` |
| Provenance | `_inputs_l104.json` pins each complete checkpoint, prediction archive, identity and result; raw-file SHA, processed-array byte hashes and split hashes checked |
| Data | Full Wikipedia: 157,474 timestamped events, 9,227 nodes, 172 edge features, zero raw node vectors; ingestion/feature-version timestamps unavailable |
| Historical training | Reused L103: Adam 1e-4, batch 200, dropout .1, up to 50 epochs, release patience 3 / relative improvement .001, fixed split and seeds 0–9 |
| Split | Original L103 70%/85% time quantiles and held-node identities; preserve chronological data and published-replay populations |
| Release evaluation | Exact pre-evaluation NumPy state, full-graph history with release sampler, batch 30, one random negative, full all/new populations |
| Replay gate | Exact event IDs and negatives; probability tolerance rtol 1e-5 / atol 2e-6; AP independently rebuilt from archived scores |
| Release quirks | Omitted eligible neighbor; omitted last event; finite-mask padding; checkpoint index behavior; nonliteral old/new populations; unweighted batch-mean AP — all retained only in replay |
| Release counts | 23,620 all-test positives and 11,714 new-node-test positives, each with one sampled negative; new-node population overlaps all-test |
| Intervention baseline | Corrected strict prefix; one uniform matrix per sampler call, drawn even for empty histories; same frozen weights and question IDs |
| Same-time arm | Event time ≤ request cutoff, violating this before-event task's tie policy |
| Lookahead arm | Event time ≤ request cutoff + 86,400 seconds at every recursive hop, potentially negative elapsed times; not a single root horizon |
| Candidate roster | Inherited released negative pools, including full-graph IDs; a fixed offline candidate universe, not proof of production candidate availability |
| Random controls | Seed 104 + checkpoint seed, reset for each arm/population; identical draw shapes even for empty histories; neighborhood identities change by design |
| Availability | Assumed equal to event time in Wikipedia; synthetic fixtures alone test delayed reporting and target maturity |
| Metrics | Release: mean of per-batch AP. Intervention: pooled AP, paired by events/negative IDs/batches; never compare these two aggregations as a leakage effect |
| Audit counts | Sampled nonpast/future records relative to each local cutoff, including repeated samples and recursive requests; not unique rows |
| Selection | No new tuning or selection by test performance. All predeclared seeds included when complete; incomplete upstream seeds explicitly pending |
| Uncertainty | Sample SD of paired changes across seeds on one fixed graph/split; no cross-dataset or deployment claim |
| Runtime | GPU Python 3.12, torch 2.8.0, numpy 2.2.6, pandas 2.3.2, scikit-learn 1.7.1; modern replay, not historical environment |
| Resume | Only complete seeds; input/code/runtime identity and prediction-file checksum must agree. Interrupted evaluation reruns from checkpoint; no mid-evaluation resume claim |

## Budget and actual execution

Budget: **USD10 total**, including checks and retries. See `_budget_l104.json` for current pricing, pilot and conservative resource ceilings. Pilot took 120.73 seconds, dominated by data setup; it is excluded from results. Full evaluation projected about USD3.33 in resource time. Per-call wall-clock timeout: 3000 seconds; inner deadline: 2850 seconds. No automatic retries. At most two explicitly budgeted retries; do not repeat full batches blindly.

Initial L104 launch had nine complete checkpoints; upstream L103 seed 8 was still training. Completing L104 does not authorize modifying or restarting that separate L103 run. Fresh L104 evidence coverage and any missing seeds are in `_analysis_l104_results.json`.

**Completed author result:** all ten checkpoints were evaluated after upstream seed 8 completed. Released batch-mean AP was 95.2825% ± 0.2784 pp sample SD (all) and 93.7792% ± 0.3912 pp (new); mean targets are numerically CLOSE under the declared tolerance. All 353,340 positive questions and their paired negatives replayed with zero probability difference. Independent reconstruction checked all 40 paired AP comparisons.

The strict course baseline gave pooled AP 95.8643% (all) and 94.8299% (new). Inclusive access changed AP by +0.6431/+1.2065 pp; per-hop one-day lookahead by +1.0928/+1.8212 pp. See `_analysis_l104_results.json` for per-seed values and SDs. These are fixed-weight inference interventions, not retraining experiments.

Completed call time including the pilot implies about USD1.24 at the conservative configured resource rate; this is not an invoice and excludes startup/idle/build overhead. No L104 training or paid retries were launched.

Results are accepted only after complete inference and independent score reconstruction. A pilot or an incomplete seed is never promoted to full-population evidence. The observed successful-call time estimate is not an invoice; startup/build/retry overhead is recorded separately.

## Delivery and learning status

`_execution_l104_results.json` records notebook execution. `_delivery_l104_results.json` records browser, copied Pages and build checks. These do not establish live Colab or publication. Both remain NOT_CHECKED unless separately tested. Student mastery remains PENDING_WRITTEN_DEFENSE.
