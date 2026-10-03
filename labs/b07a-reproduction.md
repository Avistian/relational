# B07a reproduction contract

Approved2026-10-03. Exact paper target and course experiment are separate.

## Course: B07a-HYPERFAST-AMORTIZATION

Three full numeric datasets: banknote UCI(1372×4), phoneme OpenML1489(5404×5), diabetes OpenML37(768×8). Hashes and row identities are in `evidence/b07a/course-protocol.json`. Fresh stratified80/20 splits and model seeds0/1/2. Fit StandardScaler on all training rows. Sample512 train rows without replacement with torch.randperm; repeat each twice to1024, matching the release's minimum-PCA-row handling. No new evidence is created by repetition.

Full released checkpoint/model dimensions: RF32768, PCA784, hyper hidden1024, main3 layers,46 label slots. Single generated predictor, no fine-tuning/HPO. Compare NN corrections off/on using identical generated state;18 prediction arms from9 constructions. A tenth construction adds the first held-out labeled banknote row at seed0 and measures refresh only; never score that now-observed row as held-out evidence.

Query batches1/32/128, one warm-up and3 timed repeats per seed, alternating arm order; single CPU thread. Include preprocessing and release-style support recomputation in prediction time. Construction includes train preprocessing/RF/PCA/weight generation; checkpoint authentication/load is reported separately. Array bytes distinguish minimum pure-state storage from actual adapter storage retaining support for both paths; peak process RAM not measured. Primary quality is balanced accuracy; log loss also archived. Sample SD across3 splits is not a confidence interval. No model-family ranking or broad statistical significance claim.

Current release: d1f1c3b0c45572dc09173733b194c0e7384cd696. Publication-era:9a25ed34edf1d9896efde32500707fb8689e1005. Both original source trees are archived under `sources/b07a`, CC BY-NC4.0. Full source/optional fine-tuner appears in notebook; pretraining data generator/trainer is not available in the inspected release and is not reconstructed here.

Canonical adaptation delegates only output-head pooling to the live learner function. The numeric-only serving adapter implements the selected release path. Tests compare full random-weight generation at small dimensions and unmodified release predictions on identical real checkpoint-generated states. Meta allocation/memory-mapped checkpoint loading changes allocation only, preserving float32 parameters and strict state-key checking. Saved main hidden vectors/output weights allow an independent output reconstruction and two-space nearest-neighbor audit.

## Original paper: Table7 banknote

2402.14335v1 Table7 reports100.0±0.0 percent balanced accuracy,10 mini-test repetitions at300seconds each. Mini-test limits1000 train rows/100features; full banknote split1097/275. Original split/subsample IDs, seed states, selected five-minute inference/search configurations and evaluator outputs are absent in the inspected repository. Shared checkpoint URL does not prove historical byte identity. Original target:INCOMPLETE_SOURCE_PROTOCOL. Course is not a substitute. Full meta-training/all-paper benchmarks/MotherNet/iLTM runs:NOT_RUN.

The original target CLI is an authenticated preflight that refuses `--run`; it is not a complete runnable historical evaluator. No cloud operator is supplied for an unresolved scientific protocol. Missing original inputs must be resolved before dispatch.

## Commands

From repository root:

```
.venv/bin/python labs/_test_b07a.py
.venv/bin/python labs/_reproduce_b07a.py
.venv/bin/python labs/_reproduce_b07a.py --run
.venv/bin/python labs/_budget_b07a.py .venv/bin/python labs/_run_b07a.py --output /tmp/b07a-fresh
.venv/bin/python labs/_budget_b07a.py .venv/bin/python labs/_audit_b07a.py
```

`--run` intentionally exits nonzero at original source gate. Course fresh execution needs authenticated checkpoint `labs/data/b07a/hyperfast.ckpt` (or runner `--checkpoint`). The notebook downloads from official ndownloader.figshare.com/files/43484094, verifies length/SHA256 before use, and supports `HYPERFAST_CHECKPOINT` to reuse a local file. Portable archive excludes5.09GB checkpoint and includes complete code/data/saved evidence; default student lane audits evidence, gated fresh lane creates predictors. Repeating identical seeds measures repeatability, not additional independent observations. Live Colab remains NOT_CHECKED.

## Cost and completion

USD10 aggregate including all attempts/validation; USD8 execution allowance plusUSD2 reserve. Local numerical subprocesses stop after3600aggregate seconds using `_budget_b07a.py`; failures count. Paid runtime requires current-rate complete-run forecast before dispatch. Downloads are local/network, no paid API. The Figshare web URL initially returned an empty202challenge; rejected in favor of official streaming endpoint. Local setup/source retrieval is reported separately from numerical budget. No checkpoint is shipped in Pages.

Author delivery is separate from learner work: three TODOs plus written defense; PENDING_WRITTEN_DEFENSE. No deployment or learner completion inferred.
