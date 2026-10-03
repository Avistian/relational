# B06 reproduction contract

Approved 2026-10-03. Two distinct experiments; learner PENDING_WRITTEN_DEFENSE.

## B06-PRIOR-MIXTURE — complete course mechanism

Frozen configuration: [course-protocol.json](evidence/b06/course-protocol.json). Three arms (SCM, tree, equal outer mixture), seeds0/1/2,9fresh fits. Two 2D-attention blocks, width32,4heads, no dropout,4features,24support+8queries, binary task,160AdamW updates at0.001/weight_decay0.01 with4tasks per update.640episodes per fit; mixed arm exactly320of each family using shuffled stratified uniforms. Ordinary categorical sampling would fluctuate; this deliberate variance-control deviation is course-only. Final checkpoint; no early stopping or selection.

Seed-matched initial tensors; identical budgets and60evaluation tasks across all arms.20tasks each from simplified SCM/tree/hybrid generators,480query identities per fit,4320predictions total. Synthetic TierC data isolate this local mechanism. Train tasks use10000+10000*seed+episode; evaluation uses600000+1000*family+task. Disjoint seeds do not prove distributional independence; generators intentionally share code. No real datasets selected the mixture, which was fixed at0.5before results.

SCM: triangular noisy feature equations and nonlinear target. Tree: depth-two random split function. Hybrid evaluation: nonlinear plus tree target inside one task. These are transparent course proxies, not original Mitra or Mitra-v2 Hybrid SCM. Threshold labels at the support-score median. Fit standardization to support only. Model accepts only support labels and forbids query-to-query communication. Query labels enter training loss and scoring only.

Accuracy and cross-entropy: equal queries per task, equal tasks per family; per-seed results and descriptive sample SD. Paired mixed-minus-baseline effects match seeds and query identities. No Friedman/Nemenyi claim across three synthetic families sharing generators. Uniform-probability CE=ln2 is an analytical baseline. All three arms are near chance; no improvement or real-world generality is established.

From repository root:
```
.venv/bin/python labs/_budget_b06.py .venv/bin/python labs/_verify_b06.py
.venv/bin/python labs/_budget_b06.py .venv/bin/python labs/_run_b06.py --output /tmp/b06-fresh-runs
```
The fresh command refuses to overwrite existing run files. Source and protocol hashes, initialization, schedules, complete loss traces, all predictions and final checkpoint files are saved. Frozen author evidence is independently scored by `_audit_b06.py`. Standalone `evidence/b06/reproducer.zip` contains code, configuration, inputs, weights and source snapshots. Use `python labs/_verify_b06.py` from an extracted directory. The notebook defaults to saved-checkpoint inference plus live learner-code checks; it does not claim a new nine-fit experiment. Optional full fresh training is explicitly gated.

## B06-MITRA-TABLE12 — INCOMPLETE_SOURCE_PROTOCOL

Primary target: [Mitra v1 Table12](https://arxiv.org/html/2510.21204v1#A3.T12). All six SCM probabilities0,.4,.5,.6,.7,1; original TabRepo10fold classification, original tree-prior submixture. Paper rows report accuracy .808/.817/.822/.822/.818/.812 respectively and CE .451/.406/.403/.404/.406/.416. These are cited printed targets, not recomputed scores. Full ranking reproduction also needs every original comparator and the ranking evaluator, not just these six rows.

Authenticated release inventory and original config: [source manifest](sources/b06/manifest.json). Current classifier and regressor repositories list one weight file each. The current fine-tuning repository explicitly targets v2. This does not establish absence everywhere, but the inspected releases do not authenticate the six original checkpoints, complete original pretraining generators/seeds, exact dataset/fold identities or Table12 evaluator/prediction artifacts. No checkpoint weights were downloaded for this paper lane.

```
.venv/bin/python labs/_reproduce_b06.py
.venv/bin/python labs/_reproduce_b06.py --run
```
Preflight authenticates the evidence and lists gaps; `--run` deliberately exits nonzero without dispatch. This is **a runnable source gate, not a complete paper reproduction implementation**. Full original model/trainer parity cannot be claimed. Course code is fully visible and runnable; making it wider cannot fix missing original provenance.

## Version and evidence boundaries

Original Mitra:12layers,width512,4heads,72Mparameters; support quantile transformation then standardization; paper reports45million datasets and60hours on8A100GPUs for its main pretraining run. The course learner differs in scale, preprocessing, generators, task shapes, training duration and architecture details; no copied-weight source parity claimed. Released config matches dimensions but does not prove historical trajectory identity.

[Mitra-v2](https://arxiv.org/html/2609.04540v1#S2): adds within-task Hybrid SCM, broader task shapes and changed optimization; outer mixture35%baseSCM,35%HybridSCM,6%each of five tree families. The v2 fine-tuning recipe is a different downstream adaptation lane. Its performance cannot isolate the prior change because other factors changed. Full paper pretraining/full benchmark NOT_RUN; paper parity NOT_ESTABLISHED; live Colab/deployment NOT_CHECKED.

## Budget

USD10 aggregate ceiling; USD8 commitment stop and USD2 reserve. USD0 cloud/API planned and spent.3600s aggregate local numerical guard includes failing tests, all training, replay and verification. Network-only source retrieval is not numerical execution. No paid job before current-price complete-run forecast. Exhaustion stops INCOMPLETE without shrinking the frozen matrix.
