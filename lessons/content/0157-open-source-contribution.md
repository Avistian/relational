## 1 · Make your result usable by someone who was not there

**Reading route.** Choose a bounded claim → separate replay from training → inspect manifests → verify the package → prepare public review.

**Your win:** turn an experiment into a reviewable contribution whose commands, evidence and claims agree. Core lesson: about 25 minutes. Lab: one focused session; fresh training is a separate optional run.

[Lesson 156](0156-temporal-leakage-audit.html) traced information through a full F1 pipeline. Its remaining gap is practical: another researcher should not need our workspace, caches or conversation to check the result. This lesson closes that gap with a standalone package. It serves the mission by making a concrete RDL result inspectable by a skeptical reader.

A **reproduction** repeats a specified computation. An **evidence replay** checks previously saved inputs and outputs. A **contribution** gives other people something they can use and review. A public URL establishes access; it does not establish correctness, acceptance by maintainers or learner mastery.

[[WARMUP]]

Start with the [RelBench contribution guide](https://github.com/stanford-star/relbench/blob/main/CONTRIBUTING.md). Its current tests use small synthetic fixtures, rather than full training loops. Our archived experiment belongs in a reproducibility package; a proposed upstream change needs a focused regression test against current upstream. The [pinned reference implementation](https://github.com/stanford-star/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/examples/gnn_node.py) is the scientific baseline, not an assertion about today's code.

## 2 · Pick a contribution that follows from the evidence

There are two reasonable routes. A **reproducibility repository** gives readers a complete experiment, environment and report. An **upstream PR** changes an existing project and must fit its current interface, policies and tests. We choose the repository route because the evidence establishes a historical implementation and a declared policy sensitivity. It does not yet establish a defect in current upstream.

Our named experiment is **L157 — RelBench F1 temporal-audit reproducibility release**. The released baseline fits dated processors through 2010. Our intervention fits dated type discovery, vocabularies and statistics through 2005, then transforms all permitted graph rows. Sampling still uses each query's cutoff. Static versions and schedule publication histories remain unknown.

An accurate contribution title is: **“Reproduce F1 preprocessing policies with explicit fit-horizon audits.”** An inaccurate title is: “Prove RelBench leaks and fix its benchmark.” The stronger title requires evidence that we do not have. Likewise, neither a lower nor a higher corrected score decides which information policy matches a deployment.

[[PREDICT]]

## 3 · Separate the two commands and the two manifests

[[FIG:flow]]

`cli.py replay` is cheap: it verifies saved evidence, reconstructs all archived labels, checks complete query keys and epochs, rescores predictions and applies claim gates. It does **not** rerun feature SQL or train a model. `cli.py prepare` downloads the exact inputs and independently reconstructs the SQL features. `cli.py train` builds the actual graph, fits a complete seed and saves its selected checkpoint. Ten seed commands reproduce the two full lanes.

The **experiment manifest** freezes the executable source and protocol before dispatch. Each completed run records its digest. The **release manifest** covers the final deliverable, including documentation and saved evidence. Separating them allows us to add measured results without pretending those results existed before training. Neither manifest authenticates the author's identity: an attacker who replaces both bytes and hashes can forge consistency.

Worked trace: `paper/seed-0/predictions.npz` has a frozen digest. Integrity checking detects a changed byte. The evaluator then checks every `(driver_id, cutoff)` key, aligns targets, and independently computes mean absolute error. Finally it checks that all five seeds exist. Only after these steps can a summary support the **selected experiment** claim. A SHA256 match alone cannot replace the latter checks.

[[CODE:verify_manifest]]

**TODO1:** implement this integrity function in the notebook. **CHECK:** change a byte, remove a file, and substitute a symlink. All must fail. **Recall:** why can an authentic-looking hash still fail to prove the scientist's conclusion?

## 4 · A complete recipe must survive leaving this checkout

The [downloadable package](../labs/releases/l157-f1-audit.zip) contains visible graph/model/trainer code, independent audits, original licensed reference files, exact primary dependency versions, a pinned GPU image recipe, download hashes, the full protocol and compact evidence. Its [README](../labs/releases/l157-f1-audit/README.md) contains complete commands.

The fixed environment is Python 3.11, Torch 2.5.1/CUDA12.4, Frame 0.2.3, PyG 2.6.1 and pyg-lib 0.4.0, plus the pinned requirements. This specifies primary versions and records actual installed versions; it is not a hash lock of every transitive dependency or a guarantee that upstream downloads stay online.

Code and data rights are separate. Preserve the [RelBench MIT notice](https://github.com/stanford-star/relbench/blob/9aa346267c2e1c560bd92da07d6f4ad1ca2f0639/LICENSE). Our release does not bundle raw archives or pretrained weights. The pinned user-study repository had no root license at the inspected location, so the SQL is fetched from its original URL and hash-checked rather than relicensed inside this archive. Read the [third-party notices](../labs/releases/l157-f1-audit/THIRD_PARTY_NOTICES.md) before redistributing additional inputs.

The full recipe uses every 7,453 training, 499 validation and 760 test query; five seeds per policy; ten epochs per seed; two 128-channel sum-GraphSAGE layers; uniform 128/64 sampling; Adam 0.005; batch 512; L1; first strict minimum validation MAE; and training-target percentile clipping. [Protocol](../labs/releases/l157-f1-audit/protocol.json). A smaller pilot or a subset of seeds cannot stand in for that recipe. Full model, sampler, trainer and correction are visible in the notebook's optional reproduction section.

## 5 · Count the evidence before summarizing it

[[CODE:summarize_runs]]

**TODO2:** produce a complete-seed summary. **CHECK:** delete the last intervention seed. The result must say INCOMPLETE and omit aggregate scores. Duplicate seed0 does not replace missing seed4. A ten-epoch trace must also show full training populations at every epoch; the independent evaluator checks these details before this summary receives a record.

[[TABLE]]

[[FIG:results]]

[[RESULT]]

These are new L157 runs from the standalone package, separate from L156's earlier measurements. Seed SD describes variability across these five fits, not uncertainty over all possible relational databases. Integer seed matching does not ensure identical initialized weights when preprocessing changes encoder dimensions. The policy difference is descriptive; it does not establish superiority or equivalence.

[[COVERAGE]]

The released lane's ±0.20 descriptive comparison is with the [RelBench paper](https://arxiv.org/html/2407.20060v1) Table 7 basic GNN target, validation 3.193 / test 4.022. The fixed-horizon intervention is not that paper experiment. Historical archive identity and actual feature availability remain NOT_ESTABLISHED. The source numeric encoder's nonfinite first-backward entries are retained and disclosed, not silently repaired.

[[BUDGET]]

## 6 · Review the claim as carefully as the code

[[WIDGET]]

[[CODE:review_claim]]

**TODO3:** gate the requested statement using its required evidence. **CHECK:** ten verified fits can support selected reproduction while historical availability remains unknown and the public contribution stays PENDING_PUBLICATION. Supplying a URL alone only moves publication to NOT_CHECKED: a reviewer still needs to visit it and verify the released artifact.

The [contribution draft](../labs/releases/l157-f1-audit/CONTRIBUTION_DRAFT.md) states the question, source pin, exact commands, expected/observed contract, complete result and limitations. It invites discussion of preprocessing policy rather than declaring a general benchmark defect. Community feedback can clarify intended semantics; a local agent review cannot impersonate maintainer agreement.

To pursue an upstream patch: reproduce the issue against current upstream; check existing issues; agree on expected behavior; write a tiny failing fixture; implement the narrow change; follow project tests and formatting; then submit with the evidence. Keep the full GPU experiment outside a routine PR test suite. The [current contribution guide](https://github.com/stanford-star/relbench/blob/main/CONTRIBUTING.md) is the authority for repository mechanics and may change.

## 7 · Run it from a clean directory, then defend it

Open the [student notebook](../labs/0157-open-source-contribution.ipynb), implement the three functions and run all CHECK cells. The default notebook embeds the compact package, so it works outside this repository without fetching training data. It exports your claim review. The [executed solution](../labs/html/0157-open-source-contribution.html) is author evidence, not your demonstrated mastery.

For full retraining, use the package's pinned CUDA environment, set `RUN_FULL_REPRODUCTION=True` in the notebook and run all. It preserves the author evidence, performs the complete preparation audit and executes all ten fits. This optional manual gate has no dollar meter; the author's managed dispatcher used the separately recorded $10 cap. Do not turn it on accidentally during a CPU replay.

**EXIT:** write 250–400 words defending (1) the exact supported claim, (2) why replay differs from a fresh run, (3) why the correction is a policy experiment, and (4) what remains before a public contribution is complete. Grade 0–2 each for scope, executable recipe, integrity/coverage, limitation handling and contribution clarity. Pass at 8/10 with no zero, after review. Learner status remains PENDING_WRITTEN_DEFENSE.

[[TEACHBACK]]

Tomorrow, reconstruct the two manifests and two commands from memory. In one week, diagnose a package with one missing seed and an overstrong headline. Ask the teaching agent to review your code or defense whenever a boundary is unclear.

## 8 · What is ready, and what remains public work

[[DELIVERY]]

The local lesson and contribution artifact can be checked now. A public URL, maintainer response and accepted PR are separate milestones. This prepares the public-evidence habit needed for the Year 4 synthesis and later research launch. Continue to [Lesson 158](0158-year-4-synthesis.html) to turn this reviewable package into a bounded thesis argument. The curriculum also plans Lesson 157b on whether a schema graph is the right graph to build. That lesson is not available yet and is not a prerequisite for this route; an impeccable reproduction package cannot by itself justify that modeling assumption.

[Quick reference](../reference/reproducibility-contribution.html) · [Full protocol](../labs/l157-reproduction.md) · [Machine-readable report](../labs/evidence/l157/report.json).
