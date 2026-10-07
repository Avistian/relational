# Two-lesson review pilot — October 7, 2026

Scope: L081 first, then L082; stop after two and report actual session usage. No new benchmark training or paid cloud jobs. Authoring and review use the teach, lesson-pedagogy, lesson-visuals, lab-authoring and PDF skills. The user's one-by-one audit request authorizes the concrete repairs below.

## L081: MPNN framework

- Prerequisites: added the row-to-node and [2,E] edge-list mapping before indexing tensors; explained gathering into edge order and reducing back into node order.
- Explanation and visual: replaced the B-only horizontal trace with a native mobile table and one vertical computation from A—B—C plus D, through all four edge messages, to all node updates and graph sums [13.5,5]. Explained why 18.5 is correct only when the four nodes belong to the same graph. The earlier symmetry figure uses that single-graph convention.
- Exercises: the old reducer check claimed multiple destinations but used only destination 1. Added multiple destinations, two signed coordinates, both reductions, zero-valued empty results and interleaved graph membership. Added a directed transfer problem, because reversing a symmetric edge list leaves it unchanged.
- Source audit: [Gilmer §2, Eqs. 1–3](https://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf) supports the message/update/readout separation, old-state dependency and invariant readout. The hand mean operator is explicitly a teaching choice. [Supplement Tables 1 and 3](https://proceedings.mlr.press/v70/gilmer17a/gilmer17a-supp.pdf) gives the GG-NN dipole target 3.94 × 0.1 = 0.394 Debye. Checked the pinned released `mpnn.py` against the two message banks, reset-before-recurrent-multiplication GRU and gated sum. Historical split/search gaps remain open.
- Independent verification: 40 random directed, multi-feature graphs against an explicit NumPy edge loop; six deliberately broken implementations rejected. Existing dense bond-message and NumPy GRU oracles pass. Fresh portable solution: 16 core code cells. QM9 training not rerun.

## L082: GCN

- Prerequisites: defined channel, logit, ReLU, eigenvector and eigenvalue inline; added the two-node Laplacian's constant and alternating patterns.
- Explanation and visual: the existing scalar W=1 example hid channel mixing. Added a second coordinate on the same four-node graph and signed W; show H, HW, S(HW), then ReLU, with B highlighted throughout and a native mobile version of the computation. Numbers are generated from the illustrated matrix operations and rounded only for display. The new text includes exact B values and a different-width transfer exercise.
- Exercises: the previous positive scalar check accepted an erroneous ReLU inside `propagate`. Added signed multichannel dense/sparse checks. Added a loss-mask trace with nonzero gradients at neighboring unlabeled inputs, zero direct gradients at held-out output logits and no use of held-out labels.
- Source audit: [Kipf & Welling Eqs. 2, 8–10 and Table 2](https://arxiv.org/pdf/1609.02907) support symmetric augmented-degree propagation, the renormalization distinction, two-layer classifier and masked supervision. The [pinned release](https://github.com/tkipf/gcn/tree/39a4089fe72ad9f055ed6fdb9746abdcfebc4d81) has ReLU only in its first layer, identity final activation, first-layer regularization, and previous-ten mean stopping with last weights. Saved 100-run PyTorch results remain separate from TensorFlow parity.
- Independent verification: 40 random graphs against explicit node/feature loops; analytic input-gradient calculation; five deliberately broken implementations rejected. The existing verifier also re-executes original preprocessing, verifies source/data hashes, and compares full-Cora sparse and dense forward calculations. Fresh portable solution: 23 core code cells. Full Cora training not rerun.

## Delivery and evidence boundaries

The verification was a separate numerical/source/render pass by the same agent, using independent formulations and mutation checks; it was not an independent human or second-agent review. Student notebooks retain three live TODOs apiece. No learner mastery is inferred.

Changed exercise code invalidates historical notebook execution. Builders now support an explicit `--reset-execution`; the default still rejects changed code when preserving saved outputs. Fresh core execution records identify unexecuted training cells; prior execution receipts are archived here. Canonical model/trainer implementations and their source hashes are unchanged. L082's benchmark caption now says saved evidence.

Commands, from the repository root:

```sh
.venv/bin/python reviews/two-lesson-pilot-2026-10-07/verify.py 81
.venv/bin/python reviews/two-lesson-pilot-2026-10-07/verify.py 82
.venv/bin/python reviews/two-lesson-pilot-2026-10-07/execute_core.py 81
.venv/bin/python reviews/two-lesson-pilot-2026-10-07/execute_core.py 82
.venv/bin/python labs/_verify_l081.py
.venv/bin/python labs/_verify_l082.py
.venv/bin/python labs/_delivery_l081.py
.venv/bin/python labs/_delivery_l082.py
python3 scripts/refresh_lesson_visuals.py --check
```

Browser checks use the local Playwright installation and its shared-library directory. `l81-browser.json`, `l82-browser.json`, and `rendering.json` record 375/1200px resource and interaction checks, native mobile tables with at least 15px type, print fitting, and portable previews. The full diagrams retain the enlarge controls. Screenshots were visually inspected. Delivery checks verified 100 and 107 local publication references respectively. Clean Git-index build is checked after staging, so ignored workspace files cannot mask missing publication inputs.

Usage records contain actual session token deltas and account allowance snapshots. Cached input is a subset of input; reasoning output is a subset of output. The account percentage is rounded and can include other activity; it is not exact per-task credit billing. No expansion beyond these two lessons is part of this pilot.

The final pre-push usage checkpoint records 3,480,640 total tokens: 3,458,018 input (3,305,216 cached) and 22,622 output. Account weekly usage moved from 23% to 25%; purchased-credit balance remained 0. This includes selection, source audit, implementation, notebook execution and delivery checks through the recorded timestamp; final commit/push overhead follows it.
