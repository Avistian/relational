# Lesson 140 Implementation Plan

**Goal:** Deliver the approved two-task reproduction checkpoint and defensible evidence.
**Architecture:** Shared task-parameterized released trainer; new audit/report functions; existing model primitives retained. Budgeted Modal runs read prior materializations, write only L140 evidence. A canonical builder produces lesson/reference and standalone notebooks.
**Tech stack:** Python, pinned Torch/PyG/Frame/RelBench, Modal, nbformat/nbclient, matplotlib, plain HTML/JS, Playwright.
**Spec:** 2026-09-30-lesson-140-design.md.

- [ ] Add behavior checks and live checkpoint functions in labs/_check_l140.py and labs/relkit/checkpoint_l140.py. Verify incomplete runs, ties, key mismatch, and protocol gaps fail safely.
- [ ] Add labs/_full_l140.py, modal/l140_repro.py and budget ledger. Audit source/config equivalence; preflight archive/graph identity. Run full seed0 pilots; forecast and reserve remaining eight fits before dispatch.
- [ ] Collect all results/checkpoints in labs/_collect_l140.py. Independently score all predictions, verify query sets/selection/epoch counts/timestamp audits/source parity; retain gaps.
- [ ] Author lessons/content/0140-rdl-reproduction-checkpoint.md, assets/reproduction-gates-viz.js, labs/_figures_l140.py, labs/_build_l140.py, reference and notebooks. Three live tasks plus written defense; model/trainer visible and portable.
- [ ] Execute solution, test full gate in pinned cloud runtime, verify inline AST identity, figures, links, browser/mobile/keyboard/print/noJS, copied Pages and clean index. Update manifest, curriculum, resources and NOTES without claiming learner completion.
