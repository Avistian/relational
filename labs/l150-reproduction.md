# Lesson 150 — Q3 full reproduction checkpoint

Approved 2026-09-30. Target: [RelGNN v2 §5.2/Table 2](https://arxiv.org/html/2502.06784v2#S5.T2), full `rel-f1/driver-position`, raw test MAE 3.798. Reproduce one named task; whole paper NOT_RUN. Historical training identity and current competitive standing NOT_ESTABLISHED. Learner PENDING_WRITTEN_DEFENSE.

## Source and protocol ledger

- Architecture release [cffdb8b54627e92c7dd112c1243dde739c90d35b](https://github.com/snap-stanford/RelGNN/tree/cffdb8b54627e92c7dd112c1243dde739c90d35b). Verbatim source packet `sources/l141`; freshly checked URLs and hashes in `evidence/l150/sources.json`.
- Released checkpoint revision `321e6f6e7af5d7546b637f147783fc28ab5d4a7a`, SHA256 `3ba2b6e5c99bc0939d13debb6b06d8d0f0828148361662d54bab83f6c23943df`.
- Database SHA256 `ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482`; task `775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e`. GloVe revision `e5e8fec6971be8960cfaa853a77a6ddc62a265d7`. Fresh materialization hash and package versions: `evidence/l150/prepared/prepared.json`.
- Python3.11, Torch2.5.1+cu124, PyG2.6.1, PyTorchFrame0.2.3, RelBench1.1.0, pyg-lib0.4.0+pt25cu124; remaining pins in `requirements-l117-runtime.txt` and Modal image.
- Complete 7,453/499/760 train/validation/test queries; 74,063 graph rows, 169,421 forward FK edges, 338,842 directed edges, 20 atomic routes. Target is average `positionOrder` in `(cutoff, cutoff+60 days]`; future participation determines eligible query rows. Raw label and population reconstruction checks all8,712 targets.
- Model: one composite layer,128 channels, four128-channel heads,512→128 projection, sum routes, per-node LayerNorm/ReLU, scalar output. Uniform temporal128/64 sampling, query-owned cutoffs, bidirectional subgraphs, batch512, no workers.
- Each training fit: ten complete epochs,15batches per epoch, all7,453queries each epoch, Adam, unclippedL1 loss. Initialize lazy parameters with one evaluation-mode training batch before optimizer. Clip only evaluation predictions to training-target percentiles2/98: [2,31]. First strict validation minimum saves checkpoint. Final validation is resampled and can differ from selection-time validation.

## Four execution tracks

| Track | Protocol | Use |
|---|---|---|
| Reference | seeds0–4, lr0.005, five full fits | Historical score comparison |
| Search | seed100, lr0.001/0.003/0.005, three full fits | Validation-only choice |
| Selected | seeds10–14, chosen lr, five full fits | Course procedure measurement |
| Replay | one released checkpoint, reconstructed feature layout | Inference compatibility only |

Candidate selection uses each fit's minimum validation score, lower learning rate breaks exact ties. Search physically rejects `get_table('test')`; prediction files contain validation only. Freeze candidate scores, SHA256 hashes and final seed IDs before selected dispatch. `evidence/l150/frozen.json` records lr0.003. Reference scores do not enter selection. No post-test retuning. Five-run means and sample SD are kept separate; search, replay and notebook validation never join primary means.

Historical descriptive closeness: `abs(mean-3.798) <= .20` with1e-12rounding allowance. This is neither equivalence nor current near-SOTA. The current [RelBench leaderboard](https://star-project.stanford.edu/relbench/leaderboard/) labels regression normalized MAE; [RelArena](https://star-project.stanford.edu/relarena/) has different data-state, tuning and refit rules. Dated page/JavaScript/data snapshots and the metric/protocol mismatch are retained in `evidence/l150/current-context.json`. No conversion or competitive claim is made without reconciliation.

## Deviations and limits

Training choices absent from the release are frozen course reconstructions. The released test-cutoff database snapshot supplies preprocessing statistics; this is not training-only preprocessing. Event-time filtering cannot reconstruct historical feature arrival/mutation histories. The test population was already seen in earlier lessons: the selected course experiment is exploratory, not a pristine confirmatory holdout.

Fresh inference makes `qualifying.position` categorical; released weights require it numerical. Preserve the original load failure. Reconstruct only the replay feature layout `[number,position]`, comparing ordered means/populationSD to checkpoint buffers at1e-6absolute/relative tolerance. Do not change fresh training inputs. Matching moments support compatibility but do not uniquely identify historical preprocessing.

Compare original architecture outputs at identical weights and sampled batches (atol/rtol2e-4), plus operator output/input/parameter-gradient differential checks. A separate matched-RNG first-real-batch gradient audit preserves nonfinite masks. Source parity is not healthy gradients:640nonfinite entries matched in that batch, finite-entry maximum error1.1920928955078125e-7. No silent source repair. Per-fit nonfinite counts remain in raw results.

## Executable package and evidence

`_full_l150.py` contains full preprocessing and training. `relkit/relgnn_l143.py` is the visible source-faithful model; `relkit/checkpoint_l150.py` provides the three live learner contracts. Notebook includes all graph/model/trainer/compatibility code inline, pinned original source as differential oracle, and all15,346author predictions. Default execution runs model fixtures, source differential checks and keyed scoring. It does not train fresh models.

Set `RUN_FULL_REPRODUCTION=True`, keep `VALIDATE_ONE=False`, in the pinned GPU environment to run fresh preparation, compatibility replay, five reference fits, three validation-only candidates, freeze selection, then five selected fits. Output directories must be fresh. `l150-own-run-report.json` is distinct from the default author-evidence `l150-checkpoint-report.json`. Neither supplies a written defense. Additional notebook validation uses the newly prepared graph and one complete seed1000 plus replay; excluded from primary statistics. Live Colab remains NOT_CHECKED.

Primary histories, prediction arrays, sampled traces, source/graph/checkpoint hashes and immutable run IDs are retained under `evidence/l150`. Selected weight binaries were collected locally under ignored `results/l150`; public artifacts carry their hashes and executable reconstruction. `_audit_l150.py` reconstructs raw labels; `_analyze_l150.py` independently scores predictions; `_verify_l150_results.json` records scientific checks. `_execution`, `_notebook` and `_delivery` reports distinguish actual tested surfaces. No deployment requested.

## Aggregate budget and stop rule

USD10total cap including preparation, thirteen fits, replay, notebook check, diagnostics, failures and overhead. [Modal pricing](https://modal.com/pricing), checked2026-09-30: T4+2physicalCPU+16GiB =USD0.00022572/s. Sixteen1800second reservations (prepare+13fits+replay+notebook) plus600second diagnostic =USD6.636168; USD3reserved for startup/build/commit/storage/unitemized overhead. Conservative totalUSD9.636168. Worker-body estimates are not itemized invoices.

Pilot ref0 counts toward reference; require duration×1.25+120<1800before further dispatch. No automatic retries. Reserve before dispatch, reject duplicate phase IDs and aggregate-cap overflow. Stop and retain honest incomplete status if bounds fail. Credits do not increase the cap. Ledger `_budget_l150.json` retains all attempts and scientific-code hashes. Local packaging changes do not consume new paid execution; notebook scientific code must match the validated code hash.

## Commands

Existing run IDs are immutable. For a new paid run, copy the operator into a fresh named experiment namespace with a new volume and empty reservations after reviewing total cost; never erase this evidence. The standalone notebook is the simplest fresh full-execution route. Repository author sequence:

```bash
.venv/bin/python labs/_check_l150.py
.venv/bin/modal run modal/l150_repro.py --phase prepare
.venv/bin/python labs/_collect_l150.py prepare
.venv/bin/python labs/_audit_l150.py
.venv/bin/modal run modal/l150_repro.py --phase ref-0
.venv/bin/python labs/_collect_l150.py ref-0
.venv/bin/modal run modal/l150_repro.py --phase reference
.venv/bin/modal run modal/l150_repro.py --phase search
.venv/bin/modal run modal/l150_repro.py --phase replay-compatible
.venv/bin/modal run modal/l150_audit.py
.venv/bin/python labs/_collect_l150.py reference
.venv/bin/python labs/_collect_l150.py search
.venv/bin/python labs/_collect_l150.py replay-compatible
.venv/bin/python labs/_collect_l150.py diagnosis
.venv/bin/python labs/_freeze_l150.py
.venv/bin/modal run modal/l150_repro.py --phase selected
.venv/bin/python labs/_collect_l150.py selected
.venv/bin/python labs/_audit_l150.py
.venv/bin/python labs/_analyze_l150.py
.venv/bin/python labs/_figures_l150.py
.venv/bin/python labs/_build_l150.py
.venv/bin/python labs/_execute_l150.py
.venv/bin/modal run modal/l150_notebook_check.py
.venv/bin/python labs/_collect_notebook_l150.py
.venv/bin/python labs/_verify_l150.py
.venv/bin/python labs/_delivery_l150.py
.venv/bin/python labs/_check_pages_checkout.py
```

## Observed execution

All thirteen primary/search training fits completed ten full epochs. Reference seeds0–4: test4.314567±0.371880MAE; selected seeds10–14 at lr0.003: test4.079708±0.099947MAE (sampleSD). Both OUTSIDE_TOLERANCE. Replay test3.737381MAE is kept separate. All8,712labels and15,346primary/search/replay predictions independently checked, with 1,049,106 query occurrences audited and no future violations.

Default26-code-cell notebook PASS. The exact same code passed in the pinned GPU runtime with one full extra seed1000 and replay;2,518additional predictions independently aligned and scored. Thirty-two illustrative browser states passed across desktop/mobile, with keyboard/reset, no-JS, print, portable figures and copied Pages links checked. LiveColab/deployment NOT_CHECKED. Learner PENDING_WRITTEN_DEFENSE.

Reserved aggregate including overhead: USD9.636168. Recorded worker-body estimate: USD0.161936; excludes startup/build/commit/storage, invoice NOT_ITEMIZED. No paid retries. Local packaging review corrected the no-JS fallback-count assertion and improved table spacing; neither changed scientific code.
