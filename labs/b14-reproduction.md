# B14 · Flattening challenge reproduction contract

Approved2026-10-03:USD10aggregate,USD8stop,USD2reserve;3600aggregate local execution seconds. No deployment. Status and measured output are in `evidence/b14/paper-status.json`; course and paper evidence are distinct.

## Selected paper/release experiment

B14-TABPFNREL-OSS-F1-DNF: RelArena paper2608.16319v2 Table4, local text-free TabPFN-Rel on rel-f1/driver-dnf, target0.7145AUROC. Original RelArena tagv0.0.1 commit `e89002200e18be6d8d7a55f8a5ab50c993ce4d5d`; TabPFN8.0.8, FastDFS1.1, RelBench2.1.2. Complete original source/lockfile and published result CSV are archived under`sources/b14/`. The current main branch has later model aliases and is not silently substituted.

Seed0; DFS depths2/3/4; hard_pool context, K100000,M400000,8estimators,no text. All three full validation candidates run; select largest validation AUROC with original candidate-order tie handling. Refit selected and default on train+validation; if identical the original runner shares the refit. Train11411/val566/test702. Inner DB cutoff2005-01-01; outer2010-01-01. Frozen DB plus per-query DFS cutoff, with original transformations and target-history construction. All eight original release checksums match, including unmasked held-out label content. Complete target-history schedule has zero unfinished earlier30daywindows under anchor+30days availability; actual publication clocks and all raw-field historical availability remain NOT_ESTABLISHED.

Checkpoint: public `Prior-Labs/tabpfn_3` revision `24a16a89d245878b846555110985634aa2e656d7`, `tabpfn-v3-classifier-v3_default.ckpt`, SHA256 `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988`. Download authenticated against Hub LFS metadata. Original CSV does not include checkpoint SHA256, so historical byte identity is NOT_ESTABLISHED. This lane executes the pinned available checkpoint with the original release protocol; it cannot manufacture historical identity.

Hardware A10080GB,4CPU,24GiB replaces the reference RTX PRO6000. Package versions and prediction receipts are recorded. No preprocessing repair, depth reduction, support subsample or seed substitution is permitted. Floating-point/hardware and dependency differences remain explicit deviations. Descriptive CLOSE tolerance0.02AUROC is set before inference; it is neither significance nor historical identity. Exact rounded agreement is reported separately if observed.

The release CSV selects depth3 with validation0.6194376417 and test0.7144683550. These reference scores are not fresh B14 measurements and do not bypass validation search. Prediction scoring must authenticate full(driverId,date)keys and labels and independently recompute AUROC with positive-negative pair comparisons.

## Course diagnostic

12fresh fits: seeds0/1/2 × entity-only/relational features × linear ridge/RBF kernel ridge.128entities; one event per5days over0..115; one independent local feature; latent history values plus temporal variation. Support dates20/40/60/80 give512rows; query dates100/110 give256rows. Course label horizon10days, so final support has matured by day90. Target=.4local+1.5sin(history_mean)+.15last+Normal(0,.1), constructed under a rolling world; fixed test features stop at day100. This intentionally favors relational and nonlinear signal and includes staleness for later queries.

Feature rule:event_time<min(query_time,snapshot) and available_time≤boundary. This explicit two-clock rule is a teaching contract, not an undocumented change to original paper preprocessing. Reject duplicate event IDs/query keys; each physical event contributes once. Empty aggregates are0. Fixed alpha1; support-only mean/populationSD; RBFgamma1/F. No hyperparameter selection. Same support/query arrays within each feature arm, hashed once and shared across predictors. Report all3072fixed and3072rolling predictions. Rolling changes query information only, does not refit, and is not an admissible backbone gain. Outcomes are MSE, not paper AUROC.

## Original RDBLearn source diagnostic

Pinned original0.1.2 code from commitb5b03ebf8091547285a6e06cba53d2d1a40cb171 and FastDFS0.2.1 copied with authenticated original ledger. Fresh full AutoGluon preprocessing probe fits b/c/d, transforms unseen a/z/0/e, then rechecks the known categories and numeric control. The environment-qualified semantic failure remains separate from the adapted RelArena wrapper, which supplies categoricals directly to TabPFN. No measured benchmark AUROC effect is inferred from a synthetic category probe.

## Executable commands

Run from repository root with the repository Python for author operators. Course:

```bash
.venv/bin/python labs/_budget_b14.py .venv/bin/python labs/_test_b14.py
.venv/bin/python labs/_budget_b14.py .venv/bin/python labs/_run_b14.py
.venv/bin/python labs/_budget_b14.py .venv/bin/python labs/_verify_b14.py
.venv/bin/python labs/_reproduce_b14.py --show-plan
```

Isolated paper CPU environment (do not replace course dependencies):Python3.12,torch2.13.0+cpu,relbench2.1.2,tabpfn8.0.8,fastdfs1.1,pandas2.3.3,numpy2.5.3,scikit-learn1.6.1,setuptools80.9.0,configspace,jsonschema,pyyaml. Reconstruct withuvvenv anduvpipinstall; use the PyTorch CPU wheel index. Exact observed package snapshot is archived. Set OMP/OPENBLAS/MKL threads1 for local preparation.

```bash
.venv/bin/python labs/_budget_b14.py /tmp/b14-paper-env/bin/python labs/_preflight_b14.py
.venv/bin/python labs/_budget_b14.py /tmp/b14-paper-env/bin/python labs/_warm_b14.py
.venv/bin/python labs/_budget_b14.py .venv/bin/python labs/_download_b14.py
.venv/bin/python labs/_budget_b14.py .venv/bin/python labs/_prepare_cloud_b14.py
.venv/bin/python labs/_budget_b14.py .venv/bin/python labs/_reproduce_b14.py --phase pilot
# Full phase requires recorded measured pilot admission and enough total budget:
.venv/bin/python labs/_budget_b14.py .venv/bin/python labs/_reproduce_b14.py --phase full
```

Temporary environment/cache paths are explicit author locations, not portable dependencies. The source archive includes the complete original code; the notebook runs independently with NumPy. A new paper execution needs reconstruction, checkpoint download and a new budget ledger. Operators refuse implicit repeated phases to preserve original evidence. Do not delete past reservations to obtain extra budget.

## Accounting and boundaries

Modal public rate verified2026-10-03: A10080GB0.000694USD/s +4CPU×0.0000131 +24GiB×0.00000222=0.00079968USD/s. Reserve full function timeouts:pilot360s/full1260s, total1.2954816USD; overhead allowance2USD. Planned envelope3.2954816USD, belowUSD8stop, with separateUSD2margin to theUSD10cap. A successful pilot and conservative full-grid forecast are required before full dispatch. Reservations are conservative envelopes, not itemized invoices. Failed setup,300second preprocessing timeout and successful cache retry remain in the local ledger.

No full21-task benchmark, hosted text-enabled inference or backbone pretraining. Source parity, complete selected inference, historical identity, production availability, deployment and learner mastery remain separate. Learner PENDING_WRITTEN_DEFENSE; liveColab/deployment NOT_CHECKED. Author figures and solution execution are not learner evidence.

Initial cloud image construction rejectedscikit-learn1.9.0 because RelBench2.1.2 requires≤1.6.1. No GPU trial dispatched in that attempt; logs retained. Corrected1.6.1 matches the release lockfile. NumPy2.5.3 is newer than lockfile2.4.6; observed packages are explicit, so full historical environment parity is not claimed.

## Observed selected reproduction

COMPLETE_SELECTED_RELEASE_CLOSE. Fresh complete three-depth search and both final refits; validation selects depth3. Selected test AUROC0.712838530 versus published0.7145 (absolute difference0.001661470). All5evaluations/3102predictions independently scored. Historical checkpoint byte identity remains NOT_ESTABLISHED; full suite and pretraining NOT_RUN. The full worker completed in75.289seconds including process startup; pilot43.199seconds. Reserved ceiling3.2954816USD remains conservative; this is not an itemized invoice.


### Reconstruct the isolated environment

With `uv` installed, the following reproduces the author CPU dependency choices. Replace the temporary directory only when creating a new isolated run; preserve the original evidence and budget ledgers.

```bash
uv venv --python 3.12 /tmp/b14-paper-env
uv pip install --python /tmp/b14-paper-env/bin/python \
  'torch==2.13.0+cpu' 'numpy==2.5.3' 'pandas==2.3.3' \
  'scikit-learn==1.6.1' 'relbench==2.1.2' 'fastdfs==1.1' \
  'tabpfn==8.0.8' 'setuptools==80.9.0' configspace jsonschema pyyaml \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match
```

The portable source archive includes the budget wrapper and Modal worker/launcher. Model weights and derived DFS caches are deliberately reconstructed by the download/warmer operators. The standalone notebook's default path needs only NumPy, authenticates the source packet and replays all saved paper predictions; it does not invoke the expensive setup or GPU lane. New paper runs need a separate output checkout and budget ledger; the current ledger refuses duplicate pilot/full reservations.
