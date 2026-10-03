# B15 reproduction contract

Approved 2026-10-04; push and Pages deployment authorized. Two separate tracks follow.

## B15-LABEL-VISIBILITY: complete finite course experiment

No random sampling, fitted parameters, seeds or hyperparameter selection. Every declared world has equal weight. Python standard library only. Full implementation is `relkit/labels_b15.py` and is visible in both notebooks.

Rule worlds: `(theta,b)` in `{0,1}²`; query truth `b XOR theta`. Local observed neighbor label is `b`; its own generating neighborhood is not supplied as a labeled training pair. A legal external support pair `(0,theta)` identifies the task rule. Query key `(query,10)`, external key `(remote,2)` available at4, future key `(future,3)` available at11. At cutoff10 only the external pair is usable by the expanded head. Context membership is declared in advance. Local-only prediction averages both rules; expanded prediction averages rules compatible with the permitted pair. Leaky control reads query truth directly.

For each world enumerate all four assignments to the stored query/future labels. Legal predictions must be invariant:16 interventions. The leaky control follows the changed query label and is rejected. Event time must be strictly before cutoff; available-at may equal cutoff. Exclude query identity explicitly; reject duplicate full keys, invalid bits and nonfinite times. These are course availability rules, not claimed paper preprocessing.

Column worlds: `(S,a,b)` in `{0,1}³`; truth is query feature `x[S]`, where `x=(a,b)`. Both local support rows `(0,0)->0` and `(1,1)->1` are fixed across S. Average all compatible columns; expanded support adds `(0,1)->S`. Measure mutual information between S and observed context in bits, counting every world equally. Four of eight query worlds have equal features; those remain predictable without identifying S. Thus local query accuracy is75%, not50%, despite zero information about S.

Accuracy uses threshold `p>=0.5`; Brier is mean squared probability error. This exact finite distribution has no sampling uncertainty: no seed SD or population confidence interval is claimed. Predetermined expected values: rule local accuracy.5/Brier.25; expanded1/0; leaky1/0. Column local.75/.125; expanded1/0. Local/expanded information about S:0/1bits.

These mechanisms illustrate an indistinguishability argument, not the complete parity-based graph construction, general theorem, learned-encoder comparison, or empirical superiority. Expanded support changes available information; its gain cannot be attributed to encoder training. Labels at the prediction head differ from labels fed into the encoder.

## B15-RDBLEARN11-TRIAL: selected paper target

Paper arXiv2607.05476v2 Table 5; RDBLearn v1.1, rel-trial/study-outcome, published AUROC72.71%=0.7271. Source tag object46a725de458205962b9eda3ffcd52fe7482ae519 resolves to commit78561f0a9c1dd231d44659e761d5d85e18c82f6e. Complete24 tracked release files plus paper HTML are archived and individually SHA256-pinned under`sources/b15/`.

Source audit:README says target history defaults ON. Actual `RDBLearnConfig.enable_target_augmentation` defaults False; estimator both stores history and adds it to the database only under that flag. No actual benchmark run configuration is in the24-file release. The default agrees with the paper's statement that its empirical encoder does not use local labels. This resolves the *default*, not the exact executed benchmark configuration.

Missing: complete trial candidate grid (backbones/depths/aggregations/categorical encoding), seed and training-subsample mapping, checkpoint revisions/hashes, validation-selection and final-support/snapshot policy, data/environment identity and prediction receipts. `random_seed=None` in config is not an authenticated experimental seed. The example runs rel-f1/driver-dnf with defaults, not trial. Appending a different dataset name cannot recreate an unknown search protocol.

Status:INCOMPLETE_SOURCE_PROTOCOL_GATE; fresh paper inference NOT_RUN. Operator `--phase paper` authenticates the archive then refuses execution. Full original executable source is included; a complete aligned Table 5 runner cannot honestly be supplied until missing settings are recovered. No paper-score tolerance or observed score is fabricated. Full six-benchmark suite and backbone pretraining NOT_RUN; historical identity NOT_ESTABLISHED.

## Commands and next step

From repository root:

```bash
.venv/bin/python labs/_budget_b15.py .venv/bin/python labs/_test_b15.py
.venv/bin/python labs/_budget_b15.py .venv/bin/python labs/_run_b15.py
.venv/bin/python labs/_budget_b15.py .venv/bin/python labs/_verify_b15.py
.venv/bin/python labs/_reproduce_b15.py --phase audit
.venv/bin/python labs/_reproduce_b15.py --phase paper  # expected refusal
.venv/bin/python labs/_build_b15.py
.venv/bin/python labs/_budget_b15.py .venv/bin/python labs/_execute_b15.py
.venv/bin/python labs/_budget_b15.py .venv/bin/python labs/_delivery_b15.py
```

The notebook embeds the mechanism code, frozen evidence and source archive and executes in an empty directory with no repository imports or downloads. Its guarded paper cell remains off; changing it cannot bypass source admission. Recover the original run receipt/config, pin checkpoint and dataset identities, implement the full source-aligned candidate/search/refit procedure, then set tolerance before execution. A timed pilot and current-rate estimate must cover every candidate/seed/retry before any full dispatch. No cloud worker is advertised as ready while the experiment is undefined.

## Budget and evidence

3600 aggregate local numerical seconds, including failed tests and notebook validation; actual attempts in`evidence/b15/local-budget.json`. Initial and actual cloud spendUSD0. MaximumUSD10 total, stopUSD8/reserveUSD2; no paid forecast or dispatch warranted while source admission fails. Architecture/source checking, portable replay, browser checks, Pages CI and live deployment are separate from paper-score reproduction and learner mastery. Learner PENDING_WRITTEN_DEFENSE. Live Colab NOT_CHECKED.
