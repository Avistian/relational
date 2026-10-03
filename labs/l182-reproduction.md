# Lesson 182 — frozen reproduction and research boundary

Approved2026-10-02. **COMPLETE_SELECTED_REPRODUCTION**: fresh RDB-PFN v5 Table9 rel-f1/driver-dnf,512support,three released checkpoints/configurations ×ten support seeds ×702test queries =21,060predictions. This is inference from released weights, not fresh foundation pretraining or hybrid training. It repeats L166 on the same data, not an independent dataset replication.

## Fixed protocol

The complete [L166 protocol](l166-reproduction.md) is inherited unchanged. Code `a95378225478daa262b85f180d482da7516b0af6`; data `d6a88c0a8cce79607cfc0fca0dcba78ba262ffad`; TabICL weights `eaf789a9b25ee8486d6f48997ba076f850bbc30b`. Paper [2603.03805v5 Table9](https://arxiv.org/html/2603.03805v5#A6). Authenticating ledger: [source-ledger](sources/l182/source-ledger.json), [input manifest](evidence/l182/input-manifest.json).

Full11,411training/566validation/702test queries; released evaluator takes support from training and never tunes on validation or test. Seeds0–9 derive from SHA256(`rel-f1-dfs-2:driver-dnf:{s}`),first4bytes,big-endian; NumPy default_rng chooses512training rows without replacement. All three arms share each support. Fixed RDBPFN model_eval00528 and single-table model_eval00360; no checkpoint search. TabICL0.1.3/v1.1,32estimators and default random_state42. Global source RNG42. Fill missing values from support medians,all-missing→0; source model uses support population moments and±100clamp. Numeric category codes retained; no target/entity-key inputs. Six transformer blocks,width96,fourheads,MLP192,692,738parameters. Source paper appendix says width128; released checkpoint takes precedence and this discrepancy remains explicit. Float32 model input; no real-task optimizer updates. Complete query keys are(driverId,date).

**Deviations and unresolved provenance:** the released label is1−current raw 30-dayMAX(statusId!=1). Preserve release orientation for numerical comparison; complement both labels and probabilities for current DNF interpretation. L166 independently reconstructed all12,679raw labels; L182 authenticates/reuses that audit, not a new SQL execution. The three MAX-timestamp features passed owner-cutoff checks in L166, but full raw DFS reconstruction, feature availability and historical fit population are not established. Numerical reproduction does not certify a deployable leak-free dataset. Historical identity NOT_ESTABLISHED; full DFS reconstruction NOT_RUN. RelGNN training-health failure in L178/L181 is not repaired by this lesson.

## Fresh execution and independent checks

Immutable fresh volume `l182-rdbpfn-evidence`,pilot-1 seed0 and full-1 seeds1–9. Apps `ap-9Npqn0drBLYxIqicxc53z3` and `ap-mpZjQE71VmLQPL6zFFquSi`. Python3.11,Torch2.5.1CUDA12.4,NumPy1.26.4,TabICL0.1.3; full dependency versions in receipts. Original authenticated `_run_l166.py` unchanged. Independent pairwise AUROC agrees with sklearn within1.12e-16 for all30runs. All query identities, labels, support identities and file hashes checked. Fresh probabilities exactly equal inherited L166 probabilities in this run; no universal determinism claim.

Mean±sampleSD AUROC: RDBPFN .721937735±.019982668; single-table .663989655±.034176777; TabICLv1.1 .717568438±.009769595. Means round to paper targets .7219/.6640/.7176. Predeclared mean-distance tolerance .02; all CLOSE. These10support draws share one test population and are not10independent databases. RDBPFN−TabICL mean .004369297,positive6/10draws; no broad superiority claim.

## Re-run and replay

From repository root, clean Python3.11 environment using `labs/l166-requirements.txt` and Torch2.5.1:

```sh
python labs/_fetch_l166.py --out /tmp/l166-input
python labs/_run_l166.py --input /tmp/l166-input --out /tmp/l182-new-evaluation
```

This executes all30evaluations, not a saved-score replay. Keep output directory fresh. The source directory is `labs/sources/l166/upstream`; archived code and fixed checkpoints are required. The portable notebook's default is offline rescore plus live mechanism work. Its post-EXIT opt-in fetches pinned source/data/weights and runs the same evaluator, with explicit dependency checks. Live Colab NOT_CHECKED.

The author cloud operator is `modal/l182_repro.py`; completed attempt names intentionally reject re-dispatch. A new paid run requires a new approved ledger/volume/attempt identity; never reset the completed ledger. Recorded commands were `modal run modal/l182_repro.py --phase pilot`,then `--phase full` after admission.

```sh
.venv/bin/python labs/_budget_l182.py .venv/bin/python labs/_report_l182.py
.venv/bin/python labs/_budget_l182.py .venv/bin/python labs/_verify_l182.py
```

## Composite mechanism and hypothesis

RelGNN v2 §4 Equations3–5 and original operator at `cffdb8b54627e92c7dd112c1243dde739c90d35b`. Course kernel specializes to one head,identity Q/K/V projections,no destination skip. Source SAGE/attention/projection layers are set to exactly that state for differential validation; finite differences check input gradients. Full multi-head model or fresh RelGNN training is not reproduced in this lane. Temporal filtering is an explicit strict-before course policy, not a claim about all source sampling defaults.

Deterministic interventions: preserve complete FK rows and receiver-specific cutoff; permute legal message order; alter excluded future features; duplicate one legal message; deliberately merge another route. These probe order invariance,time isolation,multiplicity sensitivity and role mixing. They do not establish universal expressivity or accuracy superiority. Two fabricated score rows illustrate the factorial contrast; no model was trained to obtain them.

Proposed future hybrid: synthetic-task pretraining for a graph encoder plus a shared support/query ICL head. Compare single-table versus relational priors and conventional two-hop versus composite encoders with a fixed schema-agnostic parameter-sharing policy. Need newly trained checkpoints,matched labeled context,loss,training tasks,parameter/compute accounting,held-out schemas and test tasks. A generator-side operator replacement instead changes the task distribution and must be a separate experiment. Both hybrid alternatives NOT_RUN.

## Budget and failures

USD10cap,USD8plannedstop,USD2overhead/reserve;6000aggregateworkerseconds maximum. L4+2physicalcores+16GiB .00028372USD/s. Reserved630+5330=5960seconds; compute ceiling1.6909712USD plus2USDoverhead =3.6909712USD. Known worker-body50.78029912seconds; invoice NOT_ITEMIZED. Runtime reservations include30seconds per attempt for loading/startup beyond function timeout; this is a conservative execution ledger, not a billing invoice. Local numerical cap3600seconds,attempts in local-budget.json.

No failed cloud execution. Collection first wrote a single file to a non-directory destination,then used a directory and normalized the nested folder; the admission failure before correction is retained in local accounting. Original-PyG parity first needed its dynamic module registered in sys.modules; failed check and corrected check are both accounted. No prediction,checkpoint,seed or budget substitutions.

Additional validation detail: a strict float32 visible/original forward check differed by 1.55e-6 on one logit and failed its 1e-6 threshold. The mathematical parity lane was then explicitly run in float64 (as in L166), with maximum errors below 9e-15 across both checkpoints. The notebook uses float64 for that small parity check; the unchanged fresh full evaluator uses float32. This does not claim bitwise equality between the two float32 implementations. The failed parity/build attempts are retained in local accounting.
