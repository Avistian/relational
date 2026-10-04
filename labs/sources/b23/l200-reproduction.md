# L200 — Year 5 exit exam reproduction

Approved 2026-10-02: fresh complete selected **RDB-PFN v5 Table 9, rel-f1/driver-dnf**, all three configurations × ten support draws × all702 queries =21,060 predictions. Released-checkpoint inference, not foundation pretraining. Old predictions never populate the new run directory.

## Frozen protocol

Inherit [L166](l166-reproduction.md) and [L182](l182-reproduction.md) unchanged. Code a95378225478daa262b85f180d482da7516b0af6; data d6a88c0a8cce79607cfc0fca0dcba78ba262ffad; TabICL checkpoint revision eaf789a9b25ee8486d6f48997ba076f850bbc30b. RDBPFN model_eval00528, single-table model_eval00360, TabICL0.1.3/v1.1,32estimators. Python3.11,Torch2.5.1CUDA12.4,NumPy1.26.4,pandas2.2.3,sklearn1.6.1,pydantic1.10.26. Per-worker dependency receipts retain actual versions.

11,411training/566validation/702test queries. Support draws0–9 use first4bytes big-endian SHA256(rel-f1-dfs-2:driver-dnf:{s}); NumPy default_rng chooses512training rows without replacement. All configurations share each support; global model RNG42. No checkpoint search or test-based tuning. Support-only median imputation,all-missing→0; support population normalization,±100clamp; numeric category codes retained. No labels/identity keys enter feature inputs. Float32 evaluation, full(driverId,date) query keys. Six blocks,width96,fourheads,MLP192. AUROC target means .7219/.6640/.7176; absolute mean tolerance .02 descriptive, not statistical equivalence. Mean and sampleSD across draws, paired differences; one fixed test population.

## Deviations and boundaries

Released labels complement the currently reconstructed raw30dayDNF indicator. Preserve released orientation for comparison; complement both probabilities and labels to reinterpret. Source checkpoint width96 takes precedence over appendix128. L166's label/selected timestamp audit is authenticated inherited evidence, not new SQL execution. Full DFS regeneration NOT_RUN; full availability and historical identity NOT_ESTABLISHED. RDBLearn remains INCOMPLETE_SOURCE_PREPROCESSING_GATE. No claim of whole-paper reproduction, broad superiority, pretrained-weight retraining, liveColab, deployment or learner mastery.

Visible model parity is float64 on a small fixed input against both real checkpoints; full fresh inference stays on the unchanged float32 evaluator. Notebook default rescoring replays the new L200 artifacts; it is not itself another fresh inference run. The optional complete evaluator below is separate. L198/L199 author reports are frozen for proposal context, not numerically re-executed or learner-submitted.

## Fresh and replay commands

From repository root, clean Python3.11 environment with [requirements](l166-requirements.txt),Torch2.5.1:

```sh
python labs/_fetch_l166.py --out /tmp/l200-input
python labs/_run_l166.py --input /tmp/l200-input --out /tmp/l200-fresh-evaluation
```

The fetcher pins source/data/weights. Keep a fresh output directory. Full default is all30evaluations. Source checkout lives at labs/sources/l166/upstream. Read its license/source ledger. User reruns require their own aggregate budget; do not reset completed author ledgers.

Author cloud operator: modal/l200_repro.py; new immutable volume l200-rdbpfn-evidence. Pilot includes allthree configurations at seed0; the remaining27run dispatch follows independent full pilot admission. Commands: `modal run modal/l200_repro.py --phase pilot`, collect raw files, `_admit_l200.py`, then `--phase full`. Source hashes, input manifest and reservations are in evidence/l200. Disabled automatic retries,duplicate attempts/source drift rejected before dispatch.

```sh
.venv/bin/python labs/_budget_l200.py .venv/bin/python labs/_audit_l200.py
.venv/bin/python labs/_budget_l200.py .venv/bin/python labs/_verify_l200.py
```

The portable [audit archive](evidence/l200/reproducer.zip) includes raw new predictions, prepared data, receipts, manifests, visible contracts and independent verifier. Install NumPy, extract, then run `python labs/_audit_l200.py` and `python labs/_verify_l200.py` from the extracted root. For repeated work use the included budget wrapper. The complete portable solution embeds this same archive and runs from an empty directory.

## Budget

USD10all-in cap,USD8plannedstop,USD2overhead reserve;6000aggregateworker seconds. L4+2physicalcores+16GiB at USD0.00028372/s,checked at https://modal.com/pricing on2026-10-02. Reserved630+5330=5960seconds includes30seconds allowance per call beyond function timeout,upper worker reservation USD1.6909712; plus2 allowance USD3.6909712. Startup/idle/build/storage/collection are not identical to worker-body time: measured bodies are a lower accounting component,not the invoice. Invoice NOT_ITEMIZED. Full pilot projection must admit remaining scope. Any integrity failure or over-cap projection stops; no seed/model substitutions. Local numerical/delivery cap3600seconds; local-budget includes failed checks and preparation allowance.

## Exam

Independent rank-based AUROC oracle verifies all30runs. Mutation tests reject changed bytes,duplicate/missing runs,wrong keys/supports/labels and nonfinite probabilities. Three live learner contracts feed final reporting. Proposal and defense remain PENDING_WRITTEN_DEFENSE. All evidence slots complete means READY_FOR_TEACHER_REVIEW,never automatic mastery. No future proposal experiment is authorized here.
