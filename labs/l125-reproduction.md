# L125 reproduction and evidence contract

## Result

**Full selected paper experiment: NOT_RUN.** The original database, task and raw-data
endpoints return 404 (2026-09-27; six probes). No checksum-matching mirror was recovered.
Paper target: Hu et al., arXiv:2404.00776v2 §5.3/Table 2, `rel-stackex-engage`,
Frame ResNet + PyG HeteroSAGE, ROC-AUC .854. No measured score is claimed.

Executed separately: complete released F1 feature path, all 74,063 rows across nine
tables through 2010-01-01, and one training-query gradient step. Those are course
mechanism results, INCOMPARABLE to the paper's ROC-AUC.

## What is pinned

`_sources_l125.json` records full upstream revisions and SHA-256 hashes for 20 source
files. `sources/l125/relbench/` preserves task SQL, dataset registry, graph conversion,
row/GNN encoders, model, training/selection/evaluation loop, text adapter and MIT
license. The RelBench revision is the last selected contemporary snapshot at the
paper's first release period, **not** an author-attested Table 2 commit. Frame 0.3.0
is the teaching/operator reference, **not** a claim about the historical environment.
`_audit_l125.py` compares installed Frame load-bearing files byte for byte.

## Protocol ledger

| Component | Paper/historical candidate | Executed course path |
|---|---|---|
| Data | Seven-table historical rel-stackex; unavailable original archives | Complete checksum-pinned released F1 database through 2010-01-01 |
| Task | Contribute vote/comment/post in next 730 days | Feature export; one step on 23 driver-position TRAIN queries |
| Split | Archived validation 2019-01-01, test 2021-01-01 | Gradient cutoff 2004-09-03; no test performance measured |
| Preprocessing | Archived graph routine materializes database censored at test time | Statistics fit through 2004-09-03 for dated tables; static histories unavailable |
| Row model | Paper names ResNet; candidate four width-128 blocks | Two width-8 blocks, LayerNorm, dropout0 |
| Text | Candidate GloVe SentenceTransformer, revision unpinned | Fixed SHA-256 bucketed token counts, 16 dimensions; not semantic/pretrained |
| GNN | Paper names HeteroSAGE; candidate two layers + relative time | One mean-aggregation results-to-drivers relation for gradient exercise |
| Training | Candidate Adam .01, batch512, 10epochs, fanouts[128,64], seed42 | One Adam .001 update; exported vectors precede this update |
| Selection | Candidate first best validation ROC-AUC | None; no benchmark fit |
| Reporting | Published .854; seed aggregation not attested | Shapes, equality, IDs, finite values, gradients; no ROC-AUC |
| Missingness | Historical behavior depends on missing runtime | Fixture: mean numeric, unknown/missing categorical -1; F1 string missing values explicitly become empty strings |
| Historical identity | NOT_ESTABLISHED | No historical identity claim |

The archived candidate regenerates tasks (`process=True`) with a comment that
correct tasks still need uploading. Matching the archived task ZIP alone therefore
does not establish that it contains the paper's labels. The source makes historical
recovery concrete but is not a ready-validated paper replay environment.

## Exact local commands

From the repository root, the existing `.venv` is the tested author runtime:

```bash
.venv/bin/python labs/_check_l125.py
.venv/bin/python labs/_source_check_l125.py
.venv/bin/python labs/_run_l125.py
.venv/bin/python labs/_audit_l125.py
.venv/bin/python labs/_figures_l125.py
.venv/bin/python labs/_build_l125.py
.venv/bin/python labs/_execute_l125.py
.venv/bin/python labs/_delivery_l125.py
.venv/bin/python labs/_verify_l125.py
```

`_run_l125.py` uses the delivered archives under `evidence/l125/`. On the initial
author run it copied them from the existing RelBench cache. Loading verifies the
original hashes. It never uses Stack Exchange or cloud compute.

Recheck original archive URLs and repin source (requires git and network):

```bash
git clone --filter=blob:none https://github.com/pyg-team/pytorch-frame.git /tmp/l125-frame-upstream
git clone --filter=blob:none https://github.com/snap-stanford/relbench.git /tmp/l125-relbench-upstream
.venv/bin/python labs/_prepare_l125.py
```

If those directories already exist, use the existing checkouts. `_prepare_l125.py`
reads exact revisions rather than their current working-tree heads. Its report
must be reviewed if a URL becomes available; HTTP success alone is not a hash check.

## Recovery preflight

Original registry hashes:

- `db.zip`: `deb00ccdf825e569b34935834444429cd1c0074b50226b12d616aab22d36242d`
- `engage.zip`: `9afce696507cf2f1a2655350a3d944fd411b007c05a389995fe7313084008d18`
- optional raw archive: `ad3bf96f35146d50ef48fa198921685936c49b95c6b67a8a47de53e90036745f`

Place recovered database/task archives under a directory of your choice, then:

```bash
.venv/bin/python labs/_recover_l125.py --archive-dir labs/data/l125-stackex --report labs/_recovery_l125_results.json
```

Exit 2 means missing/mismatching input. Exit 0 means archive identity only; the
Table 2 protocol and environment remain unresolved. Do not substitute modern
`rel-stack/user-engagement` data based on the name alone.

After archive/config/environment recovery, the preserved candidate's original
entrypoint is `examples/gnn_node.py --dataset rel-stackex --task rel-stackex-engage`.
The complete pinned repository can be checked out using the revision in
`_sources_l125.json`. That command is **not validated in this package** and cannot
run against the missing historical inputs. Establish a compatible pinned Frame,
PyG/native sampling, Torch and text-model environment before the bounded pilot.
No executable stub is represented as a completed reproduction.

## Cost and runtime

Approved aggregate limit USD10. Current Modal T4+2CPU+16GiB rate USD.00022572/s;
eight one-hour reservations would cost at most USD6.500736 in worker resources,
with USD3.499264 reserved for setup/storage/retries/other overhead. A pilot must
fit within those reservations before full seeds are scheduled. No cloud function
was launched: measured paid-compute spend **USD0**, remaining budget USD10.
Do not claim this resource estimate is an invoice or guarantees an unknown workload.
Source: https://modal.com/pricing (checked 2026-09-27).

The author CPU runtime and exact package versions are saved in
`_environment_l125.json`. Colab bootstrap is a separate pinned proposed runtime;
live Colab is NOT_CHECKED and cross-runtime identical floating point is not claimed.

## Artifacts and teaching boundaries

- `evidence/l125/summary.json`: complete feature and gradient report.
- `evidence/l125/encoded-reg.npz`: table-specific IDs and all width-8 vectors.
- `evidence/l125/f1-db.zip`, `f1-task.zip`: checksum-pinned released input archives.
- `_source_check_l125_results.json`: token/row outputs, gradients and optimizer parity.
- `_audit_l125_results.json`: all-row original Frame comparison and four killed mutations.
- `_paper_audit_l125_results.json`: original-source/data recovery evidence and gaps.
- `relkit/frame_l125.py`: visible load-bearing code used by both notebook variants.

The notebook repeats the complete feature path independently from an empty directory.
Its four learner functions are genuinely used in that path. The full historical
source appendix is visible but not executed. The model architecture figures describe
the executed course model; the historical candidate is documented separately.
Dated rows pass the stated cutoff but static mutable attributes have no independent
arrival/version history. One training update is not a predictive evaluation.
Author execution does not establish learner mastery: PENDING_WRITTEN_DEFENSE.
