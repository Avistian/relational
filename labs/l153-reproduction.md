# Lesson153: recommendation portfolio reproduction contract

Approved 2026-10-01: fresh RelBench v1 Table8 GraphSAGE, rel-trial/site-sponsor-run, five seeds0–4. Published validation14.09±.77%,test10.70±1.10%MAP@10. Frozen descriptive tolerance±.02MAP on the mean, not a statistical equivalence test. Whole-paper and ContextGNN reproductions are outside this selected target.

## Sources and runtime

Commit9aa346267c2e1c560bd92da07d6f4ad1ca2f0639; original trainer, loader, graph, neural classes, task SQL and MIT license in `sources/l153/`, SHA256 manifest and immutable source hashes in `_budget_l153.json`. Full visible encoder/model classes exactly match original class ASTs. Torch2.5.1 CUDA12.4,PyG2.6.1,Frame.2.3,RelBench1.1.0,pyg-lib.4.0+pt25cu124,numpy1.26.4,pandas2.2.3; complete pins in requirements-l117-runtime.txt and modal/l153_repro.py.

GloVe sentence-transformer revisione5e8fec6971be8960cfaa853a77a6ddc62a265d7; fresh full feature/graph materialization at seed42. Registry-pinned database archive9fb5ba14f7cbca8115f3dfe0800415f98d6ddc15561e56c35ee614da6b89552a and task archive64c8b039b81d45b708e85c71c319e95372dd8f9a832e4255aad448ca0fe0a879. Prepared artifacts retain hashes per parquet plus graph hash. Historical archive/RNG identity NOT_ESTABLISHED.

## Exact selected procedure

| Component | Contract |
|---|---|
| Population | 669310train,37003validation,27428test;53241sponsor catalog |
| Query key | facility_id and exact cutoff |
| Label | distinct sponsor IDs via facility-study date in(t,t+365days], source dangling filter; no sponsor-study date filter |
| Graph | full primary/foreign-key graph; reverse edges; source column/type proposal; GloVe text |
| Model | per-table Frame ResNet, relative-time encoder, sponsor shallow ID vectors, two128channel sum GraphSAGE layers,128output head |
| Training | Adam.001,20epochs,batch512,fanouts128/64,uniform temporal sampling |
| Objective | BPR softplus(negative−positive),B×Bshared negatives at a common cutoff |
| Positives / negatives | one randomly selected positive per query; uniform catalog negatives without positive rejection |
| Source batching | TimestampSampler drops each timestamp group's incomplete batch; source stops after2001steps if reached |
| Selection | latest tied validation MAP maximum (`>=`), test not used to choose weights |
| Evaluation | full-catalog dot products and top10, no sampled-candidate shortcut; fresh neighborhoods for final val/test |
| Repeats | seeds0–4; one fixed recipe, no HPO |
| Audits | all sampled dated nodes and edges checked against owner cutoff/global identity; first32training batches receive positive-collision diagnostics |

Evaluation source uses fixed split timestamps; independent task evidence checks they equal all corresponding query cutoffs. Benchmark populations omit no-positive queries. Timeless features have no measured arrival history; temporal audit is not a historical availability guarantee. Source numeric encoder gradient behavior is recorded, not repaired inside reproduction. Caches are fresh materializations; same prepared feature graph is reused across training seeds with seed42preprocessing, which is a disclosed reconstruction choice.

## Instrumentation and pilot boundary

Full trainer instrumenter changes only model injection to the identical visible class, cached graph materialization, progress display, passive audits and artifact recording. Source objective, sampling, optimizer, epoch loop, tied selection and ranking remain intact. Pilot additionally uses one epoch, stops after32training batches and does not execute test; it performs complete validation. Pilot weights/rankings do not count as a completed experiment.

Save executed source, all final ranking arrays with keys, traces, selected checkpoints and hashes. Checkpoints stay on Modal volume `l153-recommendation-evidence`; small evidence is collected under `labs/evidence/l153`. Output directories and attempt markers are single-use; no implicit resume/retry. A fresh attempt requires a new reservation. Source mutations fail the budget freeze guard rather than silently mixing evidence.

## Cost gate

Aggregate capUSD10 including preparation,pilot,seeds,retries,validation and overhead. USD3reserved for overhead/contingency. T4+2physicalCPU+32GiB=.00026124USD/sec at https://modal.com/pricing (checked2026-10-01). Prepare and pilot each reserve1800s before dispatch; full fits each require4200s allocations and a passed forecast. Full five-seed work plus notebook checking must fit the remaining cap. No automatic retries. Reservations are ceilings; worker-body estimates exclude unitemized startup/build/storage and are not an invoice.

Pilot extrapolation:32batch training time → all released timestamp batches ×20epochs;22full-validation-equivalent passes; measured setup; five seeds. Apply1.25safety and120sperfit; reserve600sfor notebook validation. This conservative first-batch scenario includes diagnostic/cold-start overhead, not a mathematical lower bound or guaranteed bill. Stop if it exceeds aggregate remaining allowance or per-worker timeout. No reduction in seeds,epochs,data,candidates or required evidence is called full reproduction.

```bash
# From repository root; each paid attempt is reserved before launch.
.venv/bin/modal run --detach modal/l153_repro.py --phase prepare
.venv/bin/python labs/_collect_l153.py
.venv/bin/modal run --detach modal/l153_repro.py --phase pilot
.venv/bin/python labs/_collect_l153.py
.venv/bin/python labs/_analyze_l153.py
# Refuses if measured decision is STOP, or sources/reservations fail:
.venv/bin/modal run --detach modal/l153_repro.py --phase full
```

The portable notebook has an explicit fresh five-seed gate, with original model/trainer/preparation/loader code visible. Default execution rescoring author evidence is separate from own fresh training. A larger manual run needs acknowledged resource use; it cannot bypass the managedUSD10dispatch guard. Full source/protocol remains available when execution is INCOMPLETE.

## Verification

`_check_l153.py`: query-key alignment, multi-positive ranking denominators, candidate uniqueness/ranges, negative collision versus observed arrival, complete same-protocol seed evidence. `_source_check_l153.py`: original bytes/classes and instrumented selection/objective/schedule. `_mechanism_l153.py`: synthetic full neural output/gradient source parity and learned embedding update. Source-SQL and independent join audits run before training. `_analyze_l153.py` independently recomputes saved metrics with a second vectorized implementation and checks all query identities. Browser/notebook/publication checks are recorded separately, never promoted to benchmark completion.

Final measured evidence, decision and exact reproduction status are in `evidence/l153/summary.json` and `cost-decision.json`; final delivery report `_verify_l153_results.json`. Learner PENDING_WRITTEN_DEFENSE; historical identity and feature-arrival legality NOT_ESTABLISHED; whole paper/fresh manual-FE NOT_RUN; liveColab/deployment NOT_CHECKED unless explicitly verified. No publication requested.

## Measured outcome and delivery

STOP. Pilot32training batches took38.187561s; full validation20.110557s. Released full epochs have1299batches(665088query occurrences);4222timestamp-tail rows are dropped per epoch. The five-run linear scenario isUSD41.232549beforeoverhead, safety-adjustedUSD51.854174including notebook allowance; remaining compute allowance at decisionUSD6.059536. No full fits dispatched. Final pilot validationMAP.010508151,Hit.070399697,Recall.056482860; testNOT_RUN. The selection passMAP.007062707differs because final validation resamples neighborhoods.

Independent reconstruction733741labels, rescoring37003rankings, temporal audit450batches/229640roots/4503923datednodes/6783360edges:PASS. Shared negative collisions356/8388608; nonfinite real gradients384; source forward maxabsolute2.384e-7. All20default cells pass locally and in pinned runtime. First notebook attempt failed before cell0due missingIPython; separately recorded retry passed. Full notebook training gateNOT_RUN. Browser/mobile/keyboard/reset/noJS/print/deterministic regeneration and40copiedPages linksPASS.

Final reservations plus overheadUSD4.253952; measured worker-body estimateUSD0.289883,invoiceNOT_ITEMIZED. Clean Git-index publication verification is recorded separately.

Reporting correction (2026-10-08): the declared inclusive ±2 percentage-point tolerance now accepts both exact endpoints despite binary floating-point rounding. The original module and default-notebook code are archived in `sources/l153/*before_boundary*`; `_boundary_provenance_l153.py` verifies that only reporting, its live check and the embedded copy changed. Revised default cells execute locally; the historical pinned default execution remains separately identified. No full fits were run and no final paper verdict was added.
