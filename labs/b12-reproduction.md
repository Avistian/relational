# B12 · Frozen reproduction contract

Approved2026-10-03: USD0 paid execution.3600 cumulative local execution seconds including preparation, failures, validation and delivery. `_budget_b12.py` reserves remaining time before running a command, accounts failures, and kills the process group on timeout. No cloud or paid API call is authorized. Larger funding requires a separately approved protocol.

## Complete course experiment

**Name:** B12 support reachability × label corruption × channel diagnostic.
TierC synthetic mechanism isolation. Seeds0/1/2; NumPy default_rng(PCG64), binary64. Each seed draws20×4 standard-normal row features,20×4 parent features, a4×4 projection `I + Normal(0,.12)`, then one permutation of8support IDs. First8rows are supports, last12queries. Labels equal the indicator that the first coordinate of `(x+.25parent)` is positive. Row embeddings `(x+.25parent)W` are fixed, untrained features. Support event times0…7, arrival2…9, horizon1, all query cutoffs10. Query identity `(entity,cutoff)` retained in every condition; truth is excluded from prediction inputs.

High reachability: query indexi reads support indices(i+j)%8 forj0…3; low reads none. Same8eligible support rows in both arms. This aligns available input budgets, not used label counts: relational high uses4, relational low0, dual uses8 unless labels hidden. Label modes: intact, one fixed-per-seed permutation over8rows (multiset preserved), or hidden (allunknown and masked). Unchanged parameters and all432predictions retained.

Attention: query/support embedding dot product /sqrt(4), masked softmax, weighted support labels; empty permission set returns declared .5prior. Dual prediction=.5×(relation prediction+all-support prediction). This equal fusion, fixed embedding and absence of pretraining are deliberate course departures, not recovered OpenRFM defaults. Three seeds ×2reachability ×3interventions ×2channels=36conditions. No optimizer, checkpoint selection, HPO, train/validation/test fit or model-ranking claim.

Report per-condition Brier and mean absolute change from intact. Summaries use mean and sampleSD over3fixture seeds, not confidence intervals. Independently verify all432complete keys and labels with scalar math, tolerance1e-12. Check query-label exclusion, late events/arrival/outcome windows, hidden/empty supports, label permutation, independent query batching, and unchanged parameter bytes. No claim that these tests prove historical availability or generalization.

## Exact local commands (repository root)

```sh
.venv/bin/python labs/_budget_b12.py .venv/bin/python labs/_test_b12.py
.venv/bin/python labs/_budget_b12.py .venv/bin/python labs/_prepare_b12.py
.venv/bin/python labs/_budget_b12.py .venv/bin/python labs/_reproduce_b12.py
.venv/bin/python labs/_budget_b12.py .venv/bin/python labs/_run_b12.py
.venv/bin/python labs/_budget_b12.py .venv/bin/python labs/_verify_b12.py
.venv/bin/python labs/_budget_b12.py .venv/bin/python labs/_build_b12.py
.venv/bin/python labs/_budget_b12.py .venv/bin/python labs/_execute_b12.py
.venv/bin/python labs/_budget_b12.py .venv/bin/python labs/_delivery_b12.py
```

Frozen `sources/b12/source-ledger.json` authenticates source bytes; `_prepare` verifies instead of overwriting an existing packet. Prediction generation refuses differing existing output. Reruns count against cumulative budget. `evidence/b12/artifact-manifest.json` authenticates delivered code/evidence; changed code is not an authenticated rerun until reviewed and resealed. Notebook has no repository import requirement and installs NumPy only if absent; author environment recorded separately. Live Colab NOT_CHECKED.

## Griffin · full selected target retained, NOT_RUN in B12

Paper arXiv2505.05568v1 Table12, Others-2 FULL SFT versus no-pretrain on rel-f1-driver-dnf.512/4096labels ×5subsetseeds42–46 ×2arms =20fits. Fixed model seed42; complete566validation and702test rows,11,411train rows. Source commit b9d0e1fa8d89dfb1cd8bd5976b71de8a3b515427; dataset e0c54ceada75317b06f11f8dcda7aa8304fbb593; checkpoint bd8c5be5130f34e7faa31099d0bd81d95d0aa995. Preserve200maxepochs,patience10,validationevery2epochs,batch256,lr3e-4,wd2e-4,D512,fourlayers,hop2,fanout20,fewshotfanout3,reverseedgeson,gateoff. Highest validation AUROC selects checkpoint; report five-seedmean/sampleSD perarm/size. Targets: .6558/.7176 scratch and .7098/.7275 Others-2; prior descriptive closeness±.02AUROC does not prove identity.

Complete original model, trainer, data/sampler code, license, wrapper, preparation and Modal operator are copied under `sources/b12/griffin`; their inherited source hashes are checked. See `sources/b12/griffin/l164-reproduction.md` for full provenance, compatibility changes and preprocessing/temporal gaps. Archive wrappers preserve original repository-relative paths and are not standalone entrypoints. Run their original locations from this repository:

```sh
# Preparation/operator retained; NOT executed by B12.
.venv/bin/python labs/_prepare_l164.py /tmp/l164-release
# Managed twenty-fit lane refuses at preserved STOP decision:
~/.local/bin/modal run modal/l164_repro.py --phase full
```

Historical timing scenario:raw$51.112833×1.25=$63.891041, above earlier$7fit/check allowance; plus earlier reservations/reserve projected$66.7592total. This is not a fresh quote, measured full cost or B12 charge. No silent downscale; full pretraining and benchmark NOT_RUN. Original data availability/label-SQL and historical protocol identity remain unresolved as stated in L164.

## OpenRFM · named target and missing protocol

arXiv2606.04320v1 Table5: six cumulative rows(a–f), Bin-AUC/MRR/R²;17numeric cells and oneN/A. This source extraction is not a model run. Table5 adds an ICL head, changes synthetic data, adds real data, prototype objective and ensembling. These rows do not isolate a single architecture change across allsteps. Model-side support corruption is discussed separately in Table2; B12's hidden-label diagnostic is not its RandFeat intervention.

Full reproduction needs an authenticated author commit, allsix matching checkpoints or complete pretraining generators/corpus, configurations and optimization schedules, seeds and held-out identities, exact head versions, task splits, context budgets/sampling/retrieval, ensemble draws, predictions and metric aggregation. B12 has not recovered that executable identity. Do not substitute T-Lab/OpenRFM: its README identifies an independent Kumo reproduction. Failed guessed author endpoints do not establish nonexistence. Exact full commands cannot honestly be supplied without those inputs; gate INCOMPLETE_SOURCE_PROTOCOL, training/inference NOT_RUN.

## KumoRFM-2 · external model boundary

arXiv2604.12596v1 main ICL evaluation (§4, Tables3/4/7/8). The report uses up to10k context examples from training and validation; optional fine-tuning is a distinct setting. Access to a modern service would not authenticate historical weights or support selection. Need immutable model/service version, permissions/pricing, complete task keys/database snapshots, context draws, PQL/label construction, selection and output predictions. The paper-linked repository endpoint returned404 in this bounded check; no claim of permanent nonexistence. No invented Kumo trainer or guessed API operator. Historical benchmark NOT_RUN, identity NOT_ESTABLISHED.

## Exit boundary

Complete course diagnostic and source audit are author evidence. Fresh Griffin/OpenRFM/Kumo benchmark execution and whole-paper reproduction NOT_RUN. Learner PENDING_WRITTEN_DEFENSE; no deployment requested. This lesson prepares an accessible-model decision for B23; it does not guarantee any candidate passes that future gate.
