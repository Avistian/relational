# Lesson 156 · Full temporal audit and fit-horizon correction

Approved 2026-10-01. Named experiment: **L156 rel-f1/driver-position — temporal audit of released RDL and manual-FE pipelines**. This audit unit reuses the complete selected model; it does not introduce a new architecture.

## Frozen reference and correction

RelBench commit `9aa346267c2e1c560bd92da07d6f4ad1ca2f0639`. Released user-study SQL commit `445bb7a3b1230f49f8e5890ae81754d3e365680f`. Database SHA256 `ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482`; task archive `775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e`. Source/requirements/text-model/wheel pins in `evidence/l156/source-manifest.json` and `sources/l117/`.

Reference: all7453/499/760train/val/test queries; seeds0–4;10full epochs;two128channel sum-GraphSAGE layers; relative time and typed encoders;uniform128/64fanouts;batch512;Adam.005;L1;first strict validation-MAE minimum;train2nd/98thpercentile prediction clipping. Table7 targetval3.193/test4.022,predeclared descriptive mean tolerance±.20.128/64released fanouts differ from the paper's summary128. Preprocessing seed42,run seeds0–4; no test-led tuning.

The released feature graph is materialized through2010-01-01. The stricter declared fit-horizon policy permits dated-table type inference,vocabularies,numerical/time stats only through2005-01-01. `correction-protocol.json` was frozen before corrected scores; all five complete corrected runs use that population and fixed transform on all graph rows. Static-table rows remain with unavailable creation/version histories. Type/vocabulary changes can change encoder dimensions; matched seed integers do not imply identical weights. This is a course correction for that policy,not Table7 parity or proof of deployment legality. A fitted training-time processor may use matured training history later than earlier training queries; this is not a rolling-origin retraining experiment.

`audit.json.preprocessing` is an inherited descriptive string from the shared released runner. For the corrected lane it is not the final fit-scope verdict: **`audit-l156.json.preprocessing` records the actual per-table fit rows and latest fit dates**, and is what the replay checks. The shared builder first constructs the released graph; dated tensor frames and stats are replaced with the horizon-fitted processor outputs before training.

## Audit and limitations

Independent labels include all8712values,complete eligible populations and label maturity. Regenerate released SQL and independently reconstruct443552output values. Result/standings joins share verified race event dates. Audit every owner cutoff at every yielded batch,original global node/edge identity,root/task/label identity,and query isolation. Strict SQL history and inclusive sampled-event rules remain distinct. Synthetic mutant tests cover batch-max leakage,wrong equality,late arrival,missing history,duplicate keys,window endpoints and false sign-off. Real first-batch root/time mutants are rejected on all three splits of each fit.

FE `pct_laps_completed` uses maxima over eligible joined query rows within each split. Its exact released reconstruction does not establish a cohort-independent deployment feature. The task cohort itself requires future participation. Scheduled race publication,static attributes,ingestion histories and historical archive identity remain NOT_ESTABLISHED. FE values and lineage are reaudited here; fresh FE model fitting remains the L155 result,not an L156 new fit. Other portfolio entries receive a pinned contract/coverage review only: classification fresh L156 audit NOT_RUN; recommendation remains INCOMPLETE/testNOT_RUN.

Every primary held-out prediction is keyed and independently scored; every selected epoch is checked against full validation history; outputs replay against the original Model. All20primary split/seed score arrays are retained. First-backward nonfinite gradients remain visible; this is not a clean optimization-health claim. Source parity,temporal policy,score closeness,artifact completion and learner mastery are separate.

## Commands

From repository root:

```bash
.venv/bin/python labs/_check_l156.py
.venv/bin/python labs/_preflight_l156.py
.venv/bin/python labs/_audit_fe_l156.py
# Author dispatch is single-use; existing reservations refuse accidental repeats.
.venv/bin/modal run --detach modal/l156_repro.py --phase pilot
.venv/bin/python labs/_collect_l156.py --seeds 0
.venv/bin/modal run --detach modal/l156_repro.py --phase remaining
.venv/bin/modal run --detach modal/l156_repro.py --phase correction
.venv/bin/python labs/_collect_l156.py
.venv/bin/python labs/_collect_l156.py --lane fit_horizon
.venv/bin/python labs/_freeze_l156.py
.venv/bin/python labs/_replay_l156.py
.venv/bin/python labs/_figures_l156.py
.venv/bin/python labs/_build_l156.py
.venv/bin/python labs/_execute_l156.py
.venv/bin/python labs/_delivery_l156.py
```

Author archive preflight expects cached L129 raw archives from its source manifest. For standalone fresh execution use the portable notebook: it embeds complete compact audit inputs and visible model/trainer/auditor/correction source,downloads hash-checked full data for the optional gate,and checks pinned dependencies. Default offline audit requires numpy/IPython only. Full gate uses Python3.11,Torch2.5.1/CUDA12.4,Frame0.2.3,PyG2.6.1,pyg-lib0.4.0,and all exact requirements in `requirements-l117-runtime.txt`; the Modal recipe installs them. The full gate runs five fits per lane,never substitutes a short smoke test,does not enforce dollar limits,and writes into a new output directory. Source code cells use standard IPython writefile magics in a temporary directory. Packet compression transports data and unchanged licensed reference code,not the visible learner tasks or trainer.

## Budget and result

USD10 aggregate cap. Twelve1800sT4+2physicalCPU+16GiBslots atUSD.00022572/s cost at mostUSD4.875552,plusUSD3overhead allowance,marginUSD2.124448. Current rates checked2026-10-01athttps://modal.com/pricing. Pilot is complete seed0; safety forecast1.25×63.65+120=199.56s<1800. Ten primary slots used; full notebook validation gets one additional slot. Failed attempts consume reservations; no automatic retries. Worker-body estimates are not an invoice and exclude unitemized overhead. Local audits use bounded CPU threads; cloudUSD0.

Primary reference: validation3.184205±.033168/test4.128265±.222239MAE; both descriptive CLOSE. Correction: validation3.203458±.042371/test4.200381±.269608. Correction−reference test+.072116; descriptive only. All12590held-out predictions rescored. The released strict policy verdict is FAIL; corrected historical-availability verdict NOT_ESTABLISHED. Neither is a universal leak-free sign-off.

Delivery checks and notebook status are recorded in `_execution_l156_results.json`,`_delivery_l156_results.json` and `_verify_l156_results.json`. Full notebook repeats are excluded from primary means. LiveColab/deploymentNOT_CHECKED; no publication requested. LearnerPENDING_WRITTEN_DEFENSE.

## Delivered verification

Standalone30-code-cell audit executed in an empty directory with exact primary-report parity. Isolated pinned GPU notebook ran ten additional full fits and checked12590predictions; its executable-code hash matches the final delivered notebook. All validation-run predictions and checkpoints were also collected and independently checked locally. Future numerical-row perturbation changes released fitted statistics but leaves corrected statistics unchanged. These ten validation fits are excluded from primary means.

Browser1200/mobile375,72interactive states,keyboard/reset,print,noJS,portable figures,source parity,manifest galleries,37copied-site links and deterministic rebuild PASS. Clean Git-index Pages build PASS. Final reservation plus overheadUSD7.469256; worker-body estimateUSD.231397(not an invoice). LiveColab/deploymentNOT_CHECKED. Earlier staged work preserved; only the approved design was committed.

Reporting correction (2026-10-08): exact inclusive score-tolerance endpoints now pass in the replay and embedded L152 summary helper. Original notebook/replay code is archived under `sources/l156/*before_boundary*`; the two expression-only changes are verified by `_boundary_provenance_l156.py`. The revised default CPU notebook is freshly executed; the 20 historical fits and original full notebook execution remain separately authenticated, with no retraining claim. All measured report values are unchanged.
