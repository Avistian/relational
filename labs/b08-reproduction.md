# B08 reproduction contract

Approved 2026-10-03. Main skill: trace visible evidence and separate target prediction from feature reconstruction.

## Frozen course run

`B08-OBJECTIVE-ABLATION`: 3 objectives (target, feature, sum) × seeds 0/1/2. Each fit sees the same seed-paired initial parameters, 120 batches of four episodes, support24/query8, four numeric features, one masked feature per query. Model: two dual-axis blocks, width16, four task slots, one attention head, feature reconstruction at block1 and target prediction at block2. AdamW lr0.001/weight decay0.01, no clipping, 120 fixed steps, no HPO/selection. CPU float32, one thread, deterministic algorithms. Paired train and independent test episode bytes are hashed before outcomes. Evaluation:16 episodes per seed, target MSE and masked-feature MSE, per-seed points and sample SD. A support-mean baseline is scored on identical cells. Query labels and masked feature truth are loss-only.

The synthetic generator uses a shared latent variable to create three correlated features plus one independent feature. Each episode has a fresh random target coefficient vector and a small interaction. Train/test use independent random streams of the same generator, not held-out mechanisms or real datasets. Complete generator, model and trainer are visible in the notebook. Feature-only leaves the final target head untrained. Combined loss sums two averages; gradient magnitude is not held constant.

```bash
.venv/bin/python labs/_budget_b08.py .venv/bin/python labs/_run_b08.py
.venv/bin/python labs/_budget_b08.py .venv/bin/python labs/_verify_b08.py
.venv/bin/python labs/_reproduce_b08.py --lane audit
.venv/bin/python labs/_reproduce_b08.py --lane paper
```

The existing run is immutable; `_run_b08.py` refuses overwrite. Use the portable solution for a fresh independent execution. The paper command deliberately exits2 with the source blockers. Budget: USD10 aggregate (8 commitments +2 reserve),3600 aggregate numerical seconds including failures and solution validation. No paid dispatch authorized beyond that cap. Source acquisition is USD0 and separately timed. Dependencies pinned in course-protocol.json; source/checkpoint separately identified in source-gate.json.

## Named paper target

[2509.03505v2 §7.3/Table23](https://arxiv.org/html/2509.03505v2#S7.SS3): LimiX-16M, Analcatdata BroadwayMult, masked-cell normalized RMSE0.194 at5% masking. Rounded target interval [0.1935,0.1945); closeness alone does not imply reproduction. Mean/mode0.321 is cited, not locally reproduced.

**INCOMPLETE_SOURCE_PROTOCOL.** Current and publication-era snapshots do not authenticate this result's data version, train/test row IDs,5% cell masks, repetitions, typing/scaler state or historical checkpoint. Both imputation demos use a different dataset and30% masking. Do not use that demo to claim the selected result. The current BCCO-CLS release does include BroadwayMult train/test candidate files; these are pinned under sources/b08/candidate-data. Its complete325-entry listing has no BroadwayMult5% mask file. These current split bytes are not authenticated as the historical Table23 packet. Current checkpoint metadata specifies SHA2560862b8aacc4669ead009ad418c83d33c143396b7aeec069e7d0d667aa70a168a,66493201bytes, revision3a7c18ac7876fa0f4418cc077abc12630c4fefbf. Weight bytes were not downloaded because the data/protocol gate already fails. Current repository commit516bf396333feb3198cf7aff8a6c10421f218e24; pre-v2-paper snapshot c6f677f8e884e86b7638e0cda978bbccf7b45e1a.

## Executable released inference lane

`_reproduce_b08.py --lane released --packet packet.json --checkpoint LimiX-1_16M.ckpt --output fresh-predictions.npz --device cuda` runs the full pinned release, never the small course model. Use an isolated environment with the pinned release's requirements/constraints. The release source, preprocessing and inference entry points are archived under sources/b08/release; publication-era model/encoder/layer/inference are also visible in the notebook appendix. This operator's input/hash guards and paper refusal are tested; full released inference is NOT_RUN, so runtime compatibility remains unverified.

Packet JSON requires file,sha256,dataset_version,split_provenance,mask_provenance,scaler_provenance,feature_types,row_identity_provenance. Its NPZ contains X_support,y_support,X_query,true_X_query,mask,row_ids. Numeric matrices must already use the declared fitted scaler. Hidden cells in X_query must be NaN. The scorer receives truth only after prediction. Recorded outputs include row IDs and masks. A supplied current packet is RELEASED_INFERENCE_ONLY unless historical evidence is independently established; no flag bypasses the historical gate.

## Deviations and unrun work

Course differs in data, size, precision path, normalizations, heads, mask distribution and schedule; it does not reproduce LimiX pretraining. Neither a course win nor source arithmetic proves a LimiX accuracy gain. Table23 target inference, remaining Table23 datasets, full LimiX-2 benchmarks and all full-scale pretraining remain NOT_RUN. Original pretraining trainer/data are not reconstructed. Current terms differ from the old paper's Apache claim: inspect archived release/code and checkpoint licenses separately. No inference about causal identification follows from feature attention.

Author verification is separate from PENDING_WRITTEN_DEFENSE. No deployment or live Colab execution is implied.
