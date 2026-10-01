# Lesson 159 prepared: masked objectives and the transfer boundary

Date: 2026-10-01. Author preparation, not learner mastery. Mission unchanged.

Skill: define a self-supervised target by semantic identity, corrupt before encoding, trace masked loss and frozen-decoder gradients, then distinguish reconstruction evidence from transfer evidence. This follows L158's evidence-bounded thesis synthesis and prepares L162; it does not waive L160's exit gates.

Primary reading: Vogel, Hilprecht and Binnig (2023), arXiv2305.15321v1. Corrected erroneous Zahradník attribution in curriculum/resource/year plans. Named wikiTables Table1 BART-table versus GNN comparison remains NOT_RUN: exact subset/splits, matching released implementation and training/decoding configuration were not located. Historical fidelity NOT_ESTABLISHED. The audit command fails closed and is explicitly not a benchmark trainer.

Approved local lane: three paired seeds, 120 synthetic four-row tables, 84/24/12 table split, three binary tasks, 80 codec updates followed by 80 graph updates per seed. Complete 480-update experiment, 216 held-out predictions, frozen codec unchanged. Independent NumPy forward/metric checks, selected-checkpoint validation and semantic-target counterfactuals pass. Three learner mutants rejected. Tiny mean-pooled codec is not BART; synthetic contextual redundancy and additional graph training prevent a real-database or architecture-only causal claim. Zero seed SD is conditional, not zero uncertainty. USD0 cloud spending.

Next learner action: implement mask_serializations, masked_cross_entropy and freeze_codec in the portable notebook; run the declared mechanism; write the 400–600-word vision brief with a falsifiable held-out-database experiment. Rubric: five axes, 0–2 each, target 8/10 with no zero after teacher review. Learner PENDING_WRITTEN_DEFENSE.

Spacing: tomorrow retrieve three target levels and frozen-versus-detached; in one week diagnose a new schema-copy leak. Detailed notebook and browser receipts are in labs/_execution_l159_results.json and labs/_delivery_l159_results.json. Live Colab and deployment NOT_CHECKED. No new learner mastery, publication or historical reproduction inferred.

Delivery completed: exact standalone notebook parity, 50 browser states, three portable figures, independent scientific checks, deterministic rebuild and clean Git-index Pages build PASS. Actual desktop/mobile/notebook screenshots inspected after fixing architecture label overlap. Files staged; only design committed.
