# Lesson 075 delivery design

Curriculum contract: PyTorch Frame row encoding over numeric, categorical, text, timestamp and embedding columns. The user requested lesson75; the existing curriculum supplies the topic and deliverable.

Use the tool/API exception in the lab-authoring skill: the actual framework is the learning target. A framework rewrite would teach an imitation API; a benchmark-only lesson would miss the planned row-encoder interface. The selected design uses release0.3.0, source identity checks, a five-type trace fixture and actual credit_g rows. Paper benchmark results stay NOT_RUN, rather than being relabeled as tensor-shape results.

Learning sequence: retrieval → semantic types → materialization → train-only fit scope → individual encoding operations → row readout → artifact export → exit explanation. Three live TODOs cover fit scope, type dispatch and independent numeric reconstruction. The first two drive the real-data path; the numeric reconstruction is compared against the configured library primitive.

Visual questions: where do the two different-width vector columns go; can a query value change a training mean; how do numeric value and missingness alter token coordinates; where does column interaction stop and relational message passing begin? Four portable figures and two arithmetic controls answer these directly. The closest prior visual is L046's tokenizer; this lesson adds fit-state interventions, explicit missing policy and parent-type grouping.

Validation: source byte identity, exact numeric oracle, missing/unseen values, gradients, row locality, DataFrame order, solution execution and canonical-definition parity; desktop/mobile native controls, inline images and print; copied Pages links. See labs/_verify_l075_results.json, _execution_l075_results.json, _browser_l075_results.json and _delivery_l075_results.json for observed status. Live Colab/deployment remain NOT_CHECKED. No learner completion inferred.
