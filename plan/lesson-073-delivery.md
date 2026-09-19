# Lesson 073: When SSL actually helps

Authorized scope: create the planned lesson and companion lab locally. No publication or mastery claim.

Design: a paired label-budget experiment, extending L072's visible SCARF encoder. Primary comparison is SCARF fine-tuning versus the same network trained from scratch; frozen SCARF versus random features isolates representation transfer; raw logistic and fixed HistGB provide practical controls. Five nested label-blind budgets, three real offline datasets, three split/model seeds. All arms share train/validation/test IDs and count validation labels. Hold the full unlabeled training pool fixed. Fixed pretraining/fine-tuning epochs; validation-only probe selection. No test-driven protocol changes.

Alternatives: frozen probes alone are cheaper but cannot test fine-tuning gains; a full published benchmark offers broader evidence but exceeds this tightly scoped evaluation lesson. Use six-arm local experiment and explicitly label original benchmark reproduction INCOMPARABLE. No new model; show experimental access boundaries and paired arithmetic rather than introducing architecture anew.

Deliver: self-contained lesson, reference, visible student/solution notebooks, runnable canonical implementation, saved predictions/config/hashes, four computation/result figures, interactive budget accounting, crossover arithmetic, course entries and reproducibility contract. Test nested subsets, keyed pairing, missing pairs, multiple/absent crossings before implementation. Execute solution and reconcile fresh results; browser desktop/mobile and copied Pages link checks. Broader run remains optional and NOT_RUN.
