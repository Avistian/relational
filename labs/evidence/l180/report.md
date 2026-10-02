# L180 observed evidence

**Observed:** complete saved-context replay; public RT-v1 fine-tuning **NOT_RUN**; practical Q2 exit **INCOMPLETE**. The temporal gate fails and one reported full run exceeds the $10 ceiling. No new inference, training or cloud/API spend. Learner defense: **PENDING_WRITTEN_DEFENSE**.

{
  "contexts": 2106,
  "cell_slots": 2156544,
  "totals": {
    "future_cells": 385,
    "unmasked_query_targets": 0,
    "unavailable_labels": 0,
    "unknown_time_cells": 432050
  },
  "costs_gpu_only_usd": {
    "A100_40GB": "25.185600",
    "A100_80GB": "29.980800"
  },
  "decision": {
    "admission": "BLOCKED",
    "blockers": [
      "TEMPORAL",
      "TRAINING_HEALTH",
      "CHECKPOINT_BYTES",
      "SELECTION",
      "FULL_POPULATION",
      "SOURCE_PROTOCOL",
      "BUDGET",
      "ALL_IN_COST_UNKNOWN"
    ],
    "practical_exit": "INCOMPLETE"
  }
}
