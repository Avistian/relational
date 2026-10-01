# Lesson164 · Griffin evidence

| Evidence | Observed result | Permitted conclusion |
|---|---|---|
| Independent operator checks |96attention +96relation +251temporal cases;3wrong implementations rejected | Mechanism contracts pass |
| Released model parity |43parameter tensors; output max error 2.44e-15; gradient max error 1.6e-14 | Four-layer float64 eval compatibility |
| Real512-wide checkpoint |2real queries; logit max error 6.1e-05 | Checkpoint-compatible forward, not a test metric |
| Data audit |12,679unique query keys;1,494sampled temporal edges | Scoped source audit; preprocessing/history unresolved |
| L4timing probe |512train queries/2steps;566validation queries;12.72GBpeak allocated GPU | Resource evidence only |
| Complete20fit transfer comparison |0full fits; final test NOT_RUN | INCOMPLETE_BUDGET_GATE |

Conservative200epoch schedule: raw computeUSD51.11; with25%marginUSD63.89, aboveUSD7fit/check allowance. Early stopping and16worker throughput are unmeasured. Stop; no smaller experiment substituted.

Budget: runtime reservationUSD0.51816 +overhead allowanceUSD0.35 =USD0.86816. Known19.345634-second worker-body estimateUSD0.006683; invoiceNOT_ITEMIZED. USD2reserve untouched. No further paid run dispatched. App: https://modal.com/apps/pszar92/main/ap-DDJPJcsQNe8P0h2f6FROl1 .

Synthetic worked values: taskA [1.462117157, 1.075765685], taskB [0.537882843, 2.924234315]; relation message atcutoff5 [4.0, 3.0], atcutoff7 [7.0, 11.0]. These are not racing predictions.

Published Table12: no-pretrain .6558/.7176 and Others-2 SFT .7098/.7275 at512/4096. No local reproduction score is placed beside these targets. Fresh full pretraining/whole benchmarkNOT_RUN; historical identity/transfer gainNOT_ESTABLISHED. Author preparation only; learnerPENDING_WRITTEN_DEFENSE. LiveColab/deploymentNOT_CHECKED.
