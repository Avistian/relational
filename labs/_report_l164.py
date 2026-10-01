"""Produce reader-facing evidence without converting timings into model scores."""
import json
from pathlib import Path
from _lesson_run_l164 import worked_report
from relkit.griffin_l164 import cell_attention,relation_pool,eligible_edges
P=Path(__file__).resolve().parent;E=P/'evidence/l164'
r=worked_report(cell_attention,relation_pool,eligible_edges,json.loads((E/'attention-trace.json').read_text()))
(E/'report.json').write_text(json.dumps(r,indent=2)+'\n')
c=json.loads((E/'cost-decision.json').read_text());v=json.loads((P/'_verify_l164_results.json').read_text())
s=f'''# Lesson164 · Griffin evidence

| Evidence | Observed result | Permitted conclusion |
|---|---|---|
| Independent operator checks |96attention +96relation +251temporal cases;3wrong implementations rejected | Mechanism contracts pass |
| Released model parity |43parameter tensors; output max error {v['source_parity']['output_max_abs']:.3g}; gradient max error {v['source_parity']['gradient_max_abs']:.3g} | Four-layer float64 eval compatibility |
| Real512-wide checkpoint |2real queries; logit max error {v['real_checkpoint_logit_max_abs']:.3g} | Checkpoint-compatible forward, not a test metric |
| Data audit |12,679unique query keys;1,494sampled temporal edges | Scoped source audit; preprocessing/history unresolved |
| L4timing probe |512train queries/2steps;566validation queries;12.72GBpeak allocated GPU | Resource evidence only |
| Complete20fit transfer comparison |0full fits; final test NOT_RUN | INCOMPLETE_BUDGET_GATE |

Conservative200epoch schedule: raw computeUSD{c['raw_compute_usd']:.2f}; with25%marginUSD{c['safety_adjusted_compute_usd']:.2f}, aboveUSD7fit/check allowance. Early stopping and16worker throughput are unmeasured. Stop; no smaller experiment substituted.

Budget: runtime reservationUSD0.51816 +overhead allowanceUSD0.35 =USD0.86816. Known19.345634-second worker-body estimateUSD{c['pilot_worker_body_usd']:.6f}; invoiceNOT_ITEMIZED. USD2reserve untouched. No further paid run dispatched. App: https://modal.com/apps/pszar92/main/ap-DDJPJcsQNe8P0h2f6FROl1 .

Synthetic worked values: taskA {r['task_A']}, taskB {r['task_B']}; relation message atcutoff5 {r['relation_messages']['5']}, atcutoff7 {r['relation_messages']['7']}. These are not racing predictions.

Published Table12: no-pretrain .6558/.7176 and Others-2 SFT .7098/.7275 at512/4096. No local reproduction score is placed beside these targets. Fresh full pretraining/whole benchmarkNOT_RUN; historical identity/transfer gainNOT_ESTABLISHED. Author preparation only; learnerPENDING_WRITTEN_DEFENSE. LiveColab/deploymentNOT_CHECKED.
'''
(E/'report.md').write_text(s);print(r)
