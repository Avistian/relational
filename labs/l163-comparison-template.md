# L163 encoder comparison and defense

Learner: PENDING_WRITTEN_DEFENSE. Author execution is not your mastery record.

| Question | Typed features | Frozen text encoder |
|---|---|---|
| Exact inputs; excluded fields | | |
| Fitted and frozen components | | |
| Representation shape | | |
| Train-only preprocessing policy | | |
| Selected alpha for each split | | |
| Test MAE for each split; mean and SD | | |
| Reordering / renaming effect with frozen head | | |
| Encoding time and feature cost | | |
| Limitation; next falsifiable experiment | | |

Before looking at outputs, predict the effect of reversing columns and explain your assumption about field identity.

Write 200–300 words defending your choice. Identify the numeric/additive target bias, one leakage control, fresh-encoding versus cached-replay evidence, and a missing requirement for historical reproduction. Propose a test of text semantics without claiming that we ran it.

Rubric, 0–2 each: correct computation; split/selection legality; causal interpretation; source/evidence boundaries; falsifiable next test. Pass requires at least8/10 and no zero, assessed by a human/teaching agent. Code execution does not score this defense.
