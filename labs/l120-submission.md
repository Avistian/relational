# Year 3 exit submission

Status: PENDING_WRITTEN_DEFENSE

## Ownership and execution

- My implementation/commit:
- My graph drawing:
- Assistance/reference material used:
- My executed notebook and l120-task-report.json:
- Full-data evidence: [ ] independently executed [ ] audited author artifacts only
- Runtime, exact command, source/data hashes, seed IDs, checkpoint selection:
- Missing/NOT_RUN/NOT_CHECKED work:

## Hand-built graph and batch trace

Draw all people, events and merchants from the lesson. Label primary/foreign keys, row positions, typed reverse edges and both clocks. Trace person90 at days7 and8 with two hops. Show batched local edge indices, original n_id, root positions and target-query identities for two queries. Explain why context nodes may receive indirect gradients but no direct target loss.

## Full Fey 2024 reading ledger

Read the final ICML paper: https://proceedings.mlr.press/v235/fey24a.html . Complete every row, including appendices. Each claim needs a precise page/section and a distinction between proposal, proof, example and empirical evidence.

| Reading | Claim in my words + page | Evidence type | Limitation / connection to my pipeline |
|---|---|---|---|
| Abstract and §§1–2; Figures1–3 | | | |
| §§3.1–3.3; Figure4 | | | |
| §3.4; Appendices A–B | | | |
| §4; Figure5 | | | |
| §5; related work and cited lineage | | | |
| §6; Appendices C–D | | | |
| Remaining acknowledgments, impact/limitations and references | | | |

List any unread material explicitly. Reading this lesson does not count as reading the paper.

## Written defense (500–800 words plus the diagrams)

1. Trace one legal prediction from keys through graph extraction, typed encoders, both message rounds and scalar head.
2. Exhibit a late-arriving second/third-hop event and explain how your code excludes it. Contrast query time with label maturity at fit time.
3. Explain the small course model versus the pinned full-data model. Give full-data means, seed SD, checkpoint policy and exact evidence scope.
4. State the fanout, preprocessing, runtime and unavailable-history limitations. Explain why a close score is not historical or whole-paper identity.
5. Identify one fair future comparison that could falsify the proposed relational advantage.

## Rubric

Score0 (missing/incorrect),1 (correct with unresolved gaps/assistance),2 (independently correct and evidenced) for: REG, temporal validity, mini-batching, model/training explanation, reproduction evidence, complete reading/defense. Pass requires ≥10/12, no zero, complete reading, a working pipeline and no unresolved leakage or misleading evidence claim. Author-reference execution never automatically passes the learner.

## Spaced return

At1,7 and30days, redraw the legal graph and reconstruct the direct supervision mask without notes. Explain one new timestamp or batching perturbation before executing it.
