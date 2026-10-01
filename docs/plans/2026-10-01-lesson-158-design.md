# Lesson 158 approved design and implementation plan

Approved in chat 2026-10-01. Named experiment: L158 Year 4 thesis evidence replay.
CPU-only; USD0 additional cloud spend. No new fits. Standing USD10 ceiling is not authorization to resume missing experiments.

Teach an evidence-bounded synthesis essay: performance, effort, validity and scope are different claims. Replay L151–153 via the L154 adapter, L155 matched FE comparison, and L156–157 temporal lanes. Preserve source and input hashes, complete seeds, keyed populations, validation selection and conditional uncertainty. Trace identical prediction bytes and repeated database/task populations. Older L137/L149 are conceptual callbacks, not additional numerical samples. L157b is absent.

Deliver HTML, reference, portable student/solution notebooks with visible replay code and three live claim-audit functions, generated report, essay template and rubric. Reuse course visual components and show an actual FE/RDL comparison with its conditional interval. Distinguish published RelBench claims from local data; no paper-wide, effort-saving, leak-free or mastery claim.

Steps: behavioral rejection tests; frozen manifest and read-only replay; lesson/reference and interactive claim audit; portable notebooks; independent metric oracle, full solution execution, deterministic rebuild, desktop/mobile/keyboard/noJS/print and link checks; manifest/course/docs updates; stage only lesson changes and check clean Git-index Pages build. Do not publish. Preserve pre-existing staged work.

## Implementation details
- `labs/_check_l158.py` first fails against stub functions, then checks invalid lineage, unsupported claims, and test-led selection rejection in `labs/relkit/synthesis_l158.py`.
- `labs/_freeze_l158.py` writes a once-only manifest. `labs/_replay_l158.py` validates it and composes existing L154/L155 adapters with a keyed L156/L157 replay. No input report is silently regenerated.
- `labs/_build_l158.py` generates `lessons/0158-year-4-synthesis.html`, reference, figures and portable notebooks from `lessons/content/0158-year-4-synthesis.md`; learner functions are injected into real replay.
- `labs/_verify_l158.py` independently scores metrics and checks corruptions. `labs/_execute_l158.py` runs the full solution in an empty directory. `labs/_delivery_l158.py` checks browser behavior and static links; `labs/_check_pages_checkout.py` checks the staged publication tree.
- Use current checkout because upstream inputs are staged, not committed. Continue directly under existing approval; no extra execution-choice gate. No external publication.
