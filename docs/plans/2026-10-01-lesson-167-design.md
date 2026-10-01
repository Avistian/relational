# Lesson 167 — approved design and implementation plan

Approved by the user on 2026-10-01: an executable transfer audit connecting TabPFN v2 / TabICL to Lessons 164–166. Teach what carries over (representations and labeled-context prediction with frozen weights), what needs explicit database machinery (PK/FK paths, cutoff-safe joins, support-label availability), and what depends on the pretraining distribution. Do not assume optional Lessons 165b or 166b have been studied.

Reproduction: independently audit all saved results of arXiv 2603.03805v5 Table 9, rel-f1/driver-dnf, 512 context, seeds 0–9, RDBPFN / RDBPFN_single / TabICLv1.1. Verify file hashes against original receipts and a new frozen manifest, regenerate support indices from the documented seed algorithm, align complete (driverId,date) keys, recompute all 30 AUROCs and paired differences. Retain the complete fresh-inference lane in L166. This is a new audit of existing inference, not new inference or training. Historical identity NOT_ESTABLISHED; original DFS regeneration, fresh pretraining and whole paper NOT_RUN. TabPFN v2 is explained but not a measured arm.

Budget: USD0 additional paid compute; no cloud dispatch. Local verification single-threaded, each job capped at 600 seconds; halt and diagnose resource failures. USD10 standing aggregate cap remains unchanged. Do not authorize a paid extension implicitly.

Deliver short lesson with deeper optional lab, end-to-end comparison and worked cutoff trace, interactive intervention with fixed baseline, three meaningful learner TODOs, portable student/solution notebooks, reference, source/protocol/deviation ledger and transfer-map defense template. Preserve PENDING_WRITTEN_DEFENSE; author execution does not establish mastery.

Implementation sequence:
1. Pin primary sources and original evidence; implement behavioral tests before temporal/keyed audit logic.
2. Implement and independently verify full evidence audit and local intervention fixtures.
3. Build shared interactive component, figures, canonical prose, reference and portable notebooks with visible code.
4. Execute solution in an empty directory; test wrong learner implementations, hash/key/run corruptions and notebook source parity.
5. Check desktop/mobile controls, keyboard/reset, print/no-JS, figures, manifests and links; deterministic rebuild and actual Pages build from clean Git index.
6. Record exact results and limitations; stage intended delivery files only. No push or deployment requested.

The writing-plans skill is absent from the available catalog and searched skill roots. This file supplies its implementation-planning function. Existing staged Lessons 161–166 must remain intact.
