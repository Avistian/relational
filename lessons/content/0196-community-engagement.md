<p class="eyebrow">Year 5 · Lesson 196 · community engagement</p>

# Turn a finding into an answerable question

**Your win:** prepare one technical question that another researcher can run, answer, and help you investigate. About 15 minutes for the lesson; the external conversation has its own timescale.

[Lesson 195](0195-thesis-stress-test.html) narrowed our thesis to claims supported by evidence. It could not tell us which preprocessing revision the RDBLearn authors used. That is a good reason to ask the people who own the code. Your mission is to make relational ML evidence persuasive to skeptics; an explicit uncertainty is a useful starting point for collaboration.

<div class="repro-status"><strong>Author preparation:</strong> all four approved original-preprocessor cases reproduced. Model-score effect NOT_ESTABLISHED. Full RDBLearn reproduction remains INCOMPLETE_SOURCE_PREPROCESSING_GATE. Question DRAFT_ONLY; learner defense pending.</div>









A **support set** is the labeled set supplied to an in-context predictor. A **query** is an example to predict. A categorical encoder assigns integer codes to category names. If support has already been encoded, changing a name's code later can make the same name look different across support and query. That motivates a check; it does not measure what any particular predictor does with those codes.

## 2 · Start with the smallest complete observation

We fit the released full TabularPreprocessor on twelve rows: b,c,d repeated four times, plus numeric values 0–11. We cache the encoded support, transform known categories, transform one unseen category, and transform the known categories again. Every case gets a fresh fitted pipeline.

[[CASE_TABLE]]

Numeric controls stay [1,2,3] in every case. A separate fresh-pipeline check encodes query b alone and together with the unseen row. Its code is 0 alone; it is 1 alongside a or 0, and 0 alongside z or e. This shows dependence on another row in these synthetic query batches.

**Worked trace:** sorting b,c,d,a yields a,b,c,d. Now b has index 1, while the cached support still uses index 0. z sorts after the fitted categories, so their codes do not move. Keeping both controls prevents an overstatement such as “every unseen category changes known codes.”

Read the [exact original implementation](https://github.com/HKUSHXLab/rdblearn/blob/b5b03ebf8091547285a6e06cba53d2d1a40cb171/rdblearn/preprocessing.py), especially DynamicLabelEncoder.transform. The diagnostic uses the full original pipeline, including AutoGluon, rather than replacing it with the sorting explanation. [Executable protocol](../labs/l196-reproduction.md).

<details><summary>Predict, then reveal: has AUROC decreased?</summary><p>No AUROC was computed. The observation is a code-consistency counterexample. Task frequency, backend sensitivity and historical experiment impact remain unmeasured. A model might treat these integers differently; only an appropriate model comparison can establish an effect.</p></details>

## 3 · Send the question to its owner

| Question concerns | Start here | Why |
|---|---|---|
| RDBLearn preprocessing | [RDBLearn issues](https://github.com/HKUSHXLab/rdblearn/issues) | The implementation lives in that project |
| RelBench task or data contract | [RelBench community hub](https://huggingface.co/relbench) | Find the relevant dataset or Space community tab |
| PyG sampling or message passing | [PyG Q&A discussions](https://github.com/pyg-team/pytorch_geometric/discussions) | Framework-specific support and discussion |

These destinations were checked on 2026-10-02. The [RelBench README](https://github.com/stanford-star/relbench) now points to Hugging Face; the curriculum's older “mailing list” wording is a historical pointer. The [PyG forum](https://github.com/pyg-team/pytorch_geometric/discussions) has a Q&A category. This is a map of topic ownership, not a guarantee of response or posting access.

Before posting, search open and closed threads for DynamicLabelEncoder, unseen category and the release version. Inspect any existing fix and its commit. If it addresses the same question, use that thread. Preserve the old pinned result even if current main changes. Ask once in the most relevant venue, then allow time for a reply.

## 4 · Give a maintainer something they can answer

An effective question contains **goal → version → tiny reproducer → expected/observed result → one decision**. Explain the expectation as your understanding, so it can be corrected. Include unchanged controls, environment versions and the narrow evidence boundary. Avoid asking maintainers to audit your entire research thesis.

[[DRAFT]]

The [complete local draft](../labs/evidence/l196/question-draft.md) includes the minimal code and measured environment. It is ready for your review; no message has been sent. You can change its wording while preserving the evidence.

**Try a revision:** replace “This proves the paper is wrong” with a question that names the release, the observation, and what you need to learn. Compare your revision with the draft after writing it. Exact historical implementation provenance and a tested code fix would answer different parts of the uncertainty.

## 5 · A reply becomes evidence only after a check

A reply might identify an intended contract, a fixed revision, an environment difference, or a mistake in our reproduction. Record the actual link and wording, then translate it into a test. If a new commit is proposed, freeze it and rerun the same four cases alongside the original. A claim that historical results are unaffected needs historical execution evidence or a relevant comparison; a current fix alone cannot establish that.

<div id="community-feedback"></div>
<noscript><p>Feedback states: no thread → DRAFT_ONLY; posted without response → AWAITING_RESPONSE; response without a completed check → RESPONSE_UNVERIFIED; response plus documented check → FEEDBACK_CHECKED. These describe workflow, not whether a scientific claim is true.</p></noscript>

**Separate the reply from its implications.** Imagine a maintainer proposes a new commit. This hypothetical continuation has three distinct checks:

<table class="compact-trace" style="min-width:0;border-collapse:separate;border-spacing:3px"><thead><tr><th>New evidence</th><th>What it can settle</th><th>Still open</th></tr></thead><tbody><tr><td>Reply naming a commit</td><td>Which revision to test</td><td>Whether it fixes the reproducer</td></tr><tr><td>Fresh four-case check passes</td><td>The tested code-consistency failure</td><td>Benchmark score effect</td></tr><tr><td>Historical execution record</td><td>Which revision was actually used</td><td>Any remaining protocol differences</td></tr></tbody></table>

A workflow marked `FEEDBACK_CHECKED` records that a response was checked; it is not a verdict that every claim in the response is true. Keep the proposed revision, your test and the conclusion together, including a failed check if that is what happened.

**Transfer the trace.** Suppose the new code passes but the author cannot recover the historical revision. Write two sentences: the tested revision resolves this counterexample; historical paper-run identity remains unestablished. Do not combine them into “the paper has now been reproduced.”


The widget is practice, not your actual participation record. A checked reply may refute your initial interpretation. Log the evidence either way. With no reply, retain the uncertainty; silence is neither agreement nor disagreement.

## 6 · Practice, then make the real-world step

Open the [student notebook](../labs/0196-community-engagement.ipynb). Implement three small contracts: summarize the complete case set, route a question by ownership, and track feedback without inventing completion. The [solution](../labs/solutions/0196-community-engagement.ipynb) runs the original full diagnostic and displays its source. It also carries a portable offline replay of the measured observation; the two modes are explicitly separate.

Use the [question draft](../labs/evidence/l196/question-draft.md) and [response log](../labs/evidence/l196/response-log.md) to prepare your real thread. Posting is your external action, or a separately authorized agent action. The lesson package being complete does not mean that action has happened.

**Exit:** explain the code trace without looking, choose the owner, provide a real thread URL when posted, and document either a checked response or an honest awaiting-response state. Finish with: “I changed ___ because ___; I still cannot claim ___.” The community-engagement milestone remains incomplete until feedback is assessed; local exercises establish neither participation nor mastery.

**Return tomorrow:** reconstruct why a and z differ, then draft a similarly scoped question about a different uncertainty from Lesson 195. Ask the agent follow-up questions whenever the source behavior, community choice or evidence boundary is unclear.

**Next:** [Lesson 197](0197-year-5-essay.html) uses your checked findings and still-open questions in a landscape essay. You can write it while awaiting a reply; preserve that unanswered state in your argument.

[Quick reference](../reference/community-engagement.html) · [Reproduction protocol](../labs/l196-reproduction.md) · [Lesson 157: reproducible contribution](0157-open-source-contribution.html) · [Previous: thesis stress-test](0195-thesis-stress-test.html)
