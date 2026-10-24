# REL21 grouped-row evidence for human review

Status: **PACKAGED_FOR_HUMAN_REVIEW — acceptance pending**. This is a local evidence delivery by the packaging builder. It retains the separate verifier's source review; it is not a new review, configured-suite PASS, canonical acceptance, or DONE claim.

Reviewed source: `1a3752fbd2dbf6d25a2d75ac92c326cfd8626d7a`.
Baseline: `503704f30fa28e9c8896302d1423f0fcca9b539c`.
The exact delivery commits are bound in [local-pr-body.md](local-pr-body.md) and the external task handoff `evidence/piotr-review/READY.json`. The PR body is local; no GitHub PR was opened.

## Retained results and authority

| Layer | Result | Evidence |
| --- | --- | --- |
| Builder focused fixtures | PASS: 8 tests, exit 0, clean source | [Builder READY](rawproof/builder/READY.json), [raw status](rawproof/builder/focused-and-live.status.json) |
| Separate verifier code/data | PASS; RISK_FLAG NO; no confirmed defects or observed data gaps | [Verifier READY](rawproof/verifier/READY.json), [review](rawproof/verifier/blast-radius.txt) |
| Separate verifier focused fixtures | PASS: 8 tests, exit 0 | [status](rawproof/verifier/focused.status.json), [raw stderr](rawproof/verifier/focused.stderr.txt) |
| Independent inventory | 228 manifest pages; all 228 covered; no zero-row pages, planned disagreements, missing or unlisted HTML | [independent census](rawproof/verifier/independent-inventory.json), [live status](rawproof/verifier/live.status.json) |
| Independent grammar proof | PASS: 12 parser boundary checks and 6 CLI probes | [boundaries](rawproof/verifier/parser-boundaries.json), [probes](rawproof/verifier/independent-cli-probes.json), [proof output](rawproof/verifier/proof.stdout.txt) |
| Configured verify-project suite | NOT_RUN; exit_code null; configured discovery invokes prohibited model fitting | [verifier assessment](rawproof/verifier/configured-suite.NOT_RUN.json) |
| Canonical replay | NOT_RUN; expressly prohibited | [Verifier READY](rawproof/verifier/READY.json) |
| Packaging structural preflight | PASS: nonexecuting CI workflow structure only | [preflight record](structural-preflight.json) |
| Lead/human acceptance | PENDING; no merge or publication | External task handoff |

The separate verifier is `verifier:rel-21-grouped`, pane `w1:p30`, at the exact reviewed source SHA, with clean source. The builder is `builder:rel-21-grouped`, pane `w1:p3Z`. Their original statuses remain verbatim: the earlier builder record still says separate review pending; the later verifier record supplies that review and leaves acceptance pending lead. Packaging does not rewrite either historical state.

## What the source establishes

Explicit first-cell IDs, comma/slash groups and closed numeric ranges may intentionally cover related pages. Prose, incidental links and parent IDs do not cover suffix pages. Every manifest page needs a row, including unpublished entries. Planned rows contradict published pages even when another row covers them; planned absent/unpublished entries are consistent.

The reviewed related groups are 111–114 (curriculum line 302), 151–154 (line 376), and 192–194 (line 439). B19a/B19b have explicit rows at lines 495/496. Planned 179 at line 422 is absent from the manifest. The verifier manually assessed the three groups; the checker cannot prove semantic relatedness. This evidence establishes row coverage, not scientific reproduction, lesson quality or learner mastery. Grouped-row authority is retained in the verifier's review as a task-record representation of Piotr's morning decision3; it was not independently authenticated against a conversation service.

## Retained TDD history

RED: `06e82129ac40322461819cde82725cc0247c495c`, exit 1, eight assertion failures, no errors or skips. The checker was absent; RED establishes missing implementation rather than eight distinct pre-existing behavioral defects. [Raw RED](red.stderr.txt).

GREEN: `beda5136f63ea13c78634c03756b556ab4360413`, exit 0, the same eight tests pass. [Raw GREEN](green.stderr.txt). Tests remain byte-identical to RED, as confirmed in the retained [scope/history record](rawproof/verifier/scope-and-history.json). The [implementation declaration](implementation-declaration.json) and [source inventory](source-inventory.json) remain unchanged. No historical execution or canonical replay was performed during packaging.

The verifier's [first proof attempt](rawproof/verifier/proof-attempt1.txt) retains an evidence exit-file-format assertion error. The verifier corrected its external proof reader; this was neither a source defect nor a TDD RED. The successful proof and the proof script are preserved as raw files; packaging did not execute them.

## Provenance and immutability

[package-manifest.json](package-manifest.json) maps all 29 copied builder/verifier files to their exact absolute origins, byte counts and SHA-256 hashes. Both original SHA256SUMS maps remain verbatim in their rawproof directories. Packaging checked all 39 entries in those maps against copies or the original committed evidence. The package [SHA256SUMS.json](SHA256SUMS.json) additionally covers original evidence, copied READY records and all payload metadata, excluding itself and the later local PR body. The external READY binds both the payload hash map and local PR body.

Existing evidence files and all history are preserved byte-for-byte. The only repository additions are within this evidence directory; no source, course, workflow, configuration, gate or frozen inventory was edited. Existing ignored `.venv` was used only for stdlib packaging operations and the supported non-UI structural preflight. Operational handoff lives outside frozen inventories. No untracked linker copies were created.

## Exact acceptance gaps

1. Configured verify-project verification remains NOT_RUN because its discovery reaches model fitting. Focused PASS and structural preflight do not replace it. The installed skill assessed is `/home/avist/agent-stack/lead/.agents/skills/verify-project/SKILL.md`. Its required invocation cannot run within this task's training prohibition; command/profile/gates were preserved.
2. Canonical replay remains NOT_RUN and prohibited in this task. Any required canonical acceptance evidence remains absent.
3. Lead/human acceptance is pending. The retained separate review covers source `1a3752fb…`, not a fresh review of delivery documentation. No additional review loop was run.
4. No hosted CI or remote PR was run/created for this local delivery. Push and merge remain unperformed and unauthorized here.

No agents/workers spawned, installs, spend, downloads, model training, permission/profile/gate changes, or shared-ledger rewrites occurred. Human/lead resolution of these gaps is outside this packaging task.
