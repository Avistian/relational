Remove the unused lesson 58 solution-evidence funnel while preserving the lesson revision, atlas, authored route snapshot and publication consumers. The real read-only exporter and visual consumer checks remain active.

PR31's hosted runner first lacked Beautiful Soup, then exposed a missing CairoSVG import in the architecture exporter. Preserve the prior `beautifulsoup4==4.15.0` repair and add `CairoSVG==2.9.1` to the existing disposable-runner pip step. Existing pins, permissions and all checks remain intact.

- Source: `2e9d0aa91b792dbacdf90bde3ad824cd7c9d3dfe`; tests-only RED: `7a1181d4b4f579afa9dc5e380bb758c76870c514`.
- New frozen stdlib contract: genuine canonical RED (one assertion failure) → GREEN (zero failures), zero errors/skips; standalone file loading and discovery both pass. R1 is unchanged.
- Configured `/verify-project`: PASS on clean source, **74 tests**, no skips. All **5 visual consumer contracts** pass; the real exporter reports **56 authored graphs exported**.
- Source secret checks, generated-visual readonly consumer check, search index and actual Pages build from the Git index pass locally.
- [Evidence](docs/evidence/rel-02-ci-r2/rel-02-ci-r2/evidence.md), [manifest](docs/evidence/rel-02-ci-r2/rel-02-ci-r2/manifest.json), [RED](docs/evidence/rel-02-ci-r2/rel-02-ci-r2/red.log), [GREEN](docs/evidence/rel-02-ci-r2/rel-02-ci-r2/green.log), [verification proof](docs/evidence/rel-02-ci-r2/rel-02-ci-r2/verification.log).

[Hosted run 40191677588](https://github.com/Avistian/relational/actions/runs/40191677588) is historical failure input: missing `cairosvg`, 94 tests, failures=1, skipped=2. Current GitHub retrieval failed with `error connecting to api.github.com`; raw failure and access-limitation proof are retained. Local checks used the preexisting environment; disposable-runner pip resolution/native Cairo availability are not claimed as verified.

Pending: separate counted `/blast-radius` verifier and acceptance, parent update of existing PR31 without force after PASS, and genuine passing hosted CI on the exact resulting head before merge. This builder did not push or merge. Hosted CI for this repair: **NOT_RUN**.

Actual builder: `builder:rel-02-ci-r2`, pane `w1:p5P`. Full canonical logs, configured verification status/stdout/stderr, import audit, supplied hosted traceback and actual failed GitHub query are retained in `/home/avist/agent-stack/lead/.lead/tasks/rel-02-ci-r2/evidence/current-build`; the manifest binds their hashes. Committed raw excerpts are exact original output lines, including numeric canonical outcomes; the 74-test count comes from current configured verification stderr/status. The separate reviewer is explicitly pending.

The import audit finds CairoSVG as the readonly exporter's sole third-party requirement. Its default transitive dependencies come from pip; local metadata and `libcairo.so.2` lookup were inspected. Consumers also require the already-declared Beautiful Soup and nbformat; the refresh path uses stdlib. Deferred `nbconvert` is inside the writer-only `prepared()` function, beyond the real readonly consumer boundary. The initial overinclusive audit that flagged it is retained alongside the corrected audit. Hosted pip/native-library verification remains pending.
