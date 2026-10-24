# REL-16 independent verifier review

Actor: separate read-only verifier, /tmp/lead-wt/night-v-rel-16. No subagents or workers.
Exact reviewed SHA: fa4521280ab256beb622bfa3e0c8ec66e2ef68b7.
RISK_FLAG: YES
Source/code verdict: CHANGES_REQUESTED (two P2 parser findings).
Current Pages file-set parity: PASS, 18,497 generated = 18,497 actual Bash/GNU cp destinations.
Acceptance: NOT_DONE. Review complete; no source fixes, push, merge, hosted CI or canonical replay performed by this verifier.

## Findings

1. **P2: Preserve quoted wildcard semantics or reject them.** scripts/pages_publish_list.py:993 strips quote information through shlex.split; line 1024 then glob-expands every operand. Changing `cp *.txt public/ 2>/dev/null || true` to `cp "*.txt" public/ 2>/dev/null || true` yields `--compare` exit 0/equal:true, but Bash copies no files when only a.txt and b.txt exist. Both workflows complete successfully; the parser reports two files for both. A future quote change can therefore remove published files without comparison detecting drift. Evidence: drift-probes.json case quoted_glob and drift-probes.py. This is outside the current workflow's operands, so the present 18,497 equality remains valid. Before acceptance, either implement quote-aware matching or reject quoted glob syntax with exit 2; add a fixture that fails on this real false equality.

2. **P2: Reject or model wildcard/root removals instead of silently accepting them.** scripts/pages_publish_list.py:1001-1006 treats rm operands as literal destination prefixes. `rm -rf public/*.txt` removes nothing in the simulated map, and `rm -rf public/` normalizes to prefix `public`, which also removes nothing. Both fixture workflows actually delete all copied files, but `--compare` against the copy-only workflow exits 0/equal:true. Evidence: drift-probes.json cases rm_glob and rm_root. Present workflow uses literal removal public/labs/sources/l103/original, whose semantics are covered and correct. Before acceptance, support these forms or explicitly reject them with exit 2, with regression fixtures.

These are confirmed correctness risks in drift detection, not hypothetical shell execution risks. The parser executes no shell; the proof's shell execution occurs only in disposable verifier fixtures or the independently audited copy-only publication inventory.

## What changes and why current publication is safe

The change introduces a standalone generator with a frozen compatibility plan and a bounded workflow parser. Manifest missing/omitted references remain separate diagnostics and do not expand the set. It does not integrate the generator into pages.yml. Consumer census found only the CLI fixtures referring to pages_publish_list outside the generator and evidence. The live deployment workflow therefore retains its prior behavior.

The decisive safety fact for today's workflow is destination membership equality with actual shell copy behavior, proved by running code (blast-radius evidence level 4). check.py reads the real workflow's Build site block independently of the generator, omits only its two HTML-byte version mutations, and executes its audited mkdir/cp/if/loop/rm commands into a disposable publication tree using GNU cp --reflink=auto. actual-files.json records 18,497 destinations; checks.json records zero differences and empty missing_sources, missing_references and omitted_references. Empty diagnostics establish only the manifest's direct slug/labPath coverage, not site-wide links or valid published bytes. No actual deployment/app was run (level 5 NOT_RUN).

Seven existing CLI fixtures PASS (fixtures.stderr.txt). Additional independent parser-probes.py confirms recursive hidden content, optional missing input and duplicate copy behavior, plus rejection of arbitrary command/copy flag/rename syntax. The false-equality findings are preserved in separate probes, not patched into source or used to manufacture TDD RED.

Git comparison from base 503704f3 proves changes are confined to scripts/pages_publish_list.py, tests/pages and docs/evidence/rel-16/rel-16. No workflow or course-content modifications. Implementation/tests are unchanged between GREEN 1ba8e647d4ef15eb573b8caa13033cdad5e3b9ff and reviewed head. Verifier checkout remained clean throughout; only task evidence outside the checkout was written.

## Retained TDD and verification provenance

Original tests-only RED 9e09a39b909a445b420ca4ff0fbcb5fca8c7aff1 adds only tests/pages/__init__.py and test_publish_list.py. Frozen SHA-256s match RED, GREEN and reviewed HEAD. Original/canonical RED records seven tests, six assertion failures, zero runner errors; one escape rejection assertion already passed when the script was absent. GREEN records seven passing tests. checks.json verifies frozen hashes and receipt artifact hashes. Retained canonical receipt binds implementation GREEN; copied canonical logs and builder-tdd.json hash exactly to the receipt. Canonical replay is RETAINED_PASS; verifier replay NOT_RUN. No new /tdd work occurred.

The committed verify-status.json is a historical builder PASS (74 tests, exit 0) at implementation GREEN with dirty:true, preserved as-is. The recovered original builder READY.json additionally points to /home/avist/agent-stack/lead/.lead/evidence/verify/relational/20261024T191535Z. Its actual status.json, stdout.txt and stderr.txt were opened and their recorded hashes verified. This retained /verify-project result is PASS, exit_code 0, source.head fa4521280ab256beb622bfa3e0c8ec66e2ef68b7, dirty:false; stderr reports 74 tests OK. The stdout `FAIL hash check: x: sha256 mismatch` is the expected negative fixture in tests/relmini/test_relmini.py:92-100, not a failing suite exit.

Current configured argv inspected: /home/avist/Projects/relational/.venv/bin/python -m unittest discover -s tests -p test_*.py, root /home/avist/Projects/relational, timeout 600. tests/relmini/test_relmini.py:68-78 calls baseline.fit_predict; labs/relmini/baseline.py:78 calls LogisticRegression model.fit. Under this task's no-model-training limit, a new full /verify-project run is NOT_RUN. No config/profile/gate was changed to bypass that limit. Focused Pages fixtures and independent inventory were executed with the existing ignored .venv only. Retained full-suite PASS is builder evidence, not a newly executed verifier suite or experimental result. Hosted CI NOT_RUN.

## Package and authority

The requested committed docs/evidence/rel-16/rel-16/evidence/builder/READY.json and report.md are absent from this exact tree. Committed README.md is the narrative report. Original builder worktree contains READY.json as an untracked file, intentionally written after commit to bind HEAD; report.md is absent there too. original-builder-READY.json preserves that recovered receipt, and retained-evidence.json confirms every artifact hash it lists. Actual evidence integrity passes; the advertised committed package is INCOMPLETE_AS_SPECIFIED. This is separate from the source findings and TDD result; no missing file was synthesized into the builder package.

Answered grill and herder-overnight-answer-note.md establish Herder's answer under Piotr's standing order circa 21:00 Europe/Warsaw on 2026-10-24. The ANSWER_BY Piotr field is not evidence of a personal self-answer or a new lead-guard delegated overnight window. Scope remains exact existing workflow parity; omitted references must not be added to publication.

First builder outside-check hardlink failure (cross-device) remains raw and unchanged. Skill-link warning is recorded separately in skill-link-warning.md. No source/course fixes, downloads, installs/spend, permissions changes, training, push or merge. Review requests belong to the lead/builder; this verifier does not start a repair or replay loop.
