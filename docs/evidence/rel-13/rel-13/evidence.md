# REL13 builder evidence

Builder source ready; independent browser capture/review and parent acceptance remain pending. This is not lead-task accepted DONE.

The production change is one list item in the existing `nav.ref-links`: `<a href="progress.html">Course progress</a>`. It inherits the current reference navigation styling. No progress content/data, CSS, search implementation, or search assets changed.

Actor: `builder:rel-13`, actual worker pane `w1:p5T` (spawn record); inherited `LEAD_SELF_PANE=w1:p1` is the lead pane, not the worker pane. Worktree: `/tmp/lead-wt/build-rel13-headless-shell-20261025`. No browser drive or screenshots were performed by this builder. No local server was started.

## Frozen RED / GREEN

- Base: `1dd04fa41813328b404ee8c3a86081adfcac043d`.
- Test-only RED: `71026606d87663721c9a3e16d5703542bd3ff0ec`.
- One-line source fix / GREEN: `72002f56cf08cba06fdfde6fa8a22174e020173b`.
- Selected test: `tests/navigation/test_course_progress.py::CourseProgressNavigation.test_index_navigation_links_to_existing_course_progress_page`.
- Frozen test SHA256: `aa9c61e462cbfc9437ff1ac0becd7b51243f1a7b508cdcea58237aed171b0e72`.

The HTMLParser test requires an anchor in a nav element, the readable Course progress label, exact relative `progress.html` href, no explicit hidden markup on the anchor/ancestors, and an existing local destination. It checks static semantics; computed CSS visibility, layout and actual browser behavior require independent inspection. The selected file and test support are unchanged between RED and GREEN.

The same command produced one assertion failure for the missing link in RED (exit 1), then one passing test in GREEN (exit 0), with no errors/skips:

```sh
/home/avist/Projects/relational/.venv/bin/python -B -m unittest discover -s tests/navigation -p 'test_*.py' -v
```

Raw outputs: [RED](evidence/red.log), [GREEN](evidence/green.log), [exact TDD metadata](evidence/tdd.json). These are actual builder executions, not canonical replay receipts or invented LEAD_RESULT rows. Ordinary index.html production change needs no tests-helper implementation declaration.

## Current software verification

```sh
/home/avist/agent-stack/lead/bin/lead-verify relational --worktree /tmp/lead-wt/build-rel13-headless-shell-20261025
/home/avist/Projects/relational/.venv/bin/python -B scripts/search_index.py --check
```

Configured verification: PASS, exit 0, 130 tests, 57 seconds. Source head `72002f56cf08cba06fdfde6fa8a22174e020173b`, `dirty:true` because the owned evidence directory was untracked. Tracked source was clean. Receipt root is this worktree, not the main checkout. [Status](evidence/verify-status.json), [stdout](evidence/verify-stdout.txt), [stderr](evidence/verify-stderr.txt). Original durable directory: `/home/avist/agent-stack/lead/.lead/evidence/verify/relational/20261025T145444Z`.

Search check: PASS, exit 0, `search index up to date`. [Output](evidence/search-index.log). Suite stdout includes deliberate negative-fixture diagnostics (missing temporary data and hash mismatch); the actual suite verdict is PASS. This does not establish a new scientific reproduction or live experiment.

## Independent verifier handoff

The lead must launch the separate read-only verifier; this builder has spawned no agents/workers and has not performed /blast-radius.

1. Review source `72002f56cf08cba06fdfde6fa8a22174e020173b` and the final evidence-only delivery commit. Run /blast-radius and the verifier's required deterministic checks; record the actual actor/pane and invocation output.
2. Use only installed Chromium and a tracked local server. With `lead-evidence capture`, capture actual index desktop 1280x900 and mobile 390x900. Preserve both PNGs and the capture manifest/commands before the next capture: the tool writes fixed `desktop.png`, `mobile.png`, `manifest.json` and `evidence.md` names.
3. Capture actual progress desktop 1280x900 and mobile 390x900. Preserve four distinct images and both route manifests. Inspect all four images with the image viewer; verify the link in navigation (scroll to Reference as needed), navigation to the rendered progress page, mobile layout and search behavior. Initial-viewport screenshots alone may not show the Reference nav below the lesson list. Do not claim pixel visibility from this static test.
4. Update readable evidence and lowercase PR body with real screenshot links and actual independent review. Preserve the builder proof. Builder evidence.md may be overwritten by the capture tool, so retain this committed version/proof before capture.
5. Parent handles supported canonical TDD/environment/human receipts, publication/PR, actual final-head green CI and human merge. All are pending here. Do not infer acceptance from builder PASS.

Intake check passed. A task-wide `check-skills --complete` stopped at missing invocation `swarm`; existing sharpen/grill/spec/issues invocations were valid. No swarm was run because this builder is explicitly forbidden to spawn. This is a parent workflow gate, not an authorization to fabricate invocation history. Builder recorded only actual /tdd and /verify-project uses.
