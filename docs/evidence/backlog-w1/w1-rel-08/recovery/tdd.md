# REL-08 recovery TDD execution

The parent folder's blocked.json, evidence.md, verification.log and source-ready.md
are preserved historical diagnostics from the pre-repair launch, not current status.
The current source-ready handoff is written under the shared task's handoffs/w1-rel-08.

Installed skill executed: /home/avist/agent-stack/lead/.agents/skills/tdd/SKILL.md.
Actor builder:w1-rel-08, pane w1:pY. Tests use the public audit(root) interface.
The user-authorized issue requires all canonical tests committed before implementation;
this specific instruction controls over the skill's general one-test-at-a-time loop.

Command (both RED and GREEN, existing interpreter; no installation):
/home/avist/Projects/relational/.venv/bin/python -m unittest discover -s tests/notebooks -p 'test_*.py' -v

RED: 9f7f4c71; seven assertions failed, zero import errors. red.stdout.txt,
red.stderr.txt and red.exit retain raw output and status (1).
GREEN: green.stdout.txt, green.stderr.txt and green.exit retain raw output and
status (0): seven tests passed. The canonical test file is byte-identical to RED.
Disposable mutations cover malformed JSON, schema-invalid archived notebook,
deleted HTML, deleted solution, new unexplained missing solution, orphan HTML
and orphan solution. No original teaching data was edited.

Actual inventory: 410 notebooks, 221 students/templates, 222 top-level lab HTML,
148 teacher notebooks. All 410 received nbformat.validate without version conversion.
0 validation/pair errors; 464 emitted nbformat warnings are retained in GREEN stdout.
These include missing cell IDs; schema validation is not notebook execution.
Pair gaps: 1 student without HTML, 2 HTML without student, 73 students without
solution, 0 solutions without student. Exact paths and rationale for every allowed
gap appear in GREEN stdout; new gaps fail. The 72 absent teacher copies follow
labs/README.md's documented local/gitignored convention (.gitignore:6);
LAB-TEMPLATE is explicitly a template, foundation-sequence.html is explicitly a
materials directory linked from notebooks.html, and labs/_execute_l113.py:7,13
produces the extra L113 solution HTML. This checks schema/file pairing, not semantic
or cell-by-cell equivalence of HTML and notebook bodies.

Native declaration attempt returned Errno 17 File exists, as the lead's prior pin
already existed; it was preserved unchanged. The exploratory lead-verify --help
was rejected as unknown project and did not execute verification. Actual verification
is separately run and retained by lead-verify relational --worktree.
