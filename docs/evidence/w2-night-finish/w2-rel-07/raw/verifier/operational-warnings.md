# Operational notes outside frozen inventories

The initial relative evidence/w2-rel-07/READY.json lookup was absent in the source
worktree; the authoritative task evidence was read at its absolute task path.
The existing ignored .venv symlink was reused unchanged. No linker copy was made.
Some exploratory reads used absent conventional filenames (tests/test_check_links.py,
scripts/relmini.py, .github/workflows/ci.yml, optional lead-review/review skill paths);
the actual files were located and read. One broad RISK_FLAG search matched large
hash inventories and its output was truncated; no verdict depends on that output.
No inventories, original logs, historical failure records or shared ledgers were edited.
The external verifier driver completed exit 0 without a retry or assertion change.
