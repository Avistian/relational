# REL-06 builder source proof

Base: `503704f30fa28e9c8896302d1423f0fcca9b539c`.

The CLI now accepts `--check`, compares the complete parsed manifest (including
array order and metadata), and returns nonzero with a diagnostic for stale,
missing or invalid output without writing files. Normal generation uses the same
manifest builder and preserves its existing output format and metadata rules.

Frozen tests-only RED: `c0c0d69a`. The CLI assertion failed because the original
script rewrote the manifest on `--check`. The generator regression already passed.
The identical tests pass after the production change. Fixtures include formatting
and object-key variation, title/source drift, entry metadata, order, added/deleted
entries, top-level metadata, absent/invalid output and file bytes/mtime preservation.
Generator assertions cover publication/checkpoint overrides, bridge sort order,
lab fallback, output formatting and creation without an existing numeric manifest.

Command: `.venv/bin/python -B -m unittest tests.update_manifest.test_update_manifest -v`.
Raw external evidence and exact final SHA are recorded in:
`/home/avist/agent-stack/lead/.lead/tasks/w3-stdlib-finish/evidence/postrollback-rel06/`.
Doctor passed using the existing runtime. The real repository check reports stale
output and preserves the inspected manifest, lesson HTML and notebook bytes/mtimes.
No course content was regenerated to hide that baseline discrepancy.

Builder evidence is SOURCE_READY only. Separate verifier review and canonical
replay remain pending; this worktree parks before canonical replay. No push/merge.
