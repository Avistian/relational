# REL14 builder evidence

Software implementation ready for a separate verifier. Feature acceptance is
PENDING browser proof and independent review. Actor `builder:rel-14`; actual
pane `w1:p5X`. Inherited `LEAD_SELF_PANE=w1:p1` identifies the lead, not this
builder pane. Worktree `/tmp/lead-wt/build-rel14-20261025`.

REL13 base: `edebddae007db2ab6e956e9fa824bb9c5ce18fd8`.
Tests-only RED: `575b49b9bd335ce1bb64205ff6db8b5c0b22c4c9`.
Verified software source: `341875ebcad870ffec05b31b1893b01677ba923e`.
Later delivery commits contain evidence only. Final clean-delivery verification
will be retained under the external lead verification root and identified in
the builder handoff, avoiding a self-referential repository SHA.

## Change and interface

Home loads generated classic script `assets/search-index.js` before
`assets/search.js`. The data interface is `LessonSearchIndex`, an array consumed
by the existing search handler. Ordered blocking local scripts need no fetch,
modules, or full inline fallback. Missing/invalid data produces the existing
"Search index unavailable." message. Actual file:// and HTTP browser proof is
still pending.

Home is **8262 bytes**, down from 169499; below 20000. Inline CSS matches REL13
byte for byte. Course progress link is preserved. Search ranking/rendering and
`assets/search.js` remain unchanged. No required edits outside ownership were
found. Unowned historical `assets/search-index.json` remains untouched and is
unused by home; explicit `--out *.json` preserves the legacy JSON export.
The default generator and existing CI check validate the JS asset.

## Actual software proof

- `/tdd`: canonical frozen standalone regression RED/GREEN passed. RED has one
  assertion failure; GREEN passes without errors/skips. See [tdd.md](tdd.md),
  [RED](canonical-red.log), [GREEN](canonical-green.log) and the retained receipt.
- `/verify-project`: `bin/lead-verify relational --worktree
  /tmp/lead-wt/build-rel14-20261025` PASS, exit 0, dirty false, source
  `341875ebcad870ffec05b31b1893b01677ba923e`; 133 tests passed. Read
  [status](software-status.json), [stderr](software-stderr.txt) and
  [stdout](software-stdout.txt).
- `python3 scripts/search_index.py --check`: exit 0.
- `python3 -m unittest discover -s tests/search -v`: 22 tests pass, including
  lesson coverage, title/headings/glossary, ranking, multiword AND, case and
  diacritics, deterministic output, legacy JSON export and page wiring.
- Isolated default-path generator copy, using real lesson/glossary inputs:
  fresh asset exit 0; stale empty asset exit 1; missing asset exit 1 without
  recreating it. Production files were untouched by these counterexamples.
- [Focused log](focused-software.log) retains exact commands, exits, diagnostics,
  size, stylesheet comparison and selected-test digest.

The Node VM DOM adapter executes actual page scripts and input handler, checking
that "single-table assumption" produces a lesson link. This supplies software
proof, not Chrome interactions, captures, or URL-scheme proof.

## Pending independent proof

| Scheme | Width | Known-term interaction | Real screenshot |
| --- | --- | --- | --- |
| file:// | 1280 | NOT_RUN | PENDING |
| file:// | 390 | NOT_RUN | PENDING |
| HTTP | 1280 | NOT_RUN | PENDING |
| HTTP | 390 | NOT_RUN | PENDING |

Lead must launch the separate read-only verifier for actual interactions, four
screenshots and `/blast-radius`. No builder browser capture or independent
review is claimed. Hosted CI is NOT_RUN. No subagents/workers were spawned;
no push, PR creation, merge, installs, downloads, spend or permission changes.

Initial `lead-task check-skills rel-14` returned `SKILLS FAIL: missing
skills-used.jsonl`. Existing intake artifacts do not prove invocation. Builder
records cover only actual builder TDD and verification; intake and independent
review records remain the lead's responsibility. No task engine acceptance or
DONE claim is made.
