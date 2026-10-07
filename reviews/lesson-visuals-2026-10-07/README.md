# Later-lesson visual revision

User approved the design and requested a push after completion. Scope: all 179 existing later lesson pages (numeric lessons 050 onward and the research bridge). 62 pictorial guides across 61 lessons; every page gets a visual reading route. Existing architecture/evidence figures get an original-asset link and a keyboard-accessible enlarged viewer. L196 is a writing/community lesson and receives no invented architecture.

The new guides show recognizable tables, token grids, neighborhood graphs, attention axes, parallel members, state updates and prediction routes. They are conceptual first readings; existing detailed diagrams retain exact variant/configuration detail. Each guide names its primary source, explains three stages, includes a prediction/reveal question, and exports to a standalone editable SVG. No paper image reproduction was needed.

Important distinctions retained:

- RDB-PFN: relational data generation and DFS feature construction precede transformer inference on a feature table; the released checkpoint is not a GNN over raw foreign keys.
- RelGT: local tokens and global centroids are separate attention inputs.
- Griffin: retained cells can be reread as graph/task state evolves.
- GELGT: GraphSAGE and temporal attention are parallel routes mixed by a gate.
- FlexTab: the feature stream excludes target labels; the target decoder reads them separately.
- GTAlign: target adaptation updates the graph encoder while the tabular predictor remains frozen. Its diagram does not introduce DFS into this method.

## Maintenance

Run `python3 scripts/refresh_lesson_visuals.py` after any individual lesson builder, then `python3 scripts/refresh_lesson_visuals.py --check`. Authored content is in `scripts/visual_story_specs.py`; geometric primitives are in `scripts/visual_story_drawings.py`. The finishing pass only replaces its marked regions and figure tools. CI checks that generated publication files are current.

## Verification

- `coverage.json`: per-page coverage and primary-source links. Intentional mechanism reuse is explicit in the source specifications.
- `browser.json`: desktop (1200px), mobile (375px), keyboard stages, native answer reveal, enlarged view, horizontal scrolling, Escape/focus return, image loading, figure anchors, document overflow, no-JS content and print answer visibility.
- `svg-geometry.json`: all 62 standalone exports checked for text escaping the SVG viewport.
- `check_lesson_visuals.py`: navigation targets, accessible SVG titles/descriptions, three-stage explanations, source links and local assets.
- Original prose and inline scripts compared against the pre-revision Git version after removing the generated regions: identical. Notebook code, results and training artifacts were not modified.
- Actual Pages build from the staged Git index: passed. Archived-source secret check: passed.
- Representative desktop/mobile screenshots are retained for visual inspection. They are previews, not model-result evidence.

The browser tests use the existing Python environment and Playwright. On this workstation Chromium's missing runtime libraries were extracted under `/tmp/relational-browser-libs`; no system installation is required. Set `LD_LIBRARY_PATH=/tmp/relational-browser-libs/root/usr/lib/aarch64-linux-gnu` when using that environment. Other machines need the normal Playwright browser dependencies.

Publication CI and live verification are recorded separately in `deployment.json` after the push.
