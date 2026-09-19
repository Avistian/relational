# Lessons 049–070: complete solution maps and connected arguments

## What changed

All 22 lessons now introduce their central argument, link a topic-specific reading route, connect four conceptual seams, and explain what carries into the following lesson. The additions connect the mechanism to its motivation and evidence; they do not replace definitions, derivations, experiments or retrieval practice.

The 23 new diagrams replace the old primary architecture overviews (ExcelFormer and Trompt each receive their own). Layout follows the system: parallel feature routes, recurrent prompt state, retrieval and residual paths, shared ensemble members, temporal experiment lanes, nested selection boundaries, task pretraining versus inference, alternating attention axes and staged column/row/dataset compression. Evaluation lessons receive protocol maps rather than an invented neural architecture.

The diagrams label the actual pictured variant and include a trace question, explanatory answer, accessible component/connection transcript and vector download. Desktop figures use a wider canvas than prose. Narrow screens scroll the diagram at readable label size, with the text alternative and scrolling instruction placed first. Print removes the scrolling constraint. Existing numerical operator panels remain available as expandable worked examples.

The same story and portable PNG maps are present in all 22 student and 22 local solution notebooks. Prepared student HTML is rebuilt. Existing Python cells, execution counts and saved outputs are preserved exactly. Solution notebooks follow the repository's existing ignored-file policy; they are regenerated locally rather than added to version control.

The [solution atlas](../reference/solution-map-atlas.html) connects the sequence across four phases: predictor design, evaluation, learned inference, and adaptation/deployment limits.

## Canonical authoring and regeneration

- `labs/_solution_map_content.py`: individually authored graph topology, questions, scopes and prose.
- `labs/_build_solution_maps.py`: SVG/PNG export and idempotent lesson/notebook/prepared-HTML finishing pass.
- `assets/solution-maps.css`: scoped shared presentation; per-paper topology is not imposed by the stylesheet.
- `lessons/story/*.md`: inspectable story exports; edit the authoring module, then regenerate.
- Each `_build_l049.py` through `_build_l070.py` CLI ends with the finishing pass, so the established individual build commands retain the revision.

From the repository root:

```bash
.venv/bin/python labs/_build_solution_maps.py
.venv/bin/python labs/_check_solution_maps.py
.venv/bin/python labs/_check_solution_map_routes.py
.venv/bin/python labs/_browser_solution_maps.py
```

A lesson number restricts the finishing pass, e.g. `_build_solution_maps.py 66`. Authoring uses BeautifulSoup, nbformat, nbconvert and CairoSVG; browser checks use Playwright Chromium. These are authoring dependencies, not notebook runtime requirements. The old aggregate foundation builder can replace multiple lessons; run the finishing command after that older aggregate workflow.

## Verification and evidence boundaries

- Content checks cover 22 lessons, 23 maps, 88 authored transitions, valid reading-route anchors and 44 notebook code/output hashes.
- Geometry checks reject overlapping components and arrows passing through unrelated components. Real browser text bounds check label containment.
- Finishing-pass byte idempotence checked across all 22 HTML lessons and 44 notebooks. The original L052 builder was also exercised without writing delivery files: executable cells matched, and each transition appeared exactly once after the finishing pass.
- Chromium audit uses a copied tree assembled from the actual Pages workflow. Desktop 1100px and mobile 375px: lesson and prepared-notebook overflow, image loading, new links, JavaScript errors, range extrema and every select option. Keyboard disclosure/scrolling, a JavaScript-disabled atlas navigation path and print styling are checked.
- Browser results: `labs/_solution_maps_browser_results.json`. Baseline executable/output hashes: `labs/_solution_maps_baseline.json`.
- Visual review inspected the rendered TabR and TabICL maps, desktop integrated examples and the mobile temporal-protocol map. It caught and corrected text overflow, a crossing Trompt route, duplicated lane-label paint, unnecessary desktop diagram cropping, and ambiguous repetition labels in TabICL.

No fresh model training or scientific result was produced by this editorial revision. Existing paper/local/protocol evidence boundaries remain in place. Live Colab: **NOT_CHECKED**. Deployment: **NOT_RUN**.
