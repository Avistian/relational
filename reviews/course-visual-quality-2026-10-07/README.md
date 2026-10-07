# Whole-course visual teaching review

Scope: all **228 published lessons**, including 001–049 and the Year 5/6 bridge. This extends the previously shipped visual stories and 20 detailed architecture revisions. It does not claim new experiments or measured learner mastery.

## What changed

- **42 illustrated computation traces**: a complete four-step reading route, a drawing of the decisive operation, a prediction question, its answer, and an explicit schematic/protocol boundary. Examples include ordered target statistics, TabNet masks, relation-specific averaging, composite messages and support-set prediction.
- **57 existing architecture graphs across 53 lessons** now have native narrow-screen layouts. Their node bodies and directed edges come from the existing architecture authoring sources, including branches and training paths. Number labels identify blocks; they do not invent a sequential execution order. Desktop and print retain the original figures; the inspection viewer remains available on phones.
- **33 early-lesson page overflows repaired**, including tables and widget grids. The missing SVG background token is defined. Early diagrams gain keyboard-accessible inspection controls, and the shared viewer can fit a whole figure or show full-size details.
- **67 tracked notebooks and 42 prepared HTML notebooks** include the portable PNG companion. Original notebook cells, code, saved outputs and metadata are structurally identical after removing the one new Markdown cell. Existing architecture exports remain available.
- All 228 lessons receive visual navigation and shared reading support. **105 lessons retain their existing explanation without a new mechanism drawing**; their per-lesson reasons are recorded, rather than assigning a fictional architecture to an evaluation, writing or checkpoint lesson.

## Evidence

[review-decisions.json](review-decisions.json) records every lesson, teaching focus, retained explanation and change. [content-preservation.json](content-preservation.json) checks original lesson content. [notebook-preservation.json](notebook-preservation.json) checks original notebook structures; [notebook-render.json](notebook-render.json) records four prepared-notebook browser samples.

The three `after/browser-*.json` range reports cover **456 desktop/phone visits** at 1200px and 375px, including resource loading, page overflow, diagram geometry, mobile map targets, native question keyboard controls, discovered visual widget controls and figure-viewer opening, fit mode, Escape and focus return. [fallbacks.json](fallbacks.json) covers no-JavaScript mobile reading and print visibility for every lesson. Representative actual screenshots are in `after/`.

All baseline mobile visual screenshots were screened in contact sheets; the new operation drawings were also inspected at their actual reading width. Representative full panels, native maps, printed panels and prepared notebooks were inspected individually. The graph checker compares each mobile node and outgoing edge to the authored graph, and PNG manifests bind portable exports to their rendering inputs. Screenshots and checks supplement editorial judgment; they are not a user study or proof of every possible widget state.

## Reproduce

From the repository root (Python environment with BeautifulSoup and Playwright for browser checks):

```sh
python3 scripts/refresh_lesson_visuals.py --check
.venv/bin/python scripts/check_lesson_visuals.py
.venv/bin/python scripts/check_visual_details.py
.venv/bin/python scripts/check_course_visuals.py
.venv/bin/python scripts/export_architecture_routes.py --check
.venv/bin/python scripts/browser_check_course_visuals.py --start 0 --stop 76
.venv/bin/python scripts/browser_check_course_visuals.py --start 76 --stop 152
.venv/bin/python scripts/browser_check_course_visuals.py --start 152 --stop 228
.venv/bin/python scripts/browser_check_course_fallbacks.py
```

When changing an authored graph, regenerate `assets/architecture-routes.json` with `scripts/export_architecture_routes.py`. When changing a computation panel, refresh lesson HTML, run `scripts/export_course_visuals.py`, then refresh again to carry the reviewed PNGs into notebooks. The normal freshness check rejects stale exports. It only updates tracked notebook files; unrelated drafts are excluded.

## Limits

Dense original plots and detailed architecture references still use full-size inspection where appropriate. Native phone layouts preserve graph dependencies but present them as linked blocks rather than reproducing the spatial arrangement of a wide graph. Portable notebook figures are static PNGs; the lesson HTML provides responsive text and native question disclosures. Colab's remote UI was not tested, and no notebooks or paper experiments were executed. Publication and live verification are recorded separately after deployment.

## Publication receipt

Published commit [`b416223b`](https://github.com/Avistian/relational/commit/b416223b3dcf4ed7bb5f0b5bca0e6f3f66eb7c03) passed [GitHub Pages run 37649765908](https://github.com/Avistian/relational/actions/runs/37649765908). [deployment.json](deployment.json) records build/deploy success. All **386 changed public files** match that commit byte for byte: [live-hashes.json](live-hashes.json).

The live browser checks cover all **228 lessons at 1200px and 375px**. Initial checks encountered transient resource failures; affected ranges and then L071 were rechecked, with **zero unresolved issues across all 456 lesson/width pairs**. Original observations, final range reports, targeted reloads and resource byte checks are retained under `live/`; [live-verification.json](live-verification.json) explains their reconciliation. Four prepared notebooks also passed live image/render checks.

The uploaded Pages artifact is **1,780,714,274 bytes** (1.78 GB decimal), about **15.15 MB larger** than the previous deployment. The existing large site remains a capacity concern; this pass did not restructure publication storage.
