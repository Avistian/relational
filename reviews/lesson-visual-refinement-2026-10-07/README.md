# Visual quality refinement — 2026-10-07

The user challenged the previous pass's quality and approved refining the weak drawings, integrating them with the lessons and checking actual mobile reading. This continues the existing authorization to push.

## Teaching changes

The reference was the rendered L043 TabNet architecture: its distinction is the named state, actual internal operations and return paths, rather than decorative styling alone. The earlier second-pass CAST and TACO drawings still hid their important computation, and the default mobile view exposed only a horizontal fragment.

- **All 20 second-pass lessons now have a purpose-drawn narrow layout.** A `picture` source chooses a 320-unit drawing on narrow screens. The reader follows the full route vertically without sideways panning. These are separate diagrams, not crops or shrunken desktop exports. Text has a minimum 16-unit size in the portrait assets.
- **CAST (B09)** now exposes column-summary cross-attention and its residual route, feature mixing with row summaries, row-summary cross-attention, and the different support-update/query-read paths. The repeated layer and quantile output complete the route. The drawing keeps SSMax numerical weighting outside the teaching schematic.
- **TACO (B18a)** carries the same two-feature-plus-target example through support/dummy inputs and latent cells, shows row/feature mixing and the residual MLP skip, then attaches query cells and the head. Captioned distinctions cover paper latent shape, grouped-feature release and frozen inference versus joint pretraining.
- **Forecasting (B19b)** now reaches through the frozen predictor to target-bin probabilities and horizon-wise distribution readouts. The historical fit boundary, grouped feature/target slots, row attention and absence of autoregressive target feedback are visible.
- These three main lessons present one primary architecture. Their previous maps remain available inside native reference disclosure. Original lesson text, widgets and evidence are preserved.
- Their student/solution notebooks and rendered notebook HTML now embed a two-column architecture export suited to notebook width. Builders use the new PNGs. Each of the six notebooks changes only one Markdown image cell; all code cells, saved outputs and metadata match the previous commit. No model or notebook execution result was regenerated.

Primary readings: [EXAONE §2](https://arxiv.org/html/2608.25774v1#S2), [TACO §3](https://arxiv.org/html/2602.05649v2#S3), [TabPFN-TS §3](https://arxiv.org/html/2501.02945v4#S3). The existing lesson source contracts supplied the other worked examples.

## Review observations and corrections

All 20 portrait drawings were visually inspected, along with the three complete model architectures and representative full lesson/mobile/print views. Review caught caption and heading overflow, a crowded attention label, TACO's inconsistent feature count, a reversed graph-attention direction, and column headings whose whitespace collapsed. Those were corrected before publication. Source-family links are undirected; graph-message directions remain explicit. The mobile default now keeps both sides of a residual sum or information boundary visible in the same drawing.

The mobile model architectures are deliberately taller: six illustrated phases in reading order. Detailed desktop exports remain available through the existing viewer. This trades vertical length for complete, readable operations. Other lessons' original wide diagrams and the earlier visual stories were not all redesigned in this refinement.

## Verification

- `python3 scripts/refresh_lesson_visuals.py --check`: stable generated HTML/SVGs.
- `.venv/bin/python scripts/check_visual_details.py`: responsive sources, stable unique IDs, six distinct model stages and optional reference disclosure.
- `.venv/bin/python scripts/check_lesson_visuals.py`: static navigation/source accessibility contracts across all 179 later lessons.
- `.venv/bin/python scripts/export_refined_notebook_figures.py`: portable PNG and notebook image consistency. `--update` changes image payloads only.
- `scripts/browser_check_visual_details.py`: 40 SVG geometry checks (desktop + portrait), 40 lesson visits, 20 desktop no-JS/print checks and 20 mobile no-JS checks. It also checks reference disclosure, enlargement, Escape/focus return and question disclosure. `local/browser.json` records no errors.
- A 16-lesson regression sample checks the existing visual-story viewer at both widths (32 visits); `regression/browser.json` records no errors.
- `notebook-preservation.json` records unchanged code/output cells and metadata. `notebook-rendering.json` records the three portable architectures displayed at approximately 773–853 CSS pixels, with inspected screenshots. This is local rendered-notebook verification; live Colab UI was not checked.
- Stripping generated visual regions and reference wrappers yields the same original content in all 20 lesson HTML files.

The clean-index Pages build and deployment/live results are recorded at delivery. Test counts support rendering and preservation claims; the operation-specific changes above are the teaching-quality evidence. This pass makes no course-wide quality certification and marks no learner outcomes complete.
