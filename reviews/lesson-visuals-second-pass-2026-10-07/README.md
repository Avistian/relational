# Second visual improvement pass — 2026-10-07

Continuation of the approved visual design and the user's request to improve again and push. This pass adds **20 operation-level SVG supplements in 20 lessons**, adjacent to existing architecture/mechanism figures. The original detailed architectures, prose, source references, learner exercises and recorded experiment results remain in place. No training or paid inference was performed. Companion notebooks are unchanged.

The first pass provided 62 visual stories across 61 of the 179 later lessons. This pass addresses 20 additional lessons. The **118 lessons without first-pass stories** were reviewed through their content, figure captions and mechanism explanations; `review-decisions.json` records all decisions. The other 98 retain their existing visuals in this pass. Several entries explicitly identify possible refinements; this is not a claim that every remaining figure is ideal. All 20 new SVGs were rendered and visually inspected; representative lesson/mobile/print screenshots were also inspected. Content inventory review is not equivalent to screenshot inspection of every original figure.

## What changed and why

| Lessons | Previous limitation | New visual operation |
|---|---|---|
| 85, 86, 98 | Reader had to reconstruct matrix mixing, gather/scatter, or traversal direction from prose | Exact two-node mixing; source gather and destination sum; dependency expansion opposite to message computation |
| 104, 109, 123 | Several different meanings of temporal eligibility | Edge-time recursion; event/observation clocks; fixed root-owned cutoff, shown separately |
| 122, 133, 142 | Identity and aggregation details were hard to follow inside overview boxes | Foreign keys mapped to local rows; neighbor sum versus complete relation sum; bridge-path multiplicity |
| 146, 171, 172 | Information boundaries were mostly verbal | Propagation radius versus sampled tokens; transitive source-family quarantine; zero/missing/unknown states |
| 173, 174 | Dense encoder and policy overviews | Separate context pooling into a 96→64→32 encoder; 32→8→32 residual adapter with identity skip and trainability paths |
| 181, 186 | Different exclusions or clocks could be conflated | Query-target masking separate from future-row exclusion; source age separate from incorporation lag and response latency |
| B09, B12 | Summary tokens or adaptation routes required reading long cards | Table axes and summary banks; label-to-loss-to-weights versus label-to-context-to-prediction |
| B18a, B19b | Architecture overviews hid the nature of context | TACO latent cells and jointly pretrained modules; history-fitted time map, future queries and marginal forecasts without target feedback |

Sources and worked values were checked against the adjacent `lessons/content/` sources, including their pinned release/paper distinctions. Primary readings include [Houlsby adapters](https://proceedings.mlr.press/v97/houlsby19a.html), [EXAONE CAST](https://arxiv.org/html/2608.25774v1), [TabPFN-TS](https://arxiv.org/html/2501.02945v4), and [TACO](https://arxiv.org/html/2602.05649v2). The adapter is the course pooled-encoder variant; CAST counts describe the pinned release; TACO paper and grouped-feature release shapes are distinguished; forecasting pictures do not claim the notebook executed a pretrained predictor.

## Delivery and verification

- Regeneration: `python3 scripts/refresh_lesson_visuals.py --check`.
- Content/accessibility: `.venv/bin/python scripts/check_lesson_visuals.py` and `scripts/check_visual_details.py`.
- New drawings: `scripts/browser_check_visual_details.py` verifies all 20 SVGs, 40 desktop/mobile lesson visits, and 20 no-JS/print cases. Geometry detects out-of-bounds or overlapping text. The browser exercises horizontal keyboard scrolling, enlarged view, Escape/focus return and native answer disclosure.
- Broad regression: `scripts/browser_check_lesson_visuals.py --report-dir reviews/lesson-visuals-second-pass-2026-10-07/regression` checks all 179 later lessons.
- Screenshots revealed and prompted correction of desktop clipping, the hidden B09 mobile wrapper, and a temporal shading boundary. On mobile, labels retain their full size in a keyboard-accessible scrolling region; the plain-text trace remains readable without scrolling.
- Local result: all 179 lessons passed at 1200px and 375px (358 visits), with no recorded errors. All 20 edited lessons preserve their original content outside generated visual regions. The added SVG assets total about 120 KB.
- Publication uses the clean Git-index Pages build. Live verification is recorded after the push; local results alone do not establish publication.

Implementation lives in `scripts/visual_details.py`, with generated assets in `assets/visual-details/`. The existing finishing script regenerates and checks them, so rebuilding a lesson does not permanently discard its supplement. The shared figure viewer supplies enlargement; no new interaction framework was added.
