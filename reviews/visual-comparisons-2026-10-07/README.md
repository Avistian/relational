# Four operation-level visual comparisons

Follow-up to the whole-course visual review, authorized by the user's request to improve the remaining visual weaknesses.

| Lesson | Previous visual limitation | Added comparison |
|---|---|---|
| 065, query embeddings | The overview named the embedding readout but did not expose its tensor slice. | Highlight the target-token vectors in two query rows; distinguish selecting all query rows from selecting only the last row. |
| 083, GraphSAGE | The mean illustration did not expose the padded adjacency table's changed multiplicities. | True list [A,B] with values [2,8], mean 5; toy stored row [A,B,B], mean 6. The release's actual width remains explicitly 128. |
| 084, GAT | The existing trace showed one state; understanding the edge-removal intervention required remembering the prior state. | Aligned attention weights before/after removing sender 3, retaining scores and projected values; show the new denominator and output. |
| 085, over-smoothing | The collapse illustration did not contrast loss of distinction with uniform shrinking. | Original, rescaled and averaged scalar states with a scale-normalized gap; explicitly distinguish the toy metric from the general graph diagnostic. |

Each drawing is next to its mechanism, uses native responsive reading text, supplies a prediction question and print answer, and appears as a portable PNG in existing notebooks. Numeric values are generated from the illustrated arithmetic. These are explanatory examples, not measured model results. No paper experiment or notebook code was executed.

## Checks

- All visual widget controls discovered by the existing browser audit on the four affected pages were exercised at 1200px and 375px; zero errors. See `browser-64-65.json` and `browser-82-85.json`.
- Eight no-JavaScript desktop/mobile checks, four print samples and four prepared-notebook samples. Actual images were inspected; text remains inside its SVG viewBox and the page does not overflow.
- Seven notebooks retain all original code, outputs, cells and metadata after removing the added Markdown cell. See `checks.json`.
- Independent arithmetic checks cover normalization, weighted output, duplicate means and scale-invariance of the illustrated gap.
- Six existing static content/graph/export checks pass; whole-course generation is fresh (`changed: 0`). Portable-export hashes now also include the shared drawing primitive source.
- A clean Git-index Pages build and live deployment verification are recorded below when completed.

The new panels extend the course's existing 42 computation traces to 46. This is a targeted visual improvement, not a claim that every lesson's pedagogy has been validated with learners.

## Publication

The clean Git-index Pages build passed. Commit [`3e3defce`](https://github.com/Avistian/relational/commit/3e3defce39bede6288ba239c8fa40ac9138eaf43) deployed successfully in [run 37653620456](https://github.com/Avistian/relational/actions/runs/37653620456). All **20 changed public files** match the commit byte for byte (`live-hashes.json`), and all **eight live desktop/mobile visits** passed with zero errors (`live-browser-*.json`).

GitHub reported the existing artifact-size warning: **1,782,576,914 bytes**, above its stated 1 GB limit. Deployment succeeded; this follow-up adds 1,862,640 bytes. The existing site-size capacity concern remains separate from this visual revision.
