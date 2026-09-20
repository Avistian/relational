# Lessons 091–100: explanation and architecture revision

Scope confirmed by the user: 091–100. Quality references inspected: L040's evidence gates, L048's mechanism derivation, L060's controlled procedures and L070's research defense. Baseline commit: `657c4433aa608cacbeb1f002e5a3d670568c9d3b`.

## Delivered

- Ten canonical walkthroughs, approximately 6,600 words, integrated after retrieval and before the existing detailed lesson body.
- Eleven editable SVG architecture/workflow maps with portable PNG exports: R-GCN, HAN, HGT, survey taxonomy, metapath2vec, three-hop bipartite scorer, database conversion, BPR-style factorization, typed sampling, model comparison and the checkpoint.
- Explicit prior/next-lesson links, paper/release locators, additional numerical traces, prediction questions with revealable checks, and a shared sequence reference.
- All ten lesson builders integrate the walkthroughs into both notebooks and preserve each package's original student/solution preview convention.

## Validation

`labs/_check_depth_091_100.py --stage --browser` passed: ten packages; 486 local lesson links; all twenty notebooks preserve ordered code, execution counts and saved outputs against baseline; portable diagrams match canonical exports; numeric traces checked independently.

Copied the actual Pages build recipe into temporary staging. Browser checks passed at 1440px and 390px for all ten complete lesson pages: no page overflow, all images loaded, zero JavaScript errors, keyboard scrolling for wide diagrams. All eleven SVGs passed canvas and card text bounds. Ten notebook previews loaded their figures. Checkpoint batch-weighting interaction, print sizing and manifest navigation passed. Representative visual inspection included R-GCN, HAN, HGT, database conversion, BPR, comparison and mobile checkpoint views.

Representative builds L091/L094/L100 regenerate lesson, both notebooks and preview byte-identically; hashes are in `lessons-091-100-rebuild-check.json`. Existing L071 portable conversion was checked against the baseline helper and remains identical.

Visual review corrected overlong labels, separated SQL's independent oracle path from graph computation, made R-GCN's self contribution explicit, and distinguished computation from learned computation in the shared diagram legend.

## Evidence boundary

No model implementation, training result, source manifest, experiment record or mastery status changed. Existing saved outputs are preserved author evidence, not a new execution. Original execution-record notebook hashes identify the original executed artifacts; the baseline comparison establishes code/output preservation after the prose refresh. Full-paper and historical reproduction limitations remain as documented in each lesson. Live Colab was not checked. Publication is verified separately after pushing the final commit.
