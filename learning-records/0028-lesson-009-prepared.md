# Lesson 009 prepared — feature engineering

Published `lessons/0009-feature-engineering.html`, reference `reference/feature-engineering.html`, reusable `assets/feature-viz.js` (scatter that toggles a raw column vs the engineered ratio of differences — a shapeless cloud snaps into a clean line), and notebook lab `labs/0009-feature-engineering.ipynb` (blank fill-in cells, runs end-to-end verified).

**Single skill:** engineer features that encode structure a model cannot synthesize itself (ratios/differences, interactions, cyclical datetime, relational aggregations) — and keep fit-bearing features (target encoding) *inside the pipeline* so they never leak. Closes Q1's single-table-mechanics arc and is the conceptual setup for the whole thesis.

**Verified live (venv sklearn 1.9.0, 5-fold):**
- Ratio of differences \(y=(a-b)/(c-d)\): Ridge R² **0.644 → 0.998** with the ratio engineered in; HistGBDT **0.968 → 0.980** (trees locally approximate but still gain). Matches Heaton 2016: no common model synthesizes a ratio of differences alone.
- Cyclical datetime: HistGBDT on an hour-of-day signal **0.840 → 0.979** with sin/cos encoding.
- Target-encoding leakage: a pure-noise high-cardinality category mean-encoded on the full data scores CV acc **~0.76**, vs **~0.50** (chance) when `TargetEncoder` is fit per-fold inside a `Pipeline`. Textbook leak.

**Grounding:** Kanter & Veeramachaneni 2015 (DFS, primary — the algorithm behind Featuretools, the bridge manual FE → DFS → RDL); Heaton 2016 (which features models can/can't learn); Featuretools DFS docs; sklearn §6.3 preprocessing/TargetEncoder. No new deps.

**Thesis bridge:** the most powerful relational features are aggregations over child rows (count/mean/max), exactly what DFS automates by following foreign keys — and exactly what RDL learns end-to-end over the schema graph. The hand-engineered baseline this lesson teaches is the thing the thesis's RDL model must beat; building it well (PIT-correct aggregations, leak-free encoders) is what makes the comparison honest.

**Exit:** notebook prints raw-vs-engineered R² for the ratio + datetime demos and the leaky-vs-safe target-encoding gap, plus one-sentence takeaway; or "lab done." **Lesson 010 is the Q1 checkpoint** — assemble 001–009 into one reproducible sklearn baseline.
