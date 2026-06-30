# Lesson 009 started — feature engineering

User began Lesson 009 after completing Lesson 008 (metrics: ROC vs PR & calibration; lab done). Prior signal: L008 lab surfaced `cal.fit()` returns the estimator not probabilities — watch for same `fit()`-returns-self pattern if target encoding appears. L006/L007/L008 pattern of naming one effect when several apply — nudge multi-factor takeaways on exit ticket.

**Retrieval check:** warm-up **3/3** (A: target encoding inside Pipeline per-fold; A: PR baseline = prevalence; A: PIT aggregate = child rows before cutoff). L005 pipeline discipline, L008 PR baseline, and L002 PIT all retained.

**Focus this session:** engineer features models cannot synthesize (ratio of differences, cyclical datetime) vs features that are fit-bearing (target encoding); stateless vs pipeline-bound; DFS as the bridge manual FE → RDL.

**Exit:** notebook `labs/0009-feature-engineering.ipynb` prints ratio R² raw→engineered, datetime R² raw→engineered, leaky vs leak-free target-encoding accuracy, plus one-sentence takeaway; or "lab done." Lesson 010 = Q1 reproducible-baseline checkpoint.
