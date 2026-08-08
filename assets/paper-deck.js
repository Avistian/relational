/**
 * Core-paper flashcard deck (spaced re-exposure to the curriculum's papers).
 *
 * Loaded as a plain <script> (file://-safe). Assigns window.PAPER_DECK = [ card, ... ].
 * Each card:
 *   id       stable string (Leitner state is keyed on it — never renumber)
 *   paper    short citation
 *   year     publication year
 *   lesson   lesson where it was assigned (used for spacing / filtering)
 *   front    recall prompt (answer from memory)
 *   back     the claim/answer to check yourself against
 *
 * Add a card when a lesson assigns a core (★) paper. One crisp claim per card — the difference
 * between "I read it" and "I can cite it in an argument".
 */
(function (global) {
  "use strict";

  global.PAPER_DECK = [
    {
      id: "fey2024", paper: "Fey et al. — Relational Deep Learning", year: 2024, lesson: 1,
      front: "Fey et al. 2024 — what does Relational Deep Learning propose, and what problem with the standard pipeline motivates it?",
      back: "Learn directly over the database's relational entity graph (rows = nodes, foreign keys = edges) with GNNs, instead of manually flattening tables into one design matrix. The flattening step is lossy, ad hoc, and leakage-prone — RDL learns the aggregations end-to-end. Basis of the RelBench benchmark."
    },
    {
      id: "fey2024-fecost", paper: "Fey et al. — Relational Deep Learning (§1–2, the FE cost)", year: 2024, lesson: 35,
      front: "Fey et al. 2024 §1–2 — what are the issues with manually joining+aggregating tables, and which one is the 'lossy' claim?",
      back: "Manual feature engineering is (1) slow/labour-intensive, (2) arbitrary/suboptimal, (3) explores only a tiny fraction of possible features, (4) LOSES fine-grain signal by aggregating into lower-granularity features (the aggregation collision — the load-bearing claim), and (5) goes obsolete under drift. Their fix: treat the DB as a relational entity graph (node per row, edge per PK/FK link) and learn over it end-to-end."
    },
    {
      id: "cawley2010", paper: "Cawley & Talbot — Overfitting in model selection", year: 2010, lesson: 4,
      front: "Cawley & Talbot 2010 — what bias do they warn about and what is the fix?",
      back: "Selecting hyperparameters on the same CV used to report performance overfits the model-selection step, so the reported score is optimistically biased. Fix: nested CV (inner loop selects, outer loop scores the whole procedure)."
    },
    {
      id: "vanbuuren", paper: "van Buuren — Flexible Imputation of Missing Data", year: 2018, lesson: 6,
      front: "van Buuren (FIMD) — the three missingness mechanisms and why they matter?",
      back: "MCAR (missing independent of everything), MAR (depends on observed X), MNAR (depends on the unobserved value itself). MAR is imputable from X; MNAR is not fully recoverable, so keep a missingness indicator. Mechanism dictates whether imputation is safe."
    },
    {
      id: "he2009", paper: "He & Garcia — Learning from Imbalanced Data", year: 2009, lesson: 7,
      front: "He & Garcia 2009 — the core problem with imbalanced data and the families of remedies?",
      back: "Accuracy is misleading under skew (the accuracy paradox); classifiers bias toward the majority. Remedies: resampling (over/under, SMOTE), cost-sensitive learning (class weights), and threshold/metric changes (PR-AUC over accuracy). Resampling must stay inside the CV fold."
    },
    {
      id: "saito2015", paper: "Saito & Rehmsmeier — PR vs ROC", year: 2015, lesson: 8,
      front: "Saito & Rehmsmeier 2015 — their headline claim about ROC vs PR under imbalance?",
      back: "On imbalanced data PR curves are more informative than ROC: ROC is optimistic because FPR has a true-negative-rich denominator. The PR no-skill baseline is the prevalence (not 0.5), so always report PR-AUC + prevalence for rare positives."
    },
    {
      id: "nm2005", paper: "Niculescu-Mizil & Caruana — Calibration", year: 2005, lesson: 8,
      front: "Niculescu-Mizil & Caruana 2005 — which models are miscalibrated and how do you fix them?",
      back: "Boosted trees and SVMs push probabilities toward 0/1 (sigmoid distortion); bagged trees/NNs less so; logistic regression is usually well-calibrated. Fix with Platt scaling (sigmoid) or isotonic regression, fit on held-out data."
    },
    {
      id: "kanter2015", paper: "Kanter & Veeramachaneni — Deep Feature Synthesis", year: 2015, lesson: 9,
      front: "Kanter & Veeramachaneni 2015 (DFS / Featuretools) — what does it automate?",
      back: "Automatic generation of relational features by stacking aggregation + transform primitives across related tables (the manual step RDL later learns end-to-end). The bridge from hand-crafted joins to learned relational representations."
    },
    {
      id: "breiman2001", paper: "Breiman — Random Forests", year: 2001, lesson: 12,
      front: "Breiman 2001 — the two randomness sources in a Random Forest and what they buy?",
      back: "Bootstrap sampling of rows (bagging) + a random feature subset at each split. Together they decorrelate the trees so averaging drives variance down (~1/B for B independent trees). OOB rows give a free test estimate."
    },
    {
      id: "friedman2001", paper: "Friedman — Greedy Function Approximation (GBM)", year: 2001, lesson: 13,
      front: "Friedman 2001 — what is gradient boosting in one sentence, plus the role of shrinkage?",
      back: "Stagewise additive modelling: each new tree fits the negative gradient (pseudo-residuals) of the loss w.r.t. the current predictions. Shrinkage (learning rate) scales each step down; more, smaller steps generalise better."
    },
    {
      id: "chen2016", paper: "Chen & Guestrin — XGBoost", year: 2016, lesson: 14,
      front: "Chen & Guestrin 2016 — the regularized objective, the leaf-weight formula, and one systems trick?",
      back: "Objective = loss + Ω(γT + ½λ‖w‖²). Optimal leaf weight w* = −G/(H+λ) using gradients G and hessians H; split gain is the structure-score drop minus a γ toll. Systems: sparsity-aware (default direction) missing handling, weighted quantile sketch, cache/block design."
    },
    {
      id: "ke2017", paper: "Ke et al. — LightGBM", year: 2017, lesson: 15,
      front: "Ke et al. 2017 — LightGBM's two named contributions and its growth strategy?",
      back: "GOSS (keep large gradients, sample the rest, amplify by (1−a)/b to stay unbiased) and EFB (bundle mutually-exclusive sparse features). Grows leaf-wise (best-first), so num_leaves is the capacity knob. The '20× faster' is vs pre-histogram GBDT; XGBoost-hist is on par."
    },
    {
      id: "prokhorenkova2018", paper: "Prokhorenkova et al. — CatBoost", year: 2018, lesson: 16,
      front: "Prokhorenkova et al. 2018 — the leakage CatBoost fixes and how?",
      back: "Prediction shift / target leakage from greedy target statistics. Fixes: ordered target statistics (encode a row from a permutation prefix + prior, never its own label) and ordered boosting (gradient from a model that never saw the row). Uses oblivious (symmetric) trees."
    },
    {
      id: "bergstra2012", paper: "Bergstra & Bengio — Random Search", year: 2012, lesson: 17,
      front: "Bergstra & Bengio 2012 — why does random search beat grid search at equal budget?",
      back: "Under low effective dimensionality (only a few knobs matter), a grid wastes budget on √n distinct values per axis, while random search gives ~n distinct values on the important axis. The advantage grows with the number of (mostly useless) dimensions."
    },
    {
      id: "wolpert1992", paper: "Wolpert — Stacked Generalization", year: 1992, lesson: 18,
      front: "Wolpert 1992 — what is stacked generalization and the one rule that makes it work?",
      back: "Train a meta-learner on the predictions of base learners. The rule: base predictions must be out-of-fold (a row predicted by models that didn't train on it), else the meta-learner crowns the memorizer. Diversity among bases is the fuel."
    },
    {
      id: "grinsztajn2022", paper: "Grinsztajn, Oyallon & Varoquaux — Why trees win", year: 2022, lesson: 19,
      front: "Grinsztajn et al. 2022 — the three inductive biases behind tree dominance on tabular data?",
      back: "(1) Trees fit irregular / non-smooth targets (MLPs are smoothness-biased); (2) trees are robust to uninformative features (implicit feature selection); (3) trees respect orientation / are not rotationally invariant (columns are individually meaningful). Edge is a medium-data, single-table phenomenon."
    },
    {
      id: "rubachev2024", paper: "Rubachev et al. — TabReD", year: 2024, lesson: 21,
      front: "Rubachev et al. 2024 (TabReD) — what do random splits do to tabular evaluation, and what is the fix?",
      back: "On real, temporally-evolving industrial data, random train/test splits give overly optimistic estimates and shuffle model rankings versus time-based splits (XGBoost's margin shrinks under correct evaluation). Fix: use time-based splits — but most public benchmarks lack the timestamp metadata to do so. TabReD contributes 8 industry datasets with time-based evaluation."
    },
    {
      id: "kapoor2022", paper: "Kapoor & Narayanan — Leakage & the Reproducibility Crisis", year: 2022, lesson: 22,
      front: "Kapoor & Narayanan 2022 — what did they find about leakage in ML-based science, and what is their taxonomy + fix?",
      back: "Data leakage is widespread: found across 17 fields / 329 papers, often producing wildly overoptimistic conclusions. They give a taxonomy of 8 leak types in 3 families (no clean train/test separation; illegitimate features; test set ≠ distribution of interest). In their civil-war reproduction, every claim that complex ML beats logistic regression failed once leakage was removed. Fix: a 'model info sheet' with a question per leak type, filled before publication."
    },
    {
      id: "grinsztajn2022-benchmark", paper: "Grinsztajn, Oyallon & Varoquaux — Why trees win (§3–4, benchmark)", year: 2022, lesson: 24,
      front: "Grinsztajn et al. 2022 (§3–4) — how is the benchmark built so 'trees beat DL' is a fair finding, not a lucky config?",
      back: "Two pillars. (§3) Explicit dataset-selection criteria applied to a large OpenML pool: real, tabular, medium-sized, heterogeneous columns (not pixels), not too high-dimensional, not too easy. (§4) Report a random-search BUDGET CURVE, not one tuned number: for each model plot the expected test score of the best-VALIDATION config after k iterations, averaged over draw orderings — showing both the default (k=1) and the tuned ceiling. Aggregate across datasets via affine per-dataset normalization (worst→0, best→1), since raw accuracies are incommensurable. Result: GBTs beat NN families at every budget and are stronger defaults."
    },
    {
      id: "grinsztajn2022-smoothness", paper: "Grinsztajn, Oyallon & Varoquaux — Why trees win (§5.2, smoothness)", year: 2022, lesson: 25,
      front: "Grinsztajn et al. 2022 (§5.2, Finding 1) — what is the first inductive bias behind tree dominance, and the experiment that proves it?",
      back: "Neural nets are biased toward overly SMOOTH (low-frequency) solutions (the spectral bias, Rahaman 2019), while tabular targets are typically irregular — so an MLP over-smooths them and a piecewise-constant tree follows the jags. Proof: Gaussian-smooth the TARGET at a growing length-scale and refit; the tree-vs-MLP gap collapses toward zero exactly as the high-frequency variance is erased (repro: gap +0.33 R² → ~0). The tree's edge WAS the irregularity, not raw model strength."
    },
    {
      id: "grinsztajn2022-rotation", paper: "Grinsztajn, Oyallon & Varoquaux — Why trees win (§5.4, rotation)", year: 2022, lesson: 26,
      front: "Grinsztajn et al. 2022 (§5.4, Finding 3) — what is the rotation bias, and the experiment that proves it?",
      back: "A learner is rotationally invariant (Ng 2004) if applying the same orthogonal Q to train+test features leaves its score unchanged. MLPs/ResNets are invariant (W·(Qx)=(WQ)·x); trees are NOT (axis-aligned splits are tied to the original basis), nor are FT-Transformers (pointwise tokenizer). Tabular columns carry individual meaning, so the original basis is privileged — invariance is a liability. Proof: a random rotation leaves the MLP unmoved but collapses trees, REVERSING the ranking (repro: tree 0.987→0.747, MLP 0.862→0.869). Ng's theorem links it to junk features: an invariant learner's sample complexity grows ≥ linearly in the number of uninformative features. Per-feature embeddings (SAINT, FT-Transformer) break the invariance and recover much of the gap."
    },
    {
      id: "grinsztajn2022-uninformative", paper: "Grinsztajn, Oyallon & Varoquaux — Why trees win (§5.3, uninformative features)", year: 2022, lesson: 27,
      front: "Grinsztajn et al. 2022 (§5.3, Finding 2) — what is the uninformative-features bias, and the experiment that proves it?",
      back: "MLP-like architectures are NOT robust to uninformative (junk) features; trees are. Mechanism: a tree grows greedily and only splits on the highest-GAIN feature, so a pure-noise column (≈0 gain) is never chosen — implicit feature selection. An MLP wires every feature into its first layer and (being rotationally invariant, Ng 2004) needs ≥ linearly more samples per junk feature, leaking capacity onto noise. Proof (two ablations): ADD junk → the tree–MLP gap widens (repro: on a smooth target the MLP wins clean 0.986 vs GBT 0.945, but 100 junk cols cost the MLP 0.084 vs the GBT 0.032, reversing the ranking); REMOVE junk → the gap shrinks (helps the MLP more). The gate is visible as root-split gain ~118× higher on informative than junk features."
    },
    {
      id: "demsar2006", paper: "Demšar — Statistical Comparisons of Classifiers", year: 2006, lesson: 23,
      front: "Demšar 2006 — how should you compare classifiers over multiple datasets, and why not the obvious tests?",
      back: "Don't average accuracies (incommensurable across datasets) or run parametric t-tests (normality unsafe). Instead rank the models per dataset and use non-parametric rank tests: Wilcoxon signed-rank for two classifiers, and the Friedman test + Nemenyi post-hoc for many, visualized with a critical-difference (CD) diagram (CD = q_α·√(k(k+1)/6N)). Two models are not significantly different if their average ranks are within CD. (Single-dataset CV folds are correlated, so use the corrected resampled t-test there — Nadeau & Bengio 2003 — not Wilcoxon.)"
    },
    {
      id: "gorishniy2021", paper: "Gorishniy et al. — Revisiting DL Models for Tabular Data", year: 2021, lesson: 28,
      front: "Gorishniy et al. 2021 — what are the two 'simple' baselines, and what is the paper's central (methodological) finding?",
      back: "Baselines: a tuned MLP (Dropout(ReLU(Linear)) blocks) and a ResNet — a stack of pre-activation residual blocks, ResNetBlock(x)=x+Dropout(Linear(Dropout(ReLU(Linear(BatchNorm(x)))))). The skip makes the identity map free, so depth stops degrading. Central finding: once you compare against a *properly-tuned* MLP/ResNet, much prior tabular-DL 'progress' vanishes — several 'SOTA' models don't beat these baselines — and there is NO universal winner between GBDT and DL (they also introduce FT-Transformer, the stronger attention baseline, deferred to Y2). Honest baselines are the contribution."
    },
    {
      id: "feurer2015", paper: "Feurer et al. — Auto-sklearn (Efficient & Robust AutoML)", year: 2015, lesson: 29,
      front: "Feurer et al. 2015 (Auto-sklearn) — what does it automate (the CASH problem), and its two extensions over Auto-WEKA?",
      back: "It solves CASH — Combined Algorithm Selection and Hyperparameter optimization — treating 'which algorithm' as a top-level categorical hyperparameter above each model's knobs and searching the whole space with Bayesian optimization (SMAC), keeping the best-VALIDATION config. Two extensions over Auto-WEKA: (1) meta-learning WARM-START (use dataset meta-features to begin the search from configs that won on similar past datasets) and (2) automated ENSEMBLE construction (Caruana 2004 greedy selection over the pool of trained models). It does NOT do domain feature engineering, and it ties a well-tuned GBDT on typical tabular data (credit_g: default XGB 0.775 → tuned 0.806 ≈ AutoML 0.803) — it buys automation/robustness, not new accuracy."
    },
    {
      id: "guo2016", paper: "Guo & Berkhahn — Entity Embeddings of Categorical Variables", year: 2016, lesson: 31,
      front: "Guo & Berkhahn 2016 — what is an entity embedding, and what did the Rossmann state embedding reveal?",
      back: "Map each level of a categorical to a learned dense vector (dim ≈ min(50, (card+1)//2)), initialised randomly and trained by backprop jointly with the network — a d-dimensional generalisation of target encoding (the 1-D case). Because the vectors are learned to predict the target, levels that behave alike end up close together, so the model shares statistical strength across similar levels and stays compact at high cardinality (unlike one-hot). Headline result: trained only to predict store sales, the learned embedding of the German STATE variable, projected to 2-D, recovered the geographic map of Germany — a real learned representation nobody put in by hand. Embeddings help most on high-cardinality categoricals; on a small flat table they tie one-hot. The bridge from encoding a column to embedding the foreign-key entities of a relational database (RDL)."
    },
    {
      id: "huang2020", paper: "Huang et al. — TabTransformer", year: 2020, lesson: 32,
      front: "Huang et al. 2020 (TabTransformer) — what is the architecture's one new idea, and what is its HONEST verdict vs trees?",
      back: "Pass the per-column entity embeddings (L031) through a stack of N Transformer self-attention layers to turn them into CONTEXTUAL embeddings — each categorical column's vector now depends on the other columns in the same row (like 'bank' = river vs savings). Continuous features BYPASS the Transformer (just LayerNorm) and are concatenated with the flattened contextual embeddings before an MLP head. Attention = softmax(Q·Kᵀ/√d)·V. Verdict: on supervised tabular it MATCHES tree-based ensembles (a tie, not a beat); the +1.0% AUC is over other DEEP methods; its real wins are robustness to noisy/missing features and a +2.1% lift from semi-supervised pre-training on unlabeled data. The tie is the thesis: attention over a row's columns is still single-table cleverness — the same attend-to-related-information operation over the relational graph (a GNN aggregating foreign-key neighbours) is the bet. Only categoricals are contextualised in 2020; FT-Transformer (Y2) tokenises numerics too."
    },
    {
      id: "domingos2012", paper: "Domingos — A Few Useful Things to Know about ML", year: 2012, lesson: 33,
      front: "Domingos 2012 — the three 'useful things' that decide a modeling budget, and the honest stopping rule they imply?",
      back: "(1) 'Feature engineering is the key' — the features used matter more than the algorithm, and most of the time and the early gains go there. (2) 'More data beats a cleverer algorithm' — past a competent model, extra rows usually raise held-out score more per unit effort than extra cleverness/features (data cuts variance directly). (3) 'Overfitting has many faces' — more features/capacity raise variance, so held-out score eventually FALLS, and it can fall subtly. Together: front-load features (steep part of the curve), but STOP when the next feature's marginal held-out gain sinks below the CV std σ (enters the noise band — the L023 discipline), then reallocate the budget to more data / tuning. Verified on credit_g + a fixed strong GBDT: hand features peaked at k=3 (+0.005, inside the ±0.03 band → not significant) then DECLINED to 0.766 below the 0.787 baseline by k=8; a linear model barely moved. The thesis reframe: single-table FE returns die fast, so the value moved ACROSS the join — the relational aggregates DFS builds by hand and RDL aims to learn end-to-end."
    },
    {
      id: "kimball2013", paper: "Kimball & Ross — Dimensional Modeling (star schema)", year: 2013, lesson: 34,
      front: "Kimball dimensional modeling — what are fact vs dimension tables, PK/FK, grain, and how do you flatten a star schema into an ML design matrix?",
      back: "A relational database splits data across TABLES linked by keys. A PRIMARY KEY (PK) uniquely names each row of a table; a FOREIGN KEY (FK) in another table stores that value to link back (orders.customer_id → customers.customer_id). FACT / EVENT tables record things that happened (an order, timestamped, with numeric measures) and hold FKs; DIMENSION tables describe the entities they point to (customer, product). One fact table ringed by its dimensions = a STAR SCHEMA. GRAIN = what one row means (one order vs one customer). To feed a standard model you pick a grain (one customer as of prediction time t) and FLATTEN: LEFT JOIN the fact tables in via the FK, GROUP BY the entity key, and AGGREGATE the one-to-many rows into fixed-width columns (COUNT/SUM/AVG/MAX) — filtering to only rows before t (point-in-time correctness) or you leak the future. Every aggregate is lossy (AVG forgets the individual baskets), the same flatten is rewritten per task, and it discards the one-to-many cardinality / event order / identity / multi-hop paths — the structure RDL keeps as a graph. Reference: Kimball & Ross, The Data Warehouse Toolkit, 3rd ed., Ch. 2."
    },
    {
      id: "pineau2021", paper: "Pineau et al. — Improving Reproducibility in ML Research", year: 2021, lesson: 37,
      front: "Pineau et al. 2021 (JMLR 22(164)) — what did the NeurIPS 2019 Reproducibility Program actually do, what did it measure, and what does its checklist ask you to emit?",
      back: "Three interventions at NeurIPS 2019: a mandatory ML REPRODUCIBILITY CHECKLIST at submission, a voluntary CODE SUBMISSION policy, and a community Reproducibility Challenge. Measured effect: code submission at the time of the paper rose from about 50 % to 75 % of accepted papers, and reviewers rated papers with code higher — but the authors are careful that this is an ASSOCIATION, since submitting code is voluntary and correlates with rigour. The checklist's demands are the ones a run manifest can answer mechanically rather than from memory: a description of the COMPUTING INFRASTRUCTURE used; the EXACT NUMBER OF EVALUATION RUNS; a clear definition of the SPECIFIC MEASURE OR STATISTICS used to report results (not just the metric name — the estimator); a description of results INCLUDING CENTRAL TENDENCY AND VARIATION; and all HYPER-PARAMETERS and how they were chosen. Three vocabulary distinctions the paper forces: repeatability (same team, same setup) vs reproducibility (different team, YOUR artifacts) vs replicability (different team, INDEPENDENT artifacts) — ACM swapped the latter two in v1.1 (2020), so name the artifacts rather than the noun. Base rate for the hardest rung: Raff 2019 hand-implemented 255 papers without the authors' code and reproduced 63.5 %. Companion: Sculley et al. 2015 §6 on configuration debt — the argument for one diffable baseline.yaml."
    },
    {
      id: "lones2024", paper: "Lones — Avoiding common machine learning pitfalls", year: 2024, lesson: 38,
      front: "Lones 2024 (Patterns; arXiv:2108.02497) — how is the guide organised, and which sections are the reviewer's three axes for evaluating a comparison?",
      back: "A continuously-updated 'dos and don'ts' tutorial covering five stages: (1) before you build models, (2) how to reliably build models, (3) how to robustly EVALUATE models, (4) how to COMPARE models fairly, (5) how to REPORT results. The middle three are the peer-review axes. EVALUATE: do not let test data leak (preprocessing/feature-selection on all data, duplicates, temporal look-ahead), avoid SEQUENTIAL OVERFITTING / over-hyping (reusing the test set to guide the next model — the informal cousin of nested-CV selection bias), evaluate multiple times with a spread, and choose metrics carefully (accuracy lies under imbalance). COMPARE: 'do not assume a bigger number means a better model' (different splits/tuning/datasets), use MEANINGFUL baselines (naive dummy + strong established, freshly implemented and equally tuned = HP budget parity), use statistical tests, and CORRECT FOR MULTIPLE COMPARISONS (the multiplicity effect; community benchmarks are a shared version of it). REPORT: be transparent (share code), report performance multiple ways, be careful with significance vs effect size, and USE AN ML CHECKLIST (he co-authored REFORMS). The through-line: a fair review holds every compared model — especially the baseline you want to beat — to the same standard."
    },
    {
      id: "grinsztajn2022-synthesis", paper: "Grinsztajn, Oyallon & Varoquaux — Why trees win (full, synthesis re-read)", year: 2022, lesson: 39,
      front: "Grinsztajn et al. 2022 (full, L039 synthesis re-read) — what claim should a Year 1 essay make, and where does that claim stop?",
      back: "Claim: on typical medium-sized flat tabular tasks, a leak-free tuned GBDT matches or beats honest neural baselines because three inductive biases fit that regime — irregular (non-smooth) targets, privileged column orientation (trees are NOT rotation-invariant), and junk-feature robustness via gain-gated splits. Boundaries: smooth the target / rotate the basis / remove junk and the edge shrinks or reverses; against honest MLP/ResNet and AutoML the usual result is a tie or small lead, not a crush. The essay must also name the silence: trees can only win on what survived a lossy flatten (aggregation collisions destroy recoverable signal), so 'trees win on tabular' is the setup for the relational thesis, not a refutation of it. Open burden: no fair-bar RDL win yet."
    },
    {
      id: "grinsztajn2022-exit", paper: "Grinsztajn, Oyallon & Varoquaux — Why trees win (Year 1 exit re-read)", year: 2022, lesson: 40,
      front: "Grinsztajn et al. 2022 (L040 Year 1 exit) — what two deliverables close Year 1, and why is matching XGBoost within noise a pass?",
      back: "Deliverables: (1) a regenerable tuned/fixed tree baseline under a fair protocol, and (2) a written account of the three inductive biases with verified numbers and flip conditions. The lab fork is beat-XGB OR explain-why-not — ties inside a disclosed noise band (adult ±0.002 ROC-AUC; L020 pattern ref 0.9282 vs LGBM 0.9296) are full passes because Year 1 evidence says further single-table cleverness usually plateaus. Explaining a non-beat means inductive-bias match + exhaustion cascade + join silence, not 'trees are more powerful.' STAND/REVISE links the exit number to the L039 claim without inventing an RDL fair-bar win."
    },
    {
      id: "gorishniy2021-ftt", paper: "Gorishniy et al. — FT-Transformer & the deep-tabular landscape", year: 2021, lesson: 41,
      front: "Gorishniy et al. 2021 (arXiv:2106.11959), L041 landscape view — what is FT-Transformer's one new idea over TabTransformer, and what is the paper's headline verdict vs GBDT?",
      back: "FT-Transformer = Feature Tokenizer + Transformer. Its one new idea over TabTransformer (which embeds only categoricals) is the FEATURE TOKENIZER: EVERY feature becomes an embedding token — a numeric x_j → T_j = x_j·W_j + b_j (a learned per-column weight vector and bias scale the single number into a d-dim vector), a categorical → embedding-table lookup. A learnable [CLS] token is prepended, L Transformer self-attention layers run, and the [CLS] output is read by a linear head — so numerics attend to categoricals and each other, not just bypass to an MLP. That makes it the strongest, most UNIVERSAL classic neural single-table model. Headline verdict: NO UNIVERSAL WINNER — under one shared tuning protocol a tuned GBDT still wins on a large share of datasets, DL on others; fair tuning + ensembling move results as much as architecture choice (same conclusion as Grinsztajn 2022, from the architecture side). Practically: run the strong ResNet baseline first, and use the rtdl reference implementations so the neural bar is standard, not home-made."
    },
    {
      id: "gorishniy2021-protocol", paper: "Gorishniy et al. — Revisiting DL for Tabular Data (§3.2 + protocol)", year: 2021, lesson: 42,
      front: "Gorishniy et al. 2021 (§3.2, L042 'do these first') — what are the two simple baselines, how are they trained, and what is the fairness rule the paper turns into a field standard?",
      back: "The two simple baselines are a tuned MLP (a stack of Dropout(ReLU(Linear)) blocks) and a tuned ResNet — an MLP whose blocks add a residual skip, ResNetBlock(x)=x+Dropout(Linear(Dropout(ReLU(Linear(BatchNorm(x)))))), so the identity map is free and depth stops degrading (He 2015). Both are trained with AdamW + weight decay and EARLY STOPPING on the validation metric. The load-bearing rule is the SHARED TUNING PROTOCOL: a shared FRAME (same train/valid/test split, same metric, same hyper-parameter search BUDGET, selection by validation) applied identically to every model, with only the per-model search SPACE (its knobs) differing. Fairness is equal budget, not equal knobs. Its practical instruction — 'do these first' — is that you compute the tuned MLP/ResNet BEFORE any novel architecture, because a tuned ResNet alone matches many published 'novel' models, so a win over a weak or unequally-tuned net (or a GBDT-only comparison) measures effort, not quality (L038 HP-budget parity). Verified instance (credit_g, L028 evidence of record): tuned GBDT 0.793 > MLP 0.752 ≈ ResNet 0.743 — a tree lead under fair training, i.e. no universal winner on this small categorical table, ranking flips elsewhere."
    },
    {
      id: "arik2019", paper: "Arik & Pfister — TabNet: Attentive Interpretable Tabular Learning", year: 2019, lesson: 43,
      front: "Arik & Pfister 2019 (arXiv:1908.07442) — what are TabNet's four mechanisms, what makes its attention *sequential* rather than merely sparse, and how does it fare against the baseline-first bar?",
      back: "TabNet takes N_steps DECISION STEPS; each step selects a sparse subset of features, transforms them, and adds its contribution to the prediction (mask → transform → sum, a soft learned analogue of 'split on this feature, then that one' — the tree-like inductive bias an MLP lacks, L025–L027). Four mechanisms: (1) SPARSEMAX (Martins & Astudillo 2016) — the Euclidean projection onto the probability simplex; subtract one threshold tau and clip, so the mask still sums to 1 but has EXACT zeros, making it a selection rather than a weighting (softmax's exp(z) > 0 can never switch a feature off). (2) ATTENTIVE TRANSFORMER — M[i] = sparsemax(P[i−1] · h_i(a[i−1])), h_i = Linear→BatchNorm. (3) PRIOR SCALE — P[i] = ∏_{j≤i}(γ − M[j]), P[0] = 1: the memory of what earlier steps spent. THIS, not sparsemax, is what makes the attention SEQUENTIAL — sparsity and sequentiality are different properties; at γ = 1 the leftover budget is exactly 1 − M so a fully-used feature is banned from all later steps, larger γ permits reuse. (4) FEATURE TRANSFORMER — Linear→GhostBatchNorm→GLU blocks (2 shared across steps + 2 step-dependent) wired with sqrt(0.5)-scaled residuals; its output splits into d[i] (prediction, aggregated as ∑ReLU(d[i])) and a[i] (feeds the next step's attention). Plus a sparsity penalty λ_sparse on mask entropy. INTERPRETABILITY: M_agg aggregates per-step masks weighted by each step's decision contribution η[i] = ∑_c ReLU(d_c[i]), giving per-row attributions from the forward pass — no separate SHAP/LIME run. Verified on the paper's own L2X generators (L043): Syn2 (GLOBAL relevance) is a clean success, 76.8% of mask mass on the true X3–X6, top-4 exactly right; Syn4 (INSTANCE-WISE, switched by X11) only PARTIAL — the switch is found (weight 0.118) and mass moves the right way, but just 15.6% of X11<0 rows favour their own group vs 97.9% on the other side, and the paper used 10M not 10k samples for its sharp Fig. 5 masks. THE BAR: held to L042's baseline-first rule under one shared frame on 4 small tables, from-scratch TabNet ranks BEHIND the tuned simple baselines — mean ranks TabNet 2.50, MLP 1.75, ResNet 2.00, GBDT 3.75, Friedman p = 0.127. Read it precisely: p > 0.05 licenses only 'cannot distinguish on this sample', NOT 'significantly worse'; but the burden of proof is the new model's, so it did not clear the bar. The paper's own Appendix A (KDD) has TabNet tying or slightly trailing XGBoost and CatBoost — ask why that table is in an appendix (L038)."
    }
  ];
})(window);
