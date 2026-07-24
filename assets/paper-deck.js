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
    }
  ];
})(window);
