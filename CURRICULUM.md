# Expert Curriculum — Relational Deep Learning

Agent-facing plan for lesson sequencing. The student-facing version is [reference/curriculum.html](./reference/curriculum.html).

**Pace:** ~1 hour/day **baseline** (minimum on typical days) · ~360 hours/year · ~2,160 hours over 6 years.

### Why 240 units take 6 years (read this if the math looks off)

A common, correct objection: *240 lessons ÷ 365 days ≈ 0.66 years — so at one lesson/day this is an 8-month plan, not a 6-year one.* That objection is right about **lessons** and wrong about **mastery**. The fix is to stop treating a numbered row as "one day."

**A unit ≠ a day.** Each numbered row below is a **unit** = one concept *plus its lab/reproduction work plus its quiz*. Units are not uniform in size:

| Unit type | Sessions to finish | Where |
|-----------|--------------------|-------|
| Concept lesson (read + quiz) | 1–3 sessions | dense in Y1–Y2 |
| Reproduction / lab unit (train, tune, re-run, debug) | 5–20 sessions | dominant in Y3–Y5 |
| Checkpoint / exit exam | 3–8 sessions | end of each quarter/year |
| Research unit (experiment + write) | 10–40+ sessions | Y6 (mostly *not* lessons) |

**Where the ~2,160 hours actually go** (lessons are ~6% of the clock):

| Activity | ≈ hours | Share |
|----------|---------|-------|
| Lesson content itself (240 × ~0.5h) | ~120 | ~6% |
| Deep primary-paper reading (60+ papers, harder each year) | ~350 | ~16% |
| **Hands-on labs & reproduction** (GNNs, RelBench runs, tuning) | ~900 | ~42% |
| Checkpoints + year exit exams | ~150 | ~7% |
| Spaced retrieval / review | ~150 | ~7% |
| **Year 6 original research** (experiments + writing) | ~400 | ~19% |

**Density curve (front-loaded learning, back-loaded research):**
- **Y1–Y2** — concept-dense. Units arrive fast (~2–3/week). This is the only phase where "lessons/week" is the right mental model.
- **Y3–Y4** — reproduction-dominated. ~1 unit/week; each unit is days of training/debugging, not a reading.
- **Y5–Y6** — research-dominated. The ~80 units here are *milestones* spanning weeks each; the calendar is set by experiments and writing, not lessons.

So the 6 years are paced by **skill acquisition (labs/reproduction) and the final research project**, not by lesson count. If you only consumed the 240 lessons at 2–3/week, you'd "finish reading" in ~2 years — and still not be able to reproduce RelBench or run a fair benchmark. The extra four years are where expertise actually forms.

**Lesson numbering:** Year N → units `(N-1)*40 + 001` … `(N-1)*40 + 040` (approx.)  
**Rule:** Finish each quarter checkpoint before advancing. Papers are read in publication order within each year. A unit is "done" when its lab runs and its quiz/checkpoint is passed — not when the reading is skimmed.

> **Critical framing (read first).** This is a *fast-moving* field. Two truths must coexist in your head the whole way:
> 1. **GBDTs are not dead.** As of 2024–2026, tuned tree ensembles + strong-default MLPs (RealMLP, TabM) still win or tie on most *single-table* industrial data, especially under temporal splits (TabReD, TabArena). Do not let the relational thesis make you sloppy about this.
> 2. **The action has moved to two frontiers:** (a) tabular *foundation models* via in-context learning (TabPFN v2, TabICL), and (b) *relational* deep learning + relational foundation models (RelBench, RelGNN, RelGT, Griffin, RDB-PFN). Your thesis lives at frontier (b).
>
> Every architecture below is taught **with its failure mode**, not as hype. A model earns a lesson only if it changed the SOTA, exposed a real limitation, or is a baseline you must beat. See [arXiv IDs in RESOURCES.md](./RESOURCES.md).

---

## Year 1 — Tabular Foundations (Lessons 001–040)

**Goal:** Fair baselines, evaluation discipline, tree-model fluency. You cannot argue RDL is undervalued until you can tune XGBoost properly.

### Q1 · Single-table ML mechanics (Lessons 001–010)
| # | Lecture topic | Primary reading | Lab |
|---|---------------|-----------------|-----|
| 001 | Single-table assumption | Fey ICML 2024 §1 | Quiz (done) |
| 002 | Design matrix & leakage | Huyen, Ch. 2–3 | Find leakage in a join sketch |
| 003 | Train/valid/test & CV | sklearn model_selection docs | Implement stratified split |
| 004 | Grouped & nested CV | Vowels et al. 2010 (nested CV) | GroupKFold on person-ID data |
| 005 | Pipelines & preprocessing | sklearn Pipeline docs | Median impute + scale pipeline |
| 006 | Missingness taxonomy | Little & Rubin (MCAR/MAR/MNAR summary) | Structural vs stochastic missing |
| 007 | Class imbalance | He & Garcia 2009 (learning from imbalanced data) | class_weight, PR-AUC |
| 008 | Metrics that matter | Huyen, evaluation chapter | ROC vs PR for rare labels |
| 009 | Feature engineering patterns | Kaggle FE best-practices (curated notes) | Build 5 aggregate features |
| 010 | **Q1 checkpoint** | — | Reproduce sklearn baseline on one dataset |

**Papers (chronological):** none required yet beyond Fey §1 — focus on practice.

### Q2 · Gradient boosting & the tree baseline (Lessons 011–020)
| # | Lecture topic | Primary reading | Lab |
|---|---------------|-----------------|-----|
| 011 | Decision trees as partitions | ESL Ch. 9 (summary) | Visualize one split |
| 012 | Bagging & Random Forest | Breiman 2001 (skim) | RF vs single tree |
| 013 | Boosting intuition | Friedman 2001 (gradient boosting intro) | Compare to RF |
| 014 | XGBoost | Chen & Guestrin 2016 | Tune XGB on one task |
| 015 | LightGBM | Ke et al. 2017 | Speed vs XGB comparison |
| 016 | CatBoost | Prokhorenkova et al. 2018 | Categorical handling |
| 017 | Hyperparameter search | Bergstra & Bengio 2012 (random search) | RandomizedSearchCV |
| 018 | Ensembling & stacking | Wolpert 1992 (stacked generalization, skim) | Simple blend |
| 019 | When trees win | Preview Grinsztajn 2022 abstract | List 3 tree strengths |
| 020 | **Q2 checkpoint** | Chen 2016 + Ke 2017 | Match or beat published tree baseline |

**Papers:** Chen & Guestrin 2016 · Ke et al. 2017 · Prokhorenkova et al. 2018

### Q3 · Evaluation rigor & benchmark literacy (Lessons 021–030)
| # | Lecture topic | Primary reading | Lab |
|---|---------------|-----------------|-----|
| 021 | Data splits in the wild | Huyen, data chapters | Temporal vs random split |
| 022 | Label leakage patterns | Kapoor & Narayanan 2022 (leakage in ML) | Spot leakage in FE |
| 023 | Statistical comparison | Demšar 2006 (statistical comparisons) | Paired test on CV folds |
| 024 | The Grinsztajn benchmark | Grinsztajn et al. 2022 §1–4 | Run one dataset from repo |
| 025 | Inductive bias: smoothness | Grinsztajn 2022 §5.2 | Explain smoothness bias |
| 026 | Inductive bias: rotation | Grinsztajn 2022 §5.3 | Rotation experiment |
| 027 | Inductive bias: junk features | Grinsztajn 2022 §5.4 | Add noise features |
| 028 | MLP & ResNet tabular baselines | Gorishnaya et al. 2021 §3.2 | Train ResNet baseline |
| 029 | Manual FE vs AutoML | Feurer et al. 2015 (Auto-sklearn, skim) | Compare to tuned XGB |
| 030 | **Q3 checkpoint** | Grinsztajn 2022 full | Write 1-page benchmark report |

**Papers:** Grinsztajn et al. 2022 · Gorishnaya et al. 2021 (ResNet section)

### Q4 · Consolidation & bridge to neural tabular (Lessons 031–040)
| # | Lecture topic | Primary reading | Lab |
|---|---------------|-----------------|-----|
| 031 | Embeddings for categoricals | entity embeddings (Guo & Berkhahn 2016) | Target encoding pitfalls |
| 032 | TabTransformer preview | Huang et al. 2020 | Read architecture figure |
| 033 | When to stop feature engineering | Domingos 2012 (few useful things) | FE time budget exercise |
| 034 | Relational data without RDL | Star schema & joins (Kimball summary) | Sketch 3-table join |
| 035 | What joins destroy | Fey ICML 2024 §2 (feature engineering cost) | List lost structure |
| 036 | Revisit your homework pipeline | Your `homework/report.md` | Audit CV + missingness |
| 037 | Document a baseline package | — | Reproducible baseline script |
| 038 | Peer review your evaluation | — | Checklist: leakage, tuning, metrics |
| 039 | Year 1 synthesis essay | — | Write: what trees beat and why |
| 040 | **Year 1 exit exam** | All Y1 papers | Beat XGB on flat task OR explain why not |

**Year 1 exit criterion:** Reproducible tuned tree baseline + written understanding of Grinsztajn's three biases.

---

## Year 2 — Advanced Tabular Deep Learning (Lessons 041–080)

**Goal:** Know every major neural tabular architecture and tabular foundation model, when each wins, and exactly why strong trees/MLPs still often win on single tables. This is the year that makes you *honest* about baselines.

### Q1 · Neural tabular architectures (041–050)
**Papers (chronological):** Popov 2019 (NODE) · Arik 2019 (TabNet) · Huang 2020 (TabTransformer) · Gorishniy 2021 (FT-Transformer/ResNet) · Somepalli 2021 (SAINT) · Wang 2021 (DCNv2) · Chen 2023 (ExcelFormer) · Chen 2023 (Trompt)

| # | Topic | Paper | Lab |
|---|-------|-------|-----|
| 041 | Deep tabular landscape & rtdl | Gorishniy et al. 2021 full | rtdl repo setup |
| 042 | MLP & ResNet baselines (do these first) | Gorishniy 2021 §3.2 | Train ResNet baseline |
| 043 | TabNet (sequential attention) | Arik & Pfister 2019 | Train + read masks |
| 044 | NODE (differentiable trees) | Popov et al. 2019 | When irregular functions help |
| 045 | TabTransformer (contextual embeddings) | Huang et al. 2020 | Categorical embeddings |
| 046 | FT-Transformer ★ | Gorishniy 2021 §3.3 | Train FT-T |
| 047 | SAINT (row+col attention) | Somepalli et al. 2021 | Inter-sample attention |
| 048 | DCNv2 & explicit feature crosses | Wang et al. 2021 | Cross network |
| 049 | ExcelFormer & Trompt (GBDT-surpassing claims) | Chen 2023 · Chen 2023 | Read critically: claims vs protocol |
| 050 | **Q1 checkpoint** | Gorishniy 2021 | FT-T vs XGB on 3 datasets, same protocol |

### Q2 · Modern tabular DL & the honest baseline (051–060)
**Papers (chronological):** Grinsztajn 2022 (revisit) · TabR (Gorishniy 2023) · RealMLP/"Better by Default" (Holzmüller 2024) · TabReD (Rubachev 2024) · TabM (Gorishniy 2024) · TabArena (Erickson 2025)

| # | Topic | Paper | Lab |
|---|-------|-------|-----|
| 051 | Why trees still win (revisit) | Grinsztajn 2022 §5 | Re-derive 3 biases |
| 052 | TabR — retrieval-augmented DL ★ | Gorishniy 2023 (2307.14338) | kNN-attention component |
| 053 | RealMLP & strong defaults ★ | Holzmüller 2024 (2407.04491) | RealMLP vs tuned XGB |
| 054 | TabM — parameter-efficient ensembling ★ | Gorishniy 2024 (2410.24210) | Train TabM (current best DL baseline) |
| 055 | The temporal-split reality | TabReD (2406.19380) | Random vs time split flips rankings |
| 056 | Living benchmark literacy | TabArena (2506.16791) | Read leaderboard methodology |
| 057 | Ensembling across model families | TabArena §results | Build cross-model ensemble |
| 058 | Surveys & taxonomy | Borisov 2021 (2110.01889) · survey 2410.12034 | Map architecture families |
| 059 | Validation-set overfitting | TabArena critique | Diagnose overfit ensemble |
| 060 | **Q2 checkpoint** | TabM + RealMLP | Beat your Y1 XGB with a tuned DL model — or document why you can't |

### Q3 · Tabular foundation models & in-context learning (061–070)
**Papers (chronological):** Müller 2022 (PFN) · Hollmann 2022 (TabPFN v1) · LoCalPFN (Thomas 2024) · Drift-Resilient TabPFN (Helli 2024) · Hollmann 2025 (TabPFN v2, Nature) · TabICL (Qu 2025) · "Closer Look at TabPFN v2" (Ye 2025) · open-environment eval (Cheng 2025)

| # | Topic | Paper | Lab |
|---|-------|-------|-----|
| 061 | Prior-Data Fitted Networks | Müller et al. 2022 | Concept: learning Bayesian inference |
| 062 | TabPFN v1 (≤1K rows) | Hollmann et al. 2022 | Run TabPFN v1 |
| 063 | What the synthetic SCM prior encodes | Hollmann 2022 §4 | Inspect prior samples |
| 064 | TabPFN v2 (Nature) ★ | Hollmann et al. 2025 (Nature) | Compare to 4h-tuned GBDT |
| 065 | How TabPFN v2 handles heterogeneity | Ye et al. 2025 (2502.17361) | Feature-extractor mode |
| 066 | TabICL — scaling ICL to 500K rows ★ | Qu et al. 2025 (2502.05564) | Column-then-row attention |
| 067 | Retrieval + fine-tuning PFNs | LoCalPFN (2406.05207) | Local calibration |
| 068 | Distribution shift & PFNs | Drift-Resilient TabPFN (2411.10634) | Temporal shift test |
| 069 | **Critical:** where TabPFN v2 breaks | Cheng et al. 2025 (2505.16226) | Open-environment failure cases |
| 070 | **Q3 checkpoint** | Hollmann 2025 + TabICL | TabPFN v2 vs TabM vs XGB on 5 datasets |

### Q4 · Self-supervision, encoders & the bridge to relational (071–080)
**Papers:** Yoon 2020 (VIME) · Bahri 2021 (SCARF) · Ucar 2021 (SubTab) · Hu et al. 2024 (PyTorch Frame) · CARTE/cross-table transfer (skim)

| # | Topic | Paper | Lab |
|---|-------|-------|-----|
| 071 | VIME — masked tabular SSL | Yoon et al. 2020 | Masked pretraining |
| 072 | SCARF / SubTab — contrastive & multi-view | Bahri 2021 · Ucar 2021 | Contrastive views |
| 073 | When SSL actually helps | Survey + your homework VIME notes | SSL ablation design |
| 074 | Cross-table transfer (CARTE etc.) | Literature skim | Schema-agnostic embeddings |
| 075 | PyTorch Frame — the row encoder ★ | Hu et al. 2024 (2402.05964) | Encode mixed-type schema |
| 076 | Encoder → predictor stack | RDL preview | Encoder → head pattern |
| 077 | **Single-table ceiling** (synthesis) | — | Write: what rows-only models cannot represent |
| 078 | Bridge: message passing preview | Gilmer 2017 · Kipf 2017 §2 | Manual one-hop aggregate |
| 079 | Year 2 essay | — | Neural-tabular decision tree: when which model |
| 080 | **Year 2 exit exam** | All Y2 papers | Teach-back: 3 biases + TabM + TabPFN v2 + TabICL |

**Year 2 exit criterion:** Train and fairly compare FT-Transformer, TabM, and TabPFN v2 against a tuned GBDT under both random *and* temporal splits; articulate in writing why single-table models plateau and where ICL helps.

---

## Year 3 — Graph Machine Learning (Lessons 081–120)

**Goal:** Implement GNNs from message-passing principles through heterogeneous & temporal graphs.

### Q1 · Message passing foundations (081–090)
**Papers (chronological):** Gilmer 2017 · Kipf & Welling 2017 · Hamilton 2017 · Veličković 2018

| 081 | MPNN framework | Gilmer 2017 | Implement generic MPNN |
| 082 | GCN | Kipf 2017 | Cora node classification |
| 083 | GraphSAGE | Hamilton 2017 | Inductive mini-batch training |
| 084 | GAT | Veličković 2018 | Attention aggregation |
| 085 | Over-smoothing | Li et al. 2018 | Depth vs performance |
| 086 | PyG fundamentals | Fey & Lenssen 2019 | Data, NeighborLoader |
| 087 | Link prediction | Zhang & Chen 2018 (skim) | Edge split metrics |
| 088 | Graph classification | Xu et al. 2019 (GIN) | Whole-graph readout |
| 089 | Sampling at scale | Chiang et al. 2019 (Cluster-GCN) | Large-graph training |
| 090 | **Q1 checkpoint** | Kipf + Hamilton | GNN from scratch on one benchmark |

### Q2 · Heterogeneous graphs (091–100)
**Papers:** Schlichtkrull 2018 · Wang 2019 (HAN) · Hu 2020 (HGT) · Sun 2020 (HIN survey)

| 091 | R-GCN | Schlichtkrull 2018 | Relation-specific weights |
| 092 | Meta-paths | Wang 2019 HAN | Heterogeneous attention |
| 093 | HGT | Hu 2020 | Transformer on heterogeneous graphs |
| 094 | HIN survey | Sun & Han 2020 | Taxonomy of heterogeneity |
| 095 | Bipartite graphs | — | User-item as hetero graph |
| 096 | Multi-relational data | — | Connect to SQL FK semantics |
| 097 | Negative sampling | — | Link pred training |
| 098 | Hetero mini-batching | PyG HeteroData | RelBench-shaped toy graph |
| 099 | Compare R-GCN vs HGT | — | Same graph, two architectures |
| 100 | **Q2 checkpoint** | Schlichtkrull + Hu 2020 | Hetero GNN on 3+ node types |

### Q3 · Temporal & dynamic graphs (101–110)
**Papers:** Rossi 2020 (TGN) · Xu 2020 (TGAT) · Kazemi 2020 (RE-Net)

| 101 | Static vs temporal | Fey 2024 temporal REG preview | Time-respecting splits |
| 102 | Temporal Graph Networks | Rossi 2020 | Memory modules |
| 103 | TGAT | Xu 2020 | Time-encoded attention |
| 104 | Information leakage in time | Kapoor 2022 + Fey 2024 | Time-travel bugs |
| 105 | Continuous time | — | Event streams as graphs |
| 106 | Temporal link prediction | — | Future edge prediction |
| 107 | Snapshot methods | — | Discrete time slices |
| 108 | Scale: neighbor sampling over time | — | Efficient temporal batch |
| 109 | Connect to databases | — | `observed_at` semantics |
| 110 | **Q3 checkpoint** | Rossi 2020 | Temporal GNN with clean split |

### Q4 · Graph ML mastery (111–120)
| 111–114 | OGB benchmark task | Hu 2020 OGB | Leaderboard reproduction |
| 115 | Graph ML design patterns | — | Encoder → MP → head |
| 116 | Debug GNN training | — | Common failure modes |
| 117 | RDL bridge lecture | Fey 2024 full paper | REG construction walkthrough |
| 118 | Cvitkovic 2019 relational DL | Cvitkovic 2019 | Historical line to RDL |
| 119 | Year 3 synthesis | — | Essay: graphs vs flat tables |
| 120 | **Year 3 exit exam** | All Y3 papers | Build hetero temporal GNN |

**Year 3 exit criterion:** Working hetero + temporal GNN pipeline in PyG; read Fey 2024 completely.

---

## Year 4 — Relational Deep Learning (Lessons 121–160)

**Goal:** Master RDL end-to-end — REG, RelBench, reproduction, beat manual FE fairly.

### Q1 · RDL foundations (121–130)
**Papers (chronological):** Cvitkovic 2019 · Šír 2021 · Zahradník 2023 · Fey 2024 · Robinson 2023/2024

| 121 | History of relational ML | Cvitkovic 2019 | Prior art map |
| 122 | REG construction | Fey 2024 §3 | Build toy REG |
| 123 | Temporal heterogeneous graphs | Fey 2024 §3–4 | Timestamp every node/edge |
| 124 | Entity vs task table | Fey 2024 | Prediction node selection |
| 125 | PyTorch Frame deep dive | Hu 2024 | Row encoders in PyG Frame |
| 126 | RelBench beta paper | arXiv 2312.04615 | Package tour |
| 127 | RelBench v1 | Robinson 2024 NeurIPS | 7 databases overview |
| 128 | Task taxonomy | RelBench docs | Entity / link / autoregressive |
| 129 | Manual FE study | Robinson 2024 user study | Replicate FE baseline mindset |
| 130 | **Q1 checkpoint** | Fey 2024 + Robinson 2024 | Run one RelBench task end-to-end |

### Q2 · RDL architectures & baselines (131–140)
| 131 | GNN + tabular encoder stack | Fey 2024 §5 | Full forward pass trace |
| 132 | ID-GNN / IDMP variants | RelBench baselines | Compare message passing |
| 133 | Hetero conv on REG | PyG + RelBench | Layer design |
| 134 | Training at scale | RelBench sampling | Mini-batch over millions of nodes |
| 135 | Hyperparameter tuning on REG | — | Fair tuning budget |
| 136 | Leaderboard literacy | relbench.stanford.edu | Read top entries |
| 137 | Error analysis on REG | — | Where GNN fails vs FE |
| 138 | Domain: e-commerce task | rel-amazon | Full task deep dive |
| 139 | Domain: healthcare / social | RelBench v1 tasks | Second domain |
| 140 | **Q2 checkpoint** | — | Match published GNN baseline on 2 tasks |

### Q3 · Advanced RDL methods (141–150)
**Papers (chronological):** ContextGNN (Yuan 2024) · RelGNN (Chen 2025, ICML) · Griffin (Wang 2025, ICML) · RelGT (Dwivedi 2025, ICLR 2026) · RDL survey (2025)

| 141 | Composite message passing ★ | Chen 2025 RelGNN (2502.06784) | Atomic routes concept |
| 142 | Many-to-many edge pathology | Chen 2025 §motivation | Why vanilla GNNs lose signal |
| 143 | RelGNN reproduction | Chen 2025 | Run on RelBench subset |
| 144 | ContextGNN — beyond two-tower recsys ★ | Yuan 2024 (2411.19513) | Pair-wise + two-tower fusion |
| 145 | Relational Graph Transformer ★ | Dwivedi 2025 RelGT (2505.10960) | Multi-element tokenization |
| 146 | GNN vs Graph-Transformer on REG | RelGT §results | Same tasks, two paradigms |
| 147 | Survey: next-gen architectures | arXiv 2506.16654 | Map open problems |
| 148 | Ablation discipline | — | Encoder vs MP vs data contribution |
| 149 | Identify weakest RelBench tasks | — | Where the thesis might fail |
| 150 | **Q3 checkpoint** | RelGNN or RelGT | SOTA or near-SOTA on 1 task; written reproduction report |

### Q4 · RDL expertise (151–160)
| 151–154 | Multi-task RelBench portfolio | — | 5 tasks, one report |
| 155 | Compare to manual FE | Robinson user study protocol | Document human-effort ratio |
| 156 | Temporal leakage audit | — | Full REG audit checklist |
| 157 | Open-source contribution | — | PR or reproducibility repo |
| 158 | Year 4 synthesis essay | — | Evidence for/against thesis |
| 159 | Foundation model preview | Zahradník 2023 full | Pre-training objectives |
| 160 | **Year 4 exit exam** | All Y4 papers | RelBench portfolio + essay |

**Year 4 exit criterion:** Reproduce RelBench baselines; documented comparison to manual FE on ≥3 tasks.

---

## Year 5 — Foundation Relational Models (Lessons 161–200)

**Goal:** Pre-training, in-context relational learning, cross-database generalization.

### Q1 · Foundation model concepts (161–170)
**Papers (chronological):** Bommasani 2021 (FM definition) · Zahradník 2023 (vision) · Griffin (Wang 2025) · KumoRFM (2025, industry) · RDB-PFN (Wang 2026) · Fey survey 2025 §FM

| 161 | What is a foundation model | Bommasani 2021 | Scope for relational |
| 162 | The relational FM vision | Zahradník 2023 (2305.15321) | LM + GNN pre-training |
| 163 | LM encoders for rows | Zahradník 2023 §4 | Text vs typed columns |
| 164 | Griffin — graph-centric RDB FM ★ | Wang 2025 (2505.05568) | Unified encoder/decoder, cross-attention |
| 165 | KumoRFM — in-context relational learner | KumoRFM tech report (2025) | Few-shot task adaptation (proprietary) |
| 166 | RDB-PFN — synthetic-prior relational FM ★ | Wang 2026 (2603.03805) | Relational Prior Generator; reproduce (open code) |
| 167 | Tabular FM → relational FM transfer | TabPFN v2 + TabICL recap | What carries over from Year 2 |
| 168 | Cross-database generalization | Griffin + RDB-PFN experiments | Zero-/few-shot on unseen schema |
| 169 | Scaling laws & open questions | Survey 2025 | What's unknown |
| 170 | **Q1 checkpoint** | Zahradník + Griffin + RDB-PFN | Written FM design doc comparing the three |

### Q2 · Building pre-training pipelines (171–180)
| 171 | Corpus of databases | — | Curate or use RelBench multi-DB |
| 172 | Schema tokenization | — | Column types as tokens |
| 173 | Multi-task pre-training | — | Combined loss design |
| 174 | Fine-tuning protocol | — | Freeze vs full fine-tune |
| 175 | Evaluation: zero-shot | — | Unseen database task |
| 176 | Evaluation: few-shot ICL | — | k labeled examples |
| 177 | Compute budget realism | — | What's feasible at baseline vs extended sessions |
| 178 | Compare FM vs tuned GNN | — | Same task, fair budget |
| 179 | Failure modes | — | Schema shift, cold start |
| 180 | **Q2 checkpoint** | — | Fine-tune public encoder on 1 DB |

### Q3 · Research frontier mapping (181–190)
| 181 | RelBench v2 / autocomplete tasks | RelBench v2 (2602.12606) · RelGT-AC (2606.03040) | New tasks & databases |
| 182 | RDB-PFN + composite message passing | RDB-PFN × RelGNN | Hypothesis generation |
| 183 | Graph-Transformer + pre-training | RelGT × Griffin | Architecture research gap |
| 184 | Latest relational Graph-Transformers | GelGT (2605.15575) | Long-range dependency fixes |
| 185 | Causal & relational data | — | When correlation ≠ deployable strategy |
| 186 | Production constraints | Huyen | Latency, freshness, monitoring |
| 187 | Ethics & privacy on REG | — | Node-level privacy, leakage |
| 188 | Systematic literature tracking | — | arXiv alerts, RelBench leaderboard, paper log |
| 189 | Identify 3 open problems | Survey 2025 | Rank by tractability |
| 190 | **Q3 checkpoint** | — | Research gap document (5 pages) |

### Q4 · Year 5 synthesis (191–200)
| 191–194 | Reproduce best public FM baseline | — | Full reproduction |
| 195 | Thesis stress-test | — | Where is thesis wrong? |
| 196 | Community engagement | RelBench mailing list | Ask one technical question |
| 197 | Year 5 essay | — | FM landscape map |
| 198 | Propose 3 novel directions | — | Ranked research proposals |
| 199 | Select primary direction | — | One hypothesis to pursue |
| 200 | **Year 5 exit exam** | All Y5 papers | FM reproduction + proposal |

**Year 5 exit criterion:** Written research proposal with evidence from RelBench experiments; reproduced one FM-related baseline.

---

## Year 6 — Novel Research & Thesis Validation (Lessons 201–240)

**Goal:** Original contribution that supports or falsifies the contrarian thesis.

### Q1 · Hypothesis & experiment design (201–210)
| 201 | Sharpen hypothesis | — | Falsifiable claim |
| 202 | Related work deep dive | — | Position vs RelGNN/RelGT/FM |
| 203 | Experiment protocol | — | Pre-registration style doc |
| 204 | Baseline suite locked | — | Trees + FE + GNN + FM |
| 205 | Implementation sprint 1 | — | Core method |
| 206 | Implementation sprint 2 | — | Ablations |
| 207 | Negative results log | — | What didn't work |
| 208 | Mid-point review | — | Continue / pivot decision |
| 209 | Statistical analysis | Demšar 2006 | Significance across tasks |
| 210 | **Q1 checkpoint** | — | Working prototype |

### Q2 · Execution & ablations (211–220)
| 211–218 | Ablation matrix | — | Component contributions |
| 219 | Robustness checks | — | Seeds, splits, domains |
| 220 | **Q2 checkpoint** | — | Complete experiment table |

### Q3 · Communication (221–230)
| 221 | Write technical report | — | 8–12 page draft |
| 222 | Open-source release | — | Code + docs |
| 223 | RelBench leaderboard submission | — | If results merit |
| 224 | Blog / talk outline | — | Thesis for general audience |
| 225 | Respond to critique | — | Steel-man opposing view |
| 226 | Revise based on gaps | — | Second experiment round |
| 227 | Final results | — | Locked numbers |
| 228 | Limitations section | — | Honest scope |
| 229 | Future work | — | Next 3 experiments |
| 230 | **Q3 checkpoint** | — | Submission-ready draft |

### Q4 · Launch (231–240)
| 231–234 | External review | — | Peer or community feedback |
| 235 | Camera-ready / release | — | Publish or ship |
| 236 | Production pilot (optional) | — | Real deployment |
| 237 | Year 6 retrospective | — | What thesis got right/wrong |
| 238 | Expert teach-back | — | Explain full stack live |
| 239 | Next 5-year map | — | Adjacent research |
| 240 | **Graduation** | — | Public artifact + thesis verdict |

**Year 6 exit criterion:** Public artifact (paper, leaderboard, or production result) with fair baselines and honest limitations.

---

## Daily hour templates (1 h baseline)

On days with extra time, add minutes to **Practice** first, then **Input** (paper appendices). Never skip exit criteria to move faster through years.

| Phase | Retrieval (15m) | Input (25m) | Practice (20m) |
|-------|-----------------|-------------|------------------|
| Y1 tabular | Quiz prior lesson | sklearn/XGB reading or lesson | Notebook cell |
| Y2 neural tabular | Architecture recall | Paper section | TabM / TabPFN v2 / TabICL run |
| Y3 graphs | MPNN equation | GNN paper | PyG implementation |
| Y4 RDL | REG sketch from memory | RelBench / RDL paper | RelBench training run |
| Y5 FM | Pre-training objective | Zahradník / survey | Fine-tune experiment |
| Y6 research | Hypothesis restatement | Related work | Code / write |

---

## Lesson production schedule (for agent)

- Produce **1 lesson HTML per lecture row** as the student reaches it. Realistic cadence is **front-loaded**: ~2–3 lessons/week in Y1–Y2 (concept-dense), ~1/week in Y3–Y4 (reproduction units span multiple sessions), and sparse milestone lessons in Y5–Y6 (research-dominated). Do **not** pace lesson production at a fixed weekly rate across all years.
- A lesson HTML covers the *concept*; the unit isn't complete until the student finishes the lab/reproduction it points to (often days of work in later years).
- Each lesson: one concept, one quiz, one primary paper link, one lab step.
- Every 10 units: checkpoint lesson with longer exercise.
- Update [[GLOSSARY.md]] when user demonstrates term mastery.
- Track paper completion in [[NOTES.md]] with `✓ paper-id` lines.

---

## Verified paper index (arXiv IDs)

Confirmed via arXiv search, June 2026. ★ = must-read. Read in publication order within each year.

### Year 1 — Tabular foundations
- Chen & Guestrin 2016 — XGBoost — `1603.02754` ★
- Ke et al. 2017 — LightGBM — `1711.08251` (NeurIPS)
- Prokhorenkova et al. 2018 — CatBoost — `1706.09516`
- Grinsztajn et al. 2022 — Why trees beat DL — `2207.08815` ★
- Kapoor & Narayanan 2022 — Leakage & reproducibility — `2207.07048`

### Year 2 — Advanced tabular DL
- Arik & Pfister 2019 — TabNet — `1908.07442`
- Popov et al. 2019 — NODE — `1909.06312`
- Huang et al. 2020 — TabTransformer — `2012.06678`
- Yoon et al. 2020 — VIME — (NeurIPS; see proceedings)
- Gorishniy et al. 2021 — FT-Transformer / Revisiting DL — `2106.11959` ★
- Somepalli et al. 2021 — SAINT — `2106.01342`
- Bahri et al. 2021 — SCARF — `2106.15147`
- Wang et al. 2021 — DCNv2 — `2008.13535`
- Borisov et al. 2021 — DL & tabular survey — `2110.01889`
- Müller et al. 2022 — PFNs (Transformers Can Do Bayesian Inference) — `2112.10510`
- Hollmann et al. 2022 — TabPFN v1 — `2207.01848` ★
- ExcelFormer — Chen et al. 2023 — `2301.02819`
- Trompt — Chen et al. 2023 — `2305.18446`
- TabR — Gorishniy et al. 2023 — `2307.14338` ★ (retrieval-augmented)
- PyTorch Frame — Hu et al. 2024 — `2402.05964` ★ (row encoder for RDL)
- LoCalPFN — Thomas et al. 2024 — `2406.05207`
- TabReD — Rubachev et al. 2024 — `2406.19380` ★ (industry, temporal splits)
- RealMLP / Better by Default — Holzmüller et al. 2024 — `2407.04491` ★
- TALENT toolbox — Liu et al. 2024 — `2407.04057`
- Survey on Deep Tabular Learning — 2024 — `2410.12034`
- TabM — Gorishniy et al. 2024 — `2410.24210` ★ (best DL baseline, ICLR 2025)
- Drift-Resilient TabPFN — Helli et al. 2024 — `2411.10634`
- TabICL — Qu et al. 2025 — `2502.05564` ★ (scales ICL to 500K rows, ICML 2025)
- A Closer Look at TabPFN v2 — Ye et al. 2025 — `2502.17361`
- Realistic eval of TabPFN v2 (open environments) — Cheng et al. 2025 — `2505.16226` (critical)
- TabArena — Erickson et al. 2025 — `2506.16791` ★ (living benchmark, NeurIPS 2025)
- TabPFN v2 — Hollmann et al. 2025 — *Nature* `s41586-024-08328-6` ★
- _Note:_ TabDPT, CARTE, and LLM-serialization methods (e.g. TabLLM) are optional skims — track via TabArena leaderboard rather than chasing each.

### Year 3 — Graph ML
- Gilmer et al. 2017 — MPNN — `1704.01212` ★
- Kipf & Welling 2017 — GCN — `1609.02907` ★
- Hamilton et al. 2017 — GraphSAGE — `1706.02216` ★
- Veličković et al. 2018 — GAT — `1710.10903`
- Schlichtkrull et al. 2018 — R-GCN — `1703.06103` ★
- Xu et al. 2019 — GIN — `1810.00826`
- Hu et al. 2020 — HGT — `2003.01332` ★
- Rossi et al. 2020 — TGN — `2006.10637` ★
- Hu et al. 2020 — OGB — `2005.00687`

### Year 4 — Relational deep learning
- Cvitkovic 2019 — DL on relational data — `1903.06430`
- RelBench beta — 2023 — `2312.04615`
- Fey et al. 2024 — RDL position (ICML) — see PMLR v235; arXiv `2312.04615` lineage ★
- RelBench v1 — Robinson et al. 2024 — `2407.20060` ★
- ContextGNN — Yuan et al. 2024 — `2411.19513` ★ (recsys / link prediction)
- RelGNN — Chen et al. 2025 — `2502.06784` ★ (composite message passing, ICML 2025)
- RelGT — Dwivedi et al. 2025 — `2505.10960` ★ (Relational Graph Transformer, ICLR 2026)
- RDL survey — 2025 — `2506.16654`

### Year 5 — Foundation relational models
- Zahradník et al. 2023 — Towards FMs for relational DBs — `2305.15321` ★
- Griffin — Wang et al. 2025 — `2505.05568` ★ (graph-centric RDB FM, ICML 2025)
- RDB-PFN — Wang et al. 2026 — `2603.03805` ★ (synthetic-prior relational FM, open code)
- GelGT (Gaussian Relational Graph Transformer) — 2026 — `2605.15575`
- RelBench v2 — `2602.12606` · RelGT-AC — `2606.03040`
- KumoRFM — 2025 industry technical report (proprietary in-context relational FM)

**Currency rule:** This index is a snapshot. Before each new quarter, run an arXiv search for the quarter's topic sorted by `submitted` and add any paper that (a) sets new SOTA on RelBench/TabArena, (b) exposes a failure mode, or (c) is a baseline you'll be measured against. Do not add papers that merely apply an existing method.
