# Expert Curriculum — Relational Deep Learning

Agent-facing plan for lesson sequencing. The student-facing version is [reference/curriculum.html](./reference/curriculum.html).

**Pace:** ~1 hour/day **baseline** (minimum on typical days) · ~360 hours/year at baseline · 6 years to research-grade expertise at that pace. Extra study time accelerates depth and checkpoint quality, not year-skipping.  
**Lesson numbering:** Year N → lessons `(N-1)*40 + 001` … `(N-1)*40 + 040` (approx.)  
**Rule:** Finish each quarter checkpoint before advancing. Papers are read in publication order within each year.

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

**Goal:** Know every major neural tabular architecture, when it wins, and why it still often loses to trees on single tables.

### Q1 · Neural tabular architectures (041–050)
| # | Topic | Paper | Lab |
|---|-------|-------|-----|
| 041 | Deep tabular history | Gorishnaya et al. 2021 full | rtdl repo setup |
| 042 | TabNet | Arik & Pfister 2019/2020 | Train TabNet |
| 043 | TabTransformer | Huang et al. 2020 | Compare to MLP |
| 044 | FT-Transformer | Gorishnaya et al. 2021 §3.3 | Train FT-T |
| 045 | SAINT / attention variants | Somepalli et al. 2021 | Read + compare table |
| 046 | NODE & piecewise networks | Popov et al. 2019 | When irregular functions help |
| 047 | DCN & explicit crosses | Wang et al. 2021 | Feature crosses |
| 048 | Deep ensemble vs trees | Grinsztajn 2022 revisit | Same protocol comparison |
| 049 | Tabular DL survey skim | Borisov et al. 2024 or Shwartz-Ziv 2022 | Map architecture families |
| 050 | **Q1 checkpoint** | Gorishnaya 2021 | FT-T vs XGB on 3 datasets |

**Papers (chronological):** Arik 2019 · Huang 2020 · Popov 2019 · Gorishnaya 2021 · Somepalli 2021 · Wang 2021

### Q2 · Foundation tabular models (051–060)
| # | Topic | Paper | Lab |
|---|-------|-------|-----|
| 051 | In-context learning for ML | Müller et al. 2022 (PFNs) | Concept: prior-fitting |
| 052 | TabPFN v1 | Hollmann et al. 2022 | Run TabPFN |
| 053 | TabPFN Nature | Hollmann et al. 2024 (Nature) | Compare to 4h tuned XGB |
| 054 | Synthetic pre-training priors | Hollmann 2022 §4 | What prior encodes |
| 055 | Limits of TabPFN | Nature paper limitations | When PFN fails |
| 056 | Transfer & fine-tuning tabular | Recent TabPFN follow-ups | Fine-tune experiment |
| 057 | LLMs on tables (critical view) | Gorishnaya / survey LLM section | Serialize vs structure |
| 058 | Single-table ceiling | Synthesis lecture | Write: what's fundamentally missing |
| 059 | Bridge to graphs | Fey 2024 §2–3 preview | Why rows aren't enough |
| 060 | **Q2 checkpoint** | Hollmann 2024 | TabPFN vs your Y1 baseline |

**Papers:** Müller 2022 · Hollmann 2022 · Hollmann 2024 (Nature)

### Q3 · Representation & self-supervision on tables (061–070)
| # | Topic | Paper | Lab |
|---|-------|-------|-----|
| 061 | VIME / masked tabular | Yoon et al. 2020 | Masked pretraining concept |
| 062 | SCARF / contrastive tabular | Bahri et al. 2021 | Contrastive views |
| 063 | SubTab | Ucar et al. 2021 | Multi-view SSL |
| 064 | When SSL helps tabular | Survey + your homework VIME notes | SSL ablation design |
| 065 | PyTorch Frame preview | Hu et al. 2024 | Read row encoder design |
| 066 | Heterogeneous column types | PyTorch Frame docs | Encode mixed schema |
| 067 | Stacking encoders + heads | RDL preview | Encoder → predictor pattern |
| 068 | Benchmark hygiene | Re-read Grinsztajn protocol | Same splits across models |
| 069 | Error analysis | — | Confusion / calibration study |
| 070 | **Q3 checkpoint** | Hu 2024 (PyTorch Frame) | Implement row encoder stub |

**Papers:** Yoon 2020 · Bahri 2021 · Hu et al. 2024 (PyTorch Frame)

### Q4 · Year 2 synthesis (071–080)
| 071–074 | Reproduce rtdl leaderboard entry | Gorishnaya 2021 | Full reproduction |
| 075 | Write: neural tabular decision tree | — | When to use which arch |
| 076 | Prepare for graph phase | Gilmer 2017 preview | Message passing intro |
| 077 | Math: adjacency & propagation | Kipf 2017 §2 | Manual one-hop aggregate |
| 078 | Relational preview | Schlichtkrull 2018 abstract | R-GCN motivation |
| 079 | Year 2 essay | — | Single-table limits |
| 080 | **Year 2 exit exam** | All Y2 papers | Teach-back: 3 inductive biases + TabPFN |

**Year 2 exit criterion:** Train FT-Transformer and TabPFN; articulate single-table ceiling in writing.

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
**Papers (2024–2025):** RelGNN (Chen 2025) · RelGT (Dwivedi 2025) · RDL survey (2025)

| 141 | Composite message passing | Chen 2025 RelGNN | Atomic routes concept |
| 142 | RelGNN reproduction | Chen 2025 | Run on RelBench subset |
| 143 | Relational Graph Transformer | Dwivedi 2025 RelGT | Hybrid attention |
| 144 | Many-to-many edges | Chen 2025 §motivation | Schema pathology |
| 145 | Survey: next-gen architectures | arXiv 2506.16654 | Map open problems |
| 146 | LLM + RDL (critical) | Survey §4.2 | When text helps / hurts |
| 147 | Ablation discipline | — | Encoder vs MP vs data |
| 148 | Write reproduction report | — | Match or explain gap |
| 149 | Identify weakest RelBench tasks | — | Where thesis might fail |
| 150 | **Q3 checkpoint** | Chen 2025 or RelGT | SOTA or near-SOTA on 1 task |

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
**Papers:** Müller 2022 PFN · Hollmann 2024 · Zahradník 2023 · Fey survey 2025 §FM

| 161 | What is a foundation model | Bommasani 2021 (FM definition) | Scope for relational |
| 162 | Pre-training on many databases | Zahradník 2023 §3 | Schema diversity |
| 163 | LM encoders for rows | Zahradník 2023 §4 | Text vs typed columns |
| 164 | GNN over pre-encoded nodes | Zahradník 2023 Fig. 1 | Combined architecture |
| 165 | Self-supervised tasks on REG | Fey 2024 FM section | Masking / link prediction |
| 166 | In-context relational learning | KumoRFM docs / papers | Few-shot task adaptation |
| 167 | Griffin & open alternatives | Literature search | Reproduce what's public |
| 168 | Cross-database generalization | Zahradník 2023 §experiments | Zero-shot schema |
| 169 | Scaling laws (relational) | Survey 2025 | Open questions |
| 170 | **Q1 checkpoint** | Zahradník 2023 full | Written FM design doc |

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
| 181 | RelBench v2 / updates | arXiv 2602.12606 | New tasks & databases |
| 182 | RelGNN + FM combination | — | Hypothesis generation |
| 183 | RelGT + pre-training | — | Architecture research gap |
| 184 | Tabular FM → relational FM | — | Transfer lessons from TabPFN |
| 185 | Causal & relational data | — | When correlation ≠ strategy |
| 186 | Production constraints | Huyen | Latency, freshness, monitoring |
| 187 | Ethics & privacy on REG | — | Node-level privacy |
| 188 | Literature review skill | — | Systematic paper tracking |
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
| Y2 neural tabular | Architecture recall | Paper section | rtdl / TabPFN run |
| Y3 graphs | MPNN equation | GNN paper | PyG implementation |
| Y4 RDL | REG sketch from memory | RelBench / RDL paper | RelBench training run |
| Y5 FM | Pre-training objective | Zahradník / survey | Fine-tune experiment |
| Y6 research | Hypothesis restatement | Related work | Code / write |

---

## Lesson production schedule (for agent)

- Produce **1 lesson HTML per lecture row** as the student reaches it (~2–3 lessons/week at 1h/day including reading).
- Each lesson: one concept, one quiz, one primary paper link, one lab step.
- Every 10 lessons: checkpoint lesson with longer exercise.
- Update [[GLOSSARY.md]] when user demonstrates term mastery.
- Track paper completion in [[NOTES.md]] with `✓ paper-id` lines.
