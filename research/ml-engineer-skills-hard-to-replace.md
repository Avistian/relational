# Deep Research: Which ML Engineer Skills Are Hard to Replace

**Question.** Which skills of a machine learning engineer are hard to replace with AI, and what should a working (or aspiring) ML engineer develop to stay relevant?

**As of.** 31 August 2026. Evidence through mid-2026 labor data and 2025–2026 capability evaluations.

**Scope.** The unit of analysis is the *ML engineer* (MLE) as practiced in industry: people who turn messy organizational data into models that affect decisions, then keep those systems honest after launch. That is a different job from (a) foundation-model research at a frontier lab, (b) “AI engineer” work that wires LLM APIs into products, and (c) Kaggle-style modeling on a frozen CSV with a known metric. Those three jobs are being automated at different rates; treating them as one title is how career advice goes wrong.

**Method.** Exhaustive synthesis of primary labor studies, capability benchmarks, production-ML literature, regulation, and contrarian economics. Firecrawl was not available in this environment; collection used web search plus full-text retrieval of papers and official reports. Expert social-media search (X) was attempted and blocked by API access.

---

## Executive Summary

The parts of ML engineering that look like *well-specified software on a clean dataset with a known metric* are already being automated, and the automation is accelerating. Agents medal on a rising share of Kaggle-style ML engineering tasks ([MLE-bench](https://openai.com/index/mle-bench/); [MLE-STAR](https://arxiv.org/html/2506.15692)). METR’s 50% time horizon on self-contained software/ML tasks has been doubling on the order of every seven months ([METR, March 2025](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/); [live tracker](https://metr.org/time-horizons/)). OpenAI’s GDPval finds frontier models approaching expert quality on one-shot, well-briefed knowledge-work deliverables ([OpenAI, 2025](https://openai.com/index/gdpval/)). Hiring data is consistent with this: junior, codified work is being squeezed first ([Brynjolfsson, Chandar, and Chen, August 2026](https://digitaleconomy.stanford.edu/app/uploads/2026/08/Canaries_August2026.pdf); [Westby, Modestino, and Cheng, 2026](https://www.iza.org/publications/dp/18723/generative-ai-and-the-redefinition-of-entry-level-software-work)).

The parts that remain hard to replace share a different shape. They are *underspecified*, *high-context*, *silent when they fail*, and *legally or organizationally attached to a person*. That cluster includes: deciding whether a prediction problem is even the right problem; designing evaluations that measure the decision you care about rather than a leaderboard number; catching leakage and distribution shift; doing the unglamorous data work that actually determines model behavior; keeping live systems honest across feedback loops and training-serving skew; translating between domain experts, product, and risk; and remaining the accountable overseer when the system is wrong. These are not “soft skills” bolted onto ML. They *are* the ML job once the model-fitting loop is cheap. Google’s production literature has been saying this since 2015: only a small fraction of a real ML system is the learning code ([Sculley et al., 2015](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)).

Two caveats prevent this from being a comforting list. First, “hard to replace *today*” is not “hard to replace *in five years*.” Time-horizon doubling, if it continues, moves week-scale autonomous work into range before the end of the decade ([METR](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/)). Second, some of the skills that feel like expertise — architecture shopping, hyperparameter folklore, framework trivia — are the ones the Bitter Lesson predicts will lose ([Sutton, 2019](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)). The durable investment is not a bigger zoo of models. It is the ability to *pose, validate, and own* a learning system in a messy world, using models (and agents) as instruments.

---

## Key Findings

1. **Replacement is happening at the task level, not the job-title level.** Occupations are bundles of tasks. Anthropic’s usage data shows AI covering a quarter of tasks in a growing share of jobs (36% of occupations in early 2025, 49% by late 2025) while almost never covering three-quarters of a job’s tasks ([Anthropic Economic Index, Feb 2025](https://www.anthropic.com/research/the-anthropic-economic-index); [Jan 2026 primitives report](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report)). Computer programmers and data scientists are among the heaviest users *and* among the most exposed ([Anthropic labor-market impacts](https://www.anthropic.com/research/labor-market-impacts)).

2. **The junior, well-specified layer is already shrinking.** Employment of 22–25-year-olds in AI-exposed occupations stood 19% below the path of their less-exposed peers as of June 2026; experienced workers show no comparable gap. The channel is reduced hiring, not mass layoffs, and it concentrates where AI *substitutes* for tasks rather than complements them ([Canaries, Aug 2026](https://digitaleconomy.stanford.edu/app/uploads/2026/08/Canaries_August2026.pdf)). Stanford HAI’s 2026 AI Index reports software-developer employment ages 22–25 down nearly 20% since 2024 even as older headcount grows ([Stanford HAI, Apr 2026](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)). Remaining junior software vacancies have shifted toward problem solving, communication, and attention to detail — not “AI tool” keywords ([IZA DP 18723](https://www.iza.org/publications/dp/18723/generative-ai-and-the-redefinition-of-entry-level-software-work)).

3. **Kaggle-shaped ML engineering is the capability frontier, not the job.** On MLE-bench (75 Kaggle competitions), o1-preview + AIDE medaled on 16.9% of competitions in late 2024 ([Chan et al.](https://arxiv.org/pdf/2410.07095)). By 2025, MLE-STAR reported medal rates of 43.9% (Gemini-2.0-Flash) and over 60% (Gemini-2.5-Pro) on the Lite split ([Nam et al.](https://arxiv.org/html/2506.15692)). The MLE-bench authors themselves warn that Kaggle problems are clean, with clear metrics, whereas “figuring out the problem is part of the challenge” in the real world. Independent work finds medal rates inflate versus actual Kaggle leaderboards ([KompeteAI](https://arxiv.org/pdf/2508.10177)).

4. **Agents beat humans on short, scored ML research sprints and lose on long, messy ones.** On RE-Bench, the best agents scored 4× human experts at a 2-hour budget, humans narrowly led at 8 hours, and humans scored ~2× agents at 32 hours. Agents can write a faster Triton kernel than the human baseline; they get worse returns to extra time ([Wijk et al., ICML 2025](https://proceedings.mlr.press/v267/wijk25a.html)). METR is explicit that an N-hour time horizon is closer to what a *low-context contractor* can do in N hours, not what a high-context professional does as part of a job, and that agents do worse when scoring is holistic rather than algorithmic ([METR time-horizons FAQ](https://metr.org/time-horizons/)).

5. **The production ML job was never mostly model code.** Sculley et al. (2015) documented CACE (“Changing Anything Changes Everything”), hidden feedback loops, undeclared consumers, unstable data dependencies, glue code, and pipeline jungles — and the cartoon that ML code is a tiny box inside a vast infrastructure diagram. Google’s ML Test Score (28 tests across data, model, infrastructure, monitoring) exists because prediction behavior cannot be specified a priori ([Breck et al., 2017](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/)). Zinkevich’s Rule #1 is still the first production skill: don’t be afraid to ship without ML ([Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)).

6. **Silent validity failures are still a human comparative advantage — and a crisis.** Kapoor and Narayanan found leakage affecting on the order of 294–329 papers across 17 scientific fields; in a civil-war-prediction case study, claimed ML gains over logistic regression vanished after leakage was fixed ([Patterns, 2023](https://doi.org/10.1016/j.patter.2023.100804)). A 2026 methodology paper notes leakage identified in 648 published papers as of mid-2024, and argues tools still do not enforce what textbooks teach ([arXiv:2603.10742](https://arxiv.org/html/2603.10742v4)). Agents optimize the metric you give them. They do not notice that the metric is the wrong object.

7. **Data work is undervalued, pervasive, and where models actually fail.** Interviews with 53 practitioners found data cascades in 92% of high-stakes AI projects: compounding, delayed, often avoidable failures from undervaluing data quality. The cultural line is the title: “Everyone wants to do the model work, not the data work” ([Sambasivan et al., CHI 2021](https://dl.acm.org/doi/10.1145/3411764.3445518)).

8. **Live-system reliability, not peak benchmark score, is the scarce resource.** College-level tasks in Anthropic’s Claude usage data see ~12× speedup but only ~66% success (vs ~9× and ~70% for high-school-level tasks). Software-development success is estimated at 61% versus 78% on personal tasks. API traffic is ~75% automation; consumer chat is back to ~52% augmentation. Adjusting implied productivity for success rates roughly halves the gain ([Jan 2026 primitives](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report)). A 66% success rate is a tool. It is not an unsupervised employee.

9. **Law and liability keep a competent human in the loop even after capability catches up.** EU AI Act Article 14 requires high-risk systems to be *designed* so natural persons can monitor, interpret, override, and halt them, and to remain aware of automation bias. Oversight persons must understand capacities and limitations ([official text](https://artificialintelligenceact.eu/article/14/); high-risk Annex III obligations phase in 2 December 2027). This does not freeze the MLE role, but it creates demand for people who can actually exercise that oversight rather than rubber-stamp it.

10. **The hiring market is splitting, and “AI engineer” is not “ML engineer.”** Live-JD analyses in 2026 show AI Engineer postings dominated by LLM integration, Python, evals, and agents; PyTorch/TensorFlow are receding as filters ([Dexity, 390 JDs](https://dexity.com/intel/ai-engineer-career-path-2026); [425 JDs](https://dexity.com/intel/what-is-an-ai-engineer-2026)). Classical MLE remains a large installed base (HeroHunt cites Ravio: ML Engineer ~45% of AI/ML titles) but the value proposition has shifted toward deploying, monitoring, and maintaining ([HeroHunt, 2026](https://www.herohunt.ai/blog/fastest-growing-ai-roles-in-2026-data-and-rankings/)). WEF still lists AI and ML specialists among the fastest-growing roles, with AI and big data the fastest-growing *skill* — alongside analytical thinking, systems thinking, and creative thinking ([Future of Jobs 2025](https://reports.weforum.org/docs/WEF_Future_of_Jobs_Report_2025.pdf)).

11. **Complementarity, not raw capability, predicts who keeps a job.** Canaries fact (5): employment declines concentrate where AI substitutes; complementary use is associated with flat or rising employment, especially for experienced workers. Occupations involving *codified* knowledge show the junior squeeze; *tacit* knowledge occupations show faster growth for experienced workers ([Canaries](https://digitaleconomy.stanford.edu/app/uploads/2026/08/Canaries_August2026.pdf)). This is Autor’s old point in new data: tasks we cannot fully specify tend to be complemented, not replaced ([Autor 2014 / 2015](https://doi.org/10.1257/jep.29.3.3)).

12. **Specializing in “which architecture to train” is the worst bet; specializing in messy structured prediction is a better one.** Trees still dominate much of production tabular work operationally (speed, tooling, auditability), while tabular foundation models (TabPFN-class) are eating the small-data fitting loop ([Grinsztajn et al. 2022 line of results](https://andreasbergstrom.dev/posts/the-llm-shaped-hole-in-your-xgboost-pipeline); [Citi Ventures on LTMs, 2026](https://www.citi.com/ventures/perspectives/opinion/ltms-large-tabular-models-startups-enterprise-2026.html)). LLMs are a poor default regressor on rows. The remaining scarcity is *relational, temporal, leakage-safe decisioning on organization-specific tables* — which is also where most of the world’s money-making models actually live.

---

## Detailed Analysis

### 1. A test for durability (so this is not a vibe list)

A skill is hard to replace when most of the following are true:

| Test | Easy to replace | Hard to replace |
|---|---|---|
| **Specification** | Success metric is given and scorable | Deciding *what* to optimize is the work |
| **Context** | Self-contained repo / CSV / brief | Years of org, product, and domain residue |
| **Failure mode** | Unit test fails now | Looks great offline, fails six months later |
| **Feedback** | Leaderboard, loss curve, CI | Drift, feedback loops, silent leakage |
| **Accountability** | Nobody signs it | Someone is on the hook to a regulator, a CFO, or a patient |
| **Complementarity** | AI does the same task cheaper | AI makes the remaining human judgment more valuable |

This is the Autor/Acemoglu task framework applied to one occupation ([Autor, “Why Are There Still So Many Jobs?”](https://doi.org/10.1257/jep.29.3.3); [Acemoglu, Autor, and Johnson, 2026](https://doi.org/10.3386/w34854)). It also matches how METR, GDPval, and MLE-bench authors describe their own *limitations*: they evaluate well-specified, low-context, algorithmically scored work. That is exactly the slice that is disappearing from junior job descriptions.

### 2. What is already easy, or getting easy

**Fitting a model to a defined table.** AutoML (AutoGluon, H2O, NAS) automated the inner loop years ago within a predefined search space. LLM agents now search in *code* space. Medal-rate progress on MLE-bench Lite from ~17% to ~40–60% in roughly a year is the relevant number, with the caveat that offline medal rates overstate live Kaggle performance. If your weekly output is “try XGBoost, then LightGBM, then a small MLP, pack an ensemble,” you are competing with a process that is getting cheaper every quarter.

**Boilerplate software around the model.** Glue code, training scripts, simple APIs, dashboard stubs, and “write the PyTorch loop” are inside current coding-agent competence. Anthropic’s top task is literally “modifying software to correct errors” (6% of Claude.ai conversations; ~10% of API records) ([Jan 2026 primitives](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report)). GDPval’s one-shot expert-vs-model comparisons are already near parity on a large minority of deliverables.

**Framework identity and hyperparameter folklore.** Dexity’s 2026 AI-engineer JDs: LLMs 63%, Python 59%, evals 56%, agents 50%, PyTorch 33%, TensorFlow 18%, fine-tuning ~26%. 365 Data Science’s broader AI-engineer scrape puts hyperparameter tuning at ~1% of postings, which they attribute to AutoML. These are not proofs that PyTorch is useless. They are proofs that *listing PyTorch on a résumé is no longer a differentiator*.

**Prompt engineering as a standalone craft.** It is being absorbed into evals, tooling, and models. JD share is small relative to “can you measure whether the system works.”

**Junior implementation of a spec someone else wrote.** This is the Canaries/IZA result. Firms are not firing seniors en masse. They are not hiring people whose job is to turn a complete ticket into code.

### 3. Durable skill cluster A — Problem formulation

Zinkevich’s first three rules are still the highest-leverage MLE skills: ship without ML if a heuristic will do; make sure the objective is observable and attributable; prefer ML to a *complex* heuristic once you have data ([Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)). None of these are “train a model.” All of them are *deciding whether and why to train a model*.

GDPval’s own limitations section is the tell: tasks arrive as a prompt plus reference files. Real work often starts earlier — “a lawyer might have to navigate ambiguity and talk to their client before deciding that creating a legal brief is the right approach.” METR says the same about jobs: most are not well-specified algorithmic tasks; they require interacting with other people and success metrics that cannot be algorithmically scored.

For an MLE this looks like:

- Translating a business pain (“churn is up,” “claims cost too much,” “the ranking feels off”) into a prediction or decision problem, or refusing to.
- Choosing the unit of analysis (user, account, claim, session, entity graph).
- Choosing an action-relevant objective (calibrated probability of a *preventable* event, uplift, cost-weighted error) rather than AUC on a convenient label.
- Naming the counterfactual: is this prediction or intervention? Training on “what happened” is not the same as estimating “what would happen if we treated.”

Causal language is not decoration here. If the system will change the world it is trained on, you are already in Sculley’s feedback-loop regime. Bandits, uplift, and “do not train on your own policy’s outcomes without saying so” are formulation skills, not library skills.

**Why agents struggle.** Agents are strong when the loss is given. Formulation is the step that *creates* the loss. There is no unit test for “we framed the wrong problem.”

### 4. Durable skill cluster B — Evaluation, leakage, and metric honesty

This is the strongest empirical case in the whole file.

Kapoor and Narayanan’s taxonomy has eight leakage types, from textbook train/test contamination to open research problems (temporal leakage, non-i.i.d. grouping, illegitimate features that would not exist at prediction time). Their civil-war case study is brutal: papers claimed complex ML vastly beat logistic regression; after fixing leakage, it did not. The errors were not catchable by reading the papers.

That is the MLE’s job in production: the model will look good. Your job is to ask *relative to what, on which split, with which information set, for which decision*. Point-in-time joins, entity-grouped splits, nested cross-validation, and “split first, transform second” are not academic niceties. They are the difference between a system that survives contact with production and a system that was cheating.

The 2026 “grammar of ML workflows” paper is useful as a *negative* result too: if leakage is still so common that researchers are designing type systems to make it unrepresentable, then neither AutoML nor LLM codegen has solved it. Agents will happily preprocess before splitting if that is what the notebook they imitate does.

**Evals for generative systems are the same skill in a new costume.** Dexity finds evals in 56% of AI-engineer JDs and calls them the #1 differentiator. LLM-as-judge, labeled challenge sets, prompt-regression suites, and “what does good look like for this product” are evaluation design. The people who can build those loops are the people who used to build offline/online metric stacks for ranking and fraud. The object changed. The discipline did not.

**Develop this by doing the unglamorous thing:** take any public tabular dataset and invent three ways the standard random split lies (time, entity, leakage via a post-treatment feature). Then invent the split that would have caught it. This is higher expected value than another architecture paper.

### 5. Durable skill cluster C — Data work and data-centric judgment

Sambasivan et al. should be required reading for anyone whose plan is “I will stay relevant by knowing models.” Data cascades are pervasive (92%), invisible, delayed, and often avoidable. Prestige and citations accrue to models; care of data does not. Clients expect magic. Practitioners hack demo metrics. CS training uses UCI and Kaggle, where the data arrives already born.

Andrew Ng’s data-centric campaign and the MIT DCAI framing are the constructive version: systematically improve the dataset (label consistency, error analysis, slices) rather than only the model ([dcai.csail.mit.edu](https://dcai.csail.mit.edu/2024/data-centric-model-centric/)). Zinkevich’s early rules say the same from the other direction: get instrumentation and pipelines right *before* the first clever model.

What is durable here is not “I can pandas-clean a CSV.” Agents are fine at that. Durable data skill is:

- Knowing what the rows *mean* in the generating process (who logged this, when, under what incentive to lie or delay).
- Label ontology: what is a default, a churn, a fraud, a “good” ranking, and who decided.
- Missingness as information, not a median-impute inconvenience.
- Slice analysis: the model is fine on average and catastrophic on the slice that is the business.
- Deciding that you do not have the data, and saying so.

Sambasivan’s practitioners involved domain experts only at collection time, not end to end. The durable MLE inverts that: domain experts stay in the loop because they are the only source of ground truth that is not already in the warehouse.

### 6. Durable skill cluster D — Production systems: CACE, skew, loops, monitoring

Sculley’s list has not expired. If anything, LLM products added new versions of the same diseases (undeclared consumers of a prompt chain; hidden feedback when users adapt to the model and the model trains on the users; glue code around every vendor SDK).

The skills:

- **CACE.** Changing a feature, a threshold, a prompt, or an upstream model changes everything. You need isolation, versioning, and “what would have to be true for this not to matter.”
- **Training-serving skew.** The feature you trained on is not the feature you serve. This is still how ranking and fraud models die. Feature stores and point-in-time joins are the engineering expression of an epistemic rule: *only use information that would have been knowable at decision time*.
- **Hidden feedback loops.** Two systems influencing each other through the world. You will not see this in an agent’s Kaggle sandbox.
- **Monitoring the data, not just the model.** ML Test Score’s “minimum across categories” scoring is the right instinct: a perfect model-test suite with no data monitoring is not production-ready.
- **Rollback, canaries, and config as code.** Knight Capital is Sculley’s reminder that dead experimental codepaths are not a style issue.

This is why “MLOps certifications” are a weak proxy and “I have owned a model in production through a messy incident” is a strong one. The certification is specified. The incident is not.

### 7. Durable skill cluster E — Domain expertise, relational structure, and decision quality

Most economically valuable supervised models are still trained on tables that live in companies: claims, transactions, CRM, sensors, EHR event streams. They are relational, temporal, and hostile to i.i.d. fantasy. LLMs do not replace that stack; they sit beside it (text features, copilots, document intake). Tabular foundation models compress the *fitting* step on small-to-medium clean tables. They do not know your entity graph, your delayed labels, or your regulatory definition of a default.

That is the actual moat for an MLE who is not at a frontier lab:

- **Entity and time as first-class.** Customer–account–transaction graphs; as-of joins; delayed outcome windows.
- **Knowing when a tree is enough**, when a tabular FM is enough, and when the relational structure is the signal (the RelBench-shaped bet).
- **Calibration and decision thresholds** under cost asymmetry, not headline AUC.
- **Enough domain language to argue with a fraud analyst, an actuary, a clinician, or an ops manager without being captured by them.**

Autor’s complementarity claim is the economic version: when some tasks automate, the remaining tacit, flexible, judgment-heavy tasks become *more* valuable if they are still needed in the bundle. Canaries’ 2026 payroll data is consistent with that for experienced workers in complementary occupations.

The risk, which Acemoglu, Autor, and Johnson name *expertise-leveling*, is that AI also flattens some expertise by making novices “good enough.” That is happening to junior coding. It will happen to junior modeling. It is much slower to happen to “I know why this claims triangle is lying” or “this join invents a feature from the future.”

### 8. Durable skill cluster F — Oversight, governance, and being the person who can say no

Article 14 of the EU AI Act is unusually concrete. High-risk systems must be designed so a natural person can: understand capacities and limits; watch for anomalies; resist automation bias; interpret outputs; disregard or reverse them; halt the system safely. Deployers must assign people with competence, authority, and resources. This is not a “ethics module.” It is a job description.

Even outside the EU, banks, health systems, and insurers already have model-risk management, validation independent of development, and documentation burdens. The MLE who can write a model card that a validator cannot punch through, who can explain a slice failure to a risk committee, and who will refuse to ship a leaked metric, is scarce relative to the MLE who can fine-tune.

Anthropic’s reliability numbers are the operational reason this does not go away when models get better: 61–66% success on complex work is exactly the regime where unsupervised automation is malpractice and supervised automation is a productivity gain. Their own analysis says weighting by success roughly halves implied productivity.

**Automation bias is now a listed legal hazard.** The skill is noticing that you, too, will trust the dashboard.

### 9. Durable skill cluster G — Taste over long horizons, and using agents as labor

RE-Bench is the cleanest picture of the current human/agent division of labor in ML itself. Agents explore faster and sometimes find local engineering wins humans miss (the Triton kernel). Humans get more from additional hours: they reframe, abandon a dead approach, and keep a thread of “what would actually constitute progress.” That is research taste. It is also what a senior MLE does when an investigation is not a ticket.

The complementary skill — new since 2024 — is **being good at delegating to agents without being fooled by them**. That means:

- Writing specs and evals the agent can be scored on.
- Knowing which tasks are “2-hour RE-Bench-shaped” (delegate) vs “32-hour messy” (stay).
- Catching reward hacks, leakage, and “it passed the unit test and missed the point.”
- Keeping context the agent does not have (the incident last quarter, the political constraint, the data-retention rule).

This is the non-cynical reading of the 2026 JD spike in “agents + evals.” Firms want people who can operate a fleet of cheap interns that hallucinate.

### 10. What the labor market is actually paying for

Putting the JD evidence next to the payroll evidence:

- **Do not try to enter as a human AutoML.** That layer is shrinking.
- **Do not try to enter only as an “LLM wrapper” engineer without evals and production judgment.** That title is crowded and the easy work will compress.
- **Seniors who combine systems ownership + evaluation discipline + domain context are not showing the Canaries gap.**
- **The remaining junior door is “I can formulate, communicate, and notice,”** which is exactly what IZA finds in leftover junior vacancies.
- WEF’s rising-skill list (analytical thinking, systems thinking, creative thinking, resilience, curiosity) is vague, but it is directionally the same as the durability table. “AI and big data” as a skill is table stakes, not a moat.

McKinsey-style trillion-dollar productivity headlines and Acemoglu’s ~0.66% TFP over ten years cannot both be gospel ([Acemoglu, *The Simple Macroeconomics of AI*](https://doi.org/10.3386/w32487)). For career planning, you do not need to pick a world GDP number. You need the *micro* fact both sides share: **well-specified cognitive tasks compress; judgment, context, and new-task creation do not compress at the same rate.**

### 11. Implications for someone training on relational / tabular ML

This repository’s own mission already points at several of the durable clusters: leakage-free temporal splits, honest baselines, relational structure the single-table paradigm misses. That is a better career bet than “I learned the latest LLM agent framework,” with two conditions.

**Condition 1.** Do not let the relational thesis become architecture shopping. CURRICULUM.md already says GBDTs are not dead and that trees plus strong-default MLPs still win a lot of single-table industrial data. The durable skill is *knowing when the relational inductive bias pays for itself*, with a baseline that would embarrass you if you skipped it. Agents will generate RelGNN code. They will not reliably tell you that XGBoost on a leaky point-in-time join is lying.

**Condition 2.** Pair the modeling stack with production-shaped evaluation and a domain. A RelBench reproduction is training. A RelBench reproduction plus a write-up of where leakage would have entered, what decision the metric proxies, and what would be monitored in production is a portfolio piece that still looks like 2028 work.

The undervalued relational-FM bet in MISSION.md is, in labor-market language, a bet that **messy multi-table prediction remains economically central and technically unsolved**. The evidence in this report is consistent with that bet. It is not consistent with the bet that “knowing the paper list” is the scarce resource.

---

## Contrarian Views And Risks

**1. The Bitter Lesson says your craft is a trap.** Sutton’s argument is that human-knowledge approaches win the short run and lose the long run to search plus learning plus compute ([2019](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)). Tabular foundation models and MLE agents are that story applied to the MLE’s inner loop. If you invest in feature-engineering folklore the way speech researchers invested in vocal-tract knowledge, you may be the person the lesson is bitter for. *Partial agreement:* do not build your identity on tricks that scale with your patience. *Disagreement:* the Bitter Lesson is about *how to get better predictions given a well-posed problem*. It does not pose the problem, define leakage, or take responsibility for a live loop.

**2. Time horizons may eat the “messy, long” residual.** METR’s own authors warn people over-interpret the number, note 40–100× lower horizons on visual computer-use, and flag that measurements above 16 hours are unreliable with the current suite. They also say the *slope* (doubling every ~6–7 months historically, possibly faster recently) is the thing to take seriously. If that slope holds, “humans win at 32 hours” is a 2024–2025 fact, not a law. *This is the most important risk to the present report.* Treat every “durable” skill as “durable at current horizons,” and keep re-testing against agents.

**3. Expertise-leveling can devalue the middle without replacing the expert.** Acemoglu, Autor, and Johnson distinguish labor-augmenting, automating, expertise-leveling, and new-task-creating technical change. Only new-task creation is unambiguously pro-worker. Generative AI is very good at expertise-leveling: it makes a novice’s first draft look like a median MLE’s. That compresses wages for the middle even if the top still signs the model. Career implication: **be closer to “defines new tasks and owns outcomes” than to “executes known ML recipes faster.”**

**4. Polanyi’s paradox is already half-defeated.** Autor (2014/2015) argued that what we cannot specify we cannot automate, and that ML is the path that infers tacit rules from examples. That path worked. The remaining paradox is different: we can *transfer* tacit skill into a network without being able to *audit* it. That makes oversight harder, not easier, and it is why evaluation and governance become more valuable as models absorb tacit tricks.

**5. “Durable skills” advice is often motte-and-bailey career coaching.** The motte: “judgment and problem formulation matter.” The bailey: “therefore a six-month thought-leadership course.” This report’s list is only as good as the evidence that (a) current evals fail at these tasks, (b) production incidents still come from them, and (c) hiring/payroll data show complementarity rather than substitution. If in two years agents reliably catch temporal leakage, write model-risk memos that pass validators, and operate production loops with low incident rates, the list is wrong. That is a falsifiable claim, not a brand.

**6. Regulation is a lagging, geographic moat.** Article 14 helps EU-facing high-risk work. It does not protect a growth-stage SaaS ranking model in a lax jurisdiction. Do not confuse compliance theater with skill.

**7. Macro uncertainty cuts both ways.** If Acemoglu is right and ten-year TFP gains are modest, firms will automate the easy 15% of tasks and keep most headcount, which *supports* “evolve the job.” If the METR extrapolation plus API-side 75% automation is the better forecast, task coverage deepens fast and titles compress. The Canaries paper is explicit that it is descriptive, not causal, that some gaps predate ChatGPT, and that ADP patterns are stronger than household-survey benchmarks. Do not overfit your career to one payroll series.

**8. Relational / tabular specialization could be a shrinking niche if LTMs + agents close it.** TabPFN-class models and relational FMs are themselves automation of the specialist. The hedge is to own *evaluation + production + domain* on top of that stack, not to own a particular architecture that a foundation model may absorb.

**9. The title “ML engineer” may lose to “AI engineer” in search even if the work remains.** Labor-market navigation is not the same as skill durability. You may need to speak evals, agents, and LLM product language to get in the room where the durable work happens.

---

## What to develop (practical, ranked)

Ranked by expected durability × how much the market already signals it. “Hard to replace” is not the same as “easy to get hired for next month”; the list notes both.

1. **Evaluation as a first-class engineering discipline.** Offline splits that respect time and entities; leakage hunts; slice metrics; calibration; online evals; LLM eval harnesses. Portfolio: a public write-up where a popular dataset’s standard split is shown to be invalid, plus the valid alternative. *Hiring signal: already loud (evals in ~half of AI-engineer JDs).*

2. **Problem formulation and metric design with a domain.** Pick one vertical and stay long enough to know when the label is a political object. Fraud, credit, claims, clinical risk, supply chain, and ranking are better than generic “I did Titanic with RAG.”

3. **Production ownership of a learning system.** Not a tutorial MLOps repo — a system you can describe by its incidents: skew, drift, a bad deploy, a feedback loop, a rollback. Study Sculley and the ML Test Score until they are instinct.

4. **Data generation-process literacy.** How rows get into the warehouse; delayed labels; selection bias; point-in-time correctness. This is the Sambasivan gap and the reason trees-plus-leaky-joins still embarrass deep models in the wild.

5. **Agent supervision.** Use coding/ML agents constantly, but keep a personal log of how they fail (wrong split, silent metric change, invented column, reward hack). That log is the training set for the skill firms actually need.

6. **Communication that changes a decision.** IZA’s leftover junior vacancies shifted to problem solving, communication, and care. Canaries’ survivors are experienced workers in complementary jobs. The mechanism is: you alter what the org does. Practice writing the one-page “do not ship / ship with this threshold / this is the wrong problem” memo.

7. **Enough math to not be bluffed — and to bluff less.** Not a new architecture zoo: probability, causal diagrams, experimental design, calibration, generalization under shift. This is what lets you argue with both a vendor and a paper.

8. **Legal-adjacent literacy if you work in high-stakes domains.** Article 14, model-risk management, data protection. Enough to design for oversight rather than bolting a human on at the end.

**Deprioritize as identity (keep as tools):** from-scratch implementations of standard architectures; hyperparameter folklore; being “the PyTorch person”; prompt-only expertise; certificates that attest to specified, scorable procedures.

**A weekly habit that matches the evidence:** one real evaluation bug hunt (leakage, skew, slice failure) for every three model-training sessions. The market is automating the three. It is not yet automating the one.

---

## Open Questions

1. **Does the METR doubling continue, slow, or hit a messiness wall?** METR already sees much lower horizons on visual computer use and worse performance on holistic scoring. The career-relevant question is the horizon on *unspecified* organizational ML work, which nobody has a clean benchmark for.

2. **Will evals themselves be automated?** If agents become reliable leakage detectors and metric critics, cluster B compresses. Watch whether MLE-bench-class agents start *refusing* invalid problem statements rather than optimizing them.

3. **Is the junior hiring gap transitory (firms pausing while they learn tools) or a new steady state?** Canaries say it has widened from 2025 to 2026 and loads on substitution occupations. Another two years of ADP/BLS data will tell.

4. **New-task creation vs expertise-leveling.** Will MLEs mostly supervise automated fitting, or will new tasks (oversight tooling, eval platforms, relational FMs in production, AI-for-science loops) create enough demand to absorb the people squeezed out of junior modeling?

5. **How far tabular/relational foundation models go on dirty, multi-table, delayed-label data.** If they only win on clean small tables, classical MLE plus RDL remains a craft. If they absorb entity graphs and time, the craft moves further up into formulation and monitoring.

6. **Accountability without competence.** Article 14 requires competent overseers. Organizations may still appoint unqualified ones. Whether that creates a real labor market for competent overseers is a political question as much as a technical one.

7. **Wage vs employment adjustment.** Canaries find adjustment through headcount, not base pay. If that persists, remaining MLEs may not capture the productivity gains they enable — a reason to prefer roles where you are close to revenue or to scarce domain expertise.

---

## Sources

Every URL used, with a one-line note. Primary sources marked ★.

1. ★ [Sculley et al., “Hidden Technical Debt in Machine Learning Systems,” NeurIPS 2015](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) — canonical map of CACE, loops, glue code, 5%/95% cartoon.
2. ★ [Breck, Cai, Nielsen, Salib, Sculley, “The ML Test Score,” IEEE Big Data 2017](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/) — 28 production-readiness tests.
3. ★ [Zinkevich, “Rules of Machine Learning”](https://developers.google.com/machine-learning/guides/rules-of-ml) — Google’s engineering rules; Rule #1 ship without ML.
4. ★ [Zinkevich PDF](https://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf) — same document, archival PDF.
5. ★ [Kapoor and Narayanan, “Leakage and the reproducibility crisis in ML-based science,” Patterns 2023](https://doi.org/10.1016/j.patter.2023.100804) — 17 fields, ~294 papers, eight leakage types, civil-war case.
6. ★ [Kapoor and Narayanan arXiv](https://arxiv.org/abs/2207.07048v1) — preprint of the leakage survey.
7. ★ [Sambasivan et al., “Everyone wants to do the model work, not the data work,” CHI 2021](https://dl.acm.org/doi/10.1145/3411764.3445518) — 92% data-cascade prevalence; prestige vs data work.
8. [Sambasivan et al. PDF](https://www.shivanikapania.com/assets/chi2021paper.pdf) — full paper text.
9. ★ [Sutton, “The Bitter Lesson,” 2019](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) — search/learning/compute beat built-in knowledge.
10. ★ [Autor, “Why Are There Still So Many Jobs?” JEP 2015](https://doi.org/10.1257/jep.29.3.3) — Polanyi’s paradox; complementarity of unspecified tasks.
11. ★ [Autor, “Polanyi’s Paradox and the Shape of Employment Growth,” 2014](https://economics.mit.edu/sites/default/files/publications/polanyis%20paradox%202014.pdf) — tacit knowledge vs computerization.
12. ★ [Acemoglu, “The Simple Macroeconomics of AI,” NBER w32487](https://doi.org/10.3386/w32487) — ≤0.66% TFP over 10 years under task-based accounting.
13. ★ [Acemoglu, Autor, and Johnson, “Building Pro-Worker Artificial Intelligence,” NBER w34854](https://doi.org/10.3386/w34854) — five categories of technical change; new-task creation vs expertise-leveling.
14. [MIT PDF of pro-worker AI](https://economics.mit.edu/sites/default/files/2026-03/Building%20Pro-Worker%20Artificial%20Intelligence.pdf) — full text.
15. [Acemoglu, Autor, Johnson 2023 policy memo](https://computing.mit.edu/wp-content/uploads/2023/11/Pro-Worker-AI-Policy-Memo20.pdf) — earlier complementarity argument.
16. ★ [Brynjolfsson, Chandar, Chen, “Canaries in the Coal Mine?,” Aug 2026](https://digitaleconomy.stanford.edu/app/uploads/2026/08/Canaries_August2026.pdf) — ADP payroll; 19% young-worker gap; substitution vs complementarity.
17. ★ [Stanford HAI, “Inside the AI Index: 12 Takeaways from the 2026 Report”](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report) — junior software employment; jagged capabilities.
18. ★ [Westby, Modestino, Cheng, IZA DP 18723, 2026](https://www.iza.org/publications/dp/18723/generative-ai-and-the-redefinition-of-entry-level-software-work) — 14–15% relative drop in junior software vacancies; remaining JDs shift to problem solving/communication.
19. ★ [Anthropic Economic Index launch, Feb 2025](https://www.anthropic.com/research/the-anthropic-economic-index) — 36%/4% task-coverage figures; 57/43 augmentation/automation; coding concentration.
20. ★ [Anthropic, “Labor market impacts of AI”](https://www.anthropic.com/research/labor-market-impacts) — observed exposure; programmers highly exposed; tentative junior hiring slowdown.
21. ★ [Anthropic Economic Index, Jan 2026 primitives](https://www.anthropic.com/research/anthropic-economic-index-january-2026-report) — 49% jobs at 25% task use; 12×/66% vs 9×/70%; API 75% automation; success-adjusted productivity.
22. ★ [OpenAI GDPval](https://openai.com/index/gdpval/) — 44 occupations, expert-blind grading; models approaching expert on one-shot deliverables; limitations on ambiguity/iteration.
23. [Patwardhan et al. / GDPval paper via doi](https://doi.org/10.70777/si.v2i4.17197) — full evaluation write-up.
24. ★ [Eloundou et al., “GPTs are GPTs,” 2023](https://arxiv.org/pdf/2303.10130.pdf) — 80%/19% task-exposure; LLM+software 47–56% of tasks.
25. ★ [Chan et al., MLE-bench](https://arxiv.org/pdf/2410.07095) — 75 Kaggle tasks; o1-preview 16.9% medals; authors note real-world messiness.
26. [OpenAI MLE-bench announcement](https://openai.com/index/mle-bench/) — official summary.
27. [openai/mle-bench GitHub](https://github.com/openai/mle-bench) — later leaderboard; 2026 submission freeze.
28. ★ [Nam et al., MLE-STAR](https://arxiv.org/html/2506.15692) — 43.9% medals (Flash) / >60% (2.5-Pro) on Lite; AutoML search-space contrast.
29. [KompeteAI paper](https://arxiv.org/pdf/2508.10177) — MLE-bench medals inflate vs real Kaggle LBs.
30. ★ [Wijk et al., RE-Bench, ICML 2025](https://proceedings.mlr.press/v267/wijk25a.html) — 7 ML research-eng environments; 4× agent lead at 2h; humans 2× at 32h.
31. [RE-Bench arXiv](https://arxiv.org/abs/2411.15114) — full paper.
32. [METR RE-Bench blog, Nov 2024](https://metr.org/blog/2024-11-22-evaluating-r-d-capabilities-of-llms/) — qualitative: agents faster, worse long-horizon returns.
33. ★ [METR, measuring long-task ability, Mar 2025](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — ~7-month doubling; week-scale extrapolation.
34. ★ [METR time-horizons live page](https://metr.org/time-horizons/) — FAQ: low-context interpretation; messier tasks worse; 16h unreliability note (2026).
35. [METR, limitations of time horizon, Jan 2026](https://metr.substack.com/p/2026-01-22-time-horizon-limitations) — author cautions against over-precision.
36. [METR Time Horizon 1.1](https://metr.org/blog/2026-1-29-time-horizon-1-1/) — 228 tasks, more long tasks.
37. ★ [WEF Future of Jobs Report 2025 PDF](https://reports.weforum.org/docs/WEF_Future_of_Jobs_Report_2025.pdf) — AI/ML roles and AI/big-data skills among fastest growing; analytical/systems thinking.
38. [WEF skills-outlook chapter](https://www.weforum.org/publications/the-future-of-jobs-report-2025/in-full/3-skills-outlook/) — same, HTML.
39. ★ [EU AI Act Article 14](https://artificialintelligenceact.eu/article/14/) — human oversight legal text; 2027/2028 application.
40. [EU AI Act Service Desk, Article 14](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-14) — official explainer.
41. [Dexity, AI Engineer career path from 390 JDs](https://dexity.com/intel/ai-engineer-career-path-2026) — skill frequencies; evals as differentiator.
42. [Dexity, 425 JDs on what an AI Engineer is](https://dexity.com/intel/what-is-an-ai-engineer-2026) — RAG baseline vs agentic 36%.
43. [HeroHunt, fastest-growing AI roles 2026](https://www.herohunt.ai/blog/fastest-growing-ai-roles-in-2026-data-and-rankings/) — MLE share of titles; shift to deploy/monitor.
44. [365 Data Science, AI engineer outlook](https://365datascience.com/career-advice/career-guides/ai-engineer-job-outlook-2025/) — skill frequencies including hyperparameter tuning ~1%.
45. [GCN write-up of Canaries 19% gap](https://gcn.com/ai-hiring-gap-workers-ages-widens/20887/) — secondary summary of Stanford Digital Economy Lab.
46. [MIT DCAI, data-centric vs model-centric](https://dcai.csail.mit.edu/2024/data-centric-model-centric/) — Ng-influenced curriculum framing.
47. [Databricks on data-centric platforms, citing Zinkevich/Ng](https://www.databricks.com/blog/2021/06/23/need-for-data-centric-ml-platforms.html) — industry restatement.
48. [Zschech et al., “Data-Centric Artificial Intelligence,” BISE 2024](https://doi.org/10.1007/s12599-024-00857-8) — academic survey of the paradigm.
49. [IBM on data leakage](https://www.ibm.com/think/topics/data-leakage-machine-learning) — standard definition; time-based validation.
50. [“A Grammar of Machine Learning Workflows Rejecting Data Leakage at Call Time,” arXiv:2603.10742](https://arxiv.org/html/2603.10742v4) — 648-paper leakage count; type-system intervention.
51. [Bergström, “The LLM-shaped hole in your XGBoost pipeline”](https://andreasbergstrom.dev/posts/the-llm-shaped-hole-in-your-xgboost-pipeline) — practitioner synthesis of tabular vs LLM evidence.
52. [Citi Ventures on large tabular models, 2026](https://www.citi.com/ventures/perspectives/opinion/ltms-large-tabular-models-startups-enterprise-2026.html) — enterprise interest in LTMs vs LLMs.
53. [AIMultiple tabular benchmark 2026](https://aimultiple.com/tabular-models) — trees still operational default; TFMs leading some accuracy regimes (vendor-adjacent; treat cautiously).
54. [Brookings, AI workforce policy framework](https://www.brookings.edu/articles/ai-workforce-policy-framework/) — effects-not-predetermined; steering vs brakes.
55. [IMF F&D, Acemoglu and Johnson on rebalancing AI](https://www.imf.org/-/media/files/publications/fandd/article/2023/december/26-29-acemoglu-final.pdf) — inequality-first impact on current path.
56. [CEPR Policy Insight 123, pro-worker AI](https://cepr.org/system/files/publication-files/191183-policy_insight_123_can_we_have_pro_worker_ai_choosing_a_path_of_machines_in_service_of_minds.pdf) — tax bias toward automation.
57. [Nature comment, Acemoglu/Johnson 2026](https://www.nature.com/articles/d41586-026-02566-6) — stop talking AGI, build complementary tools.
58. [Albertoni et al., reproducibility overview, AI Magazine](https://doi.org/10.1002/aaai.70002) — leakage taxonomy restated; non-experts + no-code tools as a driver.

**Collection gaps.** X/Twitter expert discourse could not be queried (developer-app access). McKinsey Global Institute full 2025/2026 PDFs were not retrieved; McKinsey figures appear only as cited inside Acemoglu (2024). Chip Huyen’s *Designing Machine Learning Systems* was not scraped (book; cited indirectly via the same production tradition as Sculley/Zinkevich).

---

## Rerun Inputs

```
workflow: firecrawl-deep-research
topic: Which machine learning engineer skills are hard to replace with AI, and what should an MLE develop to stay relevant?
depth: exhaustive
output: markdown
notes: Firecrawl API was unavailable; used WebSearch + full-page retrieval. Re-run when (a) METR publishes a time horizon reliably above 16h, (b) Canaries/ADP updates past 2026H2, (c) MLE-bench or RE-Bench successor shows agents refusing invalid metrics, or (d) EU AI Act Annex III oversight labor-market studies appear after Dec 2027.
```
