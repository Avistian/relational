"""Authored solution topology and causal reading routes for lessons 49–70.

Coordinates are deliberate: two-lane comparisons, feedback systems, nested
selection, temporal lanes and axis-changing networks must not share one graph.
The original lesson/source is authoritative for every pictured local variant.
"""
from dataclasses import dataclass, field

@dataclass
class Map:
    key: str
    title: str
    subtitle: str
    height: int
    anchor: str
    question: str
    answer: str
    scope: str
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    groups: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    def node(self,key,x,y,w,title,*lines,role='compute'):
        self.nodes.append(dict(key=key,x=x,y=y,w=w,h=86,title=title,lines=lines,role=role));return self
    def edge(self,a,b,label='',ports='bt',via=None,control=False):
        self.edges.append(dict(a=a,b=b,label=label,ports=ports,via=via,control=control));return self
    def group(self,x,y,w,h,label,role='compute'):
        self.groups.append(dict(x=x,y=y,w=w,h=h,label=label,role=role));return self
    def note(self,x,y,text):self.notes.append((x,y,text));return self

MAPS={}
def diagram(n,key,title,subtitle,height,anchor,question,answer,scope):
    m=Map(key,title,subtitle,height,anchor,question,answer,scope);MAPS.setdefault(n,[]).append(m);return m

m=diagram(49,'feature-routing','ExcelFormer: make feature influence directional','A learned importance order changes who may send information.',645,'Model architecture — ExcelFormer',
 'If the weak feature changes, which strong-feature state can it reach?',
 'Follow the mask: weak → strong is blocked inside attention, but weak → pooled output remains open. Feat-Mix changes training examples; it is not another inference layer.',
 'Numeric local ExcelFormer mechanism. Importance uses training labels only; the diagram separates Feat-Mix from the prediction path.')
m.group(30,140,600,380,'PREDICTION · feature states stay separate until pooling').group(665,140,305,380,'TRAINING ONLY','learn')
m.node('x',65,185,230,'Feature vector x','B × F numeric inputs','train-fitted feature order',role='query').node('mix',700,185,235,'Feat-Mix','donor feature subsets','information-weighted targets',role='learn')
m.node('emb',65,310,230,'Gated tokenizers','one embedding per feature','B × F × d').node('att',350,310,245,'Directed attention','strong reads strong','weak reads strong + weak')
m.node('pool',350,425,245,'Residual blocks → pool','attention + gated FFN × L','learned feature pooling')
m.node('out',350,550,245,'Normalization → head','class logits → softmax','B × K probabilities',role='query').node('loss',700,425,235,'Supervised loss','mixed targets vs prediction','update learned parameters',role='learn')
m.edge('x','emb').edge('emb','att',ports='rl').edge('att','pool').edge('pool','out').edge('mix','emb',ports='lb',via=[(645,228),(645,405),(180,405)],control=True).edge('mix','loss').edge('out','loss',ports='rt',via=[(645,593),(645,400),(817,400)],control=True)
m.note(350,290,'Only higher-ranked senders pass.')

m=diagram(49,'prompt-state','Trompt: the row is reread; prompt state evolves','Distinct cells share a prediction head, not their entire computation.',625,'Model architecture — Trompt',
 'Why can the second cell route the same columns differently for two rows?',
 'The first routing weights use shared identities and zero state. Its gathered values make O¹ row-dependent. O¹ then changes the next cell’s prompt routing, even though both cells reread the original x.',
 'Local numeric mirror: L=2 cells, P=8 prompts, d=16. Training sums cell losses; local inference averages cell logits before softmax.')
m.node('x',40,170,220,'Original row x','B × C','reread at every cell',role='query').node('id',380,170,235,'Prompt + column IDs','learned identities','initial state O⁰ = 0')
m.group(20,290,450,200,'CELL 1 · O⁰ → O¹').group(500,290,480,200,'CELL 2 · O¹ → O²')
m.node('c1',45,345,220,'Fuse identities + O⁰','score columns; softmax C','routing B × P × C').node('v1',285,345,160,'Expand x','gather over C','O¹: B × P × d')
m.node('c2',525,345,220,'Fuse identities + O¹','new column routing','sample-dependent').node('v2',765,345,190,'Expand x','gather over C','O²: B × P × d')
m.node('head',380,525,235,'Same head on O¹, O²','attention pool prompts','cell logits → mean → p',role='query')
m.edge('id','c1',ports='lb',via=[(275,213),(275,450),(155,450)],control=True).edge('c1','v1',ports='rl').edge('v1','c2',ports='rl').edge('c2','v2',ports='rl')
m.edge('x','v1',ports='rt',via=[(365,213)]).edge('x','v2',ports='rt',via=[(280,213),(280,270),(860,270)]).edge('v1','head',ports='bt',via=[(365,505),(497,505)]).edge('v2','head',ports='br',via=[(860,568)])

m=diagram(50,'checkpoint-fit','A fair comparison contains a complete fitted procedure','The FT-Transformer is one contestant inside the selection boundary.',650,'Model architecture',
 'Which decision may validation labels change, and which path must test labels never enter?',
 'Validation may choose a configuration or stopping point. The test features pass through that frozen choice; test targets meet predictions only at the final score. A stronger attention block cannot repair a breached boundary.',
 'Numeric local FT-T: d=32, two blocks, four heads; first attention pre-norm omitted. Tree/MLP contestants use their own declared recipes.')
m.group(25,145,640,370,'FIT / SELECT · outer training and validation').group(700,145,275,370,'FROZEN EVALUATION','query')
m.node('data',55,185,245,'Training X and y','fit preprocessing on train','validation stays separate',role='learn').node('tok',55,320,245,'Feature tokens + CLS','xⱼwⱼ + bⱼ; d=32','B × (F+1) × 32')
m.node('block',360,320,270,'FT-T block × 2','attention residual','ReGLU FFN residual')
m.node('head',360,425,270,'CLS → LN → ReLU → head','binary logit; BCE training','validation selects recipe')
m.node('test',725,185,225,'Held-out test X','same saved transform','same frozen model',role='query').node('pred',725,425,225,'Frozen predictions','sigmoid(logit)','test y still hidden',role='query')
m.node('score',540,550,280,'Paired test comparison','test y enters metric here','aggregate by dataset',role='query')
m.edge('data','tok').edge('tok','block',ports='rl').edge('block','head').edge('test','pred').edge('head','pred',ports='rl',control=True).edge('pred','score',ports='bl',via=[(837,530),(515,530),(515,593)])
m.note(65,285,'Other contestants: tree and MLP recipes fit here.')

m=diagram(51,'interventions','Change the problem; observe which bias helps','Three controlled interventions, not three new architectures.',630,'A bias is',
 'Which intervention changes coordinates while preserving predictive information?',
 'Rotation preserves information under an invertible map but changes axis alignment. Smoothing alters the target; irrelevant columns alter the input burden. Their different controls let us test different explanations for a tree–network gap.',
 'Experimental design from the lesson. Each branch has its own declared intervention and matched comparison; no universal winner is encoded.')
m.node('base',365,155,270,'One dataset, one split','baseline features and labels','freeze candidate recipes',role='query')
for k,x,title,lines in [('smooth',30,'Smooth targets',('remove short-scale variation','the target itself changes')),('rotate',365,'Rotate features',('invertible coordinate map','information is retained')),('noise',700,'Add irrelevant columns',('keep target fixed','increase nuisance inputs'))]:
 m.node(k,x,305,270,title,*lines).edge('base',k,ports='bt',via=[(500,275),(x+135,275)])
 m.node(k+'fit',x,440,270,'Refit tree and network','same declared budget / split','compare paired errors',role='learn').edge(k,k+'fit')
m.note(40,580,'Interpret the changed gap only for the intervention performed.')

m=diagram(52,'retrieval-system','TabR-S: retrieve evidence, then correct its meaning','The query has a direct route and a label-bearing memory route.',735,'Model architecture',
 'If the nearest labels stay fixed but their keys move, what can change downstream?',
 'Keys affect both selected identities/weights and the relative-key correction T(k−kᵢ). The query representation also bypasses retrieval. A prediction is therefore not just a weighted label vote.',
 'Complete numeric local TabR-S. Query-ID exclusion applies during supervised fitting; legal labeled memory remains part of the fitted predictor at inference.')
m.group(20,145,305,400,'QUERY PATH','query').group(360,145,620,400,'MEMORY PATH · shared encoder weights')
m.node('q',50,190,245,'Query features x','unknown y never supplied','B × F',role='query').node('mem',700,190,245,'Legal memory X, y, IDs','training records only','N × F; labels retained')
m.node('hq',50,315,245,'Encode → h; key K(h)','save h for residual','B × d').node('hm',700,315,245,'Encode → hᵢ; key K(hᵢ)','same encoder and key map','N × d')
m.node('search',390,315,270,'Nearest eligible keys','top-m of −‖k−kᵢ‖²','exclude own record ID').node('val',390,445,270,'Correct each label value','vᵢ = E(yᵢ) + T(k−kᵢ)','softmax scores → Σ pᵢvᵢ')
m.node('sum',50,590,245,'Add direct + retrieved','h + Σ pᵢvᵢ','B × d').node('head',390,590,270,'Residual predictor → head','classification probability','or regression output',role='query')
m.edge('q','hq').edge('mem','hm').edge('hq','search',ports='rl').edge('hm','search',ports='lr').edge('search','val').edge('mem','val',ports='rr',via=[(965,233),(965,488)]).edge('hq','sum').edge('val','sum',ports='br',via=[(525,563),(325,563),(325,633)]).edge('sum','head',ports='rl')
m.note(405,705,'Training target → loss → weights; never into its own memory value.')

m=diagram(53,'recipe-system','RealMLP-TD-S: the predictor and how it is learned','Preprocessing, parameterization and schedule form one recipe.',620,'Model architecture',
 'Why does matching hidden-layer widths fail to reproduce this procedure?',
 'The transform fixes input scale, trainable feature scales adapt it, NTP parameterization controls layer scale, and the optimizer schedule controls the path to a validation-selected checkpoint. Width describes only part of that system.',
 'Numeric TD-S: local width64; paper recipe width256. Three hidden layers; SELU for classification, Mish for regression. Categorical/full-TD paths are outside this diagram.')
m.group(25,140,640,380,'FORWARD PATH').group(700,140,275,380,'LEARNING RECIPE','learn')
m.node('raw',55,190,255,'Numeric row','train median / IQR','range fallback; smooth clip',role='query').node('scale',55,325,255,'Learned feature scales','one multiplier per feature','B × F')
m.node('mlp',355,325,275,'Three hidden layers','NTP linear → activation','F → d → d → d').node('head',355,440,275,'Zero-initialized head','class logits / target value','saved checkpoint output',role='query')
m.node('opt',725,190,225,'Parameter groups','initialization + decay','learning-rate policy',role='learn').node('sched',725,325,225,'coslog4 schedule','four settling opportunities','training loss → updates',role='learn').node('select',725,440,225,'Validation selection','choose checkpoint','freeze for test',role='learn')
m.edge('raw','scale').edge('scale','mlp',ports='rl').edge('mlp','head').edge('opt','sched').edge('sched','mlp',ports='lr',control=True).edge('sched','select').edge('select','head',ports='lr',control=True)
m.note(65,580,'A saved model needs the fitted transform as well as its weights.')

m=diagram(54,'shared-members','TabM-mini: many prediction paths, one expensive backbone','Member diversity enters before shared layers; outputs stay separate.',650,'Model architecture',
 'Which quantities are shared, and at what point do member predictions finally mix?',
 'Members multiply the same input by different learned first-layer R vectors, then use shared matrices. Their heads stay independent. Training averages member losses; classification inference averages probabilities, not logits.',
 'Corrected local TabM-mini: first-layer R only, shared W/biases, independent heads. Full TabM has additional member adapters in every block.')
m.node('x',370,155,260,'Transformed row x','train-only impute / z-score','broadcast B × k × F',role='query')
for k,x,title in [('a',40,'Member 1: x ⊙ R₁'),('b',370,'Member 2: x ⊙ R₂'),('c',700,'Member k: x ⊙ Rₖ')]:
 m.node(k,x,285,260,title,'learned feature multipliers','different effective functions').edge('x',k,via=[(500,255),(x+130,255)])
m.group(25,400,945,125,'ONE SHARED BACKBONE · N linear / ReLU / dropout layers; dropout off at inference')
for k,x in [('a',40),('b',370),('c',700)]:
 m.node(k+'h',x,430,260,'W shared → head '+k.upper(),'independent output projection','logits → probability p',role='compute').edge(k,k+'h')
m.node('mean',370,550,260,'Average probabilities','p = (p₁ + … + pₖ) / k','one prediction per row',role='query')
for k,x in [('a',40),('b',370),('c',700)]:m.edge(k+'h','mean',via=[(x+130,533),(500,533)])

m=diagram(55,'time-contract','TabReD: the split is part of the question','The same rows can define two different deployment experiments.',700,'Two tests, two questions',
 'Can an earlier event still be illegal training evidence?',
 'Yes. Its feature or label may not yet have been available at the cutoff. Temporal eligibility must check availability clocks before preprocessing, fitting or validation selection—not just before test scoring.',
 'Protocol schematic, not a new model. Match row universes and partition sizes when isolating split policy; released splits and stricter time-group policies remain distinct.')
m.group(25,140,450,425,'RANDOM PARTITION · exchangeable-row question').group(525,140,450,425,'TIME PARTITION · future-row question','query')
m.node('pool',55,185,390,'Same audited row universe','IDs, event time, feature-ready time','label-ready time; fixed row counts')
m.node('clock',555,185,390,'Cutoff → eligibility','event occurred AND features ready','training labels revealed by cutoff',role='query')
m.node('random',55,330,390,'Random train / validation / test','fit transforms on train only','validation chooses model',role='learn').node('time',555,330,390,'Train → validation → future test','preserve declared time boundary','selection uses past information only',role='learn')
m.node('rscore',55,470,390,'Held-out mixture error','estimates this partition policy','same budget as temporal arm',role='query').node('tscore',555,470,390,'Future-period error','estimates this horizon / availability','test labels enter only after prediction',role='query')
m.edge('pool','random').edge('pool','clock',ports='rl').edge('clock','time').edge('random','rscore').edge('time','tscore')
m.node('gap',305,595,390,'Compare errors; diagnose separately','a gap establishes policy sensitivity','it does not identify the cause of drift',role='query').edge('rscore','gap',via=[(250,575),(500,575)]).edge('tscore','gap',via=[(750,575),(500,575)])

m=diagram(56,'benchmark-reduction','TabArena: follow the unit of evidence','A score becomes a rank only within a declared comparison pool.',650,'Follow one error',
 'Would duplicating a dataset’s folds give it more influence on the final mean rank?',
 'Not with the displayed reduction: rank methods within dataset/split, average splits within each dataset, then give datasets equal weight. Bootstrap paired datasets, retaining their method results together.',
 'Frozen-score reanalysis for the declared four-method pool. These ranks are not the paper’s full-pool Elo; no fresh model fitting is pictured.')
m.node('ledger',40,165,270,'Frozen score ledger','dataset / split / method / regime','verify complete matched coverage',role='query')
m.node('rank',365,165,270,'Rank each dataset / split','lower error is better','ties receive average rank').node('within',690,165,270,'Average over splits','one rank per method / dataset','folds do not become datasets')
m.node('mean',690,345,270,'Average over datasets','equal dataset weights','one score per method',role='query').node('boot',365,345,270,'Paired dataset bootstrap','resample whole dataset records','retain method pairing',role='learn').node('scope',40,345,270,'Name the fitted regime','default / tuned / ensemble','different fitted procedures')
m.edge('ledger','rank',ports='rl').edge('rank','within',ports='rl').edge('within','mean').edge('within','boot',ports='bl',via=[(825,300),(340,300),(340,388)]).edge('scope','ledger',ports='tb',control=True)
m.node('read',365,525,270,'Rank + uncertainty','comparison pool changes rank','neither rank nor Elo is accuracy',role='query').edge('mean','read',via=[(825,495),(500,495)]).edge('boot','read')

m=diagram(57,'oof-system','Cross-family stacking: each training row needs an honest prediction','The combiner trains on predictions made without that row’s target.',720,'Model architecture',
 'Why is a held-out base prediction still unsafe if its preprocessing saw the row?',
 'The entire fitted base procedure must exclude the held-out fold. After scattering OOF predictions into original row order, the combiner may use those rows’ labels. Test scoring waits until both base-serving and combiner policies are frozen.',
 'OOF procedure with XGBoost, TabM and provided TabICL predictions. The serving refit/fold policy is a declared part of the artifact, not inferred from this schematic.')
m.group(25,145,950,250,'OUTER TRAINING ONLY · repeat for every held-out fold','learn')
m.node('split',50,200,250,'Fold f held out','other folds fit preprocessing','and each base procedure',role='learn').node('families',370,200,280,'XGB | TabM | TabICL','fit / construct context without f','predict only excluded fold f').node('scatter',725,200,220,'Scatter by row ID','repeat all folds','Z: N × 3 OOF p')
m.edge('split','families',ports='rl').edge('families','scatter',ports='rl')
m.node('combine',365,430,280,'Fit / select combiner','Z and outer-training y','freeze weights / policy',role='learn').node('serve',40,430,265,'New row → base models','declared serving fit policy','same class / column order',role='query').node('out',700,430,265,'Base p → combiner','one probability per row','test y is absent',role='query')
m.edge('scatter','combine',via=[(835,405),(505,405)]).edge('combine','out',ports='rl',control=True).edge('serve','out',ports='bb',via=[(172,575),(832,575)])
m.node('score',700,615,265,'Test labels → final score','compare to frozen single model','never select the oracle winner',role='query').edge('out','score',ports='rb',via=[(985,473),(985,710),(832,710)])

m=diagram(58,'evidence-funnel','From a literature map to a testable experiment','A survey locates ideas; matched evidence tests a claim.',615,'A survey is',
 'Which missing denominator could reverse a “best model” statement?',
 'The task roster, eligible method coverage and evaluation regime define the denominator. Reconstruct them before ranking. Then use a proposed explanation to design an intervention, rather than treating a meta-feature correlation as a cause.',
 'Evidence-synthesis workflow. The small benchmark in this lesson does not inherit the scope of the surveys it cites.')
m.node('papers',35,165,280,'Paper claims','task type / model operation','what question did each study ask?',role='query').node('protocol',360,165,280,'Recover the experiment','datasets; splits; tuning budget','metric; missing methods')
m.node('pool',685,165,280,'Matched denominator','which tasks / methods survive?','state exclusions before ranking')
m.node('ranks',685,345,280,'Within-task comparisons','then equal-task aggregation','retain effect sizes and cost')
m.node('hyp',360,345,280,'Candidate explanation','meta-feature association','a hypothesis, not identification')
m.node('test',35,345,280,'Discriminating experiment','hold a baseline fixed','change the proposed mechanism',role='learn')
m.edge('papers','protocol',ports='rl').edge('protocol','pool',ports='rl').edge('pool','ranks').edge('ranks','hyp',ports='lr').edge('hyp','test',ports='lr')
m.note(45,535,'The output is a next experiment, not a universal leaderboard winner.')

m=diagram(59,'nested-selection','Validation overfitting: nest the selector, not just the fit','A low error can select lucky noise even when each candidate is valid.',725,'Reconstruct the measuring instrument',
 'Does a leave-one-out score remain an unbiased report after it chooses the best candidate?',
 'Deleted-row residuals score a fixed candidate. Taking their minimum adapts to those scores. The outer held fold must be absent from candidate fitting AND PRESS selection, so it evaluates the whole adaptive procedure.',
 'Lesson’s KRR measuring instrument: kernel/regularization candidates, exact deleted residuals and nested selection. This is an evaluation architecture, not a new tabular network.')
m.group(25,140,630,425,'OUTER-FOLD TRAINING COMPLEMENT','learn').group(690,140,285,425,'OUTER HELD FOLD','query')
m.node('train',55,180,260,'Outer-training rows','features and labels','define candidate grid',role='learn').node('krr',355,180,260,'Fit each KRR candidate','kernel + regularization','solve coefficients')
m.node('press',355,330,260,'Deleted-row residuals','αᵢ / inverse diagonalᵢ','mean squared PRESS')
m.node('select',55,460,260,'Select minimum PRESS','keep selected fitted model','freeze choice for held fold',role='learn').node('held',715,180,235,'Held features','held targets excluded','from the entire left region',role='query')
m.node('pred',715,460,235,'Selected KRR predicts','no further selection','then reveal held targets',role='query')
m.edge('train','krr',ports='rl').edge('krr','press').edge('press','select',ports='bl',via=[(485,430),(35,430),(35,503)]).edge('select','pred',ports='rl',control=True).edge('held','pred')
m.node('eval',350,610,300,'Outer predictions → error','repeat; preserve pairing','report selector performance',role='query').edge('pred','eval',via=[(832,585),(500,585)])

m=diagram(60,'comparison-system','Compare complete procedures on the same task','The experiment freezes who may learn from which partition.',680,'Freeze the experiment',
 'What must stay matched for a model comparison to answer its declared question?',
 'Use the same task partitions and explicit budgets, retain each baseline’s essential recipe, select on validation, and compare frozen predictions with paired dataset-level summaries. Quality and lifecycle cost answer different parts of the decision.',
 'Broad model-comparison protocol; not a novel architecture. The lesson’s small experiment and the cited papers retain separate evidence scopes.')
m.node('contract',350,150,300,'Freeze comparison contract','tasks / split / metrics / budgets','eligibility and preprocessing',role='query')
for k,x,t,desc in [('t',30,'Tree procedure','fit splits / leaves'),('n',365,'Neural procedure','fit weights + training recipe'),('f',700,'Foundation procedure','context + checkpoint + wrapper')]:
 m.node(k,x,300,270,t,desc,'select within allowed validation',role='learn').edge('contract',k,via=[(500,270),(x+135,270)])
 m.node(k+'p',x,445,270,'Frozen test predictions','same test row IDs','record fit and inference cost',role='query').edge(k,k+'p')
m.node('compare',350,570,300,'Paired errors + cost','dataset is the aggregate unit','retain scope and uncertainty',role='query')
for k,x in [('t',30),('n',365),('f',700)]:m.edge(k+'p','compare',via=[(x+135,550),(500,550)])
m=diagram(61,'learn-inference','PFN: train on tasks; infer from a labeled set','The outer optimizer learns a reusable conditional prediction rule.',745,'Model architecture',
 'What changes when the same pretrained PFN receives a different labeled context?',
 'Its activations and predictive density change. At inference its weights stay fixed. During pretraining only, query targets enter the negative log likelihood, whose gradient updates those shared weights across sampled tasks.',
 'Local GP row-PFN: d=64, three postnorm blocks, four heads, 64-bin full-support density. The GP is the auditable task prior here, not the TabPFN SCM mixture.')
m.group(25,140,950,165,'PRETRAINING · repeated sampled regression tasks','learn')
m.node('prior',50,190,260,'Sample one GP task','joint draw at all row inputs','split context and query',role='learn').node('labels',375,190,250,'Context (Xc, yc)','query Xq; hide yq','Ex shared; Ey context only').node('truth',715,190,235,'Hidden targets yq','used only by loss','never a forward input',role='learn')
m.group(25,340,620,285,'FORWARD · same operator on new real contexts').group(680,340,295,285,'PRETRAINING ONLY','learn')
m.node('tokens',50,390,260,'Context: Ex(x) + Ey(y)','query: Ex(x) only','B × (C+Q) × 64').node('blocks',365,390,250,'Row blocks × 3','residual → LayerNorm','context + own query eligible')
m.node('density',365,525,250,'Head → 64 bin masses','divide finite-bin mass by width','tail bins keep full support',role='query').node('loss',715,525,235,'Query density loss','−log pθ(yq | Xq, context)','backpropagate into θ',role='learn')
m.edge('prior','labels',ports='rl').edge('prior','truth',ports='tt',via=[(180,170),(832,170)]).edge('labels','tokens',via=[(500,325),(180,325)]).edge('tokens','blocks',ports='rl').edge('blocks','density').edge('truth','loss').edge('density','loss',ports='rl').edge('loss','blocks',ports='tl',via=[(832,365),(340,365),(340,433)],control=True)
m.note(60,690,'NEW TASK: real context + query → frozen forward path → density. No optimizer.')

m=diagram(62,'historical-v1','TabPFN v1: one row token, many labeled-context interactions','The wrapper constructs views; the same frozen network reads each view.',720,'Model architecture',
 'At which point can a context label affect a query prediction?',
 'Only context tokens receive label embeddings. The query then reads those context states through the masked row blocks. After decoding each view, undo class permutations before combining probabilities.',
 'Historical v1 checkpoint; 12 row blocks, d=512, four heads, 100-feature input contract, up to ten native classes. View count and preprocessing are the declared lesson recipe.')
m.group(25,140,950,150,'WRAPPER · context-fitted transformations / feature and class views')
m.node('context',55,180,265,'Labeled context','Xc and yc','fit the declared transform').node('query',365,180,265,'Unlabeled queries','Xq transformed identically','no query target supplied',role='query').node('views',695,180,250,'Create declared views','feature / class permutations','record inverse class maps')
m.node('ctok',55,345,265,'Context row token','pad / scale to 100 features','Ex(x) + Ey(y): 512 wide').node('qtok',365,345,265,'Query row token','same feature projection','Ex(x): 512 wide',role='query')
m.group(25,485,630,135,'FROZEN HISTORICAL NETWORK')
m.node('blocks',55,525,265,'12 row blocks','masked attention + FFN','residuals and normalization').node('head',365,525,265,'Query decoder','512 → 1024 → 10','slice active classes; temperature')
m.node('combine',695,525,250,'Undo class remapping','softmax in original class order','average view probabilities',role='query')
m.edge('context','ctok').edge('query','qtok').edge('views','qtok',ports='bl',via=[(820,315),(340,315),(340,388)],control=True).edge('ctok','blocks').edge('qtok','blocks',via=[(497,470),(187,470)]).edge('blocks','head',ports='rl').edge('head','combine',ports='rl')
m.edge('views','ctok',ports='bt',via=[(820,300),(187,300)],control=True)
m.note(65,680,'Pretraining learned θ on synthetic tasks. A new table changes context, not θ.')

m=diagram(63,'prior-generator','The synthetic prior: sample a world, then sample its rows','Task-level randomness creates a shared mechanism across observations.',735,'The released generator',
 'If you resample the graph for every row, are you sampling the same kind of task?',
 'No. A task has shared graph, weights and observation choices; rows receive fresh root/noise draws under that shared world. Resampling the mechanism per row destroys the within-task relationship the PFN is trained to infer.',
 'Released SCM-style generator concept and lesson reconstruction. Visible variables need not be causal parents of the target; predictive inference is not causal identification.')
m.group(25,140,950,155,'ONCE PER TASK · shared generating world','learn')
m.node('hyper',55,185,255,'Draw hyperparameters','depth / width / sparsity','noise and mechanism choices',role='learn').node('graph',375,185,255,'Sample graph + weights','fixed within this task','shared structural assignments',role='learn').node('obs',695,185,255,'Choose observations','features; target variable','class threshold / class mapping',role='learn')
m.group(25,335,950,190,'FOR EACH ROW · different observations, same world')
m.node('noise',55,395,255,'Draw roots and noise','fresh exogenous values','row-specific randomness',role='query').node('apply',375,395,255,'Evaluate the graph','parents + noise → child','follow directed graph order').node('xy',695,395,255,'Observe X; discretize Y','causes / effects / proxies','imbalanced classes possible')
m.node('split',375,595,255,'Split context and query','reveal only context labels','PFN learns query likelihood',role='query')
m.edge('hyper','graph',ports='rl').edge('graph','obs',ports='rl').edge('noise','apply',ports='rl').edge('graph','apply',control=True).edge('apply','xy',ports='rl').edge('obs','xy',control=True).edge('xy','split',via=[(822,560),(502,560)])
m.note(55,715,'Repeated tasks vary the world; repeated rows reveal one sampled world.')

m=diagram(64,'axial-network','TabPFN v2: keep the feature axis alive through inference','A query learns from context at each token position across 12 blocks.',815,'Model architecture',
 'How does a query feature influence its target readout without becoming a sender to other rows?',
 'Within-row attention connects the query’s feature tokens to its target token. Row attention lets that row read context keys/values only. Repeating the two axes propagates context-conditioned feature interactions to the query target state.',
 'Complete historical default classifier: 12 blocks, d=192, six heads, two-feature groups. Numeric single-view lab uses temperature0.9; historical wrapper variants remain separate.')
m.node('x',35,150,285,'Context / query features','context-fitted numeric wrapper','B × (C+Q) × F',role='query').node('y',680,150,285,'Context y / query missing','value + missingness channels','query flag distinguishes unknown')
m.node('groups',35,285,285,'Two-feature group encoder','2 values + 2 flags → 192','add projected group identity').node('target',680,285,285,'Target encoder','2 channels → 192','append one target token')
m.group(25,425,950,215,'REPEAT 12 DISTINCT BLOCKS · B × (C+Q) × (G+1) × 192')
m.node('feature',50,480,270,'1 · Within each row','feature / target attention','residual + LayerNorm').node('row',365,480,270,'2 · At each token position','context ↔ context; query ← context','query uses first-head K/V reuse').node('ffn',680,480,270,'3 · Token-wise FFN','192 → 768 → 192; GELU','residual + LayerNorm')
m.node('read',350,690,300,'Query target states → head','192 → 768 → 10; keep K classes','temperature → softmax → Q × K',role='query')
m.edge('x','groups').edge('y','target').edge('groups','feature').edge('target','feature',ports='bl',via=[(822,405),(35,405),(35,523)]).edge('feature','row',ports='rl').edge('row','ffn',ports='rl').edge('ffn','read',via=[(815,660),(500,660)])
m.note(60,610,'Row attention: residual + LayerNorm. Repeat the complete three-sublayer block.')
m.note(60,795,'Pretraining: synthetic query loss updates θ. This inference path holds θ fixed.')

m=diagram(65,'embedding-probe','Query embeddings: freeze the encoder, learn a new readout','Embedding extraction is itself a fitted-context procedure.',755,'Model architecture',
 'Why must a training row be extracted as a query rather than as labeled context?',
 'A labeled context state can already contain that row’s target. Query-role extraction removes this direct path: each training fold is embedded using only other-fold labels, then scattered to original row order before probe fitting.',
 'Lesson’s ten-fold query-role training extraction and full-training evaluation context. Frozen historical TabPFN v2; preprocessing/representation mismatch is measured, not assumed away.')
m.group(25,140,455,370,'TRAINING EMBEDDINGS · 10 folds','learn').group(520,140,455,370,'VALIDATION / TEST EMBEDDINGS','query')
m.node('fold',55,185,395,'Hold fold f in query role','other folds provide context labels','f supplies features only',role='learn').node('full',550,185,395,'All training rows as context','validation or test rows are queries','their labels never enter encoder',role='query')
m.node('encode1',55,325,395,'Frozen v2 encoder','12 feature / row blocks','read final query target states: 192').node('encode2',550,325,395,'Same frozen v2 encoder','same 192-wide readout','different context size/composition')
m.node('scatter',55,450,395,'Scatter embeddings into row order','N × 192; attach training y','no self-target in extraction').node('probe',550,565,395,'Fit / select linear probe','fit on training vectors; validation selects','freeze probe; test once',role='learn')
m.edge('fold','encode1').edge('full','encode2').edge('encode1','scatter').edge('scatter','probe',ports='bl',via=[(252,608)]).edge('encode2','probe')
m.note(55,720,'The probe changes; the pretrained encoder does not. Context mismatch remains testable.')

m=diagram(66,'three-transformers','TabICL: build row representations before learning from labels','Three transformers change the unit of attention: column → row → dataset.',875,'Model architecture',
 'Where does the feature axis disappear, and where do labels first enter?',
 'The row stage concatenates four 128-wide CLS outputs into one 512-wide row vector. Labels enter only after that compression, before dataset ICL. Column memory uses context values; the later dataset stage uses context row keys.',
 'Original February checkpoint, numeric single-view lesson path. Three column blocks, three row blocks, twelve ICL blocks. Hierarchical class extension and view ensembles are outside this pictured wrapper.')
m.node('table',35,150,290,'Numeric Xc and Xq','context-fitted preprocessing','N × F scalar cells',role='query').node('labels',680,150,285,'Context labels yc','kept out of column / row stages','not available for query rows',role='learn')
m.group(25,280,950,175,'1 · COLUMN NETWORK · 3 inducing blocks per column · d=128')
m.node('u',50,325,260,'Shared scalar projection','cells → U: N × 128','context slice Uc only → readers').node('mem',370,325,260,'128 inducing readers I','M = MAB(I, Uc)','fixed-size context memory').node('cell',690,325,260,'Every cell reads M','V = MAB(U, M)','then affine: x·W(V)+b(V)')
m.group(25,500,620,180,'2 · ROW NETWORK · 3 attention blocks, then one CLS readout')
m.node('row',50,545,260,'Feature interactions','prepend 4 CLS; apply RoPE','attention within each row').node('compress',370,545,240,'Concatenate 4 CLS','4 × 128 → 512','N × 512; feature axis gone')
m.group(680,500,295,180,'3 · DATASET ICL × 12')
m.node('icl',705,545,245,'Context gets labels','query rows read context only','512-wide row attention + FFN')
m.node('head',680,745,285,'Query row head','512 → 1024 → 10','active-class softmax → Q × K',role='query')
m.edge('table','u').edge('u','mem',ports='rl').edge('mem','cell',ports='rl').edge('cell','row',via=[(820,475),(180,475)]).edge('row','compress',ports='rl').edge('compress','icl',ports='rl').edge('labels','icl',ports='rr',via=[(985,193),(985,588)]).edge('icl','head')
m.note(50,435,'Repeat the MAB pair 3 times; apply the conditioned affine once, after the final block.')
m.note(50,735,'Column/row representation: no label input.')
m.note(50,768,'MAB = attention + residual + normalization + FFN.')
m.note(50,802,'Pretraining changes weights; new-table ICL does not.')

m=diagram(67,'local-context','LoCalPFN: retrieval wraps a complete pretrained network','Selecting a context and adapting the weights are different operations.',800,'Model architecture',
 'Does changing the neighbor set require changing the PFN weights?',
 'No. Frozen local inference composes retrieval with the pretrained PFN. Fine-tuning adds a separate query-loss gradient path. Approximate training shares anchor neighborhoods; exact inference retrieves for each query.',
 'Historical v1 model: 12 blocks, d=512. Fixed standardized Euclidean retrieval; this is not TabR’s learned key/value correction. Local normalization follows the retrieved context.')
m.group(25,140,950,160,'RETRIEVAL GEOMETRY · training-fitted scaler and clipping')
m.node('q',50,185,255,'Query x','transform with saved scaler','do not provide its target',role='query').node('bank',690,185,255,'Memory X, y, IDs','transform with same scaler','exclude query ID during fitting')
m.node('knn',365,340,270,'Exact nearest k records','distance in fixed scaled space','C(x) contains features + labels').node('norm',365,475,270,'Local normalization','local context mean / sample SD','pad to100; feature-count scaling')
m.node('pfn',365,610,270,'Full historical v1 PFN','row tokens → 12 blocks → head','softmax → query probability',role='query')
m.node('train',35,610,280,'Optional fine-tuning','shared-anchor neighborhoods','query loss updates PFN θ',role='learn').node('val',695,610,270,'Validation-only selection','choose tuning checkpoint','then frozen exact test retrieval',role='learn')
m.edge('q','knn',via=[(177,320),(500,320)]).edge('bank','knn',via=[(817,320),(500,320)]).edge('knn','norm').edge('norm','pfn').edge('train','pfn',ports='rl',control=True).edge('pfn','val',ports='rl',control=True)
m.note(55,760,'k=N approaches full-context inference under matched preprocessing and class order.')

m=diagram(68,'temporal-pfn','Drift-resilient PFN: time conditions a learned inference rule','The prior changes relationships; the forward model encodes the domain clock.',855,'Model architecture',
 'Is Time2Vec a separate token, and does it alone establish useful extrapolation?',
 'No: its 100 coordinates concatenate with each pair of feature values before projection. The temporal prior supplies the assumption about changing tasks. A timestamp channel alone does not identify tomorrow’s relationship.',
 'Complete released Drift checkpoint: 12 axial blocks, d=192. The prior path is paper-grounded reconstruction; the original pretraining generator is unavailable in the release.')
m.group(25,140,950,155,'PRETRAINING IDEA · reconstruct the assumption; do not claim a fresh pretraining run','learn')
m.node('g',50,185,255,'Base structural graph G','relationships generate X, y','selected edges may drift',role='learn').node('h',370,185,255,'Drift graph H(time)','coherent changes to G weights','sample ordered-domain tasks',role='learn').node('pre',690,185,255,'Query likelihood loss','learn weights across tasks','freeze released θ for inference',role='learn')
m.group(25,340,950,365,'RELEASED FORWARD PATH · historical labeled context + future query')
m.node('time',50,395,255,'Domain c → Time2Vec','1 linear + 99 sine coordinates','normalize clock using context').node('encode',370,395,255,'Time in every pair','[2 values; 100 time coords]','102 → 192; add group identity').node('target',690,395,255,'Target encoder','context y; query label hidden','2 → 192 target token')
m.node('blocks',370,565,255,'12 axial blocks','feature attention → row attention','FFN; residual + LN each time').node('head',690,565,255,'Query target → head','192 → 768 → 10','keep K logits → softmax',role='query')
m.edge('g','h',ports='rl').edge('h','pre',ports='rl').edge('time','encode',ports='rl').edge('encode','blocks').edge('target','blocks',ports='bl',via=[(817,520),(345,520),(345,608)]).edge('blocks','head',ports='rl')
m.note(50,755,'Query row attention reads source K/V only. Future labels are never memory.')
m.note(50,795,'Matched tests compare released Drift, NoT2V and Base procedures separately.')

m=diagram(69,'failure-contract','Open environments: keep failures inside the evaluation','A changed class, schema or distribution challenges a different contract.',700,'Trace the whole evaluation',
 'Can a classifier with only seen-class outputs correctly name an unseen class?',
 'Its native output axis cannot name that class. A confidence detector may flag uncertainty, but detection is a separate evaluated task. Keep unsupported rows in all-query accounting instead of silently making the test easier.',
 'Stress-test workflow for the lesson’s frozen predictors. These tests identify conditional failures, not a universal ordering of foundation models.')
m.node('contract',350,150,300,'Freeze prediction contract','known class vocabulary / schema','context, fallback, metric, cost',role='query')
for k,x,title,l1,l2 in [('class',30,'Emerging classes','hold a class out of context','separate naming from detection'),('feat',365,'Feature removal','same query IDs; declared fallback','preserve paired comparison'),('shift',700,'Distribution shift','declare the changed law','keep temporal / label eligibility')]:
 m.node(k,x,315,270,title,l1,l2).edge('contract',k,via=[(500,280),(x+135,280)])
 m.node(k+'p',x,455,270,'Frozen predictions','retain unsupported / hard rows','score declared objective',role='query').edge(k,k+'p')
m.node('audit',350,595,300,'Absolute score + change','failure size ≠ baseline quality','state the violated contract',role='query')
for k,x in [('class',30),('feat',365),('shift',700)]:m.edge(k+'p','audit',via=[(x+135,568),(500,568)])

m=diagram(70,'decision-architecture','The final checkpoint: choose a defensible fitted system','Compare where evidence enters, then test the same deployment contract.',780,'Compare where each architecture',
 'What must be saved to replay a PFN prediction that is not contained in its checkpoint?',
 'Save context IDs and labels, preprocessing, feature/class order, view policy and checkpoint identity. Compare this full procedure with the complete tree and TabM artifacts, using the same held-out rows and declared selection budget.',
 'Decision architecture for the lesson’s measured comparison. Family diagrams summarize information routes; exact checkpoint and wrapper identities remain in the lesson evidence ledger.')
m.group(25,140,950,400,'THREE WAYS TO USE TRAINING EVIDENCE · one matched evaluation contract')
for k,x,title,body,fit,save in [('tree',35,'Tree model','features → splits → leaf outputs','training y fits splits and leaves','save transforms + tree ensemble'),('tabm',365,'TabM','adapters → W → member heads','training y updates weights; mean p','save transforms + learned weights'),('pfn',695,'PFN / TabICL','query reads labeled context','θ pretrained on synthetic tasks','save context, wrapper, weights')]:
 m.node(k,x,190,270,title,body,fit).node(k+'save',x,360,270,'Frozen fitted artifact',save,'validation choice already fixed',role='learn').edge(k,k+'save')
m.node('eval',350,580,300,'Same held-out queries','paired error; dataset-level uncertainty','lifecycle time and information stress',role='query')
for k,x in [('tree',35),('tabm',365),('pfn',695)]:m.edge(k+'save','eval',via=[(x+135,550),(500,550)])
m.node('next',350,690,300,'Relational research handoff','name information flat tables lose','test entity / event links fairly',role='query').edge('eval','next')

# Each seam is authored against a particular argument, not generated from headings.
STORIES={}
def story(n,title,opening,seams,handoff):
    STORIES[n]=dict(title=title,opening=opening,seams=seams,handoff=handoff)

story(49,'Two ways to control which features speak',
 '[[48|DCNv2]] made feature crosses explicit. ExcelFormer and Trompt ask a different question: how should a network route information among heterogeneous columns? Read them as two answers to that question. ExcelFormer constrains the direction of feature attention; Trompt repeatedly gathers feature values into learned prompt states. Only after those routes are clear can a benchmark tell us whether their extra structure helped.',[
 ('Model architecture — ExcelFormer','The claim audit identifies what needs explaining. Now start with the input-to-output route: importance ordering makes sense only because it determines which tokens may send to which other tokens. Keep the final pooling path in view while inspecting the mask.'),
 ('Feat-Mix','The mask controls communication inside a prediction. Feat-Mix instead changes the examples used to learn that predictor. Their common premise is unequal feature informativeness, but one operates on attention edges and the other on training inputs and targets.'),
 ('Model architecture — Trompt','ExcelFormer imposed a direction on feature-to-feature communication. Trompt changes the receiving objects: learned prompts gather columns, then pass their resulting states into the next cell. Trace the original row and the evolving prompt state as separate inputs.'),
 ('Audit the protocol','We now have two mechanisms, not yet two demonstrated improvements. A measured gain can also come from preprocessing, search or split choices. The protocol audit asks whether the evidence actually distinguishes architectural benefit from those surrounding choices.')],
 '[[50|The checkpoint]] turns this distinction into a common experiment: specify the fitted procedures and selection rules before asking which architecture wins.')

story(50,'An architecture earns its place through a controlled decision',
 '[[49|The previous lesson]] separated an architectural idea from the experiment used to support it. This checkpoint puts that distinction into practice. The object being compared is a complete learning procedure: preprocessing, model, tuning and selection. The FT-Transformer diagram lives inside that larger experiment because an accurate forward pass alone cannot make a comparison fair.',[
 ('“Same protocol”','A model recommendation is a decision about future predictions. To evaluate that decision, first specify which information each candidate is allowed to use. “Same protocol” becomes concrete when every fit, choice and final score has an assigned partition.'),
 ('Model architecture','The outer information boundary is now fixed. Within it, audit the actual FT-T contestant: feature tokenization, residual blocks and the CLS readout. This prevents a familiar model name from concealing a materially different implementation.'),
 ('Let validation select','Knowing the forward path tells us what each candidate computes. It does not tell us which candidate to deploy. Validation makes that choice; test must evaluate the already chosen procedure rather than participate in choosing it.'),
 ('Pair the differences','Once selection is frozen, compare models on the same test cases and datasets. Pairing removes shared difficulty from the contrast; choosing the correct aggregate unit keeps extra seeds from masquerading as broader evidence.')],
 '[[51|The next lesson]] moves from “which procedure won here?” to “which property of the problem could explain the gap?” Controlled interventions will test those explanations.')

story(51,'Turn a performance gap into a falsifiable explanation',
 '[[50|A fair comparison]] can establish a gap without explaining it. This paper makes the explanation testable by changing the learning problem in controlled ways. Smoothness, orientation and irrelevant features are connected through one question: what structure does each model find easy to learn? Keep the intervention and its preserved quantities separate in every branch.',[
 ('Smoothness:','A bias is useful only relative to a problem. Start by changing short-scale target variation while holding the comparison recipe fixed. If the performance gap changes, that is evidence about sensitivity to this intervention, not a complete theory of tabular learning.'),
 ('Orientation:','Smoothing changed the target relationship. Rotation offers a different probe: preserve information while changing the coordinate system. This helps separate a model’s preference for axis-aligned structure from the amount of signal available.'),
 ('Irrelevant columns:','Rotation redistributed signal across coordinates. Adding irrelevant columns asks whether extra input dimensions can hurt even without changing the target. The contrast connects feature selection to optimization and model bias rather than to information loss alone.'),
 ('Read the evidence','The three interventions suggest three predictions. Read each measured result against its own prediction and control, rather than repairing the explanation after seeing a winner. A failed expected pattern is a reason to inspect the intervention, as the bandwidth diagnostic illustrates.')],
 '[[52|TabR]] responds to the difficulty of fitting local variation by giving a predictor access to nearby labeled examples. That is a new information route to test, not a promise to erase every tree advantage.')

story(52,'From a difficult global function to a learned local memory',
 '[[51|The intervention study]] showed why one global fitted mapping can struggle with some tabular structure. TabR proposes an additional route: retain eligible training rows and consult a learned neighborhood for each prediction. The entire argument follows three linked choices—where to look, what to take from each neighbor, and how to combine that evidence with the query itself.',[
 ('Learn a distance','The architecture has two routes into the final head. Follow the memory route first: the key geometry chooses which records can influence a query. Selection returns identities; score normalization then assigns weights within that selected set.'),
 ('A neighbor contributes','Neighbor identities and weights answer where to look and how strongly to listen. They do not yet define the message. The label-plus-correction value makes the retrieved content depend on the relation between query and neighbor, connecting retrieval geometry to prediction.'),
 ('The memory is part','We can now trace a label from memory to the prediction head. That same path explains the evaluation danger: an ineligible label can become an input. Memory eligibility is therefore part of the fitted system, just as preprocessing and weights are.'),
 ('What did the paper show','With the forward route and legal information fixed, the experiment can ask whether retrieval helped. Compare the matched retrieval-free network, inspect the memory-label intervention, and retain the distinction between local evidence and the paper’s larger benchmark.')],
 '[[53|RealMLP]] tests a complementary response: improve how a row-only network is trained before attributing every gain to a more elaborate architecture.')

story(53,'A strong network is a training recipe, not just a diagram',
 '[[52|TabR]] added an explicit memory. RealMLP asks how far a carefully designed parametric baseline can go without that route. Its claim about defaults links three levels: stable input scales, a favorable optimization path, and transfer of the recipe to new datasets. Read the preprocessing and schedule as causes of the fitted predictor, rather than as disconnected implementation tips.',[
 ('Make input scale','The forward map starts before the first learned layer. Robust statistics and smooth clipping control the magnitudes that the optimizer will encounter. This is why a preprocessing detail can change training even when the hidden-layer graph is identical.'),
 ('Same function family','Input scale is only one side of the optimization problem. Parameterization and initialization determine how changes in weights alter the function. A network can express the same functions yet reach different ones under a fixed training budget.'),
 ('Four opportunities','The preceding parameter choices define what a gradient step does. The schedule determines when steps are exploratory and when they settle. Validation then chooses among the resulting checkpoints, completing the recipe rather than adding an unrelated final trick.'),
 ('What exactly are we comparing','A transferable default is a claim about the whole recipe across tasks. Compare that procedure with fixed and tuned alternatives under declared budgets; otherwise an architecture label can hide unequal selection effort.')],
 '[[54|TabM]] keeps the strong-baseline perspective but spends shared computation on multiple predictors. The next question is whether their errors are different enough for averaging to help.')

story(54,'Create useful disagreement without duplicating every weight',
 '[[53|RealMLP]] strengthened one fitted network. TabM pursues the benefit of several predictions while sharing expensive matrices. The connection between adapters, joint training and aggregation is the core idea: cheap member-specific parameters create distinct paths, training shapes those paths together, and averaging can reduce their combined error.',[
 ('The BatchEnsemble layer','The system view shows several member paths through a shared backbone. Now open one layer: multiplying inputs by member-specific vectors changes each effective function while retaining a common matrix. This is the computational source of the efficiency claim.'),
 ('Why weight sharing','Sharing reduces parameter duplication, but it also couples learning. Gradients from different member losses update the same matrix. The method therefore needs to be understood as jointly learned predictors, not as independent networks stored more compactly.'),
 ('Weak individually','Joint training tells us how the members arise. Whether averaging helps depends on their errors: weak but complementary members can combine well, while identical mistakes survive averaging. Inspect the ensemble prediction alongside individual predictions.'),
 ('The k knob','If complementary errors are the useful resource, more members are not automatically better. Changing k or pruning members probes how much additional diversity remains relative to its compute cost, motivating the controlled comparison that follows.')],
 '[[55|TabReD]] changes where these fitted predictors must work: from held-out rows in a mixture to genuinely later events. A strong ensemble still depends on an appropriate evaluation target.')

story(55,'A better model for which future?',
 '[[54|TabM]] improved the candidate model set. TabReD shifts attention to the deployment question. Random and temporal tests can reward different behavior because they expose different information and different distributions. The lesson connects row identities, availability clocks, validation and interpretation into one end-to-end future-prediction experiment.',[
 ('Trace the rows','Before explaining a score difference as temporal shift, establish which rows entered each experiment. A shared row universe and matched partition sizes remove some competing explanations; different universes cannot isolate the split rule alone.'),
 ('An event timestamp','Row membership is not enough for historical realism. An event may already have occurred while its features or outcome were still unavailable. Following all three clocks converts a chronological split into an information-availability contract.'),
 ('Validation is inside','The availability contract applies to model selection as well as fitting. If validation includes information that would arrive after the deployment cutoff, the selected model can benefit from the future even when its training rows look legal.'),
 ('Diagnose a gap','After enforcing those boundaries, a score gap describes sensitivity to the evaluated policies. Attribution comes next: inspect feature, label and relationship changes separately. A temporal performance drop alone cannot tell which mechanism caused it.')],
 '[[56|TabArena]] scales the accounting question across datasets and methods. The lesson moves from “what does this split measure?” to “what exactly does the aggregate ranking summarize?”')

story(56,'A leaderboard is the end of a reduction, not the beginning',
 '[[55|Temporal evaluation]] made the test distribution explicit. TabArena adds another layer: the ranking depends on which fitted procedures, tasks and splits are counted and how their errors are reduced. Follow one score through that reduction. The benchmark then becomes an auditable experiment rather than a list of model names.',[
 ('Name the fitted procedure','The comparison question specifies what evidence we want. Now identify the objects being compared: a default, tuned model and ensemble spend information and computation differently, even if they share a base-model name.'),
 ('Follow one error','Once procedures and coverage match, reduce their measurements. Ranking within a dataset/split handles incomparable error scales; averaging within datasets before across them controls the unit of influence. These are substantive evaluation choices.'),
 ('Read Elo','Mean rank and Elo both compare relative performance, but they use different constructions and depend on the comparison pool. Understanding the reduction above prevents a rating difference from being misread as an accuracy difference.'),
 ('Recompute, then interpret','The evaluator now has a precise input and output. Recompute it before interpreting the leader, then quantify uncertainty by resampling the paired dataset unit. The four-method audit answers a narrower question than the paper’s full pool.')],
 '[[57|Cross-family ensembling]] asks whether competitors in that pool can become complementary components. Their predictions become inputs to another fitted procedure, which needs its own information boundary.')

story(57,'Use complementary predictions without teaching on their answers',
 '[[54|TabM]] combined related members; [[56|TabArena]] compared whole fitted families. Cross-family ensembling connects those ideas by combining predictors with different biases. The statistical opportunity is error cancellation. The engineering requirement is equally central: every training prediction used by the combiner must come from a base procedure that excluded that row’s target.',[
 ('Different mistakes','A benchmark can show which methods lead, but it does not establish whether their predictions complement one another. Inspect disagreement at the probability level: averaging only helps when it changes the errors that matter to the chosen score.'),
 ('Model architecture','Complementarity motivates a combiner. Training one on in-sample base predictions would give it an unrealistically easy task, so the whole architecture is organized around held-out folds and row-ID alignment. This is why OOF construction precedes weight selection.'),
 ('Learn the weights','The OOF matrix now supplies honest base features for each training row. The next adaptive decision is the combiner itself: choose weights using allowed validation information and freeze both the combiner and serving policy before test.'),
 ('Measure the gain','A valid stack can still fail to improve. Compare against the single model chosen without test labels and interpret the paired differences. The best observed test model is an oracle reference, not an eligible selection rule.')],
 '[[58|The survey lesson]] steps back to organize these families and evidence types. That map will be useful only if it preserves the procedural distinctions we have just built.')

story(58,'Use the literature to choose a sharper next experiment',
 '[[49|Architecture ideas]], [[55|deployment splits]] and [[57|ensembles]] are different axes of a learning system. A survey helps locate them; a benchmark tests particular combinations. This lesson connects classification of methods to reconstruction of evidence, then to an experiment capable of distinguishing competing explanations.',[
 ('Read the original experiment','Classifying a method tells us what operation it introduces. The original experiment tells us whether that operation was isolated, which tasks were used, and what other choices changed. This is the bridge from a conceptual map to a supported claim.'),
 ('Reconstruct the denominator','A result applies only to the tasks and methods that contributed to it. Before normalizing errors or calculating ranks, recover missing entries and exclusions. Otherwise a clean aggregate can conceal a changed question.'),
 ('Make selection bias','Matched scales do not remove selection bias. Dataset filtering, candidate search and repeated analysis can favor particular conclusions. The next step asks how the set of reported comparisons came to exist.'),
 ('From meta-features','The small benchmark produces patterns worth investigating. A meta-feature correlation proposes an explanation; a controlled intervention tests it. Use the evidence map to choose what to hold fixed and what to change in the next experiment.')],
 '[[59|Validation overfitting]] examines one especially subtle selection mechanism: repeated choices can train on the validation set even when no gradient ever touches it.')

story(59,'The selector is a learner too',
 '[[58|Evidence synthesis]] exposed choices hidden behind reported results. This paper focuses on the choices made using validation. Even if each candidate is fitted legally, choosing the minimum noisy score favors lucky candidates. The lesson’s KRR experiment makes that effect measurable, then nests the whole selection procedure to evaluate it honestly.',[
 ('A minimum selects','The validation labels influence which model is kept. Taking the minimum score makes that dependence visible mathematically: the winning noise term is unlikely to average to zero. More choice can improve apparent validation performance without improving the true problem.'),
 ('Reconstruct the measuring','The negative control states what should happen without signal. Now inspect the actual instrument used to test it: KRR coefficients, deleted-row residuals and PRESS candidate selection. The exact computation matters because a training residual answers a different question.'),
 ('Nest the decision','Leave-one-out residuals can evaluate a fixed candidate while still overfitting when used to select one. The outer exclusion must contain both candidate fitting and candidate choice. Nesting follows the adaptive decision, not merely a familiar model-fit function.'),
 ('Audit an ensemble','The same logic applies when the selected object is an ensemble. Diversity is not the source of the evaluation bias; adaptive use of the selection evidence is. Audit what decisions were made and what data evaluated those decisions.')],
 '[[60|The broad comparison]] uses this distinction to freeze a complete selection contract. A leaderboard is defensible only if its winners were selected inside that contract.')

story(60,'Finish the baseline comparison before changing the learning paradigm',
 '[[59|Validation overfitting]] showed that selecting a procedure is itself learning. This checkpoint assembles the earlier architecture, recipe and evaluation lessons into a reproducible decision. Only after task, budget and information boundaries are fixed can quality and cost inform a recommendation. That gives the foundation-model sequence a strong baseline to challenge.',[
 ('Freeze the experiment','The four papers ask different questions, so their conclusions cannot simply be pooled. State the question for this experiment explicitly, then bind datasets, splits, metrics and selection budgets before fitting candidates.'),
 ('Preserve each baseline','A common comparison contract does not mean erasing a method’s essential recipe. Keep its intended preprocessing and fitting procedure visible while matching the declared resources and allowed information.'),
 ('Test labels enter','Each contestant now has a specified fitting and selection path. Freeze its chosen state before revealing test labels. That single boundary turns candidate development into evaluation of a reproducible decision.'),
 ('Quality and cost','Paired errors describe predictive quality under this experiment. A deployment choice also depends on fitting, inference and reuse costs. Reading both prevents a small error advantage from becoming an unsupported universal recommendation.')],
 '[[61|PFNs]] introduce a different place to pay for learning: train an inference procedure across synthetic tasks, then condition on a new dataset without refitting its weights.')

story(61,'Learn the act of inference across many tasks',
 '[[60|Conventional model comparison]] treated fitting on the target dataset as part of each procedure. A PFN moves expensive optimization to a distribution of synthetic tasks. This connects a probabilistic target—the posterior predictive distribution—to a neural training objective. The GP example supplies an exact answer, so architecture, density head and learning can be checked against a common reference.',[
 ('Condition the joint','Sampling a joint function gives related observations, not independent labels. Conditioning that joint Gaussian tells us the correct prediction after context labels arrive. This analytical conditional is the target the neural approximation will be asked to match.'),
 ('Why held-out likelihood','We now know what an ideal conditional predictor returns. Averaging held-out negative log likelihood over sampled tasks rewards the network for approximating that answer. The prior defines which tasks it must learn to infer.'),
 ('Model architecture','The objective specifies what to learn; the architecture specifies how context can affect the query. Separate context label embeddings from query feature tokens and follow the allowed row-attention edges into the query density head.'),
 ('Turn bin masses','The Transformer produces a finite vector, but the target is a continuous density. Bin masses need an explicit within-bin density and tail policy before a likelihood is meaningful. The head therefore connects the forward computation back to the probabilistic objective.')],
 '[[62|TabPFN v1]] applies the same learn-inference idea to tabular classification. The synthetic task prior, input contract and class head change; the context/query distinction remains.')

story(62,'Make the PFN idea a concrete tabular classifier',
 '[[61|The GP PFN]] established why synthetic query likelihood can teach a conditional predictor. TabPFN v1 turns that idea into a historical classifier with an explicit input contract and a frozen row Transformer. Treat preprocessing, row encoding, masked attention and view aggregation as one predictor: each determines what the fixed weights actually receive.',[
 ('Model architecture','The PFN principle explains why weights can stay fixed. The complete prediction route now explains how: construct context and query tokens, mix only permitted rows, decode the queries, then return probabilities in the original class order.'),
 ('Preprocessing is part','The attention mask controls token communication, but the tokens come from a fitted transformation. A checkpoint trained on a particular scale and feature layout does not make arbitrary preprocessing equivalent. Audit the wrapper before blaming the network.'),
 ('Combining several views','A preprocessing view can change feature order or class coding while preserving the task. Run the same fixed model on each declared view, undo class permutations, and only then combine outputs. Aggregation is the final part of the forward system.'),
 ('What the experiment measured','The historical package and checkpoint make the function identifiable. The experiment measures that function under a particular context, view policy and task set. Source correspondence and benchmark performance remain different claims.')],
 '[[63|The synthetic-prior lesson]] opens the world behind those weights: what relationships did the training generator make the network expect?')

story(63,'The generator defines the problems the model learns to solve',
 '[[62|TabPFN v1]] made the inference mechanism concrete. Its fixed weights still embody assumptions learned before the real table arrived. This lesson follows those assumptions back to the generator. The crucial connection is between task-level structure and row-level observations: the PFN must infer one sampled world from several rows, not memorize unrelated random labels.',[
 ('Sparsity is a task','The released generator produces a shared computation for a whole task. Sparsity changes that computation’s graph before observations are drawn, so it changes which relationships all the task’s rows can reveal.'),
 ('Observed features','A sampled graph contains more variables than the final table reveals. Choosing observed columns can hide causes, expose effects or retain proxies. The resulting predictive table therefore does not automatically identify the causal graph that generated it.'),
 ('How a continuous','The structural computation yields a continuous target mechanism. Converting it to classes changes boundaries and class balance, connecting the generator to the classification likelihood the PFN actually optimizes.'),
 ('The query’s features','If both features and labels depend on the sampled world, a query’s features can carry information about that world. This probabilistic observation is distinct from whether a particular released architecture lets query rows influence one another.')],
 '[[64|TabPFN v2]] changes how the table is represented and processed. Keep the prior, architecture and inference wrapper as separate explanations for any change in performance.')

story(64,'Keep the table structure visible while reasoning from context',
 '[[62|v1]] compressed each row before row attention; [[63|the prior]] explained what synthetic tasks teach the weights. v2 changes the computational route: feature-group and target tokens survive through alternating attention axes. The lesson follows how a query feature reaches its target state, then checks the surrounding preprocessing boundary that an attention mask alone cannot guarantee.',[
 ('Encode values','The architecture tells us which objects interact. Encoding defines what those objects mean: observed zero, missing value and unknown target need distinct channels. Without this distinction, the later attention would mix ambiguous evidence.'),
 ('Alternate two attention','The tokens now retain both row and feature-group identity. Within-row attention connects measured features to the target token; across-row attention brings in labeled-context evidence at each position. Repetition lets the two kinds of information combine.'),
 ('A correct mask','The row mask excludes query senders from attention. That does not automatically exclude query influence in earlier numerical preprocessing. The counterexample traces the whole function to distinguish a correct local mask from global query isolation.'),
 ('New local evidence','After identifying the forward function and its boundaries, test whether supplied labels actually change useful predictions. The label intervention examines one information source while keeping the complete pretrained network fixed.')],
 '[[65|Query embeddings]] reuse the final hidden states. The same information-route analysis determines whether those vectors are legitimate features for a new supervised head.')

story(65,'A representation inherits the information used to produce it',
 '[[64|The v2 forward pass]] ends in a query target state before the class decoder. That state can be reused as a feature vector, but its meaning depends on context and role. This lesson connects hidden-state extraction to out-of-fold learning: a representation that already received its own label is not an honest input for evaluating a new probe.',[
 ('Why context and query','A frozen encoder is still a conditional function of its supplied data. Context states can contain their labels; query states are produced without that direct target input. Freezing weights therefore does not remove the need to audit extraction roles.'),
 ('Ten folds','To use query-role vectors for supervised training, each row must temporarily leave the labeled context. Fold extraction implements that requirement and scatters one vector back to each original row ID, connecting the encoder to the later probe dataset.'),
 ('Derive what the linear','With valid vectors in hand, the new head becomes an ordinary supervised learner. Its fitted coefficients can improve the readout without changing the pretrained representation. Keep the encoder context policy separate from the head’s optimization.'),
 ('Selection spends','Choosing a probe or regularization strength adapts to validation. That familiar selection boundary applies even though the encoder is frozen. Evaluate the chosen head once, then diagnose whether extraction-context mismatch contributed to its result.')],
 '[[66|TabICL]] makes row representation construction an explicit early stage. Its pre-ICL vectors are label-free, while its later ICL states are not—the stage name now matters as much as “embedding.”')

story(66,'Compress features before expensive labeled-row reasoning',
 '[[64|v2]] retained feature tokens throughout its repeated blocks; [[65|embedding extraction]] showed why the stage and information boundary of a vector matter. TabICL deliberately separates representation construction from label-based inference. Column statistics inform cell embeddings, row attention compresses features, and only then do labels enter dataset-level attention.',[
 ('Column stage:','The complete block defines how a query reads a memory. The column stage uses that operator twice: learned inducing readers summarize context values, then every cell reads the summary. This gives shared scalar encoders column-specific behavior without using labels.'),
 ('Row stage:','Column processing has contextualized each cell but still leaves a feature axis. Row attention relates those feature states and collects them into four CLS outputs. Concatenating them creates the fixed-width row representation needed by the next stage.'),
 ('Dataset stage:','The feature axis has now disappeared into one row vector. This is the first point where labels are added. Dataset attention can therefore spend its work on relations between labeled examples rather than repeating full feature-axis reasoning.'),
 ('Count the remaining','Separating stages changes where computation is spent; it does not remove attention costs. Count column memory construction, row compression and dataset attention separately before interpreting speed or the effect of more context.')],
 '[[67|LoCalPFN]] attacks context cost and relevance differently: retrieve a neighborhood for each query, then optionally fine-tune the inference rule. Compare that outer retrieval route with TabICL’s internal compression.')

story(67,'Localize the evidence before adapting the inference rule',
 '[[66|TabICL]] reorganized computation inside the pretrained model. LoCalPFN changes what surrounds it: retrieve a query-specific context, normalize that local problem, then run a complete PFN. Optional fine-tuning changes the weights as well. Keeping those interventions separate connects this lesson back to [[52|TabR]] while avoiding the false impression that all retrieval models use the same geometry.',[
 ('Two normalizations','Local context is useful only relative to a distance rule. The global training-fitted scale defines who is close; the second, local scale defines what the pretrained PFN receives. Swapping these roles changes the whole composed predictor.'),
 ('Model architecture','The two transforms specify the retrieval and input contracts. Now trace the complete composition: legal memory, nearest records, local normalization, all pretrained layers and the class head. Retrieval chooses evidence; the PFN still performs the conditional inference.'),
 ('Exact inference','Per-query neighborhoods describe the desired prediction rule. Training can share an anchor neighborhood across several queries to reduce cost, but that is an approximation with a different batch layout. Compare what is reused before comparing runtimes.'),
 ('Fine-tune the query','Retrieval can change predictions with fixed weights. Fine-tuning adds a second adaptation route through query loss and gradient updates. Validation selects its checkpoint; test must evaluate the frozen combination of retrieval and adapted weights.')],
 '[[68|Temporal PFNs]] ask whether useful context must also be interpreted through time. A nearby historical record can still belong to an obsolete relationship.')

story(68,'Extrapolation needs an assumption about how tasks change',
 '[[55|Temporal splits]] made future evaluation explicit; [[67|local contexts]] changed which evidence a predictor reads. Drift-resilient PFNs ask how historical evidence should be interpreted when the relationship itself evolves. The explanation connects a prior over changing mechanisms, a domain-clock encoding, and a frozen predictor evaluated on future domains.',[
 ('The prior:','Naming the shifted distribution tells us what may differ. The temporal prior goes further: it specifies how related domains can differ coherently. A second graph changes selected relationships of the base graph, supplying a learnable extrapolation assumption.'),
 ('Time2Vec:','Synthetic pretraining can teach temporal inference only if the forward model receives the requested domain. Time normalization and encoding connect the clock to the feature tokens; they do not independently guarantee a correct trend assumption.'),
 ('Model architecture','Follow that encoding into the actual released network. Time coordinates join each feature pair rather than forming a separate time token. The target still receives labeled historical evidence through the same explicit source/query boundary.'),
 ('Decide the experiment','A temporal mechanism suggests several tests, but they answer different questions. Freeze domains, legal historical context and checkpoint variants before reading scores. A zero-time intervention is not equivalent to a separately pretrained no-Time2Vec model.')],
 '[[69|Open-environment failures]] broadens the stress test beyond time: new classes, removed features and changed objectives each challenge a different part of the prediction contract.')

story(69,'Name the failed contract before explaining the failed score',
 '[[68|Temporal extrapolation]] challenged the relationship between historical context and future queries. Open environments can also change the class vocabulary, feature schema or evaluation objective. The common thread is the prediction contract: identify what the frozen procedure can represent, preserve difficult rows in the score, then distinguish failure detection from successful prediction.',[
 ('Emerging classes','The released evaluator defines what is actually computed. Begin with the output vocabulary: a native head cannot name a class absent from its supported mapping. Detecting that a query is unfamiliar is a separate decision and requires a separate metric.'),
 ('Keep the unsupported','A confidence score may identify some unfamiliar rows without predicting their new labels correctly. The all-query loss must still account for those rows; dropping them silently replaces the intended open-environment question with an easier closed one.'),
 ('Feature removal','Emerging classes changed the output contract. Removing features changes the input contract. Keep query identities matched and state the fallback transformation so the measured degradation refers to a reproducible change rather than a different dataset.'),
 ('A different objective','A method can retain ranking quality while assigning poor probabilities, or suffer a large relative drop from a high baseline. Connect the score to the deployment objective before treating one failure metric as a general verdict on the model.')],
 '[[70|The final checkpoint]] combines these contracts with predictive quality and lifecycle cost. The deliverable is a replayable model decision and a precise next relational experiment.')

story(70,'Carry a defensible tabular baseline into relational research',
 'The sequence has built three kinds of knowledge: [[49|how a model computes]], [[55|what information it may use]], and [[69|where its deployment contract can fail]]. This checkpoint connects them into one model decision. The output should be a saved, replayable procedure with measured limits—not a preference for a family name.',[
 ('A checkpoint identity','Freezing the prediction question is the first step. Next freeze the object that answers it: package, checkpoint, context, transformations and aggregation policy. Otherwise two runs with the same family name may implement different functions.'),
 ('Compare where each','Once identities are fixed, compare the actual information routes. Trees and TabM store target-task learning primarily in fitted parameters; PFNs also consume labeled context at prediction. That difference explains what must be saved and where computation is paid.'),
 ('Choose the statistical','A reproducible metric still needs an appropriate evidence unit. Preserve method pairing and aggregate within each dataset before across datasets; repeated seeds describe conditional variation, not a multiplied task population.'),
 ('Turn the evidence','Quality, lifecycle cost and information stress now bound what the single-table system establishes. The relational handoff should identify a specific missing entity or event relationship and test its value against this strong baseline under the same availability rules.')],
 'Return to [[55|the temporal availability contract]] when designing relational joins, and to [[52|the retrieval baseline]] when claiming gains from access to other records. The next research question is which explicit relationships add information that these strong flat-table procedures cannot already use.')
