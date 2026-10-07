"""Lesson-specific reading routes; values are copied from the lesson or labeled schematic.

Each route names the complete computation; the focus opens one decisive operation.
No experiment or paper-parity status is inferred from these illustrations.
"""
SPECS={}
def add(key,title,steps,op,question,answer,scope,anchor=None):
 SPECS[key]=dict(title=title,steps=[s.split('|',1) for s in steps],op=op,question=question,answer=answer,scope=scope,anchor=anchor)
add('1','Four tables do not become one row for free',[
 'Ask one question|Customer c at cutoff t: will they cancel in the next 30 days?',
 'Follow three relationships|Orders, sessions and tickets connect to the customer by key.',
 'Summarize eligible history|Aggregate each child table before combining customer features.',
 'Predict from the flat row|The learner sees counts and averages; it does not see the discarded event identities.'
],('join',),'If two customers have the same aggregates, can this model distinguish their event histories?','Not from those aggregates alone. Add a distinguishing feature or preserve the relevant relational context.','Illustrative schema from this lesson. Aggregating each child separately also avoids multiplying rows in a many-to-many join.', 'Three ways the assumption breaks')
add('12','A forest averages trees trained on different views',[
 'Bootstrap rows|Draw a training sample with replacement for each tree.',
 'Randomize each split|Consider a random subset of features at each split.',
 'Grow different trees|Each tree partitions its sampled feature space.',
 'Average predictions|Classification averages class probabilities; regression averages numeric predictions.'
],('ensemble',),'Does averaging help equally when every tree makes the same error?','No. Correlated errors survive averaging; row and feature randomization seek useful diversity.','Schematic three-tree probabilities, not the measured experiment. OOB scoring uses only trees that omitted the scored row.', 'Out-of-bag')
add('13','Boosting keeps the old prediction and adds a correction',[
 'Start with a constant|The squared-error example begins at the target mean.',
 'Compute what remains|For each training row, residual r = y − current prediction.',
 'Fit a small tree|The next tree predicts residuals, not the original target again.',
 'Add a shrunken correction|Fnew(x) = Fold(x) + η × tree(x); repeat, then sum at inference.'
],('boost',),'If the residual is +2 and the tree predicts +2, what does η = 0.1 add?','It adds +0.2 to the old prediction. The learning rate scales the correction, not the accumulated prediction.','Worked squared-error schematic; other losses use their corresponding negative gradients.', 'Why the learning rate matters')
add('14','XGBoost turns gradients into a regularized tree',[
 'Differentiate the current loss|Each training row supplies gradient g and curvature h.',
 'Compare candidate splits|Sum G and H on each side; compare regularized child and parent scores.',
 'Assign leaf values|Each leaf outputs −G / (H + λ); γ penalizes added structure.',
 'Update the ensemble|Add η times this tree. At prediction time, sum the selected trees.'
],('xgb',),'With G = −6 and H = 2, how does λ changing from 1 to 4 change the leaf?','The leaf shrinks from 2 to 1: 6/(2+1) versus 6/(2+4).','Illustrative leaf sufficient statistics. The figure opens the regularizer already derived in this lesson.', 'Why XGBoost actually won')
add('15','LightGBM changes how the next tree is built',[
 'Accumulate histogram statistics|Bin feature values; store gradient and curvature sums per bin.',
 'Evaluate candidate leaves|The same regularized gain logic scores potential splits.',
 'Expand the best leaf|Leaf-wise growth spends the next split on the greatest available gain.',
 'Add the tree to the ensemble|GOSS and EFB are separate efficiency mechanisms; their applicability depends on the configured training path.'
],('leaf',),'If left and right leaves offer gains 2 and 7, which expands next?','Leaf-wise growth chooses gain 7, subject to constraints. It does not require both leaves to advance to the same depth.','Schematic gains. GOSS samples rows; EFB bundles features. Neither is the definition of leaf-wise growth.', 'GOSS')
add('16','CatBoost keeps a row’s own target out of its encoding',[
 'Choose a training permutation|Earlier positions define the permitted prefix for a training row.',
 'Encode a category from its prefix|Use preceding target statistics plus the prior; exclude the current target.',
 'Build oblivious trees|Within one tree depth, the same split is used at every node.',
 'Sum fitted tree predictions|Ordered boosting also protects gradient construction; serving uses fitted training statistics.'
],('ordered',),'For the second A row, may its own target enter the training category mean?','No. Only preceding A targets and the declared prior can enter that row’s ordered statistic.','Illustrative A-category targets 1, 0, 1; unsmoothed prefix means are shown only to expose the exclusion. Real encoding includes a prior.', 'Ordered boosting')
add('19','Noise creates more chances to chase an accident',[
 'Hold the signal fixed|The target depends on the same informative feature.',
 'Append irrelevant columns|Additional features carry no population signal about the target.',
 'Fit using finite observations|A noise column can correlate with the target by chance in this sample.',
 'Evaluate held-out rows|The accidental association need not persist; robustness is empirical, not immunity.'
],('noise',),'Does an unused noise column necessarily change a fitted tree prediction?','No. The point is the opportunity for a spurious selected split, not a claim that every appended column changes every tree.','Schematic feature-target associations; no measured score is drawn. Compare the actual noise-feature experiments in L027 and L051.', 'Strength 3')
add('76','Trace customer 7 through the relational stack',[
 'Keep query ownership|Customer 7 and its cutoff select eligible event rows; keys remain metadata.',
 'Encode two schemas|Independent customer and event encoders each emit four coordinates.',
 'Pool incoming events|Mean eligible event vectors by customer; concatenate the customer’s own vector.',
 'Read out one target|The 8 → 8 → 1 head produces a logit; BCE trains both encoders through the aggregation.'
],('mean',),'If customer 7 has two eligible events, what fraction of the mean’s gradient reaches each event?','One half of each arriving coordinate gradient, before the event encoder’s own derivatives. Excluded events have no path.','Original one-hop course model, d = D = 4. The scalar mean below illustrates one embedding coordinate.')
add('78','A GCN mixes neighbors and feature coordinates',[
 'Prepare Cora|Row-normalize 2,708 × 1,433 features; add graph self-loops once.',
 'Normalize graph routing|S = D⁻½(A + I)D⁻½ supplies fixed neighbor weights.',
 'Apply two graph layers|Dropout → S X W₀ → ReLU, then dropout → S H W₁.',
 'Read seven class scores|Training uses 140 labels; inference disables dropout and keeps the feature graph.'
],('gcn',),'Does changing W change which edges exist?','No. W mixes feature coordinates; S routes between nodes on the declared graph.','Cora teaching/reproduction path. The small graph opens the two axes; its values are schematic.')
add('95','A three-step walk returns to candidate items',[
 'Build the fitting graph|Likes define B with users on one side and items on the other.',
 'Normalize both directions|PUI = Du⁻¹B; PIU = Di⁻¹Bᵀ.',
 'Walk user → item → user → item|Multiply PUI PIU PUI while preserving intermediate identities.',
 'Blend, mask and rank|Validation chooses popularity weight; remove seen items before top-k evaluation.'
],('walk',),'Why does a two-step user → item → user walk not directly rank items?','It ends at users. A third transition maps the returned user mass back to the item catalog.','Deterministic course scorer, with no neural weights. Degree-zero fallback and official split caveats remain in the protocol.')
add('99','Separate relation averaging from learned attention',[
 'Freeze the comparison input|Use the same ACM features, typed edges and label split.',
 'Train complete alternatives|Two-layer R-GCN, HGT, uniform-HGT and feature-only MLP.',
 'Open the aggregation rule|R-GCN averages within relations; HGT normalizes learned scores at a receiver.',
 'Select before testing|Validation chooses epoch and learning rate; compare frozen target-aligned predictions and costs.'
],('relations',),'Is replacing HGT scores with uniform weights the same as relation-wise averaging?','No. A uniform receiver softmax weights all incoming edges equally; relation-wise means first normalize each relation separately.','Schematic three-edge example. Equal widths or epochs do not imply equal capacity or cost.')
add('100','Sampling shrinks context without changing seed identity',[
 'Select paper seeds|The first B paper occurrences are the supervised targets.',
 'Sample typed context|Two hops, fanout 8 per relation; retain global-to-local maps.',
 'Encode and pass messages|Typed adapters feed the two-layer R-GCN/HGT alternatives.',
 'Train seeds; validate globally|Cross-entropy uses only seed labels. Full-graph validation selects the frozen test predictor.'
],('seed',),'Should a sampled context paper automatically contribute its label to this batch loss?','No. Context can carry features without being a supervised seed. Keep the first-B seed boundary explicit.','ACM course checkpoint. Full-neighbor gradient parity holds at fixed weights; sequential optimizer updates are a different trajectory.')
add('110','Score the event before it changes memory',[
 'Consume old queued messages|A 688-value message updates the 172-value GRU memory.',
 'Read strict history|Attention uses ten historical edges with timestamps less than the current event time.',
 'Score real and negative pairs|Two-head attention yields embeddings; the decoder supplies positive and negative probabilities.',
 'Queue the real event afterward|Current edge features create messages for real endpoints only; negatives never update memory.'
],('time',),'May the current event’s edge features help score that same event?','Not in this declared TGN path. They enter the message queue after scoring.','Wikipedia TGN-attn: learned weights can be frozen while temporal memory continues to evolve.')
add('112','Three graph layers, one masked training objective',[
 'Prepare the full feature graph|169,343 × 128 features; binary undirected citations, loops and normalized S.',
 'Hidden layer one|S X W₁ᵀ + b₁ → BatchNorm → ReLU → dropout; width 256.',
 'Hidden layer two and head|Repeat the hidden block, then graph-linear 256 → 40 and log-softmax.',
 'Train and restore correctly|NLL uses 90,941 training labels. Validation selects weights and BatchNorm buffers together.'
],('gcn',),'Are held-out feature rows removed merely because their labels are masked?','No. This transductive GCN uses the full feature graph; the supervised loss uses only training labels.','Complete ogbn-arxiv GCN path. The operator sketch is a small graph, not the benchmark adjacency.')
add('113','Sampling and GraphSAGE answer different questions',[
 'Partition the products graph|METIS makes 15,000 disjoint groups.',
 'Choose an induced training graph|Union 32 groups; retain edges between selected groups.',
 'Compute two message branches|Transform the neighbor mean and the root features separately, then add.',
 'Repeat and predict|100 → 256 → 256 → 47; inference uses all neighbors, one complete layer at a time.'
],('sage',),'Does the sampler replace the learned root branch?','No. Sampling chooses the computation graph. The root and neighbor maps still define the layer inside that graph.','Released products path. ReLU and dropout apply to hidden layers, not final logits.')
add('114','The MLP comparison changes the normalization population',[
 'Start from paper features|Each row has 128 numeric coordinates; there is no adjacency input.',
 'Apply hidden blocks|128 → 256 → 256, with BatchNorm, ReLU and dropout after each affine map.',
 'Map to forty scores|The output affine layer and log-softmax produce class scores.',
 'Separate fitting and evaluation|Training BN sees training rows; evaluation uses its saved statistics.'
],('bn',),'Is removing graph edges the only change relative to the released GCN?','No. The training normalization population also differs: the GCN sees all feature rows, the MLP trains on training rows.','Source comparison from this lesson; schematic population dots do not represent benchmark row counts.')
add('117','One driver query owns every sampled occurrence',[
 'Ask for driver 90 at day 7|Keep result at day 5; exclude result at day 11, at every hop.',
 'Encode tables and relative ages|Per-table encoders emit width 128; dated rows get query-relative time vectors.',
 'Run two typed message layers|Within each relation sum neighbor states and add its learned root map; combine relations, normalize and activate.',
 'Predict query roots|The head emits finishing position; L1 trains the stack and validation selects the saved state.'
],('cutoff',),'Does reaching a row through an old first hop make its future descendants legal?','No. Every hop must still satisfy the original query’s cutoff.','RelBench regression route. Evaluation clips predictions to declared training-label percentiles; that is not the training loss.')
add('127','A relational prediction begins with an entity and a time',[
 'Form a query-owned neighborhood|Use the task entity and cutoff; keep future target values separate.',
 'Encode heterogeneous rows|Per-table feature encoders and relative-time features create common-width vectors.',
 'Pass typed messages|Two layers aggregate eligible context and update root representations.',
 'Select the task readout|Root head → task loss for training; validation selection → frozen test scoring.'
],('cutoff',),'Can two occurrences of the same entity share arbitrary context when their cutoffs differ?','No. They ask different historical questions; context must remain attached to its query owner.','RelBench task pipeline. Read the lesson’s task-specific head, metric and timestamp contract with the full diagram.')
add('129','A manual feature baseline still learns an additive model',[
 'Build cutoff-safe aggregates|SQL/relational feature engineering produces one numeric row per question.',
 'Fit preprocessing on training data|Keep held-out information out of learned transforms.',
 'Fit boosted corrections|Each tree adds a learning-rate-scaled update to the current score.',
 'Select and evaluate|Validation chooses the fitted procedure; test estimates its task performance.'
],('boost',),'If the relational join drops a useful distinction, can a larger booster reconstruct it from identical feature rows?','No. Identical model inputs cannot reveal which discarded history produced them.','Manual-feature baseline, not an end-to-end learned join. The numeric correction is illustrative.')
add('131','The batch is a set of questions, not just node IDs',[
 'Create query occurrences|An entity key and cutoff identify each target occurrence.',
 'Sample and encode|Each occurrence owns legal typed context, encoded to width 128.',
 'Update through typed layers|Relation-specific neighbor and root maps combine into a seed representation.',
 'Read and supervise seeds|The task head acts on queried roots; context rows are not automatically targets.'
],('seed',),'Why is a node ID alone insufficient to identify a temporal prediction?','The same node queried at different times can have different available context and different future targets.','RelBench GNN–tabular stack. The tiny seed/context diagram shows ownership, not actual fanout.')
add('138','A customer score needs both a query and a book representation',[
 'Sample time-valid neighborhoods|Keep customer and book occurrences tied to their query time.',
 'Encode rows and graph context|Shared learned components produce the vectors used by the selected recommendation path.',
 'Compare candidate books|Score customer–book pairs under the lesson’s declared candidate universe.',
 'Rank and evaluate|Validation selects the model; held-out target keys and ranking protocol determine the reported metric.'
],('dot',),'Can two recommendation scores be compared fairly if one ranks a small candidate sample and the other the full catalog?','Not as the same evaluation claim. Candidate-universe changes can change the difficulty and metric.','Read the complete source-specific architecture and candidate contract below; the dot-product sketch illustrates pair scoring only.')
add('139','A clinical query keeps one time boundary through the graph',[
 'Identify the trial query|Attach the prediction cutoff to the root occurrence.',
 'Filter at every relation|Study and related dated records must satisfy that same cutoff.',
 'Encode and aggregate context|Typed encoders and message passing update the root state.',
 'Predict the declared target|Train only on the permitted target split; evaluate with the task’s fixed contract.'
],('cutoff',),'May a future neighbor enter through an eligible older study record?','No. The path does not reset the query clock; every dated occurrence must pass the same boundary.','Temporal eligibility sketch. Availability and missing-clock caveats remain explicit in the lesson.')
add('143','One composite layer spans two graph edges',[
 'Select an ordered route|Constructor → result fact → driver destination; retain query ownership.',
 'Fuse source and fact|Map source context and fact attributes into one route-specific value.',
 'Attend within the destination route|The driver query scores incoming facts; weighted values combine before route outputs are summed.',
 'Predict at the root|Four heads concatenate, project to 128 and combine routes; the root head predicts finishing position.'
],('composite',),'Does traversing constructor → result → driver require two composite model layers?','No. One composite layer uses that two-edge route. Graph-hop distance and model-layer count differ.','The scalar 3 and 4 trace is illustrative; the actual model uses four 128-coordinate heads and an L1 objective.')
add('153','One shared encoder trains from pairwise comparisons',[
 'Form three root groups|Facility queries, positive sponsors and sampled sponsors each get temporal neighborhoods.',
 'Reuse the same encoder|Typed encoders and two sum-GraphSAGE layers emit 128-coordinate vectors.',
 'Compare positive and sampled scores|For each query, dot products feed softplus(negative − positive).',
 'Rank the full sponsor catalog|Selected weights encode all 53,241 sponsors; score blocks and retain top ten.'
],('bpr',),'If the positive score is 2 and the negative score is 0, does reversing them increase the loss?','Yes: softplus(−2) ≈ 0.127 becomes softplus(2) ≈ 2.127.','Recommendation portfolio path. Scores are ranking quantities, not calibrated probabilities.')
add('162','Frozen language weights can still pass gradients',[
 'Serialize and encode rows|BART encodes row text and schema information.',
 'Exchange relational context|The GNN refines representations using connected rows.',
 'Decode masked text|A reconstruction loss measures whether the missing target can be recovered.',
 'Distinguish the two stages|Adapt the language model first; in the graph stage freeze language weights while updating graph weights.'
],('frozen',),'Does freezing the decoder’s weights mean blocking the gradient at its input?','No. A fixed differentiable decoder can still transmit the loss gradient to the GNN that supplied its input.','Conceptual source-inspired route. The historical graph–decoder tensor interface remains unresolved; the lab does not train BART.')
add('167','Relational preparation and pretrained attention are different layers',[
 'Construct the task matrix|Strict cutoff-safe joins and aggregates produce the same 72 features for the compared runs.',
 'Provide labeled support|512 support rows and labels accompany 702 query rows.',
 'Apply the selected frozen predictor|TabICL reads column/row/ICL states; RDB-PFN alternates feature and support-row attention.',
 'Decode the query target|Each measured arm emits binary probabilities; TabPFN is a conceptual comparator here.'
],('support',),'Does using relationally engineered features mean the predictor itself attends over the original database graph?','No. The shown predictors receive the constructed table. Graph structure has already been summarized by preparation.','Selected TabICL/RDB-PFN comparison. Shared preprocessing and support identities are part of fairness.')
add('175','Value masks and attention masks do different jobs',[
 'Sample the query context|1,024 cells contain values, column names, row identities and foreign-key links.',
 'Encode cells to width 256|The query target uses a mask embedding instead of its answer.',
 'Apply twelve relational blocks|Column, feature, neighbor and full attention each define different access; residual updates and SwiGLU transform states.',
 'Read the boolean head|Final RMSNorm and a 256 → 1 head give DNF probability under frozen weights.'
],('mask',),'If the query answer is masked, can other future-dated cells still leak?','Yes. Hiding the target value does not certify the sampler’s context. That is why this lesson stops before model scoring.','Actual RT-v1 architecture; the input audit failed. This drawing does not imply that the stopped evaluations ran.')
add('183','Transfer must preserve parameter meaning, not just shape',[
 'RelGT builds row tokens|Type, hop, time, features and local structure concatenate and project to width 512.',
 'RelGT mixes local and global context|Local sampled-row attention runs beside attention to learned global centroids.',
 'Griffin builds a different state|Task-conditioned cell attention and relation-aware messages use distinct modules.',
 'Transfer within an architecture|A Griffin checkpoint initializes Griffin; a RelGT adapter must be built and trained as a new proposal.'
],('transfer',),'Can matching width 512 justify loading Griffin weights as a pretrained RelGT?','No. Equal tensor widths do not establish that modules or coordinates have the same function.','Source architectures and an unrun proposed transfer path. No new pretraining or fine-tuning is implied.')
add('192','A frozen predictor still depends on a stable fitted map',[
 'Choose study keys and cutoffs|Collect the permitted relational history for each question.',
 'Construct DFS features|Follow key paths and aggregate; for example mean(3,5) = 4 and count = 2.',
 'Fit once; preserve meaning|Support and query must use the same fitted preprocessing map.',
 'Call the selected frozen backend|Labeled support plus query features produce predictions only after the preprocessing gate passes.'
],('codes',),'If b was encoded as 0 during support fitting, may it become 1 after seeing a new query category?','No. That changes the meaning of an existing coordinate/code. The measured instability blocks backend admission here.','Source diagnostic on synthetic categories. Backend inference and the full model search remain unrun.')
add('193','One failed shared transform can invalidate every backend arm',[
 'Declare the query and support|Entity/cutoff keys and label availability define the legal context.',
 'Run deterministic DFS|Depth H produces support n × p and query q × p matrices.',
 'Audit the fitted transform|An unseen query category must not change the codes of known support categories.',
 'Select only admitted candidates|The planned backend search is downstream of this gate; the gate stopped the declared experiment.'
],('codes',),'Does repeating the same preprocessing error across three backends make their comparison valid?','No. Shared code is not proof of correct input semantics. All dependent arms inherit the failed admission gate.','Measured synthetic preprocessing diagnostic, not task-level accuracy evidence.')
add('b18','Context sufficiency is a property of the question',[
 'Declare the target question|Fix the query, support population and permitted information.',
 'Choose a context representation|A subset or summary retains some distinctions and removes others.',
 'Predict from the retained state|The predictor cannot recover a distinction absent from all its inputs.',
 'Test the lost distinction|Construct paired histories with identical retained state but different relevant structure.'
],('collision',),'Do equal means guarantee that two contexts answer every future question equally well?','No. Histories [0,4] and [2,2] have the same mean but different variance. Sufficiency depends on the target question.','Illustrative collision, not a universal claim that raw context is always superior.')
add('b20','Curriculum changes the training trajectory',[
 'Generate declared synthetic tasks|A task provides support features/labels and hidden query targets.',
 'Choose the task order|Hold the task pool and budget fixed while changing presentation order.',
 'Train the compact course PFN|Support labels enter context; query labels enter only the training loss.',
 'Evaluate frozen parameters|Use the declared held-out task set and compare like-for-like evidence lanes.'
],('support',),'If query targets are supplied as predictor inputs, are you testing a better curriculum?','No. You have changed information access and leaked answers. Task order cannot repair that contract violation.','Source-shaped compact course model; not a historical large-model curriculum reproduction.')
add('b21','A structural edit changes who supplies a message',[
 'Construct the declared graph|Keep node features and target split fixed for the structural intervention.',
 'Encode node rows|The course GraphSAGE-style model produces hidden node features.',
 'Aggregate in both directions|Each layer combines the root with transformed neighbor information.',
 'Compare matched predictions|Changing one edge changes a message route; the experiment must say what else is held fixed.'
],('sage',),'Does deleting an edge necessarily delete the destination’s own contribution?','No. The root branch remains. The intervention removes or changes a neighbor contribution and its normalization.','Course bidirectional GraphSAGE-style model; paper/tutorial differences remain documented.')
add('b23','A query target is decoded, never supplied',[
 'Build declared relational features|Use the frozen feature contract and paired support identities.',
 'Encode support and query cells|Labels are present for support; query target inputs are masked.',
 'Alternate the checkpoint’s axes|Feature mixing works within rows; row attention reads permitted support states.',
 'Decode and compare|Query target states produce probabilities; keyed predictions support the declared evaluation.'
],('support',),'Can two runs use different support draws and still isolate only the model change?','Not in the paired comparison claimed here. Support identity is part of the controlled input.','Selected released-checkpoint inference. This is not pretraining or whole-paper reproduction.')
add('b24','An architecture defense needs a distinguishing example',[
 'Name the retained representation|Mean-only flattening maps a history to one scalar.',
 'Construct a collision|Two different histories can map to the same scalar.',
 'Repair the relevant distinction|Adding variance separates this particular pair.',
 'Limit the conclusion|A repaired example motivates a test; it does not prove universal sufficiency or model superiority.'
],('collision',),'After adding variance, have you preserved every possible distinction between histories?','No. It repairs this collision only. Different histories can still share both mean and variance.','Illustrative architecture-defense example. B24 remains saved-evidence replay, not fresh model execution.')
add('28','A residual block learns a change to the carried state',[
 'Encode the row|Numeric values and categorical embeddings form the input representation.',
 'Compute a learned branch|Normalization, affine maps, nonlinearities and dropout transform the state.',
 'Keep the identity route|Add the branch output to the unchanged incoming state.',
 'Repeat and predict|The final representation feeds the task head; the skip does not bypass the loss.'
],('residual',),'If the learned branch outputs zero, what does the residual block return?','It returns its input. This identity route is why a block can learn a small refinement without relearning the carried state.','ResNet baseline schematic; consult the full diagram for the configured block ordering.', 'The full architecture')
add('32','Categorical tokens share context; numeric values take a bypass',[
 'Embed each categorical value|Value and column identity create the feature tokens.',
 'Mix categorical tokens|Repeated Transformer blocks contextualize categorical features within the row.',
 'Join the numeric branch|Concatenate contextual categorical states with normalized continuous values.',
 'Apply the prediction head|The MLP sees both branches and emits the task output.'
],('bypass',),'Can a numeric value change categorical attention weights in this architecture?','Not through the pictured input path. Numerics join after categorical attention, at the MLP input.','TabTransformer route. This limitation motivates the all-feature tokenizer in L046.', 'Lab —')
add('43','TabNet carries a feature-use prior from step to step',[
 'Normalize and initialize|The input and initial feature transform provide the first attention state.',
 'Select a sparse mask|The attentive transformer combines its state with the carried prior; sparsemax can set entries exactly to zero.',
 'Transform selected values|Multiply the mask by the original input, then split transformed state into decision and next-attention parts.',
 'Accumulate and repeat|Add decision outputs; update the feature-use prior before the next step, then predict from the accumulated decision.'
],('sparse',),'Does the mask multiply the previous hidden vector or the original feature input?','The feature mask selects the original feature input at each step. The attention state and prior govern that selection.','TabNet encoder schematic. The feature-use prior encourages reuse control; it is not a hard one-use rule.', 'The mechanism, piece by piece')
add('44','A soft oblivious tree sends weight to every leaf',[
 'Select a feature at each depth|Entmax produces differentiable feature-selection weights.',
 'Compute soft left/right decisions|The selected value, learned threshold and scale feed entmoid.',
 'Multiply routing probabilities|A depth-d tree assigns one product weight to each of its 2^d leaves.',
 'Combine learned responses|Weighted leaf responses form tree outputs; ensembles and stacked layers compose the NODE network.'
],('softtree',),'With two right-branch probabilities both 0.5, how much mass reaches each of four leaves?','Each gets 0.25. Products over left/right choices sum to one across all leaves.','Two-depth routing schematic. The actual lesson widget lets you change the feature and gate behavior.', 'NODE the network')
add('45','Training and pretraining use the same categorical route',[
 'Embed categorical columns|Column-aware embeddings represent observed categorical values.',
 'Contextualize within the row|Transformer blocks mix the categorical tokens.',
 'Attach continuous values|Normalized numerics join contextual states at the MLP head.',
 'Choose the objective|Supervised prediction fits labels; optional pretraining learns from corrupted/unlabeled rows under its own objective.'
],('bypass',),'Does optional pretraining move continuous features into categorical self-attention?','No. A different training objective does not by itself change the input topology.','TabTransformer architecture. Pretraining and supervised comparison remain separately reported in the lesson.', 'Training it:')
add('46','FT-Transformer lets every feature enter attention',[
 'Tokenize numeric values|For feature j, multiply scalar xj by learned vector wj and add bj.',
 'Embed categories and prepend CLS|All feature tokens share width d; CLS is a learned readout token.',
 'Mix all token types|Repeated attention and feed-forward blocks update feature and CLS states.',
 'Decode CLS|The final CLS state feeds the prediction head; training labels enter the task loss.'
],('token',),'If one numeric value changes, can its token influence the final CLS state?','Yes. The numeric token participates in the same attention route as the other features.','FT-Transformer topology, contrasted with the numeric bypass in TabTransformer.', 'Training it:')
# The later baseline revisit uses the same operator and explicitly names its own scope.
SPECS['42']={**SPECS['28'],'title':'Read the skip before comparing the trained baselines','anchor':'Why the skip','scope':'Residual baseline reused from L028. This figure explains the operator, not a new benchmark result.'}
# The Amazon lesson predicts customer churn, not product recommendations.
SPECS['138'].update(title='A book influences churn through its review path',steps=[
 ['Choose a customer query','The review history determines eligibility; the next outcome window defines the churn label.'],
 ['Encode the three row types','Customer, review and product attributes become learned 128-coordinate representations.'],
 ['Pass context through reviews','A product informs a review; a later layer can carry that state to the customer.'],
 ['Predict customer churn','The customer root head produces one logit; binary cross-entropy trains against the churn target.']
],op=('churn',),question='Is this output a ranking over books?',answer='No. It is one churn prediction for the queried customer. Books supply relational context through review rows.',scope='rel-amazon customer-churn path. Eligibility and future outcome windows are distinct; future reviews are not input features.')
