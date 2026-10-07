"""Pictorial reading guides. Detailed lesson figures remain the variant authority.

Each stage specifies a drawing, not a generic box of prose. All small tables,
weights, and graphs are schematic; no measured model result is encoded here.
"""
STORIES = {}
def story(key, title, why, stages, question, answer, source, scope='Conceptual overview; the detailed architecture below specifies the pictured implementation.'):
    STORIES[str(key)] = dict(title=title, why=why, stages=stages, question=question, answer=answer, source=source, scope=scope)

def stage(title, drawing, label, explanation):
    return dict(title=title, drawing=drawing, label=label, explanation=explanation)
S=stage
story(52,'TabR: a prediction can look things up','Follow two information paths: the query itself and the labeled rows it retrieves.',[
S('Query + training memory','retrieval','Features find neighbors; labels supply values','Encode the query and eligible training rows. Search in learned key space; the query must not retrieve itself during training.'),
S('Read the neighbors','values','Value = label embedding + key correction','Distance-based weights combine values E(yᵢ) + T(k − kᵢ). The label and directed difference play different roles.'),
S('Keep the direct path','residual','Query state + retrieved context → head','Add the retrieved representation to the query representation, then use the prediction network. Retrieval is inside the model, not a replacement for its head.')],
'If two neighbors have equal weights, must they contribute equally?','No. Their learned value vectors can differ in both magnitude and direction. A weight alone does not describe a contribution.','https://arxiv.org/html/2307.14338v2#S3.SS2')
story(53,'RealMLP: the recipe reaches beyond the layers','See the input geometry, learned network and training schedule as one recipe.',[
S('Scale the coordinates','scale','Training statistics → robust input transform','Fit preprocessing on training rows. Smooth clipping limits extreme transformed values without a hard discontinuity.'),
S('Fit the network','mlp','Input scaling → hidden layers → output','The lesson implements numeric TD-S. The fuller TD recipe contains additional choices; do not treat this picture as a diagram of every variant.'),
S('Control the updates','schedule','Learning-rate schedule changes over training','Initialization, parameterization, regularization and the schedule affect what the architecture learns. A comparison must specify the recipe as well as its widths.')],
'Would identical layer widths guarantee an identical RealMLP baseline?','No. Preprocessing and training choices can change the fitted function even when the layer sizes match.','https://arxiv.org/html/2407.04491v3#S2', 'Numeric TD-S teaching route. This is a recipe overview, not the full TD architecture.')
story(54,'TabM: several predictors share the expensive part','Trace the member-specific paths through one shared weight matrix.',[
S('One row, several members','fork','The same row enters k member paths','Each ensemble member sees the row. Member-specific scaling lets the paths behave differently without storing k independent large matrices.'),
S('Share W, vary the scales','ensemble','Member i: ((x ⊙ rᵢ) W) ⊙ sᵢ','The small r and s vectors belong to members; the large W is shared. Repeat the configured ensemble layers through the backbone.'),
S('Train separately, combine','mean','Member losses during fitting; average at inference','Each member receives a learning signal. Combine the member predictions at evaluation; averaging before the training loss is a different objective.')],
'If all member-specific parameters and stochastic choices were identical, what would averaging buy?','The members could make identical predictions. Parameter sharing alone does not guarantee useful diversity.','https://arxiv.org/html/2410.24210v3#S3.SS2')
story(61,'PFNs: learn an algorithm from many small tasks','Separate learning shared weights from conditioning those weights on a new table.',[
S('Generate a task','prior','Sample a function, then its labeled rows','A task prior produces a dataset. Split its rows into observed support and targets whose labels will be predicted.'),
S('Learn across tasks','episodes','Many tasks → shared predictive weights','During pretraining, query-label loss updates one network across sampled tasks. The network learns how support relates to query predictions.'),
S('Condition on a new table','context','Labeled support + unlabeled query → prediction','In the basic PFN use case, inference supplies a new context to fixed weights. No query label enters that context.')],
'What changes when you replace the support labels at inference?','The input context changes, so predictions can change even if no parameter is updated.','https://arxiv.org/html/2112.10510v7#S3')
story(62,'TabPFN v1: a table becomes a context','The row is the token; support labels are observed information.',[
S('Encode rows and labels','context','Support: x + y; query: x without its y','Feature and label encodings provide observed context. Query labels stay hidden.'),
S('Use masked row attention','row-attention','A query reads support; queries stay separate','The attention mask determines which rows can exchange information. Do not confuse access to query features with access to query targets.'),
S('Read query predictions','distribution','Shared pretrained transformer → class scores','A query representation feeds the prediction head. Any preprocessing and ensemble transformations remain part of the inference recipe.')],
'Why must the diagram draw the label boundary, not just a stack of transformer blocks?','The same blocks can solve a different, leaking problem if query labels are allowed into context.','https://arxiv.org/html/2207.01848v6#S2')
story(64,'TabPFN v2: attend along two table axes','Keep rows and feature groups visible before reading out the target.',[
S('Make a token grid','grid','Rows × feature groups × hidden width','Preprocessing and feature grouping produce a structured token grid. The target is represented with the required support/query masking.'),
S('Alternate attention axes','axial','Across features, then across data rows','Feature attention combines information within an example. Row attention supplies evidence from the observed examples; the attention mask controls access.'),
S('Decode the query target','distribution','Target representation → predictive head','Read the query target representation after the repeated blocks. Classification and regression have different output contracts.')],
'Are feature attention and row attention interchangeable?','No. They mix different axes: features within examples versus evidence across examples.','https://www.nature.com/articles/s41586-024-08328-6','Historical TabPFN v2 conceptual route; grouping, masks and ensemble settings are detailed in the lesson.')
story(66,'TabICL: change the unit of attention','First understand a column, then a row, then the labeled dataset.',[
S('Learn column context','inducing','Column values ↔ inducing points','Column processing uses a compact set of inducing points to construct contextual cell representations.'),
S('Compress each row','row-pool','Cell representations → row representation','Row attention combines a row’s feature representations. The dataset-level learner then works on rows rather than the full cell grid.'),
S('Learn from labeled rows','context','Support row vectors + labels → query class','Dataset-level in-context learning conditions on labeled support representations and predicts the query labels.')],
'Where does reducing the number of features stop being the same as reducing context length?','After row compression, the dataset transformer sees row representations. Feature count and support-row count affect different stages.','https://arxiv.org/html/2502.05564v1#S3')
story(67,'Local PFNs: choose the evidence around a query','Retrieval and parameter adaptation are separate decisions.',[
S('Find a local context','retrieval','Query features select eligible support rows','Retrieve using allowed features and the declared fitted transform. A close row is still inadmissible if its label is unavailable.'),
S('Condition the PFN','context','Local support + query → pretrained network','The selected neighborhood replaces a global support set. This changes what evidence the network can see.'),
S('Optional fine-tuning','adapt','Frozen inference or declared weight updates','If fine-tuning is used, name its data and budget separately. Changing context and changing model weights are different mechanisms.')],
'Can retrieval repair a missing or late-arriving label?','No. Similarity does not make unavailable information admissible.','https://github.com/layer6ai-labs/LoCalPFN/tree/ff8803c57cd277380b2444f0f5ed4856b46f47f5', 'Conceptual local-context/adaptation route. Consult the lesson sources for the exact retrieval and fine-tuning variants.')
story(71,'VIME: learn what was changed','One corrupted row creates two complementary pretraining targets.',[
S('Corrupt some cells','corruption','Clean row + donor values → changed row','Replace selected cells using donor values. Preserve both the clean values and the corruption information for the pretext task.'),
S('Share an encoder','two-heads','Hidden state → mask head AND value head','One head predicts which cells changed; another reconstructs values. Both losses train the encoder.'),
S('Reuse the representation','transfer','Retained encoder → supervised task head','The pretext heads are not the downstream predictor. The released route and the lesson specify how the encoder and task head are trained or frozen.')],
'Why does predicting the mask differ from reconstructing the row?','The mask asks where corruption happened; reconstruction asks what the values should be. These are different supervision signals.','https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html')
story(72,'SCARF and SubTab: preserve row identity across views','The common idea is agreement; the view construction and objectives differ.',[
S('SCARF: replace cells','corruption','Clean row ↔ marginally corrupted row','SCARF pairs a clean row with a corrupted view. Both pass through the same encoder and projector.'),
S('SubTab: select columns','subsets','Overlapping column subsets of the same row','SubTab constructs several views from subsets of columns and combines reconstruction with configured representation objectives.'),
S('Transfer the encoder','contrast','Match views by row identity, then fit the task','Contrastive similarities identify same-row pairs. Downstream use retains the learned representation route, not necessarily the pretraining projector.')],
'Does a different row with the same class automatically become a positive pair?','No. In the instance-contrastive construction, positive pairs are views of the same row, not merely rows sharing a downstream label.','https://arxiv.org/html/2106.15147v2#S3','Comparison of two view constructions, not a serial pipeline from SCARF into SubTab. SubTab source: arXiv:2110.04361.')
story(74,'CARTE: turn a row into a small graph','Column names become relations; values become the neighboring content.',[
S('Build a star for one row','star','Center = row; leaves = cell values','A schema-flexible representation connects a row center to its cells. Encodings of column names describe the edges.'),
S('Condition the messages','edge-product','Value vector ⊙ column-name vector','The edge representation changes the message before attention aggregates it at the center. The names affect computation, not just display labels.'),
S('Read the row center','transfer','Pretrained encoder → task-specific prediction','Reuse learned graph computation and fit the chosen downstream head. The local route need not reproduce the full pretraining stack.')],
'What would be lost if the graph kept values but discarded column names?','Messages would lose the relation semantics that tell the model what each cell value describes.','https://arxiv.org/html/2402.16785v2#S3.SS3')
story(75,'PyTorch Frame: types choose the encoder','One table contains several kinds of values; keep their conversion explicit.',[
S('Group columns by type','typed','Numbers | categories | text','A fitted materializer records statistics and vocabularies from training data. Held-out rows reuse that state.'),
S('Create aligned tokens','tokenize','Type-specific encoders → common width','Numerical, categorical and text encoders convert their inputs into column representations with a compatible hidden dimension.'),
S('Read out a row','row-pool','Column tokens → backbone → task head','The configured backbone mixes or pools the tokens. A row encoder by itself does not pass messages between database rows.')],
'Which part makes the model relational?','None of this row-only diagram. Relationships must enter through a further graph or context stage.','https://arxiv.org/html/2404.00776v2')
story(81,'Message passing: the graph routes the computation','Track one node’s neighbors instead of imagining one dense layer over all nodes.',[
S('Choose incoming neighbors','graph','Edges define who can send to the center','Each node has a hidden state. For this update, the graph chooses which neighbor states and edge attributes are available.'),
S('Build and reduce messages','aggregate','Messages → permutation-invariant reduction','Apply a message function on each incoming edge, then combine messages without depending on their arbitrary storage order.'),
S('Update, repeat, read out','recurrent','Old state + aggregate → new state','An update function creates the next state. Repeated rounds expand the receptive field; the task chooses a node, edge or graph readout.')],
'Would swapping two entries in a neighbor list change a sum aggregator?','No. A sum is invariant to that storage order, although changing the actual edges or values can change the result.','https://proceedings.mlr.press/v70/gilmer17a.html')
story(82,'GCN: smooth along edges, learn across features','Graph mixing and feature mixing act on different axes.',[
S('Include the node itself','self-loop','Adjacency A → A + I → degree scaling','Self-loops retain a self contribution. Symmetric degree normalization controls how neighbor contributions are scaled.'),
S('Apply S H W','gcn','S mixes nodes; W mixes feature channels','Multiply node states by the shared feature map and normalized graph operator. A hidden-layer nonlinearity follows.'),
S('Repeat to produce logits','gcn-stack','Hidden GCN layer → output GCN layer','The usual two-layer classifier has different weights per layer, shared across nodes. Only permitted labels enter the training loss.')],
'Which object changes if you rewire edges but keep all learned weights fixed?','The graph operator S changes. The same feature weights can therefore produce different node predictions.','https://arxiv.org/html/1609.02907v4#S2')
story(83,'GraphSAGE: sample, aggregate, combine','A learned neighborhood function can operate on newly encountered nodes.',[
S('Sample a neighborhood','sample','Bound the neighbors considered at each hop','Sampling determines the computation neighborhood. Multi-hop expansion still needs an explicit fanout budget.'),
S('Keep self and neighbors','concat','Self state ∥ aggregated neighbor states','The chosen aggregator summarizes the neighbors. Concatenate this with the node’s own state before the learned update.'),
S('Reuse the learned rule','new-node','Same weights on a new node’s features','The model learns an update function, rather than requiring one independent learned embedding for every node identity.')],
'Can an unseen node be encoded without any features or usable neighborhood?','The function can be reused, but it still needs its defined inputs. Inductive weights do not create missing evidence.','https://arxiv.org/html/1706.02216v4')
story(84,'GAT: neighbors receive learned weights','The graph still controls access; attention controls the mixture.',[
S('Transform allowed neighbors','graph','Project the center and its neighbors','A shared feature projection prepares node states. Only graph-allowed pairs enter the attention calculation.'),
S('Normalize edge scores','weighted-graph','Pair score → softmax over this neighborhood','Compute attention scores and normalize them over the receiving node’s neighbors. The drawn widths illustrate weights, not measured importance.'),
S('Combine attention heads','heads','Several weighted sums → layer output','Each head forms its own neighborhood mixture. Concatenation or averaging depends on the layer and variant.')],
'Can a large attention weight reveal information from a node with no allowed edge?','No. Weighting operates over the allowed neighborhood; it does not add missing graph connections.','https://arxiv.org/html/1710.10903v3#S2')
story(87,'SEAL: classify a neighborhood around an edge','The candidate pair defines the graph that the classifier will inspect.',[
S('Mark the candidate pair','pair','Two endpoints + enclosing neighborhood','Extract a local subgraph around the candidate link. The split and target-edge removal rules prevent the candidate from revealing its own label.'),
S('Label structural positions','labeled-graph','Node labels encode position relative to endpoints','Endpoint-relative structure distinguishes nodes that identical raw features might otherwise make indistinguishable.'),
S('Predict one link','graph-pool','GNN → subgraph readout → link score','A graph classifier summarizes this candidate-specific subgraph. Different candidate links can induce different computations.')],
'Why is this more than taking the dot product of two fixed embeddings?','The input is a candidate-specific subgraph, so the model can use structural patterns around the pair.','https://arxiv.org/abs/1802.09691')
story(88,'GIN: keep multiplicities in the neighborhood','Summing distinguishes some multisets that averaging collapses.',[
S('Read a multiset','multiset','Neighbors may repeat feature values','A neighborhood is an unordered collection with multiplicity. Two copies of one state are different input from one copy.'),
S('Sum, then learn','gin','(1 + ε) self + Σ neighbors → MLP','The sum and self term feed a learned MLP. The expressiveness argument depends on the relevant assumptions, not just naming the block GIN.'),
S('Read the whole graph','graph-pool','Pool node representations across the graph','A graph-level readout combines the node states, potentially from multiple depths, before the task head.')],
'Why can mean aggregation lose a difference between one and two identical neighbors?','Their mean is the same, while their sum changes. This is the multiplicity distinction pictured here.','https://arxiv.org/html/1810.00826v3#S4')
story(89,'Cluster-GCN: batch a connected piece of the graph','The partition changes the sampled computation, not the GCN update rule.',[
S('Partition the graph','clusters','Dense connections stay mostly inside groups','A graph partitioner groups connected nodes so that sampled blocks retain many useful internal edges.'),
S('Choose clusters for a batch','cluster-batch','Selected groups → induced batch graph','Train on the graph induced by selected clusters. Edges outside that batch are absent from its computation.'),
S('Run the GCN on that batch','gcn','Standard graph convolution, bounded working set','The local GCN forward/backward pass uses the batch graph. Sampling and normalization choices still belong to the experiment contract.')],
'Are all cross-cluster edges necessarily used in every minibatch?','No. An edge is available only when its endpoints and the batching rule admit it into the batch graph.','https://arxiv.org/html/1905.07953v2#S3')
story(91,'R-GCN: the edge type chooses the message map','Different relationships should not all use one indistinguishable transformation.',[
S('Preserve relation types','relations','Different edge colors mean different relations','A multi-relational graph distinguishes, for example, an authorship edge from a citation edge.'),
S('Transform by relation','relation-maps','Neighbor h → Wᵣ h → relation-wise reduction','Each relation contributes through its configured map and normalization. Basis sharing can reduce how many independent parameters are stored.'),
S('Combine with a self term','residual','Sum relation contributions + self update','Combine incoming relation messages and the node’s own transformed state, then apply the nonlinearity.')],
'What changes if you preserve endpoints but swap two relation labels?','The selected relation transformations change, so the same connected nodes can yield different updates.','https://arxiv.org/html/1703.06103v4')
story(92,'HAN: attend within paths, then between paths','A meta-path chooses a semantic neighborhood before attention weights it.',[
S('Choose a relation sequence','metapath','Example: author → paper → author','A meta-path specifies which multi-step relation connects neighbors. Different paths create different semantic neighborhoods.'),
S('Attend within each path','heads','One neighborhood summary per meta-path','Node-level attention builds a representation for each chosen path. These summaries do not yet decide which path matters most.'),
S('Mix semantic summaries','semantic','Path representations → semantic attention','Semantic-level attention combines the path-specific representations into a node representation for the task.')],
'Could two meta-paths connect the same endpoints but convey different semantics?','Yes. Their relation sequences can describe different reasons for the connection, which is why path-specific summaries are retained.','https://arxiv.org/html/1903.07293v2#S4')
story(93,'HGT: types enter both attention and messages','Follow the source type, relation type and destination type through one edge.',[
S('Keep the edge triplet','relations','Source type — relation → destination type','The node and relation types are computational inputs, not merely graph annotations.'),
S('Build typed attention','hgt','Typed Q/K/V + relation transformations','Type-specific projections and relation-specific transformations determine attention scores and message content.'),
S('Reduce into the target','weighted-graph','Weighted messages → target-type update','Aggregate the allowed incoming messages and apply the target-side transformation and residual route.')],
'Why is one shared projection for every type a different model?','It removes a type-dependent part of HGT’s attention and message parameterization.','https://arxiv.org/html/2003.01332v1#S3')
story(94,'metapath2vec: types constrain a random walk','The walk defines the context pairs used to learn embeddings.',[
S('Select a schema path','metapath','A repeating sequence of permitted node types','The schema guides the random walk through heterogeneous relations rather than accepting any adjacent type.'),
S('Walk to create contexts','walk','Walk sequence → center/context windows','Nearby positions in sampled walks become training contexts. This is a sampling construction, not a supervised task label.'),
S('Fit node embeddings','embedding','Typed skip-gram objective → embedding table','Learn embeddings from those co-occurrences. Downstream evaluation requires a separate task and split contract.')],
'If two graphs share their node types but differ in edges, do they generate the same walks?','Not generally. The schema restricts type transitions, but actual edges determine which transitions are available.','https://ericdongyx.github.io/metapath2vec/m2v.html')
story(97,'BPR: train a relative preference','The training object is a user–positive–negative triple.',[
S('Sample a comparison','pairwise','User u, observed item i, sampled item j','The sampler chooses the comparison. An unobserved item is not automatically a proven dislike.'),
S('Score the difference','difference','s(u,i) − s(u,j)','Use one scoring model for both pairs. The objective depends on their score gap, not two independent class labels.'),
S('Optimize a ranking loss','ranking','Increase log σ(score gap), with regularization','Update the configured parameters to favor the observed item over the sampled alternative; evaluate with a declared candidate set.')],
'Would replacing the negative sampler leave the training problem unchanged?','No. It changes which comparisons the loss emphasizes.','https://arxiv.org/abs/1205.2618')
story(102,'TGN: memory changes after events','Read time in one direction: predict from admissible history, then update state.',[
S('Read past memory','timeline','Events before cutoff → node memory','Each node’s memory summarizes processed events. Event order and the memory-update schedule are part of the model contract.'),
S('Build temporal embeddings','memory','Memory + temporal neighborhood → embeddings','The embedding module can combine memory with neighbor information at the query time. A decoder scores the requested interaction.'),
S('Update after observation','recurrent','New event → message → memory update','Once an event is available under the chosen schedule, create and aggregate messages, then update memory for subsequent predictions.')],
'Why can updating memory with the event before scoring it leak information?','The target event may enter the state used to predict itself. The update schedule must preserve the intended information boundary.','https://arxiv.org/html/2006.10637v3#S3')
story(103,'TGAT: encode elapsed time inside attention','A recursive neighborhood query carries a timestamp at every level.',[
S('Retrieve earlier neighbors','timeline','For query (v,t), only events before t','Each recursive call has its own time boundary. A root-valid edge can still be invalid inside an earlier nested query.'),
S('Encode time differences','waves','Elapsed time → sinusoidal feature vector','Functional time encoding gives attention a representation of relative event time alongside node and edge features.'),
S('Attend recursively','temporal-attention','Earlier neighbor representations → query state','Temporal attention combines the eligible representations across layers. This route differs from maintaining a persistent TGN memory.')],
'Can a recursive neighbor use an event that occurred before the root query but after that neighbor’s query time?','No. Every recursive query must enforce its own cutoff.','https://arxiv.org/html/2002.07962v1#S3')
story(106,'EdgeBank: remember observed edges','A deliberately simple temporal baseline makes the value of recurrence visible.',[
S('Read historical events','timeline','History up to the prediction cutoff','Admit only events that are available before the query. Decide whether the memory covers all history or a finite window.'),
S('Store edge membership','edge-memory','Was this endpoint pair seen in memory?','The baseline stores edge occurrence, rather than learning a node-feature transformation.'),
S('Score candidate edges','pair','Seen versus unseen pair → baseline score','Candidate sampling and the time window can change apparent performance. State them alongside the score.')],
'Would a repeated edge receive the same treatment after it expires from a finite memory window?','No. Its membership changes once it leaves the admitted window.','https://arxiv.org/html/2207.10128v2#S4')
story(107,'EvolveGCN: the evolving state is a weight matrix','Distinguish changing graph features from changing the graph-convolution parameters.',[
S('Observe graph snapshots','snapshots','G₁, G₂, G₃ … with node features','Each snapshot supplies its graph and node features. Node sets can differ between snapshots.'),
S('Evolve convolution weights','weight-state','Recurrent state → next GCN weight matrix','In EvolveGCN-H, summarized node embeddings help the GRU evolve weights; the O variant uses a different recurrence.'),
S('Convolve with current weights','gcn','Current graph + evolved Wₜ → embeddings','Apply the GCN with the time-specific weights, then the task head. This is not the same state as TGN’s per-node memory.')],
'Which object is carried through time here: one memory vector per node, or convolution parameters?','The convolution parameters. The precise recurrent inputs depend on the EvolveGCN variant.','https://arxiv.org/html/1902.10191v3#S3')
story(118,'Relational GNNs: keep the database connections','Rows become nodes; foreign keys route information between tables.',[
S('Map rows and keys','database','Table rows → nodes; foreign keys → edges','Construct the relational graph without replacing the full neighborhood by one manually aggregated flat row.'),
S('Encode, then pass messages','relations','Row attributes → states → typed messages','Encoders handle row attributes. Relational message passing uses the schema connections to propagate information.'),
S('Read the task entity','target-graph','Target-node state → prediction head','The task chooses which entities are scored. Available features, neighborhood extraction and evaluation splits remain explicit.')],
'Does turning rows into nodes automatically prevent temporal leakage?','No. The graph construction still needs feature-availability and task-time rules.','https://arxiv.org/abs/2002.02046v1')
story(132,'Identity-aware messages: mark the query’s role','The same database neighborhood can be interpreted relative to different targets.',[
S('Extract a rooted neighborhood','target-graph','Task entity → sampled computation graph','The task identifies a root. Neighborhood sampling decides which database records can participate.'),
S('Mark the target identity','root-mark','Give the root a distinct input signal','An identity signal tells the message-passing network which node is the current target instead of relying only on its ordinary attributes.'),
S('Propagate and read the root','aggregate','Identity-aware states → root representation','Messages can now depend on the target’s role. The root readout produces the task prediction.')],
'What ambiguity can remain when the root looks exactly like another node?','Without a role signal, an otherwise symmetric computation may fail to distinguish which node the task refers to.','https://arxiv.org/html/2407.20060v1','Conceptual identity-aware relational route; the lesson defines the exact marking and sampling implementation.')
story(141,'RelGNN: compose a relation through a bridge','A many-to-many relationship needs its intermediate records to stay visible.',[
S('Keep the bridge table','bridge','Entity → bridge row → related entity','A bridge row identifies an individual association. Its columns may carry information that a collapsed endpoint edge would discard.'),
S('Compose the messages','composite','Follow the relation sequence before reduction','Composite message passing treats a relation path as a structured route. Follow where each intermediate state is formed and aggregated.'),
S('Update the target entity','target-graph','Relation-path evidence → task-node state','Combine the admitted messages at the target and apply the downstream prediction head.')],
'Why can replacing all bridge rows by a single endpoint connection lose information?','It can remove multiplicity and bridge attributes, so distinct relational situations become the same graph input.','https://arxiv.org/html/2502.06784v2#S4')
story(144,'ContextGNN: combine local context and global identity','The two prediction routes cover different candidate situations.',[
S('Build query-local context','target-graph','Query entity → sampled relational neighborhood','Local graph computation builds representations conditioned on the query neighborhood.'),
S('Keep two scoring routes','dual-score','Contextual candidates | global candidate embeddings','Candidates available in local context can use contextual representations. A global embedding route supplies representations beyond that neighborhood.'),
S('Combine into candidate scores','ranking','Route-aware scores → ranked candidates','The prediction layer combines the relevant routes. Candidate eligibility and ranking evaluation are separate from the architecture.')],
'Why retain a global candidate route if the local graph is informative?','The sampled local neighborhood need not contain every eligible candidate. The global route covers candidates outside it.','https://arxiv.org/html/2411.19513v1#S4')
story(145,'RelGT: make a relational neighborhood into tokens','A token needs both its content and its place in the relational context.',[
S('Sample around the target','target-graph','Target entity + typed, time-valid neighbors','The local relational neighborhood defines the input set. Temporal and ownership rules determine which records are eligible.'),
S('Encode relational position','rel-tokens','Content + type + position/time information','Token encodings preserve the structural roles that a plain unordered collection of row vectors would omit.'),
S('Mix tokens, read the target','token-attention','Transformer blocks → target representation','Attention exchanges information across the admitted token set. The target representation then feeds the task head.')],
'Why are ordinary feature vectors insufficient to describe this input?','They do not by themselves encode each record’s relational role, type or temporal position relative to the target.','https://arxiv.org/html/2505.10960v1#S3')
story(159,'Mask a cell, learn from its context','A relational pretraining objective must define both the hidden target and the visible evidence.',[
S('Choose the hidden value','masked-cell','Observed database + explicitly hidden target','Select a cell or target to predict, then remove its value from the visible input.'),
S('Build the allowed context','database','Related rows can supply predictive evidence','The schema can connect informative records. Copies of the hidden value and unavailable future information still need exclusion.'),
S('Predict the missing content','two-heads','Shared representations → configured prediction head','A loss teaches the model to recover the target from allowed context. The objective alone does not establish transfer to a new database.')],
'Is replacing one displayed cell with a mask sufficient if the same target is copied elsewhere?','No. A duplicate or derived value elsewhere can still reveal the target.','https://arxiv.org/html/2305.15321v1#S2','Conceptual relational masked-prediction objective, not a claim that one fixed architecture implements every vision paper.')
story(163,'Semantic row encoders: names become model inputs','A numerical value and its meaning are separate pieces of information.',[
S('Keep names beside values','typed','Column name + cell value + declared role','Serialization or typed encoding must specify which schema text and cell content the model receives.'),
S('Use a language representation','embedding','Text / semantic encoder → cell or row vectors','A pretrained encoder can supply representations of names and text. Numeric handling still needs an explicit design.'),
S('Connect to the task','row-pool','Combine encoded content → downstream predictor','The downstream model consumes those representations. A name-ablation comparison tests the contribution of semantic information.')],
'Would replacing all column names by arbitrary IDs preserve the model input?','No. It removes or changes semantic information, even if the numeric table and split stay identical.','https://arxiv.org/html/2305.15321v1#S2','Conceptual semantic-encoding route; not a single universal serialization or model architecture.')
story(164,'Griffin: reread cells as graph context changes','The row state evolves, but individual cells remain available for another read.',[
S('Keep cell keys and values','cell-memory','Column metadata → keys; cell contents → values','Frozen input encoders produce cell representations. Keep the cell arrays rather than permanently collapsing them into a single row summary.'),
S('Alternate reading and routing','cell-graph','Cell attention ↔ relational row-state updates','The current task context guides cell reads. Relational messages update row and task state, which can change what is useful at the next read.'),
S('Read the requested roots','dual-head','Final root representation → class or numeric head','Gather the task roots after the final updates. Classification and regression use different output routes; the lesson specifies the released transfer configuration.')],
'What is lost if you average the cells once and discard them before message passing?','Later graph context can no longer change how the model reads individual cells. That removes the repeated cell-access mechanism.','https://arxiv.org/html/2505.05568v1#S3')
story(165,'KumoRFM: condition on a relational task','Draw the labeled examples and query as part of a relational context.',[
S('Declare the prediction task','database','Database + target entities + prediction time','The task tells the system what to predict and when. Admissible database history is determined relative to that task.'),
S('Supply labeled support','rel-context','Support examples + query neighborhood','Support labels convey the current task. The relational context can carry information beyond a flattened query row.'),
S('Predict with shared weights','context','Task-conditioned inference → query output','The system uses a pretrained relational model. This functional overview does not invent undisclosed layer details.')],
'Does an in-context support example allow using its label before that label is available?','No. Support membership must obey the same availability contract as the rest of the context.','https://kumo.ai/research/','Functional information-flow view. Public descriptions and the lesson’s evidence boundary govern what is known about internals.')

# Bridge introductions have their own lesson framing. Reuse a picture only when
# the mechanism really is the same, and label that reuse in the coverage report.
def reuse(key, base, title=None):
    import copy
    STORIES[key]=copy.deepcopy(STORIES[str(base)])
    if title: STORIES[key]['title']=title
reuse('b02',54,'Numerical embeddings and shared ensembles: locate the ensemble mechanism')
reuse('b03',61,'PFN generations: separate pretraining from task context')
reuse('b04',66,'TabICL: trace column, row and dataset computation')
STORIES['b04']['scope']='Shared three-stage TabICL idea. The bridge lesson details TabICLv2; this schematic does not equate its exact blocks with the historical L066 model.'
reuse('b11',118,'Supervised relational baselines: from database rows to predictions')
reuse('b14',52,'A retrieval baseline sets a stronger flat-table bar')
STORIES['b14']['scope']='TabR mechanism reminder for baseline comparison; this is not the architecture of TabPFN-Rel or RDBLearn.'
story('b05','TabDPT: a real table becomes a prediction task','The chosen target column changes what retrieval is permitted to see.',[
S('Choose a target column','masked-cell','Real table → features and prediction target','Construct episodes from real tables by choosing a target. Prevent that target from entering the features used to retrieve neighbors.'),
S('Retrieve a local episode','retrieval','Feature-only neighborhood → support + queries','The neighborhood supplies labeled context. Retrieval is an information-selection step, not an exemption from leakage rules.'),
S('Learn across episodes','context','Shared transformer → query predictions','Train on episodes and use context for prediction. The lesson separates released inference evidence from historical pretraining identity.')],
'What happens if neighbor retrieval includes the very column being predicted?','The retrieval can depend on the answer, leaking target information even if that column is later masked inside the transformer.','https://arxiv.org/abs/2410.18164')
story('b06','Mitra: change the distribution of training tasks','The mixture lives before the learner sees an episode.',[
S('Choose a generator family','mixture','Several task families → declared mixing weights','The training prior determines which kinds of prediction problems the model practices.'),
S('Sample a task and episode','prior','Generator draw → support and query rows','Each draw instantiates a task. Split observed evidence from targets within that episode.'),
S('Train a shared predictor','episodes','Episode losses → shared predictive model','The controlled lesson experiment holds the learner and budget fixed while changing its task distribution.')],
'Is mixing training-task generators the same operation as averaging fitted models?','No. One changes the tasks seen during learning; the other combines predictions from already fitted models.','https://arxiv.org/html/2510.21204v1#S3','Mitra prior-mixture concept; the lesson separately discusses Mitra-v2 and the bounded course experiment.')
reuse('b07',163,'Semantic transfer: follow the name into the prediction')
story('b07a','HyperFast: write a predictor from a support table','The generated weights are an output of one network and an input to another.',[
S('Summarize labeled support','context','Support features + labels → task representation','Process the support table to represent the current prediction problem.'),
S('Generate network weights','hyper','Hypernetwork → parameters θ for a predictor','A parameter-generating network writes the downstream predictor. This differs from merely storing support tokens for attention.'),
S('Run the generated predictor','mlp','New query x → network with θ → output','Queries use the generated predictor, with any declared refinement accounted for separately. Repeated-query cost differs from one-time generation cost.')],
'What is the critical output of the hypernetwork?','Parameters of another predictor. A task embedding alone is not the complete generated predictor.','https://arxiv.org/abs/2402.14335','Conceptual HyperFast route. Random projection, dimensionality reduction and refinement details remain in the full architecture.')
story('b08','LimiX: make the hidden cells explicit','An objective is defined by what is observed, what is hidden and what is scored.',[
S('Choose observed cells','masked-cell','One table, a declared visibility mask','The mask specifies which cells are input evidence and which are prediction targets.'),
S('Process available context','axial','Table-aware computation over visible evidence','Representations combine allowed information across the table. Hidden targets must not re-enter through another feature or preprocessing fit.'),
S('Score the hidden targets','two-heads','Prediction heads → configured reconstruction losses','The training objective can reward more than one target pattern. The lesson’s ablation isolates what objective changes, holding other choices fixed.')],
'If you change both visibility and the network, can an observed gain identify the objective’s effect?','No. The comparison confounds the information supplied, the computation and the learning objective.','https://arxiv.org/html/2509.03505v2#S2','Conceptual objective/visibility guide. Exact LimiX architecture and objective scope are in the cited lesson sources.')
story('b10','Relational Transformer: cells communicate through the schema','A foreign key determines routes between cells in different rows.',[
S('Tokenize database cells','cell-memory','Each cell carries content and its schema role','Represent individual cells rather than committing immediately to one fixed vector per row.'),
S('Route allowed attention','cell-graph','Within-row routes + schema-linked routes','Sparse relational attention routes connect the permitted cell groups. A link changes who can communicate.'),
S('Read the masked task cell','masked-cell','Contextual task-cell state → prediction head','Predict the masked target from the resulting representation. The visibility mask must exclude target values and inadmissible copies.')],
'Why does removing a foreign-key route change more than the drawing?','It changes the allowed information paths into the masked task cell.','https://arxiv.org/html/2510.06377v1#S3.SS1','Conceptual cell-level routing view. Use the source-linked detailed diagram for the exact Relational Transformer operator.')
story('b16','AutoGrable: choose whether to build a graph','Graph construction is a selected modeling decision.',[
S('Propose candidate columns','subsets','Column subsets define candidate relationships','Each candidate construction proposes a way for table rows to become neighbors.'),
S('Score the constructions','selection','Evaluate candidates under the declared rule','Construction selection must use the allowed evidence and budget. Keep an explicit option to decline graph edges.'),
S('Train on the chosen input','clusters','Selected graph or no-edge baseline → predictor','The downstream GNN receives the selected structure. A schema or similarity function does not guarantee useful connections.')],
'Can a correct graph-selection procedure return no edges?','Yes. If the selection rule favors the no-edge alternative, inventing a graph would contradict that decision.','https://arxiv.org/html/2608.11431v1#S4','Conceptual selection pipeline; see the lesson’s primary source for the exact AutoGrable scoring rule.')
story('b22','RefineICL: support representations become working state','Hold the inputs fixed and follow a write to the internal support state.',[
S('Encode support and query','context','Observed support labels + unlabeled query','The same task context starts the comparison. Keep the input table and model weights fixed.'),
S('Refine internal support','support-write','Read support → update support representations','An internal write changes the support state used by later computation. This is distinct from updating model parameters.'),
S('Read the refined state','row-attention','Later query reads → prediction','Later reads can depend on the updated support. A skip or permutation intervention tests whether that write matters in the controlled experiment.')],
'If the weights and raw inputs are fixed, can predictions still change after an internal support write?','Yes. Later computation consumes a different internal state; fixed parameters do not imply fixed activations.','https://arxiv.org/html/2609.27679v1#S3','Conceptual support-state mechanism. The lesson distinguishes the paper from its randomly initialized course diagnostic.')

# Version-pinned primary sources already used by the individual lessons.
for key, source in {
 'b05':'https://arxiv.org/html/2410.18164v3#S3',
 'b07a':'https://arxiv.org/html/2402.14335v1#Sx4',
 'b08':'https://arxiv.org/html/2509.03505v2#S2',
 'b10':'https://arxiv.org/html/2510.06377v1#S3.SS1',
 'b16':'https://arxiv.org/html/2608.11431v1#S4',
 'b22':'https://arxiv.org/html/2609.27679v1#S3',
}.items():
    STORIES[key]['source']=source
STORIES['145']['stages'][2]=S('Mix local and global context','local-global','Local token attention + global centroid attention','The local module mixes the sampled tokens. A global module reads learned centroid context; the task representation combines these routes before prediction.')
STORIES['b10']['stages'][1]=S('Apply four attention routes','cell-graph','Column → feature → neighbor → full attention','RT-v1 applies separately parameterized attention sublayers in sequence, then a feed-forward update. Relational masks structure some routes; the full-attention sublayer also mixes across the admitted context.')
STORIES['159']['stages'][2]['drawing']='distribution'
story(184,'GELGT: keep local edges and attend farther away','Two parallel summaries meet at a learned gate.',[
S('Encode and select tokens','sample','Protect nearby nodes; rank distant candidates','Encode table, hop, time, field and structural information. Preserve nearby nodes, then use seed compatibility to select distant candidates.'),
S('Compute two summaries','gnn-attention','GraphSAGE branch ∥ temporal-attention branch','GraphSAGE follows retained edges. In parallel, attention compares tokens with time-dependent score bias.'),
S('Gate the two routes','gate','g × GNN + (1 − g) × attention → head','A learned sigmoid gate mixes the summaries before the regression head. The detailed lesson records source/paper depth differences.')],
'Can temporal attention recover a useful row that the sampler discarded?','No. Both branches operate on the admitted context; the missing row supplies no token to attend to.','https://arxiv.org/html/2605.15575v2#S3','Released regression route, shown conceptually. Exact token budgets, depths and cutoff policies remain in the detailed diagram and protocol ledger.')

# Preserve the critical distinction: RDB-PFN consumes relationally derived
# tabular features; it is not a graph encoder at released-checkpoint inference.
story(166,'RDB-PFN: relational tasks become tabular context','Synthetic relational structure enters through generated tasks and relational features.',[
S('Generate linked records','rel-prior','Synthetic schema + foreign keys + table contents','Pretraining samples relational databases and prediction tasks. These generators determine which dependencies the learner practices.'),
S('Construct relational features','flatten','Related records → DFS aggregates → feature table','Deep Feature Synthesis turns relational neighborhoods into tabular attributes such as counts and means. The released checkpoint receives this feature matrix, not raw graph edges.'),
S('Predict from labeled support','axial','Feature tokens + support labels → query logits','The transformer mixes features and support rows. Query labels remain hidden; a decoder reads the query label token. Evaluation uses fixed pretrained weights.')],
'Is the released checkpoint passing GNN messages along foreign keys during this forward pass?','No. Its input is the released feature matrix. Relational feature construction and transformer inference are separate stages.','https://arxiv.org/html/2603.03805v5#S5','Synthetic-relational pretraining and released DFS-feature inference route. See the detailed diagram for the selected checkpoint’s six blocks and token dimensions.')
reuse('b13',166,'Synthetic relational tasks: follow structure into the training table')
reuse('182',166,'RDB-PFN: separate relational feature construction from inference')
STORIES['182']['scope']='RDB-PFN route only. The lesson separately contrasts RelGNN composite message passing; the two are not one combined architecture.'
reuse('125',75,'Typed row encoding: materialize, tokenize, predict')
STORIES['141']['stages'][1]=S('Compose before reducing','composite','Source state + bridge state → destination message','A composite message combines the source row with the intermediate bridge row before destination aggregation. Distinguish this from two generic layers that prematurely compress the intermediate context.')
STORIES['b10']['stages'][1]['drawing']='rt-routes'
story(191,'KumoRFM-2: task information enters early','Four axes of information exchange connect cells, rows, relations and context examples.',[
S('Condition the cell encoding','axial','Column attention + row attention within tables','The smaller table-level network alternates attention axes. Known context labels enter early so feature extraction can depend on the task; query targets stay hidden.'),
S('Mix relational and task context','rel-context','Foreign-key routes + labeled context examples','The larger network exchanges information along relationships and across context examples. Every example’s context must respect its own time cutoff.'),
S('Read the query target','distribution','Task-conditioned representation → prediction','Frozen base weights produce in-context predictions in the selected paper comparison. Fine-tuning, lagged targets and ensembling require separate protocol descriptions.')],
'Would adding the task label only at the final prediction head depict the same mechanism?','No. Early task conditioning lets intermediate feature extraction depend on the labeled context.','https://arxiv.org/html/2604.12596v1#S3','Published information-flow overview. No unreleased widths, layer counts or checkpoint availability are implied.')
story(63,'An SCM prior: sample a world, then observe it','The generator’s causal graph is a way to make training tasks, not a recovered graph of your data.',[
S('Sample a mechanism graph','causal','Directed dependencies + random mechanisms','Sample a directed acyclic graph and functions or noise variables that determine its values.'),
S('Generate rows in that world','prior','One sampled mechanism → many observations','Rows from one task share a generating mechanism. Different sampled mechanisms create different tasks for pretraining.'),
S('Choose observed and target variables','masked-cell','Visible features + hidden target → episode','The model sees only the admitted variables. Predictive success under this task distribution does not identify the real-world causal graph.')],
'Does accurate prediction from SCM-generated training tasks demonstrate causal identification on a new dataset?','No. Predictive inference and identifying the true data-generating causal structure are different claims.','https://arxiv.org/html/2207.01848v6#S2','Conceptual SCM prior used to explain the lesson’s synthetic-task generator.')
story(68,'Drift-Resilient TabPFN: time enters the prior and the input','The model practices tasks whose relationships change across domains.',[
S('Generate changing mechanisms','causal','Shared graph; mechanisms evolve across domains','The pretraining generator creates related domains with changing mechanisms. A temporal prior is a modeling assumption, not a guarantee about every real drift process.'),
S('Encode the domain coordinate','waves','Source-fitted time transform → feature/time inputs','The released route pairs a domain coordinate with feature inputs. Query coordinates use the source-fitted transformation, including extrapolation when necessary.'),
S('Read earlier labeled domains','context','Source-domain context + later query → prediction','The model predicts from the permitted labeled context. Future labels are not memory, and the domain split remains part of the evidence contract.')],
'Can a source-fitted time transform map a later query beyond the source coordinate range?','Yes. That is extrapolation, not evidence that the transformation should be refit using future labels or outcomes.','https://arxiv.org/html/2411.10634v1#S3','Conceptual drift-prior and inference route. The lesson’s historical release and time encoding remain authoritative.')
story('b04a','TabFlex: summarize keys and values before the query','Associativity can change the cost of attention without building every pairwise score.',[
S('Encode support and query','context','Support keys/values; separate query features','Pretrained encoders prepare representations. Label visibility and feature preprocessing still define which information enters.'),
S('Accumulate sufficient statistics','statistics','S = Σ φ(kᵢ)vᵢᵀ; z = Σ φ(kᵢ)','Aggregate support contributions before processing each query. These statistics depend on the chosen feature map and attention formulation.'),
S('Read the summary','linear-read','Read φ(q)ᵀS, normalize with φ(q)ᵀz','A query reads the accumulated support state. This avoids explicitly materializing the full query-by-support interaction matrix in the illustrated normalized kernel calculation.')],
'Does this algebra make a kernel-attention model exactly equivalent to arbitrary softmax attention?','No. The feature map, normalization and model design determine the operator. An efficient factorization does not prove equality to a different attention kernel.','https://arxiv.org/html/2506.05584v1#S3','Normalized kernel calculation used by the lesson to explain support statistics; see the full TabFlex diagram for its encoders and output heads.')
story('b15','An information boundary: two worlds can look identical','Ask what reaches the predictor before asking how expressive it is.',[
S('Construct two possible worlds','two-worlds','Same observed features; different hidden target','The lesson pairs cases that agree on the admitted observations but require different answers.'),
S('Apply the observation mask','masked-cell','An encoder can use only the visible inputs','A learned or parameter-free encoder cannot distinguish worlds that produce exactly the same input to it.'),
S('Reveal a separating observation','context','An available support label may break the tie','The extra observation must be available under the protocol. Separating the worlds with an oracle label is not a deployable improvement.')],
'Can a more complicated deterministic encoder distinguish two exactly identical visible inputs?','No. To separate the worlds, the system needs additional information or a changed observation contract.','https://arxiv.org/html/2607.05476v2#S3','Information-boundary illustration for the lesson’s finite experiment; not a claimed reconstruction of the paper’s entire theorem or benchmark.')
story('b17','FlexTab: reuse features without reusing the labels','Keep the feature stream separate from task-conditioned decoding.',[
S('Encode features only','row-pool','Cell tokens + row token → shared feature stream','The feature encoder does not read the target labels. The local model alternates attention across columns and allowed rows.'),
S('Combine outputs across depths','depth-sum','Projected row states from multiple layers → Z','Layer-specific projections combine row representations into the shared output. Cache Z only while its encoder dependencies are unchanged.'),
S('Decode each target stream','task-decoders','Same Z + task-specific support labels → predictions','Separate target streams read the shared row representations and the permitted support labels. A label edit can change predictions without changing Z.')],
'If a support label changes while features and encoder weights stay fixed, must Z change?','Not in the feature-only encoder shown here. The decoder’s predictions can still change because its target stream receives the edited labels.','https://arxiv.org/html/2606.30336v2#S2.SS2','FlexTab mechanism; detailed dimensions below belong to the reduced untrained lab, not the published default model.')
story('b17-gtalign','GTAlign: adapt the graph encoder, then query the table model','Follow which component is allowed to learn at each stage.',[
S('Pretrain graph representations','graph','Graph views → PCA → graph encoder','Structural pretraining produces node representations. Graph-level tasks additionally need the declared pooling operation.'),
S('Align graph outputs with a TFM','graph-table','Graph embeddings + community pseudo-labels → episodes','Graph embeddings become table rows; communities provide pseudo-labels for alignment episodes. The alignment stage can update both encoder and tabular model.'),
S('Adapt one side of the interface','graph-adapt','Target support adapts graph encoder; TFM stays frozen','The target-domain prototype loss updates the graph encoder. Recompute embeddings, then perform in-context prediction with the frozen tabular predictor.')],
'Can you reuse old graph embeddings after adapting the graph encoder?','Not as outputs of the newly adapted encoder. Its changed parameters invalidate those cached representations.','https://arxiv.org/html/2607.11374v1#S2','Paper mechanism overview. GTAlign training and evaluation remain NOT_RUN in this lesson.')
EXTRA_STORIES={'b17':['b17-gtalign']}
