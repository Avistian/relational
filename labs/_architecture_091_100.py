"""Model-specific whole-computation maps using the course's editable SVG renderer."""
import _architecture_071_090 as art
art.SPECS.clear()
spec=art.spec
spec('091-rgcn','R-GCN · preserve the role of each incoming message','Entity classification: relation branches meet before the nonlinearity.',[
('input',0,0,'Typed graph + identity','N nodes; forward / inverse edges','input'),
('r1',1,0,'Relation r₁','S₁ H W₁ · per-relation mean','model'),
('r2',1,1,'Other relation branches','One Sᵣ H Wᵣ per other role','model'),
('self',1,2,'Self contribution','H W_self · exactly once','model'),
('basis',0,2,'Optional shared bases','Wᵣ = Σᵦ aᵣᵦ Vᵦ, per layer','model'),
('h',2,0,'Sum → ReLU','First layer: N × 16','model'),
('z',3,0,'Second typed layer','N × 4 class logits; no ReLU','model'),
('y',2,2,'Training IDs + labels','Only 140 labels supervise AIFB','input'),
('loss',3,1,'Masked cross-entropy','Update both layer banks','loss'),
('out',2,1,'Frozen model → argmax','36 test entities; same graph','output')],
[('input','r1'),('input','r2'),('input','self'),('self','h'),('basis','self','weights'),('basis','r1','weights'),('basis','r2','weights'),('r1','h'),('r2','h'),('h','z'),('z','loss'),('y','loss'),('z','out')],
'AIFB release lane: unrestricted weights. Basis sharing is a separate extension. The identity input is implemented without allocating I.')
spec('092-han','HAN · choose routes, attend within them, then fuse','Two independent path encoders; one shared semantic scoring function.',[
('x',0,0,'Paper keywords','X [3025,1870]','input'),
('pap',1,0,'PAP neighborhood','8 independent GAT heads × 8','model'),
('psp',1,1,'PSP neighborhood','8 other GAT heads × 8','model'),
('stack',2,0,'Stack route embeddings','Z [3025,2,64]','model'),
('sem',2,1,'Semantic scorer','Shared W,b,q → scores [N,2]','model'),
('fuse',3,1,'Path softmax + fusion','Σₚ βᵢₚ Zᵢₚ → [N,64]','model'),
('head',3,2,'64 → 3 classifier','Masked CE + L2 during fitting','loss'),
('probe',2,2,'Frozen test embeddings','Release: KNN probe fractions','output'),
('routes',0,2,'Designer supplies routes','Binary endpoints; counts lost','input')],
[('x','pap'),('x','psp'),('routes','pap'),('routes','psp'),('pap','stack'),('psp','stack'),('stack','sem'),('sem','fuse'),('stack','fuse'),('fuse','head'),('fuse','probe')],
'Release β is node-specific. Paper Eqs. 7–9 average scores across nodes before path softmax. Keep these as explicitly different modes.')
spec('093-hgt','HGT · compatibility and content take separate routes','One typed edge feeds the score branch and the message branch.',[
('target',0,0,'Receiver state zₜ','Target type selects Q','input'),
('old',3,0,'Previous receiver state','Residual path; width d','input'),
('source',0,1,'Sender state + RTE','Source type selects K and V','input'),
('score',1,0,'Typed compatibility','Qₜ · (Kₛ Aᵣ); prior / √D','model'),
('value',1,1,'Typed message','Vₛ Mᵣ · [E,H,D]','model'),
('soft',2,0,'Receiver/head softmax','All incoming relations compete','model'),
('sum',2,1,'Weighted sum of values','Concatenate heads → [N,d]','model'),
('update',3,1,'Typed output + residual','GELU, Aₜ, gate, LayerNorm','model'),
('head',3,2,'Repeat → seed task head','Field ranking; train KL loss','loss'),
('sample',0,2,'Sample eligible context','Graph budget; remove label links','input')],
[('target','old','residual'),('old','update'),('target','score'),('source','score'),('source','value'),('score','soft'),('soft','sum'),('value','sum'),('sum','update'),('update','head'),('sample','source')],
'Parameters share by node type and relation/head. RTE changes source K/V inputs; time availability is a separate data-access contract.')
spec('094-taxonomy','A method map · ask four independent questions','Survey synthesis: a route specification and an encoder are separate choices.',[
('schema',0,0,'Schema and prediction','Types, roles, target, legal access','input'),
('route',1,0,'Which structure?','One-hop / typed path / pattern','input'),
('learn',2,0,'What is learned?','Counts / ID vectors / encoder','model'),
('task',3,0,'What is predicted?','Entity label / edge / ranked item','output'),
('protocol',0,2,'Comparison contract','Same graph, task, split, metric?','input'),
('audit',1,2,'Graph-count audit','Forward, reverse, hierarchy','model'),
('evidence',2,2,'Evidence scope','Statistics ≠ trained quality','output'),
('decision',3,2,'Testable next question','Choose one controlled contrast','output')],
[('schema','route'),('route','learn'),('learn','task'),('protocol','audit'),('audit','evidence'),('evidence','decision'),('task','decision')],
'This is a workflow, not a new architecture or an accuracy ranking. R-GCN, HAN and HGT occupy several independent taxonomy axes.')
spec('094-metapath2vec','metapath2vec · from typed walks to stored vectors','Conceptual survey map; no new implementation or benchmark run in L094.',[
('g',0,0,'Typed graph + template','Author → paper → author','input'),
('walk',1,0,'Typed random walks','Next node must match path type','model'),
('context',2,0,'Context windows','Center ID and context IDs','input'),
('tables',2,1,'Embedding tables','Lookup vectors for known IDs','model'),
('objective',3,0,'Skip-gram objective','Predict context; sample negatives','loss'),
('frozen',2,2,'Learned node vectors','Freeze or fit downstream probe','output'),
('task',3,2,'Classification / clustering','No automatic new-ID embedding','output')],
[('g','walk'),('walk','context'),('context','objective'),('tables','objective'),('tables','frozen'),('frozen','task')],
'The typed walk constrains observed contexts. The enhanced ++ objective also conditions normalization on context type; do not conflate them.')
spec('095-walk','Bipartite recommendation · return to an item','Deterministic scorer: preserve the intermediate item and user identities.',[
('fit',0,0,'Fitting ratings only','Likes → B [U,I]; seen mask','input'),
('pui',1,0,'User → item','P_UI = Dᵤ⁻¹ B','model'),
('piu',2,0,'Item → user','P_IU = Dᵢ⁻¹ Bᵀ','model'),
('back',3,0,'User → item again','S = P_UI P_IU P_UI','model'),
('pop',1,2,'Fitting popularity','Item counts / all likes','input'),
('mix',2,2,'Validation chooses α','α S + (1−α) popularity','model'),
('rank',3,2,'Mask seen → rank items','Recall@10 / NDCG@10','output')],
[('fit','pui'),('pui','piu'),('piu','back'),('fit','pop'),('pop','mix'),('back','mix'),('mix','rank')],
'No neural weights or gradient objective. Degree-zero users use the declared fallback; official offline splits do not establish temporal validity.')
spec('096-database','Database → graph · preserve rows before learning','Foreign-key roles define edges; composite keys preserve distinct events.',[
('tables',0,0,'Customers and products','Separate maps for each type','input'),
('orders',0,1,'Orders / line-item tables','Tuple keys, quantity, nullable FK','input'),
('ids',1,0,'Typed row identity','Table + PK → local tensor row','model'),
('edges',1,1,'Resolve each FK role','Child → parent; reverse view','model'),
('graph',2,0,'Typed graph + audit keys','Keep isolates and line rows','output'),
('paths',2,1,'Line → order → buyer','Also line → product; sum units','model'),
('sql',0,2,'Independent SQL query','Same rows, JOIN + GROUP BY','input'),
('check',3,2,'Compare paths and sums','Existence 1 / count 2 / units 5','output'),
('learn',3,1,'Later: fit an encoder','Choose target, cutoff and split','output')],
[('tables','ids'),('orders','ids'),('orders','edges'),('ids','edges'),('ids','graph'),('edges','graph'),('graph','paths'),('paths','check'),('sql','check'),('graph','learn')],
'Construction is not training. A valid foreign key establishes referential integrity; it does not establish prediction-time availability.')
spec('097-bpr','Pairwise ranking · one scorer, two item branches','Matrix factorization with sampled comparisons; no graph message passing.',[
('pos',0,0,'Observed positive (u,i)','Fitting like; known user/item IDs','input'),
('neg',0,1,'Draw eligible j from q','Uniform / degree / hard pool','input'),
('sp',1,0,'Positive score s(u,i)','Uᵤ · Vᵢ + bᵢ','model'),
('sn',1,1,'Negative score s(u,j)','Same U, V and b tables','model'),
('delta',2,0,'Score difference','Δ = s(u,i) − s(u,j)','model'),
('loss',3,0,'softplus(−Δ) + penalty','Backprop into selected vectors','loss'),
('all',1,2,'Inference: all candidates','Frozen U,V,b; no pairwise loss','model'),
('mask',2,2,'Exclude fitting-rated items','Same catalog across trained arms','input'),
('rank',3,2,'Rank → Recall / NDCG','Sampled catalog is a diagnostic','output')],
[('pos','sp'),('pos','sn'),('neg','sn'),('sp','delta'),('sn','delta'),('delta','loss'),('sn','all','shared'),('all','mask'),('mask','rank')],
'q determines which ranking errors train the scorer. Missing items are unobserved, not certified dislikes; full-paper BPR replay is NOT_RUN.')
spec('098-batch','Typed mini-batching · expand out, compute inward','Query identity includes entity and cutoff; batch rows have type-local IDs.',[
('query',0,0,'B seed queries','Entity IDs + prediction times','input'),
('hop1',1,0,'First dependency hop','Eligible incoming typed edges','input'),
('hop2',2,0,'Second dependency hop','Enough context for two layers','input'),
('local',3,0,'Local feature/edge stores','n_id, e_id and query components','input'),
('layer1',3,1,'Shared first layer','Source features → support states','model'),
('layer2',2,1,'Shared second layer','Support states → seed states','model'),
('head',1,1,'Seed logits [B]','Gather requested customer rows','output'),
('loss',0,1,'Seed-only BCE','Context labels are not targets','loss'),
('oracle',1,2,'Full-neighbor audit','Fixed weights → dense oracle','output')],
[('query','hop1'),('hop1','hop2'),('hop2','local'),('local','layer1'),('layer1','layer2'),('layer2','head'),('head','loss'),('head','oracle')],
'All-neighbor parity needs complete dependencies and compatible normalization. Finite fanout changes computation; a temporal filter cannot repair leaked features.')
spec('099-comparison','R-GCN versus HGT · compare complete procedures','One static ACM contract; two mechanism controls make attribution narrower.',[
('graph',0,0,'Same X, edges and split','Conference edges excluded','input'),
('rgcn',1,0,'R-GCN · two layers','Relation means + learned self','model'),
('hgt',1,1,'HGT · two layers','Scored messages + gated update','model'),
('uniform',1,2,'Uniform HGT / MLP','No scorer / no graph context','model'),
('val',2,0,'Validation selects','Epoch and one LR per arm','loss'),
('frozen',3,0,'Frozen test predictions','Accuracy and macro F1','output'),
('contrast',3,1,'Interpret the contrasts','Within-HGT scorer intervention','output'),
('cost',2,2,'Report resource differences','Active parameters, time, memory','input')],
[('graph','rgcn'),('graph','hgt'),('graph','uniform'),('rgcn','val'),('hgt','val'),('uniform','val'),('val','frozen'),('frozen','contrast'),('cost','contrast')],
'Each arm trains with masked CE. Equal width and epochs do not equalize capacity or cost; one graph cannot establish a universal family ranking.')
spec('100-checkpoint','Checkpoint · make every boundary inspectable','Same typed graph → paired batches → trained candidates → frozen evaluation.',[
('data',0,0,'ACM + frozen split','804 / 403 / 2818 paper labels','input'),
('sample',1,0,'Native typed sampler','Fanout 8 × 2 hops; seeds first','input'),
('models',2,0,'Four prediction routes','R-GCN / HGT / uniform / MLP','model'),
('loss',3,0,'Seed-only CE → Adam','Update after each training batch','loss'),
('identity',0,2,'Identity / gradient audit','Typed IDs; full-neighbor parity','input'),
('val',1,2,'Full-graph validation','Select epoch and learning rate','model'),
('test',2,2,'Frozen full-graph test','Same target IDs; report metrics','output'),
('defend',3,2,'Written research defense','Mechanism, protocol, uncertainty','output')],
[('data','sample'),('sample','models'),('models','loss'),('models','val'),('identity','val'),('val','test'),('test','defend')],
'Audit gradients at fixed weights. Training steps between batches follow a different trajectory from one full-batch update; mastery requires a defense.')
_shared_draw=art.draw
def draw(key):
    # Construction and deterministic scorers also use this visual vocabulary.
    return _shared_draw(key).replace('Learned computation','Operator / computation')
art.draw=draw
if __name__=='__main__':art.build()
