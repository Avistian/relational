"""Deterministic, editable architecture drawings for the L071–L090 walkthroughs.

SVG is the web source; PNG is a portable notebook export. All coordinates and
connections are explicit: changing a model changes its topology, not only a label.
"""
from pathlib import Path
from html import escape
import textwrap
import cairosvg
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/architectures'
# node: id, column, row, title, detail, kind. Edges name their true dependencies.
SPECS = {}
def spec(key, title, subtitle, nodes, edges, footer):
    SPECS[key] = (title, subtitle, nodes, edges, footer)

spec('071-vime', 'VIME · learn what changed, then predict', 'Two pretext heads share one encoder; target labels enter a later stage.', [
('x',0,0,'Unlabeled training rows','Clean x [B,d] + column donors','input'),
('c',1,0,'Marginal replacement','x̃ = (1−m)x + m x̄','input'),
('f',2,0,'Encoder fθ','d → d · ReLU · h [B,d]','model'),
('mask',3,0,'Mask head','d logits → BCE(actual changes)','loss'),
('rec',3,1,'Value head','d sigmoid values → reconstruction','loss'),
('clean',0,2,'Clean target row','Same fitted input transform','input'),
('keep',1,2,'Retained encoder','Freeze fθ in released pipeline','model'),
('head',2,2,'New task head','Fit with train labels; optional consistency','model'),
('out',3,2,'Target prediction','Predict without pretext heads','output')],
[('x','c'),('c','f'),('f','mask'),('f','rec'),('f','keep','transfer'),('clean','keep'),('keep','head'),('head','out')],
'Pretraining updates encoder + two heads. Downstream consistency and supervised loss train the task head; inference uses clean rows.')
spec('072-scarf','SCARF · recognize the same row across views','Shared weights in both branches; the projector serves pretraining.',[
('x',0,0,'Clean view x','Training rows [B,d]','input'),('v',0,1,'Corrupted view x̃','Fixed-count marginal replacement','input'),
('f1',1,0,'Shared encoder fθ','h = fθ(x) [B,H]','model'),('f2',1,1,'Same encoder fθ','h̃ = fθ(x̃) [B,H]','model'),
('g1',2,0,'Shared projector gφ','Normalize gφ(h) → z','model'),('g2',2,1,'Same projector gφ','Normalize gφ(h̃) → z̃','model'),
('loss',3,0,'B × B similarities','Diagonal positives; N-way cross-entropy','loss'),
('pred',2,2,'Retain encoder fθ','Discard projector; attach task head','output'),('probe',3,2,'Choose transfer protocol','Paper: tune f + head; lab: freeze f','output')],
[('x','f1'),('v','f2'),('f1','g1'),('f2','g2'),('g1','loss'),('g2','loss'),('f2','pred','transfer'),('pred','probe')],
'B is batch size, H is encoder width. Different row identities supply negatives even when their downstream classes match.')
spec('072-subtab','SubTab · multiple column views, one row identity','Each subset has its own inputs but uses the same encoder, decoder and projector.',[
('s1',0,0,'Column subset 1','Select + training corruption','input'),('s2',0,1,'Column subset 2','Overlaps other views','input'),('s3',0,2,'Column subset 3','Same B row identities','input'),
('e1',1,0,'Shared encoder E','Latent h₁ [B,H]','model'),('e2',1,1,'Same encoder E','Latent h₂ [B,H]','model'),('e3',1,2,'Same encoder E','Latent h₃ [B,H]','model'),
('d',2,0,'Shared decoder D','Each h reconstructs the full row','model'),('g',2,1,'Shared projector G','Cross-view contrast + distance','model'),('mean',2,2,'Inference: mean of h','Average V views, never B rows','output'),
('l',3,0,'Joint local objective','Reconstruction + pair losses','loss'),('p',3,2,'Task predictor','Discard D and G; no training noise','output')],
[('s1','e1'),('s2','e2'),('s3','e3'),('e1','d'),('e2','d'),('e3','d'),('e1','g'),('e2','g'),('e3','g'),('d','l'),('g','l'),('e1','mean'),('e2','mean'),('e3','mean'),('mean','p')],
'Loss switches are part of the method: reconstruction-only is also an evaluated SubTab recipe. This overview shows the local joint path.')
spec('073-budget','Does SSL help? · isolate the initialization','Matched architectures, nested label sets and paired seeds make the subtraction meaningful.',[
('train',0,0,'Training features','Fit transforms; pretrain without y','input'),('ssl',1,0,'SSL initialization','SCARF pretraining → same encoder','model'),('random',1,1,'Scratch initialization','Same supervised architecture','model'),
('tune',2,0,'Matched supervised fits','Identical label subset per budget','model'),('val',2,1,'Validation labels','Choose within declared budgets','input'),('test',3,0,'Once-frozen test scores','Δseed(b) = SSL − scratch','output'),
('budget',0,2,'Label ledger','Training labels + validation labels','input'),('curve',3,2,'Gain versus budget','Paired gains; ties and brackets','output')],
[('train','ssl'),('train','random'),('ssl','tune'),('random','tune'),('val','tune','select'),('tune','test'),('test','curve'),('budget','curve')],
'This is an experimental design, not a new model. A sign change across tested budgets does not identify a universal threshold.')
spec('074-carte','CARTE · carry cell meaning across schemas','A row is a star graph. Column vectors condition messages before attention.',[
('bg',0,0,'YAGO graphlet views','Entity context + edge removal','input'),('pre',1,0,'Shared attention stack','Contrastive pretraining · width 300','model'),('weights',2,0,'Pretrained parameters','Selected encoder checkpoint','output'),
('row',0,1,'Target row → star','Values on leaves; names on edges','input'),('pair',1,1,'Node / edge input maps','zᵢⱼ = xⱼ ⊙ eᵢⱼ','model'),('att',2,1,'Relation-aware attention','Q(center); K,V(conditioned leaves)','model'),('read',3,1,'Center readout','FFN + normalization → [B,300]','model'),
('head',3,2,'Target head → prediction','Frozen ridge or local fine-tuning','output')],
[('bg','pre'),('pre','weights'),('weights','att','transfer'),('row','pair'),('pair','att'),('att','read'),('read','head')],
'The local downstream configuration uses one readout block, not the full pretraining stack. Schema flexibility does not prove transfer quality.')
spec('075-frame','PyTorch Frame · typed columns become row vectors','Materialization stores fitted state; learned encoders turn values into tokens.',[
('train',0,0,'Training DataFrame','Declare numerical / category / text','input'),('fit',1,0,'Fit materializer','Statistics + vocabulary from train','input'),('query',0,1,'Held-out DataFrame','Reuse fitted converter','input'),('tensor',1,1,'TensorFrame blocks','Columns grouped by semantic type','input'),
('num',2,0,'Numeric affine tokens','((x − μ) / s) wⱼ + bⱼ','model'),('cat',2,1,'Category / text tokens','Lookup or configured text encoder','model'),('tok',3,0,'Aligned column tokens','[B,C,d] · explicit column order','model'),('row',3,2,'Row readout → head','Flatten/project in this local stack','output')],
[('train','fit'),('fit','tensor'),('query','tensor'),('tensor','num'),('tensor','cat'),('num','tok'),('cat','tok'),('tok','row')],
'Type-specific tokenizers do not themselves connect records. L076 adds relationships after the row representation boundary.')
spec('076-rdl','A small relational stack · route before reducing','Two table encoders preserve different schemas and emit a common width D.',[
('cust',0,0,'Customer rows','IDs [42,7,99,105] + attributes','input'),('event',0,1,'Event rows','FK + event and availability clocks','input'),('ce',1,0,'Customer encoder','[4,2,4] tokens → [4,4]','model'),('ee',1,1,'Event encoder','[5,2,4] tokens → [5,4]','model'),
('eligible',2,1,'Eligible FK messages','Map keys to positions; mean by customer','model'),('join',2,0,'Concatenate self + mean','Aligned [4,4] + [4,4] → [4,8]','model'),('pred',3,0,'8 → 8 → 1 head','One logit for each customer','output'),('loss',3,2,'Training labels only','BCE backpropagates through both encoders','loss')],
[('cust','ce'),('event','ee'),('ee','eligible'),('ce','join'),('eligible','join'),('join','pred'),('pred','loss')],
'An original one-hop teaching model: event → customer only. At inference use the same fitted state and a time-valid neighborhood.')
spec('077-ceiling','An information ceiling · find the lost distinction','Two opposite histories can become the same vector before a predictor sees them.',[
('a',0,0,'History A','10 → 30 → 50 · target 1','input'),('b',0,1,'History B','50 → 30 → 10 · target 0','input'),('flat',1,0,'Order-free summary','count 3; sum 90; mean 30; max 50','model'),('same',2,0,'Identical representation','Any deterministic g gets same input','model'),('bound',3,0,'Balanced-pair ceiling','At most 1 correct out of 2','output'),
('repair',1,2,'Retain a time feature','last − first: +40 versus −40','model'),('clf',2,2,'Same classifier family','Now the inputs are distinguishable','model'),('audit',3,2,'Audit eligible history','Recovery depends on available records','output')],
[('a','flat'),('b','flat'),('flat','same'),('same','bound'),('a','repair'),('b','repair'),('repair','clf'),('clf','audit')],
'A constructive information argument, not a claim that every flat table has this ceiling or that any GNN automatically repairs it.')

def gcn(key,title,footer):
 spec(key,title,'Node-feature mixing and graph routing act on different axes.',[
 ('x',0,0,'Node features X','Cora [2708,1433] · row-normalized','input'),('a',0,1,'Eligible graph A','Add loops once; S = D⁻½(A+I)D⁻½','input'),('h',1,0,'First GCN layer','Dropout → S X W₀ → ReLU','model'),('z',2,0,'Second GCN layer','Dropout → S H W₁','model'),('p',3,0,'Class logits Z','[2708,7] → class softmax','output'),('loss',2,2,'Training objective','CE on 140 nodes + first-layer L2','loss'),('test',3,2,'Prediction','Dropout off; held-out labels score only','output')],
 [('x','h'),('a','h'),('a','z'),('h','z'),('z','p'),('z','loss'),('p','test')],footer)
gcn('078-preview','From joins to GCN · the complete prediction path','A preview of L082: shared weights transform features, graph support routes them. The graph is visible in the transductive Cora task.')
gcn('082-gcn','GCN · two layers, one shared graph support','W₀: 1433 × 16; W₁: 16 × 7. Weights differ across layers and are shared across nodes. Validation governs stopping, never test labels.')
spec('079-decision','Model choice · turn a regime into a falsifiable plan','A decision procedure comparing existing models, not a newly proposed architecture.',[
('task',0,0,'Prediction contract','Target · entity · time · metric','input'),('info',1,0,'Available information','Rows, labels, relations, shift, cost','input'),('base',2,0,'Strong baseline','Trees / simple neural / frozen PFN','model'),('challenge',2,1,'One justified challenger','Specify its expected advantage','model'),('valid',3,0,'Matched selection','Same split and explicit budget','model'),('test',3,2,'Frozen comparison','Quality + cost + falsifier','output')],
[('task','info'),('info','base'),('info','challenge'),('base','valid'),('challenge','valid'),('valid','test')],
'Carry L060 comparison discipline through L070 foundation-model evidence. A new dataset can overturn the recommendation.')
spec('080-exam','Year 2 exit · prove a decision is reproducible','Compare methods only after freezing their information and selection contracts.',[
('split',0,0,'Freeze split / budgets','Random and temporal regimes differ','input'),('fit',1,0,'Train-only transforms','Model-specific preprocessing','input'),('models',2,0,'Four declared arms','XGBoost · FT-T · TabM · TabPFN v2','model'),('val',2,1,'Validation selection','Hyperparameters / stopping / choice','model'),('test',3,0,'Final test predictions','Save row IDs + scores + provenance','output'),('repair',0,2,'Leakage intervention','Change held-out values; inspect fitted state','input'),('report',3,2,'Written defense','Paired evidence, limits, deployment rule','output')],
[('split','fit'),('fit','models'),('models','val'),('val','test'),('repair','fit','audit'),('test','report')],
'An assessment protocol, not a new model. Numerical checks support the submission; a written explanation remains required.')
spec('081-ggnn','Sparse molecular GG-NN · messages, gates, readout','An instantiated MPNN: bond types choose transforms and gates preserve atom state.',[
('atom',0,0,'Atom features','13 coordinates → zero-pad to 50','input'),('bond',0,1,'Directed bond entries','Two directions per chemical bond','input'),('msg',1,0,'Two matrix banks','Bond-specific transforms → two sums','model'),('gru',2,0,'Recurrent gated update','Message [N,100] + old state [N,50]','model'),('state',3,0,'Final atom state','Shared parameters across rounds','model'),('skip',1,2,'Original atom features','Retain alongside final state','input'),('read',2,2,'Gated atom contributions','[hT,x] → sigmoid gate × value','model'),('out',3,2,'Sum within molecule','One molecular prediction → task loss','output')],
[('atom','msg'),('bond','msg'),('msg','gru'),('gru','state'),('state','msg','repeat'),('atom','skip'),('skip','read'),('state','read'),('read','out')],
'The released gate equations are shown in the lesson; a stock GRU is not automatically equivalent. Readout never mixes separate molecules.')
spec('083-sage','GraphSAGE · expand dependencies, compute inward','The released mean variant keeps independent self and neighbor transform blocks.',[
('root',0,0,'B root nodes','Prediction targets','input'),('hop1',1,0,'10 neighbors per root','First-hop occurrences [B,10]','input'),('hop2',2,0,'25 per first-hop node','Second-hop occurrences [B,10,25]','input'),('inner',2,1,'Shared first layer','Compute root and first-hop states','model'),('agg',1,1,'Second mean layer','[self Wself || mean Wneighbor]','model'),('norm',0,2,'Final unit normalization','Embedding [B,256]','model'),('head',1,2,'256 → 121 head','Independent PPI label logits','output'),('loss',2,2,'Root-only BCE','Support nodes provide features','loss')],
[('root','hop1'),('hop1','hop2'),('hop2','inner'),('hop1','inner'),('inner','agg'),('root','inner'),('agg','norm'),('norm','head'),('head','loss')],
'Sampling travels outward; neural computation travels inward. Training uses only the training-induced graph in this inductive PPI track.')
spec('084-gat','GAT · learn a neighbor distribution for each head','Neighbor softmax precedes aggregation; class softmax follows the output layer.',[
('x',0,0,'Node states + edges','Include self once; score eligible edges','input'),('h1',1,0,'Head 1 · W₁, a₁','Project → score → receiver softmax','model'),('h8',1,1,'Heads 2–8 · own W,a','Independent learned neighbor weights','model'),('cat',2,0,'Concatenate + ELU','8 heads × 8 coordinates = 64','model'),('out',3,0,'Output attention head','64 → 7; one Cora output head','model'),('loss',3,2,'Masked task objective','Train CE + release L2; validation selects','loss'),('pred',2,2,'Inference: class softmax','Dropout off; each node has 7 logits','output')],
[('x','h1'),('x','h8'),('h1','cat'),('h8','cat'),('cat','out'),('out','loss'),('out','pred')],
'Weights are shared across edges inside a head, not between heads. Attention coefficients describe this forward pass, not causal importance.')
spec('085-depth','Repeated propagation · reach versus distinguishability','The linear limit is a diagnostic; the finite nonlinear network is a different object.',[
('h',0,0,'Initial states H₀','Different node signals','input'),('s1',1,0,'One mixing step','H₁ = S H₀','model'),('sk',2,0,'Repeated mixing','Hₖ = Sᵏ H₀','model'),('lim',3,0,'Connected-component limit','Degree-scaled stationary direction','output'),('net',0,2,'Actual Figure 2 network','Karate · identity features · untrained','input'),('layers',1,2,'Depth 1–5 GCN','ReLU hidden layers; linear final layer','model'),('plot',2,2,'Two output coordinates','Inspect geometry across depth / seeds','output'),('diag',3,2,'Separate trained extension','Accuracy + degree-adjusted collapse','output')],
[('h','s1'),('s1','sk'),('sk','lim'),('net','layers'),('layers','plot'),('plot','diag','compare')],
'No objective or fitting is used for the paper Figure 2 reconstruction. The trained Cora extension asks a separate empirical question.')
spec('086-pyg','PyG · follow one edge through the tensor program','Under source_to_target, edge_index[0] sends to edge_index[1].',[
('data',0,0,'Data container','x [N,F]; edge_index [2,E]','input'),('forward',1,0,'forward','Add loops, degrees, transform XW','model'),('msg',2,0,'message(x_j)','Gather E sender rows; weight messages','model'),('agg',3,0,'aggregate → update','Sum by receiver → [N,D]','model'),('batch',0,2,'NeighborLoader batch','Local IDs + n_id + seed count','input'),('route',1,2,'Same message program','Support nodes remain in computation','model'),('head',2,2,'Seed logits only','First batch_size rows supervise','output'),('loss',3,2,'Training loss','No supervision from context labels','loss')],
[('data','forward'),('forward','msg'),('msg','agg'),('batch','route'),('route','head'),('head','loss')],
'PyG provides routing, not an automatic split policy. Whole-graph batching and sampled-node batching have different identity contracts.')
spec('087-link','Link prediction · keep messages separate from targets','A candidate edge is an output question; its hidden existence must not enter messages.',[
('graph',0,0,'Observed training graph','Remove held-out positives both ways','input'),('enc',1,0,'Shared node GCN','Compute z for all eligible nodes','model'),('pair',2,0,'Gather pair endpoints','For each (u,v), obtain zᵤ and zᵥ','model'),('dot',3,0,'Dot-product decoder','s(u,v) = sum(zᵤ ⊙ zᵥ)','output'),('labels',1,2,'Supervision pairs','Observed positives + sampled negatives','input'),('loss',2,2,'BCE with logits','Train pair labels only','loss'),('rank',3,2,'Evaluation candidates','Scores → ranking under a fixed rule','output')],
[('graph','enc'),('enc','pair'),('pair','dot'),('labels','loss'),('dot','loss'),('dot','rank')],
'This local dot-product path is symmetric: s(u,v)=s(v,u). Candidate sampling and tie rules are part of the measured task.')
spec('087-seal','SEAL · classify a graph around each candidate link','A model overview only here; the full SEAL classifier has not been executed in this lesson.',[
('pair',0,0,'Candidate (u,v)','Observed graph + candidate endpoints','input'),('sub',1,0,'Enclosing subgraph','Extract h-hop context; remove target edge','input'),('drnl',2,0,'Root-relative labels','DRNL distances mark endpoint roles','model'),('gnn',3,0,'DGCNN node layers','Learn structural node representations','model'),('sort',2,2,'SortPooling','Order nodes; fixed-size representation','model'),('conv',1,2,'1D convolution + dense','Graph-level classifier','model'),('pred',0,2,'Link score','Train BCE; same pipeline at inference','output')],
[('pair','sub'),('sub','drnl'),('drnl','gnn'),('gnn','sort'),('sort','conv'),('conv','pred')],
'The local paper reconstruction covers CN / AA / RA baseline cells. Understanding this architecture is distinct from reproducing SEAL scores.')
spec('088-gin','GIN · count locally, read out at every depth','The MUTAG release adds graph-level class logits from the input and all hidden depths.',[
('x',0,0,'Atom-tag features H₀','One-hot node labels; graph membership','input'),('sum',1,0,'Neighbor sum + self','(1+ε)hᵥ + Σᵤ hᵤ · ε=0 for GIN-0','model'),('mlp',2,0,'MLP + normalization','Learn a nonlinear multiset map','model'),('next',3,0,'Next node states','Repeat with depth-specific weights','model'),('read0',0,2,'Input-depth readout','Sum nodes within each graph','model'),('readk',2,2,'Each-depth readout','Sum nodes → own class head','model'),('out',3,2,'Sum class logits','Training dropout per head; graph CE','output')],
[('x','sum'),('sum','mlp'),('mlp','next'),('next','sum','repeat'),('x','read0'),('next','readk'),('read0','out'),('readk','out')],
'At inference dropout is off and batch-normalization state is fixed. Parameters transfer across graphs; labels never enter the forward pass.')
spec('089-cluster','Cluster-GCN · a sampling plan plus a concrete model','The released PPI recipe has a full-training-graph first-layer cache.',[
('train',0,0,'Training-induced graph','Fit feature scaling on training nodes','input'),('cache',1,0,'Cache [Atrain X, X]','Raw adjacency; computed before partition','input'),('part',0,1,'Partition → choose q','Induce batch; recover internal edges','input'),('first',2,0,'First learned stage','100 → 2048; layer norm + ReLU','model'),('support',1,1,'Enhanced batch support','T=(D+I)⁻¹(A+I); S=T+diag(T)','model'),('later',2,1,'Later stages [SH,H]W','Four hidden stages total; width 2048','model'),('out',3,1,'Fifth stage → 121 logits','No final ReLU or layer norm','output'),('loss',3,2,'Independent-label BCE','At evaluation threshold logits > 0','loss')],
[('train','cache'),('train','part'),('cache','first'),('part','support'),('first','later'),('support','later'),('later','out'),('out','loss')],
'Batching affects messages and normalization. First-layer cache can cross training-cluster boundaries; held-out features stay excluded.')
spec('090-checkpoint','GNN checkpoint · identify which experiment you ran','GCN reconstruction and an inductive mini-batch extension answer different questions.',[
('cora',0,0,'Fixed transductive Cora','Full X,A visible; split masks for labels','input'),('gcn',1,0,'Two-layer GCN','S X W₀ → ReLU → S H W₁','model'),('loss',2,0,'140-node objective','CE + first-layer L2; validation stop','loss'),('eval',3,0,'100-run reconstruction','Fixed split; initialization variability','output'),('ind',0,2,'Inductive training graph','Remove held-out nodes and edges','input'),('sample',1,2,'Root/support mini-batch','Sample outward; compute inward','model'),('root',2,2,'Root-only supervision','Support provides eligible features','loss'),('report',3,2,'Separate extension report','Access, sampling and model differ','output')],
[('cora','gcn'),('gcn','loss'),('loss','eval'),('ind','sample'),('sample','root'),('root','report')],
'A comparable number needs an aligned protocol. Keep historical identity, measured port results, and extension evidence in separate claims.')

COLORS = {'input':('#f1eee7','#867a67'), 'model':('#e6f3f0','#14766c'), 'loss':('#fff0df','#b06122'), 'output':('#eeebfa','#7460a2')}
def draw(key):
    title, subtitle, nodes, edges, footer = SPECS[key]
    W,H=1120,650
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',f'<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle+" "+footer)}</desc>', '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#65847f"/></marker></defs>', '<rect width="1120" height="650" rx="18" fill="#fcfaf6"/>', '<rect x="0" y="0" width="1120" height="6" rx="3" fill="#14766c"/>']
    def text(x,y,s,size=17,color='#193a3a',weight='normal'):
        parts.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans,sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(s)}</text>')
    text(34,45,'RELATIONAL LEARNING  /  MODEL & INFORMATION FLOW',12,'#647b75','bold')
    text(34,83,title,min(27,1850/len(title)),'#193a3a','bold');text(34,114,subtitle,16,'#526b66')
    positions={n[0]:(34+n[1]*274,155+n[2]*139) for n in nodes}
    for edge in edges:
        a,b,*tag=edge;x,y=positions[a];xx,yy=positions[b]
        if xx>x: start=(x+230,y+49);end=(xx,yy+49)
        elif xx<x: start=(x,y+49);end=(xx+230,yy+49)
        else: start=(x+115,y+98 if yy>y else y);end=(xx+115,yy if yy>y else yy+98)
        sx,sy=start;ex,ey=end
        dash=' stroke-dasharray="6 5"' if tag else ''
        if (tag and tag[0]=='repeat') or (sy==ey and abs(xx-x)>274):
            gutter=min(y,yy)-15
            d=f'M {x+115} {y} L {x+115} {gutter} L {xx+115} {gutter} L {xx+115} {yy}'
        elif sy==ey or sx==ex: d=f'M {sx} {sy} L {ex} {ey}'
        else: d=f'M {sx} {sy} C {(sx+ex)/2} {sy}, {(sx+ex)/2} {ey}, {ex} {ey}'
        parts.append(f'<path d="{d}" fill="none" stroke="#65847f" stroke-width="2"{dash} marker-end="url(#arrow)"/>')
    for ident,col,row,heading,detail,kind in nodes:
        x,y=positions[ident];fill,line=COLORS[kind]
        parts.append(f'<rect x="{x}" y="{y}" width="230" height="98" rx="10" fill="{fill}" stroke="{line}" stroke-opacity=".45"/>')
        parts.append(f'<rect x="{x}" y="{y+13}" width="4" height="25" rx="2" fill="{line}"/>')
        hs=textwrap.wrap(heading,26);ds=textwrap.wrap(detail,28)
        for j,s in enumerate(hs):text(x+13,y+24+j*19,s,15,line,'bold')
        for j,s in enumerate(ds):text(x+13,y+28+len(hs)*19+j*18,s,14)
    parts.append('<line x1="34" y1="567" x2="1086" y2="567" stroke="#d5ded8"/>')
    for i,(kind,(fill,line)) in enumerate(COLORS.items()):
        x=34+i*220;parts.append(f'<circle cx="{x+6}" cy="590" r="5" fill="{line}"/>');text(x+19,595,{'input':'Data / fitted state','model':'Learned computation','loss':'Training objective','output':'Readout / prediction'}[kind],13,'#526b66')
    for i,s in enumerate(textwrap.wrap(footer,129)):text(34,620+i*18,s,13,'#526b66')
    parts.append('</svg>');return '\n'.join(parts)

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    for key in SPECS:
        svg=draw(key);(OUT/f'{key}.svg').write_text(svg)
        cairosvg.svg2png(bytestring=svg.encode(),write_to=str(OUT/f'{key}.png'),scale=1.5)
    print(f'Built {len(SPECS)} editable SVG diagrams and portable PNG exports')
if __name__=='__main__':build()
