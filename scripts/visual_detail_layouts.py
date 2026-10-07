"""Purpose-drawn narrow layouts and complete replacement model architectures.

Panels are independent diagrams with explicit carried state, not cropped slices of
wide SVGs. The same model panels compose desktop exports and mobile reading order.
"""
from html import escape
import textwrap

T='#087f83';P='#7656a6';G='#ab650a';R='#b54e55';I='#233c50';S='#657b8b'
PRIMARY={'b09','b18a','b19b'}
class Scene:
    def __init__(self):self.parts=[]
    def text(self,x,y,s,size=16,color=I,anchor='start'):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(str(s))}</text>')
    def rect(self,x,y,w,h,fill='#f1f6f8',stroke='#bbced8',rx=5):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>')
    def line(self,x,y,a,b,color=T,dash=False,arrow=True):
        self.parts.append(f'<path d="M{x} {y}L{a} {b}" fill="none" stroke="{color}" stroke-width="2"'+(' stroke-dasharray="5 4"' if dash else '')+(' marker-end="url(#tip)"' if arrow else '')+'/>')
    def node(self,x,y,s,color=T,r=22):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="white" stroke="{color}" stroke-width="2"/>');self.text(x,y+5,s,16,color,'middle')
    def box(self,x,y,w,s,color=T,h=34):
        self.rect(x,y,w,h,'#f5f9fb',color);self.text(x+w/2,y+h/2+5,s,16,color,'middle')
    def table(self,x,y,rows,w=48,h=30,colors=None):
        for i,row in enumerate(rows):
            for j,v in enumerate(row):
                color=colors.get((i,j),T) if colors else T
                self.rect(x+j*w,y+i*h,w-4,h-4,'#fff6df' if v=='?' else '#e5f2ef' if color==T else '#eee8f7',color)
                if v:self.text(x+j*w+(w-4)/2,y+i*h+(h-4)/2+5,v,16,color,'middle')
    def graph(self,coords,edges,labels,color=T,arrow=True):
        for a,b in edges:
            x,y=coords[a];u,v=coords[b];dx=u-x;dy=v-y;l=(dx*dx+dy*dy)**.5
            ra=max(25,len(labels[a])*4.2+5);rb=max(26,len(labels[b])*4.2+6)
            self.line(x+dx/l*ra,y+dy/l*ra,u-dx/l*rb,v-dy/l*rb,color,arrow=arrow)
        for (x,y),s in zip(coords,labels):self.node(x,y,s,G if s in ['q','seed'] else color,r=max(22,len(s)*4.2+2))

def memory_read(c,label,tokens):
    c.text(150,22,label,17,P,'middle')
    start=(300-44*len(tokens))/2
    c.table(start,36,[tokens],w=44,colors={(0,j):P for j in range(len(tokens))})
    c.line(174,73,174,93,P)
    c.box(98,99,188,'weights × values',P)
    c.box(15,160,70,'hᵢƒ',T)
    c.line(51,156,51,116,T,arrow=False);c.line(51,116,91,116,T)
    c.text(63,104,'Q',16,T)
    c.line(177,139,177,165,P)
    c.line(91,177,148,177,T);c.node(177,190,'+',G)
    c.line(177,216,177,230,P)
    c.text(60,247,'updated cell state',17,I)

# Each panel includes a visual operation and a short, nearby interpretation.
def cast_panel(i,c):
    if i==0:
        c.text(68,24,'features',16,S,'middle');c.text(149,24,'target',16,S,'middle')
        c.table(18,42,[['x','x','y'],['x','x','y'],['q','q','?']],w=53)
        c.text(187,70,'support',16,T);c.text(187,124,'query',16,G)
        c.line(93,142,93,171);c.table(28,178,[['','','','','','']],w=24,h=27)
        c.text(186,198,'d = 192',16,S)
    elif i==1:
        memory_read(c,'column f summaries',['Sƒ','Sƒ','…','Sƒ'])
    elif i==2:
        c.text(150,22,'one row i',17,S,'middle')
        c.table(20,43,[['h₁','h₂','h₃']],w=48)
        c.table(165,43,[['Sᵢ','Sᵢ','Sᵢ']],w=41,colors={(0,j):P for j in range(3)})
        c.line(34,100,275,100,T);c.line(275,118,34,118,T)
        c.text(150,151,'all six slots mix together',17,T,'middle')
        c.table(20,175,[['h′₁','h′₂','h′₃']],w=48);c.table(165,175,[['S′ᵢ','S′ᵢ','S′ᵢ']],w=41,colors={(0,j):P for j in range(3)})
    elif i==3:
        memory_read(c,'row i summaries',['Sᵢ','Sᵢ','Sᵢ'])
    elif i==4:
        c.text(150,22,'one feature f',17,S,'middle')
        c.table(23,42,[['h₁ƒ'],['h₂ƒ']],w=61);c.table(23,111,[['Sƒ'],['…']],w=61,colors={(i,0):P for i in range(2)})
        c.line(94,59,94,162,T);c.line(109,162,109,59,T)
        for y in [58,88,127,157]:c.line(120,y,210,115,P)
        c.node(243,116,'q',G);c.text(243,178,'query reads',16,G,'middle')
        c.text(20,213,'Support + summaries self-update.',16,T)
    else:
        c.table(18,25,[['q′','q′','q′']],w=48);c.line(171,37,213,37)
        c.box(215,21,65,'head',G)
        c.line(249,60,249,91,G)
        for j,h in enumerate([10,18,26,35,44,55,65]):c.rect(20+j*28,147-h,17,h,'#e9ddf7',P)
        c.text(25,176,'999 ordered quantile levels',16,P)
        c.text(20,203,'wrapper: inverse transforms',16,S)
        c.text(20,228,'+ aggregate ensemble views',16,S)

def taco_panel(i,c):
    if i==0:
        c.text(20,21,'support',17,T);c.text(181,21,'dummy rows',17,G)
        c.table(20,42,[['a','b','y']]*4,w=28)
        c.table(182,80,[['a','b','?']]*2,w=28)
        c.line(115,55,176,85,G,dash=True)
        c.text(20,193,'N observed rows',16,T);c.text(182,193,'K slots',16,G)
    elif i==1:
        c.text(15,22,'within each row',17,T);c.table(30,46,[['a','b','y']],w=70)
        c.line(46,92,220,92,T);c.line(220,108,46,108,T)
        c.text(15,150,'across rows, for each feature',16,P)
        for x in [49,119,189]:c.line(x,166,x,199,P)
        c.table(30,203,[['z','z','z']],w=70,colors={(0,j):P for j in range(3)})
    elif i==2:
        c.text(150,22,'retain only the K dummy states',16,P,'middle')
        c.table(24,45,[['z₁₁','z₁₂','z₁y'],['z₂₁','z₂₂','z₂y']],w=82,h=41,colors={(i,j):P for i in range(2) for j in range(3)})
        c.text(150,164,'K × (M+1) × L',22,P,'middle')
        c.text(150,201,'latent vectors, not copied rows',16,S,'middle')
    elif i==3:
        c.box(85,24,130,'latent context',P)
        c.line(150,64,150,86,P);c.box(80,92,140,'two-layer MLP',P)
        c.line(150,132,150,157,P);c.node(150,187,'+',G)
        c.line(83,42,35,42,P,arrow=False);c.line(35,42,35,186,P,arrow=False);c.line(35,186,121,186,P)
        c.text(215,192,'to f',17,T)
    elif i==4:
        c.text(150,21,'predictor input',17,T,'middle')
        c.table(51,36,[['z','z','z'],['z','z','z'],['q','q','?']],w=64,colors={(i,j):P for i in range(2) for j in range(3)})
        c.line(60,150,229,150,T);c.line(240,54,240,113,T)
        c.text(150,184,'feature + row attention',17,T,'middle');c.text(150,214,'12 layers · width 192',16,S,'middle')
    else:
        c.box(51,21,199,'query target state',G);c.line(150,61,150,80,G);c.box(95,87,110,'task head',T)
        for j,h in enumerate([23,83,48]):c.rect(79+j*52,211-h,28,h,'#d7eeee',T)
        c.text(150,234,'class probabilities',16,T,'middle')

def forecast_panel(i,c):
    if i==0:
        pts=[(20,125),(50,99),(80,117),(110,80),(140,98),(170,63)]
        for p,q in zip(pts,pts[1:]):c.line(*p,*q,T,arrow=False)
        c.line(192,35,192,152,G,dash=True,arrow=False)
        c.text(80,28,'history',17,T,'middle');c.text(243,28,'future',17,G,'middle')
        for x in [219,251,283]:c.text(x,100,'?',22,G,'middle')
        c.text(150,190,'fit periods from history only',17,I,'middle')
        c.text(150,220,'calendar + known covariates',16,S,'middle')
    elif i==1:
        for j,label in enumerate(['time','sin','cos','target']):c.text(50+j*69,23,label,16,S,'middle')
        c.table(18,39,[['0','0','1','y₀'],['3','1','0','y₃'],['12','0','1','?']],w=69,h=41)
        c.text(150,190,'period 12: same map at all times',16,T,'middle')
        c.text(150,219,'future time ≠ future answer',17,G,'middle')
    elif i==2:
        c.text(150,22,'feature groups + target slot',17,T,'middle')
        c.table(36,44,[['g₁','g₂','y'],['g₁','g₂','y'],['g₁','g₂','?']],w=76,h=43)
        c.line(62,195,256,195,T);c.line(266,58,266,153,P)
        c.text(150,224,'(C+H) × (G+1) × d',18,S,'middle')
    elif i==3:
        c.table(24,40,[['s₁'],['s₂'],['s₃']],w=62,h=40)
        for y in [57,97,137]:c.line(100,y,226,116,P)
        c.node(255,115,'q',G)
        c.text(150,22,'frozen TabPFN-v2',18,T,'middle')
        c.text(150,193,'feature attention → row attention',16,T,'middle')
        c.text(150,221,'→ feed-forward block; repeat',16,T,'middle')
    elif i==4:
        c.text(150,23,'query target representation',17,G,'middle')
        c.line(150,35,150,69,G);c.box(58,74,184,'decoder → softmax',T)
        for j,h in enumerate([12,32,61,76,45,18]):c.rect(47+j*36,202-h,25,h,'#e3d8f2',P)
        c.text(150,226,'probability over target bins',16,P,'middle')
    else:
        c.text(150,23,'one marginal per horizon',17,P,'middle')
        for x,y,label in [(55,119,'h=1'),(150,94,'h=2'),(245,130,'h=3')]:
            c.line(x,y-37,x,y+37,P,arrow=False);c.line(x-13,y-37,x+13,y-37,P,arrow=False);c.line(x-13,y+37,x+13,y+37,P,arrow=False);c.node(x,y,'•',G,r=9);c.text(x,187,label,16,S,'middle')
        c.text(150,225,'No ŷ₁ → ŷ₂ feedback.',18,G,'middle')

MODEL_PANELS={
'b09':[
('Encode each cell','Support targets are observed; query targets are hidden. Missingness also enters the encoder.'),
('Read the column memory','Each cell cross-attends to its 32 feature summaries and adds the result to its own state.'),
('Mix across features','Append 3 item summaries per row. Self-attend over cells plus summaries; repeat twice per layer.'),
('Read the row memory','Each cell reads its 3 item summaries through another residual cross-attention path.'),
('Mix across support rows','Support cells and feature summaries update together. Queries read them without changing them.'),
('Repeat, then predict','Repeat the CAST layer 12 times. The regression head emits 999 quantiles; the wrapper returns the prediction.')],
'b18a':[
('Seed K masked dummy rows','Embed support and dummy cells. The release initializes dummy features from the last K preprocessed support rows.'),
('Compress through attention','Feature mixing and support-conditioned row attention update the dummy cells. The compressor has 12 layers.'),
('Keep latent cells','Discard support-side outputs. Retain K dummy representations, each carrying an L-dimensional cell state.'),
('Connect the two modules','A residual MLP maps latent context into the predictor space. Its skip path preserves a direct route.'),
('Attach the query rows','Concatenate compressed context with embedded queries. Hidden query targets have their own slots.'),
('Predict; learn jointly','The predictor head reads query states. During pretraining, the query loss updates both modules; serving uses frozen weights.')],
'b19b':[
('Freeze the forecast origin','Only available history fits seasonality and preprocessing. Future timestamps are known; their targets are not.'),
('Build a regression table','Historical and future rows share the same feature map. This period-12 example shows only two phase columns.'),
('Embed cells and groups','Group features and append a target slot. Context target slots are observed; future target slots are masked.'),
('Read context in the predictor','Feature attention mixes slots within a row. Row attention brings historical evidence to query cells; feed-forward blocks update states.'),
('Decode a distribution','Read each query target state. Bin probabilities define a discretized predictive distribution over target values.'),
('Return the full horizon','Extract medians and quantiles for every future row. No predicted target is fed into the next input.')],
}
MODEL_DRAW={'b09':cast_panel,'b18a':taco_panel,'b19b':forecast_panel}

def simple_panels(key):
    """Compact, hand-composed operation traces for the other seventeen lessons."""
    specs={
    '85':[('Start with distinct nodes','The equal-degree, self-looped two-node graph has S entries equal to 1/2.'),('Apply the actual matrix','Both outputs equal 5. This exact example does not describe every unequal-degree graph.')],
    '86':[('Gather the senders','Edge columns are 0→1 and 2→1. Gathering x at the source positions gives 2 and 8.'),('Sum at the destination','Both messages enter node 1. Nodes 0 and 2 remain in the output with zero incoming sum.')],
    '98':[('Expand dependencies','Start at the seed. Find one-hop dependencies, then the nodes they require.'),('Compute back inward','Outer states feed the first hop, which feeds the seed. Only seed predictions enter the batch loss.')],
    '104':[('Move the cutoff with the edge','A root at 8 follows an edge at 5. The next neighborhood lookup therefore uses cutoff 5.'),('Reject a later child event','Event 2 is allowed; event 6 is too late for that lookup, despite preceding root time 8.')],
    '109':[('Compare the two clocks','The event happened at 2; the system first observed it at 7. The query was issued at 5.'),('Reconstruct what was known','Event time passes the cutoff. Observation time fails it, so the fact is absent from the query input.')],
    '122':[('Read role-bearing keys','Transfer 7 stores sender key 10 and receiver key 90. Keys are identifiers.'),('Resolve tensor positions','Person key 10 is local row 1; key 90 is local row 0. Preserve sender and receiver as distinct relations.')],
    '123':[('Keep the root clock','This REG query owns cutoff 8 for every sampled hop. The cutoff does not become 4 at Transfer.'),('Check availability separately','Memo event 7 can inform this query. A record arriving at 11 is excluded even if its event is earlier.')],
    '133':[('Sum within each relation','Root feature 4 enters both relation modules. Each module also transforms its neighbor sum.'),('Sum complete relation outputs','Combine 19 and −6.5 to obtain 12.5. An empty relation can retain its bias and root contribution.')],
    '142':[('One bridge path','Identity-weight propagation carries source value x to the destination once.'),('Two bridge paths','Two distinct paths contribute 2x in this linear example. One unweighted collapsed edge would lose multiplicity.')],
    '146':[('Count propagation steps','Original circuit features traverse three links to reach the driver in a synchronous root-readout GNN.'),('Check the admitted tokens','One all-pairs block can connect driver and circuit only if the sampler included both.')],
    '171':[('Follow the ancestry links','An original, its identical copy, and two family-linked derivatives form one connected component.'),('Quarantine its relatives','Keep the original in evaluation and quarantine its three relatives. Unconnected sources remain training candidates.')],
    '172':[('Keep three states distinct','An observed numeric zero, a missing number and an unknown category must retain different meanings.')],
    '173':[('Pool separate context routes','Pool same-row and dated FK-parent embeddings separately. Add a query-column embedding; each vector has width 32.'),('Encode, then choose a head','Concatenate to width 96, project through 64 and 32, then use the appropriate one of 21 local task heads.')],
    '174':[('Follow the residual branch','A frozen encoder still computes h. The trainable branch projects 32→8, applies ReLU, then projects 8→32.'),('Add back the original state','With U=0 and c=0 at initialization, the branch is zero and h′=h. The adapter and original task heads train.')],
    '181':[('Hide the query answer','Historical targets may remain available under this task contract. The query target slot is hidden.'),('Exclude future rows too','Masking a target does not make an ineligible future row legal. Apply the temporal row filter separately.')],
    '186':[('Read the four timestamps','Illustrative times in milliseconds. Source event, cache refresh, request and response are separate clocks.'),('Measure the right interval','Source age at response is 2120 ms; incorporation lag is 2000 ms; request latency is 20 ms.')],
    'b12':[('Adapt through weights','Training labels enter a loss; the optimizer changes θ to θ′. Later queries use the changed weights.'),('Adapt through context','Support labels enter the predictor as data. Changing that support can change predictions with θ frozen.')],
    }
    return specs[key]

def simple_draw(key,i,c):
    if key=='85':
        if i==0:
            c.graph([(65,100),(235,100)],[(0,1)],['2','8'],arrow=False);c.text(150,171,'difference = 6',20,P,'middle')
        else:
            c.table(32,36,[['½','½'],['½','½']],w=52,h=43);c.text(161,95,'×',24,I);c.table(190,36,[['2'],['8']],w=48,h=43);c.text(150,176,'= [5, 5]',25,T,'middle')
    elif key=='86':
        if i==0:
            c.table(35,37,[['0','2'],['1','1']],w=62,h=46);c.text(179,66,'source',17,T);c.text(179,112,'dest.',17,P);c.text(150,191,'gather → [2, 8]',22,T,'middle')
        else:
            c.graph([(57,49),(243,49),(150,147)],[(0,2),(1,2)],['2','8','10']);c.text(150,214,'output = [0, 10, 0]',20,G,'middle')
    elif key=='98':
        points=[(150,36),(75,115),(225,115),(30,204),(110,204),(190,204),(270,204)];edges=[(0,1),(0,2),(1,3),(1,4),(2,5),(2,6)]
        c.graph(points,edges if i==0 else [(b,a) for a,b in edges],['seed','a','b','1','2','3','4'],P if i==0 else T)
    elif key in ['104','123']:
        root=key=='123'
        if i==0:
            c.node(65,55,'q',G);c.node(235,55,'4' if root else '5');c.line(93,55,205,55)
            c.box(20,127,111,'cutoff 8',G);c.box(169,127,111,'cutoff 8' if root else 'cutoff 5',G)
            c.text(150,211,'root-owned' if root else 'edge-time recursion',20,G,'middle')
        else:
            c.box(20,36,260,'event 7 ≤ 8' if root else 'event 2 < 5',T)
            c.box(20,110,260,'arrival 11 > 8' if root else 'event 6 > 5',R)
            c.text(150,209,'Exclude the red path.',19,R,'middle')
    elif key=='109':
        if i==0:
            for y,name,t in [(47,'event',2),(132,'observed',7)]:
                c.text(18,y,name,17,S);c.line(20,y+26,277,y+26,S);c.node(20+t*28,y+26,str(t),T if t==2 else R,r=16)
            c.line(160,23,160,190,G,dash=True,arrow=False);c.text(160,219,'query = 5',18,G,'middle')
        else:
            c.box(25,36,250,'2 ≤ 5: event-time pass',T);c.box(25,110,250,'7 > 5: not observed yet',R);c.text(150,208,'fact excluded',23,R,'middle')
    elif key=='122':
        if i==0:
            c.text(150,30,'Transfer 7',22,I,'middle');c.box(25,67,250,'sender key = 10');c.box(25,134,250,'receiver key = 90',P)
        else:
            c.box(10,30,112,'key 10');c.box(180,30,112,'local 1');c.line(128,47,173,47)
            c.box(10,107,112,'key 90',P);c.box(180,107,112,'local 0',P);c.line(128,124,173,124,P)
            c.text(150,207,'IDs ≠ array positions',22,I,'middle')
    elif key=='133':
        if i==0:
            c.text(15,30,'orders: Σ neighbors = 1+2',18,T);c.box(10,51,280,'2×3 + 1 + 3×4 = 19')
            c.text(15,138,'tickets: Σ neighbors = 5',18,P);c.box(10,158,280,'0.5×5 − 1 − 2×4 = −6.5',P)
        else:
            c.node(65,50,'19');c.node(230,50,'−6.5',P,r=32);c.line(65,80,129,134);c.line(230,87,173,134,P);c.node(151,159,'+',G);c.text(150,224,'12.5',29,G,'middle')
    elif key=='142':
        if i==0:c.graph([(40,100),(150,100),(260,100)],[(0,1),(1,2)],['x','b','x'])
        else:c.graph([(45,115),(150,50),(150,181),(257,115)],[(0,1),(0,2),(1,3),(2,3)],['x','b₁','b₂','2x'])
    elif key=='146':
        if i==0:
            for n,(name,depth) in enumerate([('circuit',3),('race',2),('result',1),('driver',0)]):
                y=12+n*59;c.box(35,y,129,name);c.text(190,y+24,str(depth)+' hops',17,S)
                if n<3:c.line(99,y+36,99,y+55)
        else:
            c.rect(15,23,270,126,'#f5f0fa',P);c.graph([(62,87),(237,87)],[(1,0)],['driver','circuit'],P);c.text(150,191,'inside the same sample',18,P,'middle');c.text(150,224,'absent token → no evidence',17,R,'middle')
    elif key=='171':
        if i==0:
            c.graph([(70,46),(227,46),(227,191),(70,191)],[(0,1),(1,2),(2,3)],['orig.','copy','A','B'],G,arrow=False);c.text(150,111,'follow every link',17,G,'middle')
        else:
            c.box(20,22,260,'original → evaluation',G);c.box(20,86,260,'copy, A, B → quarantine',R);c.box(20,157,260,'unrelated C, D → candidates')
    elif key=='172':
        for n,(v,state,color) in enumerate([('0','observed',T),('—','missing',G),('new','unknown',P)]):
            y=20+n*76;c.box(12,y,72,v,color);c.line(91,y+17,144,y+17,color);c.box(151,y,137,state,color)
    elif key=='173':
        if i==0:
            for n,label in enumerate(['row cells → mean: 32','FK cells → mean: 32','column embedding: 32']):c.box(20,18+n*73,260,label,T if n<2 else P)
        else:
            c.box(37,16,226,'concat: 96');c.line(150,56,150,78);c.box(37,84,226,'MLP: 96 → 64 → 32',P);c.line(150,124,150,148,P);c.box(20,156,260,'21 local task heads',G)
    elif key=='174':
        if i==0:
            for y,w,s in [(12,246,'frozen encoder → h:32'),(78,200,'D: 32 → 8 + ReLU'),(144,246,'U: 8 → 32')]:
                c.box((300-w)/2,y,w,s,S if y==12 else T)
                if y<144:c.line(150,y+39,150,y+59,T)
        else:
            c.box(18,20,118,'h',S);c.box(167,20,115,'branch: 0',T)
            c.line(77,60,129,117,S);c.line(224,60,172,117,T);c.node(150,140,'+',G)
            c.line(150,168,150,186);c.text(150,221,'h′ = h → task heads',20,G,'middle')
    elif key=='181':
        c.text(15,22,'row',16,S);c.text(147,22,'features',16,S,'middle');c.text(227,22,'target',16,S,'middle')
        for n,(label,target) in enumerate([('history','y'),('query','?'),('future','×' if i else '?')]):
            y=45+54*n;c.text(15,y+23,label,17,S);c.table(109,y,[['x',target]],w=80,h=42,colors={(0,j):R if n==2 and i else T for j in range(2)})
        c.text(150,226,'future row removed' if i else 'answer slot hidden',19,R if i else G,'middle')
    elif key=='186':
        if i==0:
            for n,(s,t) in enumerate([('source event','0'),('cache refresh','2000'),('request','2100'),('response','2120')]):
                y=14+n*55;c.text(20,y+22,s,18,S);c.box(184,y,99,t,G if n>1 else T)
        else:
            for n,s in enumerate(['2120 − 0 = 2120 ms','2000 − 0 = 2000 ms','2120 − 2100 = 20 ms']):c.box(10,21+n*74,280,s,P if n<2 else G)
    elif key=='b12':
        labels=['examples + labels','prediction → loss','optimizer: θ → θ′'] if i==0 else ['support labels + query','predict with frozen θ','new support → new answer']
        for n,s in enumerate(labels):
            c.box(10,18+n*75,280,s,T if i==0 else P)
            if n<2:c.line(150,58+n*75,150,84+n*75,T if i==0 else P)

def render_board(key,title,desc,mobile=True,columns=None):
    panels=MODEL_PANELS[key] if key in PRIMARY else simple_panels(key)
    cols=columns or (1 if mobile else 3);w=cols*340-20;rows=(len(panels)+cols-1)//cols;h=rows*390+20
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc><defs><marker id="tip" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="context-stroke"/></marker></defs><rect width="{w}" height="{h}" fill="#fff"/><g font-family="system-ui,sans-serif">']
    for i,(heading,caption) in enumerate(panels):
        x=10+(i%cols)*340;y=10+(i//cols)*390
        c=Scene();c.rect(0,0,300,364,'#fbfcfd','#d2e0e5',9)
        c.text(14,28,f'{i+1:02d}',17,G)
        for n,line in enumerate(textwrap.wrap(heading,27)):c.text(49,28+n*21,line,17,I)
        d=Scene()
        if key in PRIMARY:MODEL_DRAW[key](i,d)
        else:simple_draw(key,i,d)
        c.parts.append('<g transform="translate(0 60)">'+''.join(d.parts)+'</g>')
        # Caption follows the drawing; the panel expands to its wrapped height.
        for n,line in enumerate(textwrap.wrap(caption,31)):
            c.text(12,332+n*20,line,16,S)
        out.append((x,y,c,caption))
    # Variable-height rows fit captions at readable size, on either layout.
    result=out[:1];result[0]=result[0].replace(f'height="{h}"', 'height="HEIGHT"').replace(f'0 0 {w} {h}',f'0 0 {w} HEIGHT')
    cursor=10
    for row in range(rows):
        group=out[1+row*cols:1+(row+1)*cols]
        heights=[350+20*len(textwrap.wrap(p[3],31)) for p in group];rh=max(heights)
        for (x,_,c,_),ph in zip(group,heights):
            c.parts[0]=c.parts[0].replace('height="364"',f'height="{rh-20}"')
            result.append(f'<g transform="translate({x} {cursor})">'+''.join(c.parts)+'</g>')
        cursor+=rh+18
    result.append('</g></svg>')
    return ''.join(result).replace('HEIGHT',str(cursor+5))
