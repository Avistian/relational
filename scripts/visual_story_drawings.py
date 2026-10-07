"""Small editable SVG scenes: data structures and operations, not prose boxes."""
from html import escape
from math import sin, pi
TEAL='#087f8c'; AMBER='#b46519'; PURPLE='#7660ad'; INK='#233c50'; MUTED='#597181'; LINE='#c4d5dc'
class Drawing:
    def __init__(self, prefix): self.parts=[]; self.prefix=prefix
    def add(self,s): self.parts.append(s)
    def text(self,x,y,s,size=14,color=INK,anchor='middle',bold=False):
        self.add(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="system-ui,sans-serif" font-size="{size}" font-weight="{650 if bold else 400}" fill="{color}">{escape(str(s))}</text>')
    def rect(self,x,y,w,h,fill='#e7f3f4',stroke=LINE,rx=6): self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>')
    def circle(self,x,y,r=13,fill='#e7f3f4',stroke=TEAL): self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    def line(self,x,y,x2,y2,color=LINE,width=2,dash=False): self.add(f'<path d="M{x},{y} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="{width}"'+(' stroke-dasharray="5 4"' if dash else '')+'/>')
    def arrow(self,x,y,x2,y2,color=TEAL,dash=False):
        self.line(x,y,x2,y2,color,2,dash)
        import math
        a=math.atan2(y2-y,x2-x);points=[(x2,y2),(x2-8*math.cos(a-.5),y2-8*math.sin(a-.5)),(x2-8*math.cos(a+.5),y2-8*math.sin(a+.5))]
        self.add('<polygon points="'+' '.join(f'{x:.1f},{y:.1f}' for x,y in points)+f'" fill="{color}"/>')
    def chip(self,x,y,w,label,color=TEAL):
        self.rect(x,y,w,30,'#fff',color);self.text(x+w/2,y+20,label,13,color,bold=True)
    def table(self,x=35,y=35,rows=4,cols=4,cw=42,ch=27,mask=None,labels=False):
        for r in range(rows):
            for c in range(cols):
                hidden=(mask==(r,c))
                fill='#fff0de' if c==cols-1 and labels else ['#d8eef0','#edf5f6','#cee5e9'][(r+c)%3]
                self.rect(x+c*cw,y+r*ch,cw-3,ch-3,'#fff' if hidden else fill,AMBER if hidden else '#fff',3)
                if hidden:self.text(x+c*cw+(cw-3)/2,y+r*ch+18,'?',14,AMBER,bold=True)
    def vector(self,x,y,label='',color=TEAL,n=4):
        for i in range(n):self.rect(x+i*18,y,14,24,['#c9e7e8','#8bc5ca','#459eaa','#087f8c'][i%4],color,3)
        if label:self.text(x+n*9,y-8,label,13,color)
    def graph(self,mode='graph',x=0,y=0):
        pts=[(150,106),(65,48),(60,157),(239,48),(242,157)]
        edges=[(1,0),(2,0),(3,0),(4,0)]
        for j,(a,b) in enumerate(edges):
            ax,ay=pts[a];bx,by=pts[b];col=[TEAL,PURPLE,TEAL,PURPLE][j] if mode in ('relations','hgt') else TEAL
            self.line(x+ax,y+ay,x+bx,y+by,col,[2,6,3,9][j] if mode=='weighted-graph' else 2)
        for i,(px,py) in enumerate(pts):
            self.circle(x+px,y+py,19 if i==0 else 14,'#fff0de' if i==0 else '#e7f3f4',AMBER if i==0 else TEAL)
            self.text(x+px,y+py+5,('q' if mode in ('target-graph','root-mark') else 'v') if i==0 else str(i),13)
        if mode=='self-loop':
            self.add(f'<path d="M{x+140},{y+89} C{x+107},{y+28} {x+200},{y+32} {x+162},{y+89}" fill="none" stroke="{AMBER}" stroke-width="3"/>')
            self.text(x+150,y+25,'self contribution',13,AMBER)
        if mode=='root-mark':
            self.circle(x+150,y+106,28,'none',AMBER);self.chip(95+x,183+y,110,'root flag = 1',AMBER)
        if mode=='weighted-graph':self.text(x+150,y+200,'edge width ∝ attention',13,MUTED)
        if mode=='relations':
            self.text(x+102,y+62,'r₁',13,TEAL);self.text(x+203,y+149,'r₂',13,PURPLE)
        if mode=='sample':
            self.line(x+239,y+48,x+282,y+20,LINE,2,True);self.circle(x+282,y+20,7,'#fff',LINE)
            self.text(x+150,y+204,'sampled computation',13,MUTED)
    def scene(self,kind):
        if kind in ('graph','self-loop','weighted-graph','relations','target-graph','root-mark','sample','new-node'):
            self.graph('graph' if kind=='new-node' else kind)
            if kind=='new-node':self.text(150,207,'a new node, the same update rule',13,MUTED)
        elif kind in ('grid','typed','masked-cell','cell-memory'):
            if kind=='typed':
                for i,l in enumerate(['123','category','text']):self.text(65+i*80,27,l,13,[TEAL,AMBER,PURPLE][i])
                self.table(27,45,4,3,83,32);self.text(150,196,'Different types; aligned columns',13,MUTED)
            elif kind=='cell-memory':
                for i,l in enumerate(['age','type','value']):self.chip(22+88*i,30,80,l,PURPLE)
                for i in range(3):self.vector(25+88*i,100,n=4)
                self.text(150,175,'Retain a key and value per cell',14,MUTED)
            else:
                self.table(40,25,5,5,45,30,(4,4) if kind=='masked-cell' else (4,4),True)
                self.text(150,200,'rows × columns; target hidden',13,MUTED)
        elif kind in ('context','rel-context','row-attention'):
            self.text(83,23,'support',13,TEAL);self.text(233,23,'query',13,AMBER)
            self.table(20,40,4,3,38,28,labels=True);self.table(186,76,1,3,33,30,(0,2),True)
            for y in [53,81,109,137]: self.arrow(143,y,179,91,TEAL)
            self.text(150,192,'observed y → predict unknown y',14,MUTED)
            if kind=='rel-context':
                for x in [27,80,132]:self.circle(x,167,7)
                self.line(34,167,73,167,TEAL);self.line(87,167,125,167,TEAL)
        elif kind=='retrieval':
            for x,y in [(45,65),(69,116),(130,58),(138,140),(237,45),(230,145),(196,92)]:self.circle(x,y,7)
            self.circle(109,102,13,'#fff0de',AMBER);self.text(109,107,'q',12,AMBER)
            self.add(f'<ellipse cx="108" cy="103" rx="64" ry="56" fill="none" stroke="{TEAL}" stroke-dasharray="5 4"/>')
            self.line(115,94,130,64,AMBER);self.line(98,108,73,115,AMBER);self.line(117,113,135,135,AMBER)
            self.text(150,194,'near in learned feature space',14,MUTED)
        elif kind=='values':
            for i,l in enumerate(['E(y₁)','E(y₂)','E(y₃)']):
                y=20+i*48;self.chip(12,y,60,l);self.text(88,y+20,'+',20,MUTED);self.chip(104,y,102,f'T(k − k{i+1})',PURPLE);self.arrow(214,y+15,246,98)
            self.circle(263,98,17,'#fff0de',AMBER);self.text(263,104,'Σ',19,AMBER)
            self.text(150,198,'weighted learned values',14,MUTED)
        elif kind in ('residual','aggregate','concat','gin'):
            self.vector(20,40,'self',AMBER);self.vector(188,40,'neighbors')
            self.arrow(224,72,224,110);self.chip(174,114,100,'Σ' if kind!='concat' else 'aggregate')
            self.line(60,72,60,168,AMBER);self.arrow(60,168,78,168,AMBER);self.arrow(224,149,178,170)
            self.chip(80,154,98,'concat' if kind=='concat' else '+',PURPLE);self.arrow(129,184,129,205,PURPLE)
            if kind=='gin':self.text(51,108,'× (1+ε)',13,AMBER);self.text(183,210,'→ MLP',14,PURPLE)
            else:self.text(213,209,'→ update / head',13,MUTED)
        elif kind in ('mlp','layers'):
            for i,n in enumerate([3,5,5,2]):
                x=30+i*80;ys=[110+(j-(n-1)/2)*29 for j in range(n)]
                if i:
                    for py in prev:
                        for ny in ys:self.line(x-72,py,x-9,ny,'#d4e2e8',1)
                for yy in ys:self.circle(x,yy,9,'#eee9fa' if i==3 else '#def0f0',PURPLE if i==3 else TEAL)
                prev=ys
            self.text(150,207,'shared learned maps between layers',13,MUTED)
        elif kind in ('fork','ensemble','mean','heads'):
            self.vector(5,94,n=3)
            for i,col in enumerate([TEAL,PURPLE,AMBER]):
                y=26+i*61;self.arrow(64,106,102,y+14,col);self.chip(106,y,91,('r · W · s' if kind=='ensemble' else f'head {i+1}' if kind=='heads' else f'member {i+1}'),col);self.arrow(203,y+15,251,108,col)
            self.circle(269,108,18,'#fff',PURPLE);self.text(269,113,'μ' if kind in ('mean','ensemble','fork') else '∥',19,PURPLE)
            self.text(150,214,'shared W' if kind=='ensemble' else 'combine the parallel predictions',14,MUTED)
        elif kind=='prior':
            self.chip(47,12,206,'sample one mechanism',PURPLE);self.arrow(150,51,150,86,PURPLE);self.table(68,98,3,4,43,25,(2,3),True);self.text(150,212,'generate observations in that world',13,MUTED)
        elif kind=='rel-prior':
            for i,(x,y) in enumerate([(13,35),(115,105),(215,35)]):
                self.text(x+32,y-10,['table A','table B','table C'][i],12,TEAL);self.table(x,y,3,2,31,25)
            self.arrow(58,111,120,137);self.arrow(181,137,241,112,PURPLE);self.text(150,215,'sample tables and foreign-key structure',12,MUTED)
        elif kind in ('mixture','episodes'):

            for i,col in enumerate([TEAL,PURPLE,AMBER]):
                x=17+i*94;self.table(x,30+(i%2)*22,3,3,25,24,(2,2),True)
                self.arrow(x+32,121,x+32,150,col)
            self.rect(58,163,184,35,'#f0edf8',PURPLE);self.text(150,186,'sample a generator' if kind=='mixture' else 'shared learner',15,PURPLE,bold=True)
            if kind in ('rel-prior','mixture'):
                self.line(52,15,142,15,TEAL);self.line(142,15,235,15,PURPLE)
                for x in [52,142,235]:self.circle(x,15,7)
        elif kind=='distribution':
            self.line(28,168,276,168,MUTED);self.line(28,168,28,40,MUTED)
            for i,h in enumerate([26,75,114,61,22]):self.rect(46+i*45,168-h,29,h,['#aacfd5','#66acb6',PURPLE,'#66acb6','#aacfd5'][i],'none',4)
            self.text(150,205,'predictive output',14,MUTED)
        elif kind in ('scale','schedule','waves'):
            self.line(25,169,278,169,MUTED);self.line(25,169,25,27,MUTED)
            for k in range(3 if kind=='waves' else 1):
                pts=[]
                for i in range(100):
                    x=27+i*2.48
                    if kind=='waves':y=100-40*sin((i/100)*(k+1)*2*pi)
                    elif kind=='schedule':y=40+115*(1-(1+__import__('math').cos(pi*i/100))/2)
                    else:y=151-110/(1+__import__('math').exp(-(i-50)/14))
                    pts.append(f'{x:.1f},{y:.1f}')
                self.add(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{[TEAL,PURPLE,AMBER][k]}" stroke-width="3"/>')
            self.text(150,204,{'scale':'smoothly limit extreme coordinates','schedule':'training progress →','waves':'elapsed time → periodic features'}[kind],13,MUTED)
        elif kind in ('axial','inducing','row-pool','tokenize','subsets'):
            self.table(52,26,4,5,39,29)
            if kind=='axial':
                self.arrow(53,163,243,163,TEAL);self.arrow(267,139,267,30,PURPLE)
                self.text(150,200,'feature mixing ↔ row mixing',14,MUTED)
            elif kind=='inducing':
                for x in [75,150,225]:
                    self.circle(x,177,10,'#eee9fa',PURPLE);self.arrow(x,137,x,161,PURPLE)
                self.text(150,213,'compact inducing-point set',13,MUTED)
            elif kind=='subsets':
                self.rect(49,23,78,117,'none',AMBER);self.rect(168,23,78,117,'none',PURPLE)
                self.text(150,196,'overlapping feature views',14,MUTED)
            else:
                for y in [40,69,98,127]:self.arrow(247,y,273,y);self.rect(278,y-8,15,16,'#7660ad','none',3)
                self.text(150,199,'one representation per row',14,MUTED)
        elif kind=='corruption':
            self.text(72,28,'clean',14,TEAL);self.text(226,28,'view',14,AMBER)
            self.table(14,50,4,2,37,28);self.table(199,50,4,2,37,28)
            for r,c in [(1,0),(3,1)]:self.rect(199+c*37,50+r*28,34,25,'#edc79f',AMBER,3)
            self.arrow(103,106,184,106);self.text(150,197,'same row; some values replaced',14,MUTED)
        elif kind in ('two-heads','dual-head','transfer','adapt','hyper','dual-score'):
            labels={'two-heads':('mask','values'),'dual-head':('class','number'),'transfer':('pretrain','task head'),'adapt':('frozen','fine-tune'),'hyper':('θ','predictor'),'dual-score':('local','global')}[kind]
            if kind in ('transfer','hyper'):
                self.chip(70,20,160,labels[0],TEAL);self.arrow(150,58,150,93,PURPLE);self.chip(70,103,160,labels[1],PURPLE);self.arrow(150,140,150,180);self.text(150,205,'query prediction',14,MUTED)
            else:
                self.vector(116,23,'shared input');self.arrow(145,57,74,100);self.arrow(157,57,228,100,PURPLE)
                self.chip(14,111,113,labels[0]);self.chip(173,111,113,labels[1],PURPLE)
                self.text(150,193,'two distinct routes',14,MUTED)
        elif kind=='contrast':
            for r in range(4):
                for c in range(4):self.rect(64+c*40,24+r*36,35,31,TEAL if r==c else '#e5eff2','none',3)
            self.text(150,194,'diagonal = matching row identity',13,MUTED)
        elif kind in ('star','edge-product'):
            self.graph('relations' if kind=='edge-product' else 'graph')
            self.text(150,203,'column names label the edges',13,MUTED)
            if kind=='edge-product':self.chip(87,5,130,'value ⊙ relation',PURPLE)
        elif kind=='hgt':
            for i,label in enumerate(['Q target','K source','V source']):
                y=22+i*53;self.chip(7,y,103,label,[AMBER,TEAL,PURPLE][i]);self.arrow(117,y+15,151,y+15);self.chip(160,y,131,'typed projection',[AMBER,TEAL,PURPLE][i])
            self.text(150,214,'relations transform scores and messages',12,MUTED)
        elif kind=='gcn-stack':
            self.chip(44,21,212,'S X W₀ → ReLU');self.arrow(150,60,150,99);self.chip(44,109,212,'S H W₁ → logits',PURPLE);self.text(150,200,'same graph S; different layer weights',13,MUTED)
        elif kind in ('gcn','relation-maps'):
            if kind=='gcn':
                for i,l in enumerate(['S','H','W']):
                    self.table(13+99*i,51,3,2 if i==2 else 3,25,25);self.text(43+99*i,35,l,18,[AMBER,TEAL,PURPLE][i],bold=True)
                self.text(150,180,'node mixing × states × feature map',13,MUTED)
            else:
                for i,l in enumerate(['relation 1','relation 2','self']):
                    y=24+i*57;self.chip(10,y,93,l,[TEAL,PURPLE,AMBER][i]);self.arrow(111,y+15,145,y+15);self.chip(153,y,111,['W₁ h','W₂ h','W₀ h'][i],[TEAL,PURPLE,AMBER][i])
                self.text(150,215,'typed attention + messages' if kind=='hgt' else 'combine the relation contributions',13,MUTED)
        elif kind in ('pair','labeled-graph','graph-pool','multiset'):
            if kind=='multiset':
                for i,n in enumerate([2,3]):
                    y=55+i*74
                    for j in range(n):self.circle(60+j*65,y,20);self.text(60+j*65,y+5,'a',17,TEAL)
                self.text(150,203,'same values, different multiplicity',13,MUTED)
            else:
                self.graph('graph')
                if kind=='pair':self.line(65,48,239,48,AMBER,3,True);self.text(150,28,'candidate edge',13,AMBER)
                if kind=='graph-pool':self.arrow(150,132,150,181,PURPLE);self.vector(114,186,n=4,color=PURPLE)
                if kind=='labeled-graph':self.text(150,207,'labels refer to the two endpoints',13,MUTED)
        elif kind in ('clusters','cluster-batch','database','bridge','composite','metapath','walk'):
            if kind in ('database','bridge','composite'):
                for i,(x,y) in enumerate([(15,27),(112,95),(210,27)]):self.table(x,y,3,2,34,23)
                self.arrow(66,93,119,121);self.arrow(185,122,233,94,PURPLE)
                self.text(150,206,'rows linked by foreign keys',14,MUTED)
                if kind=='composite':self.add(f'<path d="M47,14 C100,-2 202,-2 254,14" fill="none" stroke="{AMBER}" stroke-width="3"/>')
            elif kind in ('metapath','walk'):
                points=[(33,65),(105,122),(191,65),(265,122)]
                for a,b in zip(points,points[1:]):self.arrow(a[0]+8,a[1],b[0]-12,b[1])
                for i,(x,y) in enumerate(points):self.circle(x,y,19,'#eee9fa' if i%2 else '#e7f3f4',PURPLE if i%2 else TEAL);self.text(x,y+5,'P' if i%2 else 'A',15)
                self.text(150,199,'type-constrained route',14,MUTED)
            else:
                for k,(x,y) in enumerate([(57,65),(203,65),(130,155)]):
                    self.circle(x,y,41,'#f5f7fa',AMBER if kind=='cluster-batch' and k==1 else LINE)
                    for dx,dy in [(-17,10),(17,10),(0,-17)]:self.circle(x+dx,y+dy,6)
                    self.line(x-17,y+10,x,y-17,TEAL);self.line(x,y-17,x+17,y+10,TEAL)
                self.line(91,75,169,75,LINE,2,True);self.line(77,101,111,122,LINE,2,True)
                self.text(150,217,'sample groups, retain internal edges',12,MUTED)
        elif kind=='local-global':
            for i in range(3):
                y=38+i*49;self.vector(18,y,n=3);self.circle(240,y+12,13,'#eee9fa',PURPLE)
                for j in range(3):self.line(80,y+12,225,50+j*49,PURPLE,1)
            self.text(60,202,'local tokens',13,TEAL);self.text(236,202,'centroids',13,PURPLE)
        elif kind in ('semantic','embedding','rel-tokens','token-attention'):
            for i in range(4):
                y=18+i*39;self.vector(25,y+5,n=3);self.arrow(90,y+17,159,y+17,[TEAL,PURPLE,AMBER,TEAL][i]);self.vector(178,y+5,n=4)
                if kind in ('token-attention','local-global'):
                    if i<3:self.line(226,y+30,201,y+43,PURPLE)
            self.text(150,208,{'semantic':'mix meta-path summaries','embedding':'meaning → vector coordinates','rel-tokens':'content + type + hop + time + PE','token-attention':'tokens exchange information','local-global':'local tokens + global centroids'}[kind],12,MUTED)
        elif kind in ('timeline','snapshots','edge-memory'):
            self.arrow(15,121,285,121,MUTED)
            for i,x in enumerate([41,96,151,240]):
                self.circle(x,121,9,'#e7f3f4' if i<3 else '#fff0de',TEAL if i<3 else AMBER);self.text(x,155,['t₁','t₂','t₃','query'][i],14)
                if kind=='snapshots':self.table(x-17,39,2,2,17,22)
            self.line(204,31,204,178,AMBER,2,True);self.text(150,203,'history     |     prediction cutoff',13,MUTED)
            if kind=='edge-memory':self.chip(54,35,160,'seen endpoint pairs',TEAL)
        elif kind in ('memory','recurrent','weight-state','support-write','cache'):
            labels={'memory':('memory','embedding'),'recurrent':('state t','state t+1'),'weight-state':('Wₜ','Wₜ₊₁'),'support-write':('support','refined support'),'cache':('identity','cached state')}[kind]
            self.chip(6,73,110,labels[0]);self.arrow(124,88,169,88,PURPLE);self.chip(179,73,115,labels[1],PURPLE)
            self.add(f'<path d="M233,113 L233,158 L62,158 L62,109" fill="none" stroke="{AMBER}" stroke-width="2"/>')
            self.text(150,195,'state reused by later computation',13,MUTED)
        elif kind in ('temporal-attention','cell-graph','gnn-attention','gate'):
            if kind=='cell-graph':
                self.table(12,48,3,3,22,26)
                graph=Drawing(self.prefix+'-graph');graph.graph('relations')
                self.add('<g transform="translate(104 26) scale(.63)">'+''.join(graph.parts)+'</g>')
                self.arrow(88,90,148,90,PURPLE);self.arrow(189,158,55,158,AMBER);self.text(121,181,'reread using updated state',11,AMBER);self.text(150,214,'cell reads ↔ graph updates',13,MUTED)
            elif kind=='gate':
                self.chip(7,33,112,'GNN',TEAL);self.chip(180,33,112,'attention',PURPLE);self.arrow(63,72,134,133);self.arrow(237,72,164,133,PURPLE);self.circle(150,151,24,'#fff0de',AMBER);self.text(150,157,'g',21,AMBER);self.text(150,205,'g · GNN + (1−g) · attention',14,MUTED)
            elif kind=='gnn-attention':
                self.chip(50,15,200,'same retained tokens');self.arrow(150,52,62,89);self.arrow(150,52,234,89,PURPLE);self.chip(4,99,120,'GraphSAGE');self.chip(167,99,130,'time attention',PURPLE);self.text(150,194,'parallel, not serial',14,MUTED)
            else:
                self.graph('weighted-graph');self.text(150,24,'score + time information',13,PURPLE)
        elif kind in ('pairwise','difference','ranking','selection'):
            for i,l in enumerate(['candidate i','candidate j','candidate k']):
                y=24+i*53;self.text(82,y+18,l,13,INK);self.rect(149,y,115-i*32,24,[TEAL,PURPLE,AMBER][i],'none',3)
            self.text(150,212,{'pairwise':'u supplies the comparison context','difference':'compare two scores: sᵢ − sⱼ','ranking':'rank admitted candidates','selection':'select using allowed evidence'}[kind],13,MUTED)
        elif kind=='causal':
            for x,y,x2,y2 in [(45,45,143,93),(45,145,143,105),(162,97,255,145),(60,145,252,150)]:self.arrow(x,y,x2,y2)
            for x,y,l in [(40,40,'u'),(40,150,'x'),(155,98,'z'),(267,154,'y')]:self.circle(x,y,17);self.text(x,y+5,l,15)
            self.text(150,208,'sample a directed mechanism',14,MUTED)
        elif kind=='graph-table':
            self.graph('graph');self.parts=['<g transform="translate(0 15) scale(.52)">'+''.join(self.parts)+'</g>']
            self.arrow(161,76,188,76);self.table(198,35,4,3,30,26,labels=True);self.text(150,209,'node embeddings become table rows',13,MUTED)
        elif kind=='graph-adapt':
            self.chip(52,28,213,'graph encoder · update',TEAL);self.arrow(158,68,158,103);self.chip(52,113,213,'table model · frozen',PURPLE)
            self.arrow(26,134,26,43,AMBER);self.arrow(26,43,49,43,AMBER);self.text(150,196,'prototype loss updates the encoder',13,MUTED)
        elif kind=='flatten':
            self.table(9,27,3,2,25,24);self.table(25,115,3,2,25,24)
            self.arrow(75,67,128,97);self.arrow(85,143,128,111)
            self.chip(133,89,70,'DFS',PURPLE);self.arrow(207,104,231,104)
            self.table(237,65,3,2,26,27,labels=True)
            self.text(150,211,'relational records → flat features',13,MUTED)
        elif kind=='rt-routes':
            for i,label in enumerate(['column','feature','neighbor','full']):
                y=12+i*48;self.text(68,y+24,label,14,PURPLE if i%2 else TEAL)
                for r in range(3):
                    for c in range(6):
                        active=(c==r or c==r+3) if i==0 else (c<3) if i==1 else (c>=3) if i==2 else True
                        self.rect(131+c*23,y+r*12,18,9,TEAL if active else '#e5eef1','none',2)
            self.text(150,218,'separate masks; sequential sublayers',12,MUTED)
        elif kind in ('statistics','linear-read'):
            if kind=='statistics':
                for i in range(3):
                    y=25+i*53;self.vector(12,y,n=3);self.text(83,y+19,'⊗',20,PURPLE);self.vector(106,y,n=3,color=PURPLE);self.arrow(169,y+12,220,100)
                self.chip(224,87,60,'Σ',AMBER)
            else:
                self.vector(20,67,'φ(q)',AMBER,n=3);self.arrow(86,80,116,80);self.table(129,39,3,3,30,28);self.arrow(227,80,270,80);self.text(150,168,'divide by the support normalizer',13,MUTED)
            self.text(150,210,'fixed-size accumulated support state',12,MUTED)
        elif kind=='two-worlds':
            for i,label in enumerate(['world A','world B']):
                x=22+i*151;self.text(x+52,28,label,14);self.table(x,45,3,3,34,28,(2,2),True)
                self.chip(x,155,100,'y = '+str(i),AMBER if i else PURPLE)
            self.text(150,216,'same visible cells; different answer',12,MUTED)
        elif kind=='depth-sum':
            for i in range(3):
                y=21+i*53;self.chip(10,y,87,'layer '+str(i+1));self.arrow(104,y+15,133,y+15);self.chip(137,y,67,'P'+str(i+1),PURPLE);self.arrow(211,y+15,250,99)
            self.circle(267,100,17,'#fff0de',AMBER);self.text(267,106,'Σ',18,AMBER);self.text(150,212,'collect row representations across depth',12,MUTED)
        elif kind=='task-decoders':
            self.vector(116,28,'shared Z');self.arrow(142,60,71,98);self.arrow(161,60,226,98,PURPLE)
            self.chip(5,105,130,'decoder A');self.chip(165,105,130,'decoder B',PURPLE)
            self.arrow(70,181,70,143,AMBER);self.arrow(228,181,228,143,AMBER)
            self.text(70,204,'labels A',14,AMBER);self.text(228,204,'labels B',14,AMBER)
        else:raise ValueError('No authored scene for '+kind)
        return ''.join(self.parts)

def svg_scene(kind,prefix,title,description):
    body=Drawing(prefix).scene(kind)
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 225" width="300" height="225" role="img" aria-labelledby="{prefix}-title {prefix}-desc"><title id="{prefix}-title">{escape(title)}</title><desc id="{prefix}-desc">{escape(description)}</desc>{body}</svg>'
