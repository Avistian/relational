"""Editable, operation-level SVG supplements. Values follow the adjacent lessons.

Drawings are teaching schematics, never new benchmark evidence. Each has a
specific retrieval question, text alternative and link to the lesson source.
"""
from html import escape
import re

TEAL='#087f83'; PURPLE='#7656a6'; GOLD='#ab650a'; RED='#b54e55'; INK='#233c50'; GRAY='#657b8b'
DETAILS = {
 '98': ('Two passes through a sampled computation graph', 'Expand dependencies away from the seed; compute messages back toward it. Context nodes supply features, but only seed predictions enter this batch loss.', 'Which nodes contribute to the loss?', 'The seed nodes only. A sampled context node can affect their representations without supplying a supervised target to this batch.', 'sampling'),
 '104': ('A recursive cutoff travels with each edge', 'A root query at time 8 follows an edge at time 5. Its next lookup uses cutoff 5, so an event at time 6 is forbidden even though it precedes the root query.', 'Is event 6 legal below the edge at time 5?', 'No. This recursive temporal-neighborhood rule uses the traversed edge time for the next lookup.', 'recursive'),
 '109': ('One fact, two clocks', 'An event at time 2 observed at time 7 cannot appear in a prediction issued at time 5. Point-in-time reconstruction needs the observation clock as well as the event clock.', 'Can a later database snapshot reveal what the system knew at time 5?', 'Only if it preserves observation/version history and selects versions available by time 5. Filtering event time alone admits this late fact.', 'clocks'),
 '122': ('A foreign key selects a row, not an array position', 'Transfer 7 has sender key 10 and receiver key 90. Person key 10 maps to local row 1; key 90 maps to local row 0. The two edge types preserve these roles.', 'Where does sender key 10 point?', 'To Person local row 1. Treating database key 10 as array position 10 would corrupt the graph.', 'keys'),
 '123': ('A root-owned cutoff stays fixed across hops', 'For this REG contract, every sampled hop belongs to the query at time 8. A Memo at time 7 may inform a Transfer at time 4; a record arriving at time 11 remains excluded.', 'Does the second hop inherit cutoff 4?', 'No. This lesson uses the root query cutoff 8 throughout. Do not substitute the recursive edge-time rule from temporal-attention neighborhoods.', 'root'),
 '133': ('Two sums, with a root transform inside each relation', 'For root feature 4, orders contribute 2×(1+2)+1+3×4 = 19. Tickets contribute 0.5×5−1−2×4 = −6.5. The outer relation sum is 12.5.', 'If ticket edges disappear but the relation still executes, is its output zero?', 'No. The neighbor sum becomes zero, but the bias and root term remain: 0−1−2×4 = −9. Removing the relation key is a different operation.', 'relations'),
 '142': ('A bridge can repeat the same source evidence', 'In a linear path-counting illustration, one source-to-destination path carries a value once, while two bridge paths carry it twice. Collapsing the bridge can lose both multiplicity and role information.', 'Does replacing two bridge paths by one unweighted edge preserve the sum?', 'No. A source value x contributes 2x along two identity-weight paths, but only x along one unweighted collapsed edge. This is a linear illustration, not the full RelGNN operator.', 'bridge'),
 '172': ('Missing, zero and an unknown category are different inputs', 'Tokenization must preserve field identity, value type and missingness. A missing numeric field, an observed zero and an unseen categorical value are three different states.', 'Should an unseen category become the numeric value zero?', 'No. Use the fitted categorical vocabulary and its unknown-value policy. Keep numeric zero and missingness distinct according to the declared encoder contract.', 'tokens'),
 '174': ('A frozen encoder can still feed a trainable residual branch', 'The course adapter takes h with width 32 through D:32→8, ReLU and U:8→32, then adds h. The encoder is frozen; the adapter and original task heads learn.', 'Why does U=0 preserve the pretrained representation at initialization?', 'The residual branch initially outputs zero, so h′=h. D is initialized randomly; zeroing both projections can prevent the ReLU branch from learning.', 'adapter'),
 '181': ('Masking the answer and excluding the future solve different problems', 'An autocomplete query hides its own target cell. A separate temporal eligibility rule excludes future rows. Masking every historical value of the target column would change the context contract.', 'Does hiding the query target make future rows safe?', 'No. Target masking and row-time eligibility are independent gates. Apply both before constructing the model input.', 'mask'),
 '186': ('Fast response, stale evidence', 'An event can wait for a refresh before it reaches the cached representation. A quick response measures serving latency; it does not prove the evidence is fresh.', 'What is the source age when the response is returned?', '2120 ms: response time 2120 minus source event time 0. The 2000 ms incorporation lag and 20 ms request latency measure different intervals.', 'cache'),
 'b09': ('CAST exchanges summaries across the two table axes', 'Feature attention mixes columns within a row; item attention mixes support examples within a column. CAST carries 3 item-summary tokens per row and 32 feature-summary tokens per feature, with cross-axis reads before axis-wise attention.', 'What extra route do the summaries create?', 'A cell reads summaries from the opposite axis before axis-wise attention, allowing repeated exchange between row and column context. Summary counts and the 192-wide, 12-layer, 6-head configuration describe the pinned release.', 'cast'),
 'b19b': ('Forecast future rows without feeding predictions back', 'Fit time features using history only. Encode both historical and future timestamps with that map. Context rows have targets; future rows have hidden targets. The predictor returns one marginal distribution per horizon.', 'Is the prediction for horizon 1 inserted as the target for horizon 2?', 'No. This TabPFN-TS construction queries future rows without autoregressive target feedback. Marginal forecasts do not specify a joint distribution over whole future paths.', 'forecast'),
}

class Canvas:
    def __init__(self, title, desc):
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="620" viewBox="0 0 1000 620" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="context-stroke"/></marker></defs><rect width="1000" height="620" fill="#fff"/><style>text{{font-family:system-ui,sans-serif;fill:{INK}}}</style>']
        self.text(30,38,title,24)
        self.line(30,59,970,59,'#d5e3e8',arrow=False)
    def text(self,x,y,s,size=18,color=INK,anchor='start'):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" style="fill:{color}" text-anchor="{anchor}">{escape(str(s))}</text>')
    def rect(self,x,y,w,h,fill='#f0f6f8',stroke='#c4d5de',rx=9):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>')
    def line(self,x,y,a,b,color=TEAL,dash=False,arrow=True):
        self.parts.append(f'<path d="M{x} {y}L{a} {b}" stroke="{color}" stroke-width="3" fill="none"'+(' stroke-dasharray="7 5"' if dash else '')+(' marker-end="url(#arrow)"' if arrow else '')+'/>')
    def node(self,x,y,label,color=TEAL,r=27):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="white" stroke="{color}" stroke-width="3"/>')
        self.text(x,y+6,label,17,color,'middle')
    def box(self,x,y,w,label,color=TEAL,h=52):
        self.rect(x,y,w,h,'#f4f8fa',color)
        self.text(x+w/2,y+h/2+6,label,18,color,'middle')
    def footer(self,a,b):
        self.rect(30,520,940,76,'#f5f7fa')
        self.text(48,551,a,18)
        self.text(48,579,b,16,GRAY)
    def grid(self,x,y,rows,cols,w=42,h=32,high=None):
        for i in range(rows):
            for j in range(cols):
                fill='#d7eeee' if high and (i,j) in high else '#f3f5f9'
                self.rect(x+j*w,y+i*h,w-4,h-4,fill,'#aec3ce',3)
    def done(self):return ''.join(self.parts)+'</svg>'

def sampling(c):
    c.text(40,99,'1  Dependency expansion',20,PURPLE)
    c.text(525,99,'2  Message computation',20,TEAL)
    for offset,reverse in [(0,False),(485,True)]:
        coords=[(80+offset,275),(220+offset,190),(220+offset,360),(390+offset,145),(390+offset,235),(390+offset,320),(390+offset,410)]
        for a,b in [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6)]:
            p,q=coords[a],coords[b]
            if reverse:p,q=q,p
            dx=q[0]-p[0];dy=q[1]-p[1];length=(dx*dx+dy*dy)**.5
            c.line(p[0]+dx/length*29,p[1]+dy/length*29,q[0]-dx/length*31,q[1]-dy/length*31,TEAL if reverse else PURPLE)
        for i,(x,y) in enumerate(coords):c.node(x,y,'seed' if i==0 else str(i),GOLD if i==0 else TEAL)
        for x,label in [(80,'seed'),(220,'1 hop'),(390,'2 hops')]:c.text(x+offset,477,label,17,GRAY,'middle')
    c.footer('Expand from the requested output; evaluate from the outer dependencies inward.','Only the seed predictions enter this batch loss. Counts here are schematic.')

def recursive(c,root=False):
    c.text(50,106,'The clock belongs to the root query' if root else 'The clock follows the traversed edge',21)
    labels=['query','Transfer','Memo'] if root else ['root','child','event']
    for x,label in zip([120,440,760],labels):c.node(x,245,label,r=49)
    c.line(172,245,386,245);c.line(492,245,706,245)
    c.text(280,217,'edge / event 4' if root else 'edge time 5',18,TEAL,'middle')
    c.text(600,217,'event time 7' if root else 'event time 2',18,TEAL,'middle')
    for x,t in zip([120,440,760],[8,8,8] if root else [8,5,2]):c.box(x-77,330,154,f'cutoff = {t}',GOLD)
    c.node(760,425,'late' if root else '6',RED)
    c.line(492,268,729,413,RED,dash=True)
    c.text(475,455,'arrival 11 > root cutoff 8' if root else 'event 6 > recursive cutoff 5',18,RED)
    c.footer('Same root cutoff at every hop.' if root else 'A fact may precede the root and still violate a recursive lookup.', 'This REG policy differs from edge-time recursion.' if root else 'Compare L123: a root-owned REG uses a different, explicitly declared policy.')

def clocks(c):
    c.text(50,110,'Would this fact have been visible when the prediction was issued?',20)
    c.rect(565,145,350,255,'#fff1f1','#f0c8cb')
    c.text(690,180,'Unavailable at query time',18,RED,'middle')
    for y,label in [(230,'event time'),(350,'observed time')]:
        c.text(55,y+6,label,18);c.line(215,y,920,y,GRAY)
        for t in [2,5,7,9]:c.text(215+t*70,y+39,t,17,GRAY,'middle')
    c.line(565,135,565,414,GOLD,dash=True,arrow=False);c.text(565,452,'query issued at 5',18,GOLD,'middle')
    c.node(355,230,'2',TEAL);c.node(705,350,'7',RED)
    c.line(368,257,684,326,RED,dash=True);c.text(270,151,'Same fact',18)
    c.footer('event_time = 2 passes the event cutoff; observed_time = 7 fails availability.','Select the version that was available then, not the version visible in a later snapshot.')

def keys(c):
    c.text(50,110,'Transfer table',21);c.text(605,110,'Person table',21)
    for i,(a,b) in enumerate([('transfer key','7'),('sender key','10'),('receiver key','90')]):
        c.box(50,140+i*67,175,a,GRAY);c.box(235,140+i*67,95,b,TEAL if i==1 else PURPLE)
    for i,(a,b) in enumerate([('local row 0','key 90'),('local row 1','key 10'),('local row 2','key 300')]):
        c.box(605,140+i*95,155,a,GRAY);c.box(770,140+i*95,165,b,TEAL if i==1 else PURPLE)
    c.line(335,233,596,260,TEAL);c.text(455,215,'sender →',18,TEAL,'middle')
    c.line(335,300,596,169,PURPLE);c.text(465,343,'receiver →',18,PURPLE,'middle')
    c.text(55,441,'Edge identity includes source type, relation role and destination type.',20)
    c.footer('Primary keys are identifiers. Tensor indices are positions in this particular graph.','Person key 300 remains a node even when this transfer has no edge to it.')

def relations(c):
    c.node(95,160,'1');c.node(95,235,'2');c.node(95,380,'5',PURPLE)
    c.line(126,160,235,195);c.line(126,235,235,209)
    c.box(240,176,120,'sum = 3');c.line(365,202,428,202)
    c.box(434,176,340,'2×3 + 1 + 3×4 = 19')
    c.line(126,380,428,380,PURPLE);c.box(434,354,340,'0.5×5 − 1 − 2×4 = −6.5',PURPLE)
    c.box(434,83,340,'root feature = 4',GOLD)
    c.line(780,202,858,266);c.line(780,380,858,309,PURPLE)
    c.node(884,287,'+',GOLD);c.text(884,355,'12.5',27,GOLD,'middle')
    c.text(65,113,'orders',18,TEAL);c.text(65,330,'tickets',18,PURPLE)
    c.text(440,469,'The root term appears inside BOTH relation outputs.',19,GOLD)
    c.footer('First sum: neighbors within each relation. Second sum: complete relation outputs.','Teaching scalar trace of the selected sum-SAGE configuration; no normalization shown.')

def bridge(c):
    for y,m in [(200,1),(385,2)]:
        c.node(100,y,'x');c.node(825,y,'Σ',GOLD)
        ys=[y] if m==1 else [y-57,y+57]
        for i,yy in enumerate(ys):
            c.node(465,yy,f'b{i+1}',PURPLE)
            c.line(130,y,433,yy);c.line(497,yy,793,y)
        c.text(890,y+7,'x' if m==1 else '2x',24,GOLD)
    c.text(45,111,'Same source value; different numbers of bridge paths.',21)
    c.text(360,482,'Bridge nodes remain distinct.',18,PURPLE)
    c.footer('Two identity-weight paths contribute the same source twice.','A linear path-counting illustration: nonlinear aggregation need not reduce to this arithmetic.')

def tokens(c):
    c.text(50,112,'Read value AND state before embedding a cell.',21)
    for x,title,value,state,color in [(50,'numeric field','0','observed',TEAL),(370,'numeric field','—','missing',GOLD),(690,'categorical field','new value','unknown',PURPLE)]:
        c.rect(x,145,260,280,'#f7f9fc',color)
        c.text(x+130,182,title,20,color,'middle');c.text(x+130,245,value,28,color,'middle')
        c.line(x+130,267,x+130,310,color)
        c.box(x+20,330,220,state,color)
    c.footer('Preserve column identity and type alongside the value representation.','Exact reserved tokens and missing-value encodings follow the fitted tokenizer contract.')

def adapter(c):
    c.box(35,195,190,'frozen encoder',GRAY);c.line(230,221,283,221,GRAY)
    c.box(290,195,100,'h: 32',GRAY);c.line(395,221,446,221)
    c.box(452,195,105,'D: 32→8');c.line(563,221,587,221)
    c.box(592,195,95,'ReLU');c.line(693,221,717,221)
    c.box(723,195,110,'U: 8→32');c.line(839,221,875,221)
    c.node(907,221,'+',GOLD)
    c.line(338,191,338,123,GRAY,arrow=False);c.line(338,123,907,123,GRAY,arrow=False);c.line(907,123,907,187,GRAY)
    c.text(630,105,'identity skip: h',18,GRAY,'middle')
    c.line(907,253,907,323);c.box(717,329,230,'original task heads');c.line(711,355,647,355)
    c.box(493,329,147,'task loss',GOLD)
    c.line(530,324,530,258,PURPLE,dash=True);c.line(780,324,780,258,PURPLE,dash=True)
    c.text(70,350,'Frozen ≠ skipped.',23,GRAY);c.text(70,381,'Forward computation still runs.',18,GRAY)
    c.text(495,447,'Dashed: gradients update adapter + heads.',18,PURPLE)
    c.footer('h′ = h + U ReLU(Dh + b) + c      •      32 → 8 → 32      •      552 adapter parameters','Course variant: one residual adapter after the pooled encoder; U=0 and c=0 initially.')

def mask(c):
    c.text(55,111,'Rows available by the query cutoff',21)
    for j,label in enumerate(['field A','field B','target']):c.text(330+j*155,153,label,18,GRAY,'middle')
    for i,label in enumerate(['history 1','history 2','query','future']):
        y=179+i*65;c.text(65,y+32,label,19)
        for j in range(3):
            bad=i==3;masked=i==2 and j==2
            c.rect(263+j*155,y,137,51,'#fceaea' if bad else '#fff0d8' if masked else '#e1f1ed',RED if bad else GOLD if masked else TEAL)
            c.text(330+j*155,y+33,'excluded' if bad else '?' if masked else 'observed',17,RED if bad else GOLD if masked else TEAL,'middle')
    c.text(790,340,'hide answer',18,GOLD);c.text(790,405,'exclude row',18,RED)
    c.footer('Column masking does not replace the temporal row filter.','Schematic autocomplete contract: historical target visibility must follow the declared task.')

def cache(c):
    c.text(50,112,'Illustrative timing: short request, old cached evidence',21)
    c.line(85,245,924,245,GRAY)
    for x,label,time in [(120,'source event','0 ms'),(660,'cache refresh','2000 ms'),(810,'request','2100 ms'),(920,'response','2120 ms')]:
        c.node(x,245,'•',TEAL if x<810 else GOLD,r=14)
        c.text(x,180,label,17,INK,'middle');c.text(x,212,time,17,GRAY,'middle')
    c.line(120,283,920,283,GRAY);c.text(520,313,'source age at response = 2120 ms',19,GRAY,'middle')
    c.line(120,348,660,348,PURPLE);c.text(390,380,'incorporation lag = 2000 ms',20,PURPLE,'middle')
    c.line(810,418,920,418,GOLD);c.text(770,459,'response latency = 20 ms',20,GOLD,'middle')
    c.footer('The state can answer quickly while missing recent source changes.','Illustrative durations, not a measured serving benchmark. Intervals are not drawn to scale.')

def cast(c):
    c.text(40,102,'Rows × features',21)
    c.grid(105,150,5,5,52,45,high={(2,j) for j in range(5)}|{(i,2) for i in range(5)})
    c.text(391,410,'width 192',17,GRAY)
    c.line(116,260,353,260,TEAL);c.text(230,130,'feature axis →',18,TEAL,'middle')
    c.line(233,163,233,370,PURPLE);c.text(55,391,'item axis ↓',18,PURPLE)
    c.grid(434,150,5,3,28,45);c.text(482,126,'3 / row',18,TEAL,'middle')
    c.line(373,237,427,237,TEAL)
    c.grid(105,436,2,5,52,24);c.text(382,454,'32 / feature',18,PURPLE)
    c.line(237,382,237,431,PURPLE)
    c.box(635,155,305,'read opposite-axis summaries',GOLD,h=60)
    c.line(523,262,627,189,TEAL);c.line(521,453,648,223,PURPLE)
    c.box(635,276,305,'axis-wise attention',TEAL,h=60);c.line(788,220,788,270)
    c.box(635,397,305,'repeat exchange',PURPLE,h=60);c.line(788,341,788,391,PURPLE)
    c.footer('Pinned release: 12 layers • 6 heads • feature attention repeated twice per layer.','Summary glyph counts are schematic; item summaries gather features, feature summaries gather support rows.')

def forecast(c):
    c.text(40,101,'History: observed targets',21,TEAL);c.text(565,101,'Future: hidden targets',21,GOLD)
    c.line(65,224,922,224,GRAY)
    points=[(75,195),(120,165),(165,188),(210,142),(255,171),(300,132),(345,156),(390,116)]
    for p,q in zip(points,points[1:]):c.line(*p,*q,TEAL,arrow=False)
    c.line(482,116,482,459,GOLD,dash=True,arrow=False);c.text(482,489,'forecast origin',18,GOLD,'middle')
    c.box(55,282,347,'fit time-feature map on history',TEAL)
    c.line(225,230,225,276,TEAL)
    c.box(55,385,347,'context: [time features, y]',TEAL);c.line(225,339,225,379)
    c.box(562,282,372,'apply SAME map to future times',GOLD)
    c.line(408,307,554,307,GOLD)
    c.box(562,385,372,'queries: [time features, ?]',GOLD);c.line(750,339,750,379,GOLD)
    for x,dy,label in [(630,0,'h=1'),(750,-12,'h=2'),(870,10,'h=3')]:
        c.line(x,147+dy,x,211+dy,PURPLE,arrow=False);c.node(x,179+dy,'•',PURPLE,r=9)
        c.text(x,255,label,17,GRAY,'middle')
    c.footer('Context + all future query rows → frozen TabPFN predictor → H marginal distributions.','No predicted target is fed into the next horizon. Intervals are schematic, not measured forecasts.')

DETAILS.update({
 '85': ('One averaging step can erase a node distinction', 'For two connected nodes with self-loops, S has every entry 1/2. Multiplying [2,8] gives [5,5]; the common component remains while the difference vanishes.', 'Must every symmetrically normalized graph converge to identical raw states?', 'No. Unequal degrees change the stationary direction, and disconnected components cannot exchange information. This exact equal-degree example has a narrower scope.', 'smoothing'),
 '86': ('Gather by source, scatter by destination', 'For edge_index=[[0,2],[1,1]] and x=[2,4,8], the source gather yields [2,8]. Destination sum writes [0,10,0]. This is the routing primitive, before GCN loops and normalization.', 'What happens if you gather using the destination row instead?', 'You gather [4,4] rather than [2,8]. The tensor shape still looks right, but the computation is wrong.', 'scatter'),
 '146': ('Sampling reach and computation reach are different', 'On driver→result→race→circuit, three synchronous edge-local layers are needed for original circuit features to reach the driver. One local all-pairs attention block can connect those tokens only if all were sampled.', 'Can more attention heads recover an unsampled circuit?', 'No. Sampling determines which tokens exist. Attention changes communication among admitted tokens; it cannot restore excluded source data.', 'reach'),
 '171': ('Hold out the whole connected source family', 'The held-out original shares a hash with a copy. Declared family links connect that copy to two more derivatives. Following all links quarantines the entire component from pretraining.', 'Can the last derivative stay in training because it has no direct link to the original?', 'No under this conservative policy. Transitive links place it in the same component. The manifest only protects against ancestry that has been recorded.', 'family'),
 '173': ('One pooled encoder, separate prediction heads', 'The course model pools same-row cells and dated foreign-key-parent cells separately. Their vectors and a query-column embedding concatenate to width 96, then a 96→64→32 MLP feeds 21 local task heads.', 'Is the target cell allowed to reappear through a context path?', 'No. Erase its identity from every context route. The model is a course pooled MLP with local heads, not the Relational Transformer attention architecture.', 'pooled'),
 'b12': ('Follow the label to find the adaptation mechanism', 'A supervised label can change model weights through a loss, or enter a frozen predictor as labeled context. In contextual adaptation, the support state changes while the parameter state stays fixed.', 'Does reading neighbors establish label-based in-context learning?', 'No. Trace whether support labels enter the prediction computation. A neighbor sampler alone establishes neither label ICL nor task-specific weight updates.', 'adaptation'),
 'b18a': ('TACO compresses support into latent cells', 'Support cells and K dummy rows enter the compressor. The resulting latent context passes through a residual MLP to the predictor alongside embedded queries. Compressor and predictor are pretrained together.', 'Are the K compressed rows ordinary sampled observations?', 'No. They are learned latent cell representations. The paper uses K×(M+1)×L; the pinned checkpoint groups features, so its feature axis is groups plus the target slot.', 'compress'),
})

def smoothing(c):
    c.text(55,107,'Exact two-node illustration with equal degrees and self-loops',21)
    for x,y,v in [(130,211,'2'),(130,368,'8'),(832,211,'5'),(832,368,'5')]:c.node(x,y,v,TEAL if v!='8' else PURPLE,r=42)
    c.line(130,257,130,321,GRAY,arrow=False)
    c.line(181,211,373,260);c.line(181,368,373,312,PURPLE)
    c.rect(385,196,239,193,'#f1f7f9')
    for i in range(2):
        for j in range(2):c.text(447+j*103,267+i*72,'½',35,INK,'middle')
    c.text(505,162,'S',24,INK,'middle')
    c.line(635,262,782,211);c.line(635,318,782,368)
    c.text(80,457,'difference = 6',22,PURPLE);c.text(764,457,'difference = 0',22,TEAL)
    c.footer('Both nodes receive (2 + 8) / 2 = 5. Another identical mixing step leaves [5,5].','This linear equal-degree case does not imply identical raw limits on every graph.')

def scatter(c):
    c.text(45,106,'x = [2, 4, 8]',23);c.text(390,106,'edge_index',23);c.text(785,106,'output',23)
    for i,v in enumerate([2,4,8]):
        c.box(100,156+108*i,90,str(v),TEAL);c.text(62,189+108*i,i,18,GRAY,'middle')
        c.box(816,156+108*i,90,str(10 if i==1 else 0),GOLD)
    c.text(368,155,'source',17,GRAY);c.text(526,155,'destination',17,GRAY)
    for y,a in [(180,0),(339,2)]:
        c.box(382,y,84,str(a));c.box(548,y,84,'1',PURPLE)
        c.line(195,182+108*a,374,y+26);c.line(471,y+26,540,y+26)
        c.text(507,y+76,'message '+str([2,4,8][a]),17,TEAL,'middle')
        c.line(640,y+26,809,290,PURPLE)
    c.text(757,457,'2 + 8 = 10',23,GOLD)
    c.footer('Two columns in edge_index mean two edges: 0→1 and 2→1.','Declare all 3 nodes. This sum-routing primitive omits GCN loops and degree normalization.')

def reach(c):
    names=['driver','result','race','circuit']
    for i in range(4):
        x=125+i*240;c.node(x,186,names[i],GOLD if i==0 else TEAL,r=44)
        if i:c.line(x-49,186,x-188,186,TEAL)
        c.text(x,272,str(i)+' layer'+('s' if i!=1 else ''),19,GRAY,'middle')
    c.text(55,104,'Edge-local propagation: count the links back to the driver',21)
    c.line(841,345,141,345,PURPLE);c.text(489,382,'all-pairs attention: one block, if both tokens are present',20,PURPLE,'middle')
    c.rect(753,417,190,62,'#fff2f1',RED);c.text(848,454,'absent ≠ reachable',17,RED,'middle')
    c.footer('Sample membership bounds information access before either operator runs.','Schematic synchronous root-readout comparison; pooling would change these assumptions.')

def family(c):
    c.text(50,105,'Evaluation source and its declared relatives',21)
    c.rect(42,138,912,180,'#fff3e5','#d8b478')
    for i,name in enumerate(['original','copy','derivative A','derivative B']):
        x=152+i*231;c.node(x,230,name,GOLD,r=62)
        if i:c.line(x-164,230,x-67,230,GOLD,arrow=False)
    for x,label in [(265,'same hash'),(495,'family link'),(726,'family link')]:c.text(x,158,label,16,GOLD,'middle')
    c.text(153,354,'held out',18,GOLD,'middle');c.text(595,354,'quarantine all three relatives',19,GOLD,'middle')
    c.box(100,415,260,'unrelated source C',TEAL);c.box(628,415,260,'unrelated source D',TEAL)
    c.footer('Follow links until no additional snapshot is reached: a connected-component search.','Synthetic ancestry example. Different hashes alone do not prove independent source data.')

def pooled(c):
    for x,title in [(45,'same-row cells'),(357,'dated FK-parent cells'),(669,'query-column ID')]:
        c.text(x+130,110,title,19,INK,'middle')
        c.grid(x+35,135,1,4,48,44)
        c.line(x+130,188,x+130,217)
        c.box(x+5,223,250,'embedding → mean' if x<600 else 'column embedding')
        c.text(x+130,306,'32',21,TEAL,'middle')
        c.line(x+130,318,492,353,TEAL)
    c.box(347,358,290,'concat 96 → MLP 64 → 32',PURPLE)
    for x,label in [(185,'numeric head'),(493,'categorical head'),(801,'… 21 local heads')]:
        c.line(493,415,x,442,PURPLE);c.box(x-128,448,256,label,GOLD)
    c.footer('Erase the target identity on every route before pooling the context.','Course encoder with task-specific heads; this is not an RT attention block or cross-schema ICL.')

def adaptation(c):
    c.text(48,108,'Weight adaptation',23,TEAL);c.text(535,108,'Context adaptation',23,PURPLE)
    c.box(45,150,390,'training examples → prediction → loss')
    c.line(241,207,241,246);c.box(45,252,390,'optimizer updates θ → θ′')
    c.line(241,309,241,348);c.box(45,354,390,'new query uses θ′')
    c.box(535,150,390,'support features + known labels',PURPLE)
    c.line(731,207,731,246,PURPLE);c.box(535,252,390,'frozen θ reads support + query',PURPLE)
    c.line(731,309,731,348,PURPLE);c.box(535,354,390,'new support can change prediction',PURPLE)
    c.text(239,466,'label → loss → parameters',20,TEAL,'middle');c.text(731,466,'label → model input → prediction',20,PURPLE,'middle')
    c.footer('These are adaptation modes. A system can support both at different stages.','See the three model-specific architectures below for Griffin, OpenRFM and KumoRFM-2.')

def compress(c):
    c.text(47,107,'N support rows',20);c.grid(65,131,6,4,35,31)
    c.text(308,107,'K dummy rows',20);c.grid(339,157,2,4,35,31)
    c.text(312,255,'targets masked',17,GOLD)
    c.line(218,224,520,223,TEAL);c.line(483,188,520,211,GOLD)
    c.box(527,177,220,'compressor',TEAL,h=81)
    c.line(753,218,809,218)
    c.grid(821,158,2,4,31,39);c.text(883,279,'latent cells',18,PURPLE,'middle')
    c.line(883,287,883,326,PURPLE);c.box(717,332,240,'residual MLP',PURPLE)
    c.line(709,358,642,358,PURPLE);c.box(389,332,247,'predictor',TEAL)
    c.box(43,332,265,'embedded query cells',GOLD);c.line(314,358,381,358,GOLD)
    c.line(514,389,514,438);c.text(514,469,'query prediction',23,TEAL,'middle')
    c.footer('Paper latent shape: K × (M+1) × L. The release uses feature groups plus a target slot.','Compressor and predictor are jointly pretrained. Latent rows are not ordinary observations.')

DRAWINGS_MORE={'smoothing':smoothing,'scatter':scatter,'reach':reach,'family':family,'pooled':pooled,'adaptation':adaptation,'compress':compress}

DRAWINGS={'sampling':sampling,'recursive':recursive,'root':lambda c:recursive(c,True),'clocks':clocks,'keys':keys,'relations':relations,'bridge':bridge,'tokens':tokens,'adapter':adapter,'mask':mask,'cache':cache,'cast':cast,'forecast':forecast}

DRAWINGS.update(DRAWINGS_MORE)

def render(key):
    title,desc,_,_,kind=DETAILS[key];c=Canvas(title,desc);DRAWINGS[kind](c);return c.done()

START='<!-- visual-detail:start -->';END='<!-- visual-detail:end -->'
def strip(text):return re.sub(re.escape(START)+r'.*?'+re.escape(END)+'\n?', '',text,flags=re.S)

def inject(text,key):
    if key not in DETAILS:return text
    title,desc,question,answer,_=DETAILS[key]
    # Keep prerequisite/retrieval prompts before this supplement. The existing
    # architecture figure remains immediately afterwards for the full contract.
    figures=list(re.finditer(r'<figure\b[^>]*>.*?</figure>',text,re.S))
    if not figures:raise ValueError(f'No existing figure to anchor detail {key}')
    candidates=[f for f in figures if re.search(r'architecture|Architecture|whole.solution|model.map',f[0])]
    pos=(candidates or figures)[0].start()
    if key=='b09':
        pos=text.rfind('<div class="b09-desktop-architecture">',0,pos)
        if pos<0:raise ValueError('Missing B09 architecture wrapper')
    block=f'''{START}<figure class="visual-detail" id="visual-detail-{key}">
<h3>{escape(title)}</h3><p>{escape(desc)}</p>
<p class="vd-scroll-hint">On a narrow screen, scroll the drawing sideways or enlarge it. Arrow keys scroll when the drawing is focused.</p>
<div class="vd-scroll" tabindex="0" role="region" aria-label="{escape(title,quote=True)} — scrollable drawing"><img src="../assets/visual-details/{key}.svg" alt="{escape(desc,quote=True)}" width="1000" height="620" loading="lazy"/></div>
<figcaption>{escape(title)}. Teaching schematic; read alongside the original architecture and source notes below.</figcaption>
<details><summary>Predict: {escape(question)}</summary><p>{escape(answer)}</p></details>
<p class="vs-print-answer"><strong>Answer:</strong> {escape(answer)}</p>
</figure>{END}\n'''
    return text[:pos]+block+text[pos:]
