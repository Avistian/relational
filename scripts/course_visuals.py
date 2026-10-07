"""Responsive course architecture routes with an explicit operation inside each map."""
from html import escape as e
import math,re
from visual_detail_layouts import Scene,T,P,G,R,I,S
from course_visual_specs import SPECS
START='<!-- course-visual:start -->';END='<!-- course-visual:end -->'

def drawing(kind,key):
 c=Scene()
 if kind=='query_slice':
  c.text(20,21,'row',15,I);c.text(134,21,'feature groups',15,I,'middle');c.text(249,21,'target',15,P,'middle')
  for i,label in enumerate(['c₀','c₁','q₀','q₁']):
   y=40+i*40;c.text(22,y+20,label,17,P if i>1 else S)
   c.table(69,y,[['h','h','Z₀' if i==2 else 'Z₁' if i==3 else 'h']],w=67,h=34,colors={(0,2):P} if i>1 else None)
  c.parts.append('<rect x="199" y="116" width="71" height="82" rx="6" fill="none" stroke="#7656a6" stroke-width="3"/>')
  c.text(150,226,'H[:, C:, −1, :]',21,P,'middle');c.text(150,254,'two query vectors × 192',17,I,'middle')
 elif kind=='duplicate_mean':
  c.text(150,21,'true neighbor list',17,I,'middle');c.table(71,36,[['A: 2','B: 8']],w=82,h=33)
  c.text(150,96,f'mean = {(2+8)/2:g}',19,T,'middle');c.line(150,108,150,130)
  c.text(150,154,'padded table: B appears twice',16,P,'middle');c.table(20,170,[['A: 2','B: 8','B: 8']],w=88,h=33,colors={(0,1):P,(0,2):P})
  c.text(150,239,f'mean = {(2+8+8)/3:g} ≠ 5',21,P,'middle')
 elif kind=='attention_remove':
  values=[0,1,2];den=sum(math.exp(x) for x in values);weights=[math.exp(x)/den for x in values];reduced=sum(math.exp(x) for x in values[:2]);next_weights=[math.exp(x)/reduced for x in values[:2]]
  c.text(150,21,'same scores / values: 0, 1, 2',16,I,'middle')
  c.text(150,49,f'all edges · denominator {den:.3f}',16,T,'middle');c.table(22,61,[[f'{w:.3f}' for w in weights]],w=87,h=34)
  c.text(150,120,f'weighted output = {sum(w*v for w,v in zip(weights,values)):.3f}',18,T,'middle')
  c.text(150,156,'remove edge 3; scores unchanged',15,P,'middle');c.table(22,170,[[f'{w:.3f}' for w in next_weights]+['excluded']],w=87,h=34,colors={(0,j):P for j in range(3)})
  c.text(150,228,f'denominator {reduced:.3f} → output {next_weights[1]:.3f}',16,P,'middle');c.text(150,256,'remaining weights renormalize',15,I,'middle')
 elif kind=='scale_mix':
  for i,(label,values) in enumerate([('original',[2,8]),('× 0.001',[.002,.008]),('average',[5,5])]):
   y=25+i*79;color=P if i==2 else T;gap=abs(values[1]-values[0])/math.sqrt(sum(x*x for x in values))
   c.text(10,y+18,label,16,color);c.table(104,y,[[f'{v:g}' for v in values]],w=88,h=31,colors={(0,j):color for j in range(2)})
   c.text(150,y+55,f'normalized gap = {gap:.3f}',17,color,'middle')
  c.text(150,264,'shrink: same ratio · mix: gap lost',15,I,'middle')
 elif kind=='churn':
  c.graph([(50,43),(150,115),(250,43)],[(0,1)],['book','review','future'])
  c.line(150,143,150,173);c.box(56,185,190,'customer → logit',G);c.text(150,252,'future review stays excluded',16,R,'middle')
 elif kind=='join':
  for x,label,rows in [(5,'orders',[['20'],['30']]),(106,'sessions',[['4'],['8']]),(207,'tickets',[['open']])]:
   c.text(x+42,22,label,16,T,'middle');c.table(x+10,40,rows,w=70);c.line(x+42,111,x+42,140)
  c.table(15,150,[['2','6','1']],w=90);c.text(150,203,'count · mean · count',16,I,'middle')
 elif kind=='ensemble':
  c.text(150,22,'same query → three trees',17,I,'middle')
  for x,value in [(45,'0.2'),(150,'0.5'),(255,'0.8')]:
   c.node(x,65,'x',T,17);c.line(x,84,x-20,108);c.line(x,84,x+20,108);c.box(x-33,119,66,value);c.line(x,158,150,188)
  c.text(150,221,'mean = 0.5',20,P,'middle')
 elif kind=='boost':
  c.box(12,25,122,'old = 3');c.box(166,25,122,'target = 5',G);c.text(150,90,'residual = 5 − 3 = 2',18,I,'middle')
  c.box(48,119,204,'tree predicts +2',P);c.line(150,155,150,181);c.text(150,213,'3 + 0.1 × 2 = 3.2',20,T,'middle')
 elif kind=='xgb':
  c.text(150,26,'G = −6; H = 2',19,I,'middle')
  for x,lam,val in [(12,1,2),(164,4,1)]:
   c.box(x,57,124,'λ = '+str(lam),P);c.rect(x+35,183-val*37,54,val*37,'#dceee9',T);c.text(x+62,211,'leaf = '+str(val),17,T,'middle')
  c.text(150,243,'w* = −G / (H + λ)',18,I,'middle')
 elif kind=='leaf':
  c.node(150,35,'root',T,28);c.line(127,55,65,96);c.line(173,55,235,96);c.box(9,104,114,'gain 2');c.box(177,104,114,'gain 7',G)
  c.line(226,144,190,187,G);c.line(245,144,270,187,G);c.node(189,212,'L',G);c.node(271,212,'R',G);c.text(66,187,'wait',17,S,'middle')
 elif kind=='ordered':
  c.text(150,24,'permutation order →',17,I,'middle');c.table(22,44,[['A','A','A'],['1','0','1']],w=87)
  c.text(150,132,'encode second A',18,G,'middle');c.line(63,109,30,150,T,arrow=False);c.line(30,150,30,180,T,arrow=False);c.line(30,180,75,180,T);c.text(155,188,'prefix mean = 1',18,T,'middle');c.text(150,232,'current target 0 excluded',17,R,'middle')
 elif kind=='noise':
  c.text(150,24,'fixed signal + noise columns',17,I,'middle');c.table(16,47,[['x','z₁','z₂','z₃'],['1','0','1','0'],['0','1','0','1']],w=68)
  c.line(150,144,150,170,G);c.text(150,195,'more candidate splits',18,G,'middle');c.text(150,226,'≠ more population signal',17,I,'middle')
 elif kind=='mean':
  c.box(10,27,105,'event = 2');c.box(183,27,105,'event = 6');c.line(62,64,132,107);c.line(235,64,168,107);c.box(67,116,166,'mean = 4',P)
  c.box(10,183,112,'self = 3',G);c.text(165,209,'[3, 4]',21,I);c.line(123,200,155,200,G)
 elif kind=='gcn':
  c.graph([(40,53),(150,105),(260,53)],[(0,1),(2,1)],['u','v','w'])
  c.text(150,155,'S: mix neighbor rows',17,T,'middle');c.line(150,165,150,183);c.table(40,193,[['h₁','h₂','h₃']],w=76);c.text(150,248,'W: mix feature coordinates',16,P,'middle')
 elif kind=='walk':
  c.graph([(45,45),(243,45),(243,195),(45,195)],[(0,1),(1,2),(2,3)],['u','i','u′','j'])
  c.text(150,101,'retain each',17,I,'middle');c.text(150,128,'intermediate',17,I,'middle');c.text(150,155,'identity',17,I,'middle')
 elif kind=='relations':
  c.text(150,25,'A: values 2, 4 · B: value 9',17,I,'middle')
  c.table(33,46,[['2','4','9']],w=80);c.text(150,120,'relation means: 3 + 9 = 12',17,T,'middle');c.text(150,167,'uniform receiver mean:',17,P,'middle');c.text(150,202,'(2 + 4 + 9) / 3 = 5',19,P,'middle')
 elif kind=='seed':
  c.text(150,24,'first B rows are seeds',17,G,'middle');c.table(17,43,[['s₁','s₂','c₁','c₂']],w=68)
  c.line(49,82,75,122,G);c.line(117,82,110,122,G);c.box(25,133,123,'seed loss',G);c.box(163,133,123,'context',T);c.text(150,211,'features can contribute',17,I,'middle');c.text(150,237,'without extra target labels',17,I,'middle')
 elif kind=='time':
  c.box(35,18,230,'past events: τ < t');c.line(150,57,150,83);c.box(35,92,230,'score event at t',G);c.line(150,132,150,159);c.box(35,167,230,'queue current event',P);c.text(150,233,'no backward access to answer',16,R,'middle')
 elif kind=='sage':
  c.box(7,22,132,'neighbors',T);c.box(178,22,112,'root',G);c.line(73,61,73,84);c.line(234,61,234,84,G)
  c.box(7,93,132,'mean → Wₙ',T);c.box(178,93,112,'Wᵣ × self',G);c.line(73,135,128,173);c.line(234,135,175,173,G);c.node(150,194,'+',P);c.text(150,245,'two learned maps',17,I,'middle')
 elif kind=='bn':
  for j,label in enumerate(['train rows','held-out rows']):
   c.text(150,25+j*98,label,17,T if j==0 else G,'middle');c.table(55,43+j*98,[['x','x','x']],w=65)
  c.text(150,213,'fit μ, σ² on training only',17,T,'middle');c.text(150,240,'reuse saved μ, σ² at evaluation',16,I,'middle')
 elif kind=='cutoff':
  c.text(150,26,'query cutoff = day 7',18,G,'middle');c.line(25,102,278,102,S,arrow=False)
  for x,y,label,col in [(62,5,'day 5 ✓',T),(231,11,'day 11 ×',R)]:
   c.node(x,102,str(y),col);c.text(x,156,label,17,col,'middle')
  c.line(120,67,120,180,G,arrow=False);c.text(150,223,'the same cutoff at every hop',17,I,'middle')
 elif kind=='dot':
  c.text(150,24,'one query · two candidates',17,I,'middle');c.table(15,45,[['u','1','2'],['v₁','2','0'],['v₂','0','2']],w=90)
  c.text(150,174,'u · v₁ = 2',19,T,'middle');c.text(150,214,'u · v₂ = 4',19,P,'middle')
 elif kind=='composite':
  c.table(22,30,[['source','fact','fused'],['2','1','3'],['4','0','4']],w=87)
  c.text(150,159,'softmax(3, 4) ≈ (.269, .731)',16,P,'middle');c.text(150,200,'.269 × 3 + .731 × 4',18,I,'middle');c.text(150,233,'≈ 3.731',20,T,'middle')
 elif kind=='bpr':
  c.box(15,29,128,'positive 2',T);c.box(157,29,128,'negative 0',P)
  c.text(150,111,'softplus(0 − 2)',20,I,'middle');c.text(150,152,'≈ 0.127',22,T,'middle');c.line(38,180,263,180,S,arrow=False);c.text(150,219,'reverse scores → 2.127',18,R,'middle')
 elif kind=='frozen':
  c.box(43,23,215,'GNN: trainable',T);c.line(150,64,150,93);c.box(43,103,215,'decoder: frozen',P);c.line(150,144,150,173);c.text(150,197,'loss',20,G,'middle');c.line(278,187,278,40,G);c.text(150,242,'gradient passes through',17,I,'middle')
 elif kind=='support':
  c.text(150,24,'support labels visible',17,T,'middle');c.table(43,44,[['x','x','y'],['x','x','y']],w=72)
  c.text(150,137,'query label hidden',17,G,'middle');c.table(43,154,[['q','q','?']],w=72);c.line(150,191,150,211);c.text(150,247,'decode query target state',17,P,'middle')
 elif kind=='mask':
  c.table(42,20,[['x','x','?']],w=74);c.text(150,78,'value mask: hide the answer',17,G,'middle')
  c.table(35,103,[['✓','×','✓'],['×','✓','×'],['✓','×','✓']],w=78);c.text(150,225,'attention mask: allow a route',16,P,'middle')
 elif kind=='transfer':
  c.box(15,28,270,'RelGT: row tokens + centroids',T);c.box(15,105,270,'Griffin: cells + relations',P)
  c.text(150,187,'both can have width 512',18,I,'middle');c.text(150,229,'same shape ≠ same meaning',17,R,'middle')
 elif kind=='codes':
  c.text(150,25,'known categories b, c, d',17,I,'middle');c.table(30,43,[['0','1','2']],w=82);c.line(150,80,150,106,R);c.text(150,132,'query introduces a',17,R,'middle');c.table(30,151,[['1','2','3']],w=82);c.text(150,231,'reject the changed code map',17,R,'middle')
 elif kind=='collision':
  c.table(38,24,[['0','4'],['2','2']],w=112);c.text(150,125,'both means = 2',19,T,'middle');c.text(150,174,'population variances:',17,I,'middle');c.text(150,211,'4 versus 0',21,P,'middle')
 elif kind=='residual':
  c.box(65,18,170,'incoming x',T);c.line(150,59,150,84);c.box(65,92,170,'branch F(x)',P);c.line(150,133,150,158);c.node(150,186,'+',G)
  c.line(240,35,275,35,T,arrow=False);c.line(275,35,275,186,T,arrow=False);c.line(275,186,178,186,T);c.text(150,244,'output = x + F(x)',19,I,'middle')
 elif kind=='bypass':
  c.box(6,19,139,'categories',T);c.box(167,19,125,'numbers',G);c.line(75,61,75,83);c.box(6,92,139,'attention',T);c.line(75,134,130,181);c.line(229,61,229,160,G,arrow=False);c.line(229,160,175,181,G);c.box(74,194,151,'MLP → y',P)
 elif kind=='sparse':
  c.text(150,24,'original input',17,T,'middle');c.table(33,42,[['2','4','6']],w=81);c.text(150,110,'mask × input',17,P,'middle');c.table(33,127,[['0.5','0','0.5']],w=81);c.text(150,218,'[1, 0, 3] → transform',20,T,'middle')
 elif kind=='softtree':
  c.node(150,26,'x',T,18);c.line(136,43,75,81);c.line(164,43,225,81);c.node(75,105,'½',P);c.node(225,105,'½',P)
  for a,b in [(75,34),(75,109),(225,184),(225,259)]:c.line(a,131,b,170);c.box(b-27,181,54,'¼',T)
  c.text(150,246,'four leaf weights sum to 1',17,I,'middle')
 elif kind=='token':
  c.box(18,20,100,'xj = 2',G);c.table(139,22,[['1','3']],w=64);c.text(203,83,'wj',17,T,'middle');c.text(150,123,'2 × [1, 3] + [0, 1]',19,I,'middle');c.table(71,152,[['2','7']],w=82);c.text(150,224,'one numeric feature token',17,P,'middle')
 else:raise ValueError(kind)
 # Diagram-local marker IDs avoid cross-figure references in inline HTML.
 inner=''.join(c.parts).replace('url(#tip)',f'url(#cv-tip-{key})')
 return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 270" role="img" aria-labelledby="cv-title-{key}"><title id="cv-title-{key}">{e(SPECS[key]["question"]+" "+SPECS[key]["answer"])}</title><defs><marker id="cv-tip-{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#52717c"/></marker></defs><g font-family="system-ui,sans-serif">{inner}</g></svg>'

def render(key):
 s=SPECS[key];steps=''.join(f'<li><span class="cv-number">{i}</span><div><h4>{e(title)}</h4><p>{e(body)}</p></div></li>' for i,(title,body) in enumerate(s['steps'],1))
 return START+f'<section class="course-visual" id="course-visual-{key}" aria-labelledby="cv-heading-{key}"><header><p class="cv-kicker">Follow the computation</p><h3 id="cv-heading-{key}">{e(s["title"])}</h3></header><div class="cv-body"><ol class="cv-route">{steps}</ol><div class="cv-focus"><h4>Open the decisive operation</h4>{drawing(s["op"][0],key)}<details><summary>{e(s["question"])}</summary><p>{e(s["answer"])}</p></details><p class="cv-print-answer"><strong>Trace answer:</strong> {e(s["answer"])}</p></div></div><p class="cv-scope">{e(s["scope"])}</p></section>'+END

def strip(text):return re.sub(re.escape(START)+'.*?'+re.escape(END)+'\n?','',text,flags=re.S)
def inject(text,key):
 text=strip(text)
 if key not in SPECS:return text
 anchor=SPECS[key]['anchor']
 if anchor:
  match=next((m for m in re.finditer(r'<h[23]\b[^>]*>.*?</h[23]>',text,re.S) if anchor.lower() in re.sub('<[^>]+>','',m[0]).lower()),None)
 else:
  # Beside the primary architecture, not an unrelated results figure.
  match=next((m for m in re.finditer(r'<figure\b[^>]*>.*?</figure>',text,re.S) if re.search(r'(?:architecture|pipeline|relgt|076-rdl|078-preview|095-walk|099-comparison|100-checkpoint)',m[0],re.I)),None)
 if match:pos=match.start()
 else:
  h=re.search(r'<h2\b',text);pos=h.start() if h else text.index('</article>')
 return text[:pos]+render(key)+'\n'+text[pos:]
