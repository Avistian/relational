"""Portable computation diagrams; same SVG source supplies notebook PNGs."""
from pathlib import Path
from html import escape
import cairosvg
P=Path(__file__).resolve().parent/'figures/l145';P.mkdir(parents=True,exist_ok=True)
colors=['#197278','#bf6900','#7d4c9e','#176ca4','#a34255']
def text(x,y,s,size=17,color='#153342'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-family="sans-serif">{escape(s)}</text>'
def box(x,y,w,h,title,lines,color='#197278'):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="white" stroke="{color}" stroke-width="2"/>'+text(x+14,y+27,title,19,color)+''.join(text(x+14,y+54+24*i,s,15) for i,s in enumerate(lines))
def arrow(x,y,xx,yy):return f'<path d="M{x},{y} L{xx},{yy}" stroke="#506c77" stroke-width="2" fill="none" marker-end="url(#a)"/>'
def save(name,title,body,h=570):
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{h}" viewBox="0 0 1100 {h}" role="img"><title>{escape(title)}</title><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#506c77"/></marker></defs><rect width="1100" height="{h}" fill="#f4f8f7"/>'+text(24,36,title,25)+body+'</svg>'
    (P/(name+'.svg')).write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/(name+'.png')),scale=1.5)
body=box(25,62,1045,85,'Input query: (driver, cutoff) + foreign-key graph + row attributes',['Sample K=300 slots including the root. Each slot belongs to one query; timestamps require an owner-cutoff audit.'])
for i,(name,lines) in enumerate(zip(['Table type','Hop distance','Relative age','Row features','Local structure'],[['Table ID → lookup','B × 300 × 512'],['Distance → lookup','B × 300 × 512'],['Age → sinusoid','→ linear, width 512'],['Typed columns','→ ResNet, width 512'],['Random scalar','→ 4 GIN layers']])):
    x=25+i*210;body+=box(x,187,195,111,name,lines,colors[i])+arrow(x+97,147,x+97,187)+arrow(x+97,298,x+97,340)
body+=box(25,340,1045,102,'Normalize each vector → concatenate → nonlinear mixer',['[type | hop | time | row | PE]: B × 300 × 2560','Linear(2560,1024) → ReLU → Linear(1024,512). Root token is slot 0.'])
body+=box(25,490,500,113,'Local: all 300 × 300 token pairs',['L Transformer layers, 4 heads, 128 coordinates/head','Readout = root + weighted neighbor sum → B × 512'])
body+=box(570,490,500,113,'Global: root → 4096 centroids',['QKᵀ / √512 + log(occupancy) → softmax → values','EMA buffers update in training only → B × 512'])
body+=arrow(270,442,270,490)+arrow(820,442,820,490)
body+=box(25,657,1045,103,'Prediction: concatenate local and global → task head',['B × 1024 → FFN → B × 512 → scalar finishing position','Train: mean absolute error. Evaluate: train-percentile clipping; source attention and PE remain stochastic.'])
body+=arrow(270,603,270,657)+arrow(820,603,820,657)
save('architecture','RelGT • five row descriptions, two attention domains, one prediction',body,790)
body=''
for i,(name,v) in enumerate(zip(['type','hop','time','features','structure'],[1,2,3,4,5])):
    x=24+i*210;body+=box(x,75,195,106,name,[f'illustrative scalar: {v}','one slot, d=1'],colors[i])
body+=text(30,225,'Concatenate: [1 | 2 | 3 | 4 | 5]  — information stays in distinct coordinates.',21)
body+=box(30,267,500,130,'A deliberately simple linear mixer',['Weights [1, 0, 0, 2, −1]','1×1 + 0×2 + 0×3 + 2×4 − 1×5 = 4','Remove structure (set to zero): output becomes 9.'])
body+=box(562,267,507,130,'What stays fixed?',['All other inputs and projection weights.','Released model uses five 512-vectors, normalization,','then Linear(2560,1024) → ReLU → Linear(1024,512).'])
save('token','Five elements • distinguish concatenation from addition',body,430)
body=box(25,75,315,150,'Two requested queries',['(driver 7, cutoff 5)','(driver 7, cutoff 10)','Event times: 4 and 9.'])+box(388,75,320,150,'Released cache key: driver 7',['Second query replaces first.','Both rows receive the time-10 tokens.','Age(9) = 10 − 9 = 1.'])+box(756,75,315,150,'Independent audit',['For cutoff 5: event 9 is FUTURE.','Positive cached age conceals it.','Check actual event time ≤ owner cutoff.'])+arrow(340,145,386,145)+arrow(709,145,754,145)
body+=text(32,275,'A time encoder describes a token. It does not authorize access to that token.',23)
body+=box(30,310,1040,85,'The safe cache identity',['(entity ID, cutoff, sampler configuration, graph fingerprint). A repaired cache is a different experiment.'])
save('ownership','Temporal ownership • audit the raw timestamp, not the cached age',body,425)
body=box(25,75,510,150,'Local attention: within each sampled query',['Q,K,V: B × 4 × 300 × 128','softmax(QKᵀ / √128) V','Products of different FK branches can interact directly.','This does not recover rows excluded from the sample.'])+box(570,75,505,150,'Global attention: learned centroid memory',['Query: root token. Keys/values: 4096 centroids.','log(count) biases attention toward occupied centroids.','Zero-count centroids receive −∞ attention logits.','EMA: old state × .99 + batch contribution × .01.'])
body+=box(25,265,510,125,'Training',['Compute prediction from current centroid state.','Update codebook from training root features.','Save centroid buffers with the selected checkpoint.'])+box(570,265,505,125,'Evaluation',['Freeze centroids and running normalization statistics.','Release still resamples structural PE and uses','attention dropout: eval() does not make outputs fixed.'])
save('attention','Two attention domains • sampled tokens and training-time centroid state',body,425)
