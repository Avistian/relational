"""Editable SVG architecture/causal-path figures plus portable PNG exports."""
from pathlib import Path
import math
import cairosvg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent/'figures/l103';P.mkdir(parents=True,exist_ok=True)
INK='#173342';TEAL='#007f80';GOLD='#bb7121';MUTED='#52636b';PAPER='#faf9f5'
def diagram(name,w,h,body):
 svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10" fill="{TEAL}"/></marker></defs><style>text{{font-family:DejaVu Sans,sans-serif;fill:{INK}}}.title{{font-size:25px;font-weight:bold}}.head{{font-size:18px;font-weight:bold}}.body{{font-size:16px}}.small{{font-size:14px;fill:{MUTED}}}.arrow{{stroke:{TEAL};stroke-width:2;fill:none;marker-end:url(#arrow)}}</style><rect width="{w}" height="{h}" fill="{PAPER}"/>{body}</svg>'''
 (P/f'{name}.svg').write_text(svg);cairosvg.svg2png(bytestring=svg.encode(),write_to=str(P/f'{name}.png'),scale=1.6)
def text(x,y,s,cls='body'):return f'<text x="{x}" y="{y}" class="{cls}">{s}</text>'
def box(x,y,w,h,title,lines):
 return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="white" stroke="#b9cccb"/>'+text(x+17,y+29,title,'head')+''.join(text(x+17,y+57+i*24,line) for i,line in enumerate(lines))
def arrow(x,y,xx,yy):return f'<path class="arrow" d="M{x} {y}L{xx} {yy}"/>'
b=text(30,40,'TGAT · a representation is computed for a node AND a time','title')
b+=text(30,70,'Released Wikipedia variant · two layers · two heads · d = 172 · N = 20','small')
b+=box(30,95,430,100,'Query (u, t) + event store',['Retrieve 20 eligible incident records','Historical rows contain node, edge, timestamp'])
b+=box(490,95,430,100,'Raw features + shared parameters',['Zero node vectors; 172 edge features','No recurrent per-node state is stored'])
b+=arrow(245,195,245,225)+arrow(705,195,705,225)
b+=box(30,225,890,120,'Recursive layer: evaluate every child at its own interaction time',['Root: h^(ℓ−1)(u, t)     Child: h^(ℓ−1)(vᵢ, tᵢ)     Age: Δtᵢ = t − tᵢ','At layer 0: read raw features. At layers 1 and 2: repeat retrieval + attention.','Frequency and phase parameters produce Φ(Δt) = cos(ωΔt + b).'])
b+=arrow(245,345,245,375)+arrow(705,345,705,375)
b+=box(30,375,430,110,'Root record → query projection',['[h(u,t) | zero edge | Φ(0)]','B × 1 × 516 → 2 heads, width 258'])
b+=box(490,375,430,110,'History records → keys and values',['[h(vᵢ,tᵢ) | edgeᵢ | Φ(t−tᵢ)]','B × 20 × 516 → 2 heads, width 258'])
b+=arrow(245,485,245,515)+arrow(705,485,705,515)
b+=box(30,515,890,120,'Inside each head: compatibility → weighting → content',['qKᵀ / √258 → padding mask → softmax α → dropout → Σ αᵢVᵢ','Two head outputs → concatenate → projection → residual + layer norm','Merge [context 516 | root 172] → MLP 688 → 172 → embedding 172'])
b+=arrow(475,635,475,665)
b+=text(125,690,'Evaluate the same encoder θ at three endpoint queries','head')
b+=box(30,710,275,80,'Target hᵥ(t)',['B × 172 · shared encoder θ'])
b+=box(338,710,275,80,'Source hᵤ(t)',['B × 172 · shared encoder θ'])
b+=box(646,710,274,80,'Negative h⁻(t)',['B × 172 · shared encoder θ'])
for path in ['M167 790V810H245V830','M475 790V805H245V830','M475 790V805H705V830','M783 790V810H705V830']:
 b+=f'<path class="arrow" d="{path}"/>'
b+=box(30,830,430,100,'Positive pair · shared decoder ψ',['[hᵤ | hᵥ] : 344 → 172 → 1','Sigmoid → positive BCE'])
b+=box(490,830,430,100,'Negative pair · shared decoder ψ',['[hᵤ | h⁻] : 344 → 172 → 1','Sigmoid → negative BCE'])
b+=text(30,965,'Training loss = mean positive BCE + mean negative BCE. Inference freezes θ and ψ.','body')
b+=text(30,997,'Release replay preserves sampler/padding quirks. Corrected comparison is a separate lane.','small')
diagram('architecture',950,1025,b)
b=text(30,40,'Each child uses its connecting event time','title')
b+=box(30,75,290,110,'Root query A @ 8',['Available first hop: A–B @ 5','Ask for B’s state at time 5'])
b+=arrow(320,130,375,130)+box(375,75,315,110,'Child query B @ 5',['B–C @ 4: eligible','B–D @ 6: future to this query'])
b+=box(30,225,660,125,'Two boundaries, one chronological path',['A@8 → edge@5 → B@5 → edge@4 → C@4','A@8 → edge@5 → B@5 → edge@6   ✕','Filtering every edge only at root time 8 admits an invalid path.'])
b+=text(30,392,'Separate release counterexample: history [1, 3, 5], query 4','head')
b+=box(30,415,315,95,'Correct strict prefix',['Returns [1, 3]'])+box(375,415,315,95,'Released binary-search slice',['Returns [1] · one event omitted'])
diagram('recursion',720,535,b)
x=np.linspace(0,2*np.pi,350)
fig,ax=plt.subplots(1,2,figsize=(8.5,4.2),facecolor=PAPER)
for a in ax:a.set_facecolor(PAPER);a.spines[['top','right']].set_visible(False)
ax[0].plot(x,np.cos(x),color=TEAL,label='cos(t)');ax[0].plot(x,np.sin(x),color=GOLD,label='sin(t)');ax[0].set(xlabel='Time t (illustrative units)',ylabel='Feature value',title='Two coordinates, one frequency');ax[0].legend()
for delta,color in [(1,TEAL),(2,GOLD)]:
 ax[1].plot(x,np.cos(x)*np.cos(x+delta)+np.sin(x)*np.sin(x+delta),color=color,label=f'Paired, Δ = {delta}')
ax[1].plot(x,np.cos(x)*np.cos(x+1),color='#7f589b',ls='--',label='Cosine only, Δ = 1')
ax[1].set(xlabel='Common time shift',ylabel='Feature inner product',title='A shared shift tests stationarity');ax[1].legend(fontsize=10)
fig.tight_layout();fig.savefig(P/'kernel.svg');fig.savefig(P/'kernel.png',dpi=170);plt.close(fig)
svg=P/'kernel.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
print(P)
