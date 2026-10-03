"""Model-specific native SVG overviews and measured portable result plots."""
from pathlib import Path
import html,json
import cairosvg
P=Path(__file__).resolve().parent;F=P/'figures/b07';F.mkdir(exist_ok=True)
class Diagram:
 def __init__(self,title,height):
  self.h=height;self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="460" height="{height}" viewBox="0 0 460 {height}" role="img"><title>{html.escape(title)}</title><rect width="460" height="{height}" rx="14" fill="#f3f6f2"/><style>text{{font-family:DejaVu Sans,sans-serif;fill:#193c38;font-size:18px}}.title{{font-size:21px;font-weight:bold}}.box{{fill:white;stroke:#a7bdb8;stroke-width:1.5}}.arrow{{stroke:#357866;stroke-width:2;fill:none}}</style><text x="20" y="34" class="title">{html.escape(title)}</text>']
 def box(self,y,h,title,lines,fill='#ffffff'):
  self.parts.append(f'<rect class="box" x="18" y="{y}" width="424" height="{h}" rx="9" style="fill:{fill}"/><text x="30" y="{y+28}" class="title">{html.escape(title)}</text>')
  for j,line in enumerate(lines):self.parts.append(f'<text x="30" y="{y+55+j*25}">{html.escape(line)}</text>')
 def arrow(self,y):self.parts.append(f'<path class="arrow" d="M230 {y} v23 m-6 -7 l6 7 l6 -7"/>')
 def text(self,x,y,t):self.parts.append(f'<text x="{x}" y="{y}">{html.escape(t)}</text>')
 def save(self,name):
  self.parts.append('</svg>');(F/(name+'.svg')).write_text(''.join(self.parts));cairosvg.svg2png(url=str(F/(name+'.svg')),write_to=str(F/(name+'.png')))
d=Diagram('CARTE · one row becomes a graph',1020)
d.box(58,112,'Known cells → vector leaves',['volume z=2 → 2 × header vector','country=France → text vector']);d.arrow(170)
# Spatial row star with operations encoded on edges.
d.parts.append('<rect class="box" x="18" y="201" width="424" height="195" rx="9"/><path class="arrow" d="M104 274 L210 332 M355 274 L250 332"/>')
d.text(40,233,'Columns label the edges');d.text(35,266,'2 × e_volume');d.text(274,266,'v_France');d.text(30,300,'e_volume');d.text(315,300,'e_country');d.parts.append('<circle cx="230" cy="341" r="23" fill="#d6eae0" stroke="#357866"/>');d.text(209,348,'row');d.text(36,384,'center = mean(edge ⊙ leaf)');d.arrow(396)
d.box(427,136,'Map nodes, then attend',['sender message = edge ⊙ node','12 heads × 25 coordinates','center ← weighted incoming messages'],'#e1eee6');d.arrow(563)
d.box(594,111,'Readout → one 300-vector',['LayerNorm → feedforward → LayerNorm','Only the center is consumed.']);d.arrow(705)
d.box(736,111,'Fit a ridge head on training rows',['64 train: scaler + coefficients','64 validation: choose alpha 1 / 10 / 100'],'#edf0fb');d.arrow(847)
d.box(878,118,'256 test predictions per run',['Weights in graph encoder stay fixed.','Paper uses different downstream fitting.']);d.save('carte')
d=Diagram('ConTextTab · labeled context',1165)
d.box(58,112,'Known context + unknown query',['2 support rows: X and y','1 query row: X only; target hidden']);d.arrow(170)
d.box(201,137,'Cell vector + header vector',['text: MiniLM → learned projection','number: scalar or soft-bin encoding','date: day + month + year encodings']);d.arrow(338)
# miniature rows and cells
for i,label in enumerate(['S1','S2','Q']):
 d.text(30,393+i*40,label)
 for j,t in enumerate(['x₁','x₂','?' if i==2 else 'y']):
  d.parts.append(f'<rect x="{88+111*j}" y="{369+40*i}" width="94" height="32" fill="{"#fff0d7" if j==2 else "#dcece4"}" stroke="#a7bdb8"/><text x="{124+111*j}" y="{393+40*i}">{t}</text>')
d.text(32,511,'Toy hidden table: 3 × 3 × 768');d.arrow(526)
d.box(557,112,'Mix columns within each row',['Each row combines its cells.','Read labeled support across rows.'],'#e1eee6');d.arrow(669)
d.box(700,112,'Repeat attention blocks',['Base: depth 12, width 768','Default shares weights across depth.']);d.arrow(812)
d.box(843,137,'Decode query target',['classification: probabilities over classes','binning regression: Σ p(bin) × bin value','repeat contexts → average 8 predictions']);d.arrow(980)
d.box(1011,127,'New task, same weights',['Support labels influence predictions.','No query answers enter the model.','Original benchmark: source-gated.'],'#fff0d7');d.save('contexttab')
d=Diagram('TabSTAR · target-aware transfer',1250)
d.box(58,112,'Known row + all candidates',['features: volume, country','candidates: quality low AND high']);d.arrow(170)
d.box(201,137,'Verbalize each element',['text: “country: France”','number: name + bin + quantile','numeric path keeps the precise z-score']);d.arrow(338)
d.box(369,112,'Two encoders, width 384',['string → e5-small-v2 → semantic vector','scalar → small MLP → numeric vector']);d.arrow(481)
# Pair fusion not merely a name box.
d.parts.append('<rect class="box" x="18" y="512" width="424" height="183" rx="9"/>');d.text(30,542,'Per-element fusion: two-token attention')
for x,t in [(38,'text: 384'),(258,'number: 384')]:d.parts.append(f'<rect x="{x}" y="562" width="166" height="38" fill="#dcece4"/>');d.text(x+9,587,t)
d.parts.append('<path class="arrow" d="M121 605 L213 643 M340 605 L247 643"/>');d.text(97,670,'average outputs → 384');d.arrow(695)
d.box(726,137,'Six feature-interaction layers',['2 features + 2 candidates → 4 × 384','all element tokens attend within row','no feature-position encoding'],'#e1eee6');d.arrow(863)
d.box(894,137,'Shared head → class scores',['low token → score; high token → score','softmax(scores) → class probabilities','true query class is never an input']);d.arrow(1031)
d.box(1062,159,'Adaptation uses gradients',['Pretraining updates upper text layers.','Downstream: train LoRA adapters.','Eligible encoder layers can adapt.','B07 explains this; no TabSTAR fit here.'],'#edf0fb');d.save('tabstar')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=json.loads((P/'evidence/b07/course-audit.json').read_text())
fig,axes=plt.subplots(3,1,figsize=(4.6,8.6),sharey=True)
for ax,dataset,title in zip(axes,['wine_pl','wine_dot_com_prices','wine_vivino_price'],['Wine Poland','Wine.com','Vivino']):
 for j,arm in enumerate(['meaningful','anonymous','numeric_only']):
  vals=[x['r2'] for x in r['rows'] if x['dataset']==dataset and x['arm']==arm]
  ax.scatter([j-.1,j,j+.1],vals,s=40,color=['#287a68','#426fa3','#ab7433'][j]);ax.plot([j-.2,j+.2],[sum(vals)/3]*2,color='black')
 ax.set_xticks([0,1,2],['Meaningful','Anonymous','Numeric\nonly']);ax.set_title(title);ax.set_ylabel('Test R²');ax.axhline(0,color='#999',linestyle='--',lw=1);ax.grid(axis='y',alpha=.18)
fig.suptitle('27 fresh course fits\nPoints: split seeds · bars: means',fontsize=13);fig.tight_layout();fig.savefig(F/'results.png',dpi=160);plt.close(fig)
print('Three architecture SVGs/PNGs and measured result plot generated')
