"""Native architecture/mask diagrams and standalone measured plots."""
from pathlib import Path
import html,json
import cairosvg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size':12,'axes.titlesize':13,'legend.fontsize':10})
P=Path(__file__).resolve().parent;F=P/'figures/b08';F.mkdir(exist_ok=True)
class Diagram:
 def __init__(self,title,height):
  self.h=height;self.boxes=[];self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="440" height="{height}" viewBox="0 0 440 {height}" role="img"><title>{html.escape(title)}</title><rect width="440" height="{height}" rx="16" fill="#f2f5f1"/><style>text{{font-family:DejaVu Sans,sans-serif;fill:#173e36;font-size:18px}}.title{{font-size:21px;font-weight:bold}}.arrow{{stroke:#397761;stroke-width:2;fill:none}}</style><text x="18" y="32" class="title">{html.escape(title)}</text>']
 def box(self,y,title,lines,fill='#ffffff'):
  h=52+len(lines)*25;self.boxes.append((18,y,404,h));self.parts.append(f'<rect x="18" y="{y}" width="404" height="{h}" rx="9" fill="{fill}" stroke="#b5cbc1"/><text x="30" y="{y+29}" class="title">{html.escape(title)}</text>')
  for j,line in enumerate(lines):self.parts.append(f'<text x="30" y="{y+56+j*25}">{html.escape(line)}</text>')
  return y+h
 def arrow(self,y):self.parts.append(f'<path class="arrow" d="M220 {y+3} v23 m-6 -7 l6 7 l6 -7"/>')
 def save(self,name):
  for x,y,w,h in self.boxes:assert x>=0 and y>=0 and x+w<=440 and y+h<=self.h
  for a,b in zip(self.boxes,self.boxes[1:]):assert a[1]+a[3]<=b[1]
  self.parts.append('</svg>');(F/(name+'.svg')).write_text(''.join(self.parts));cairosvg.svg2png(url=str(F/(name+'.svg')),write_to=str(F/(name+'.png')),scale=2)
d=Diagram('LimiX-16M · published target',850)
y=d.box(54,'1  Supply observed evidence',['Support: known X and y','Query: observed X; hidden y','Selected task: hide 5% of query X']);d.arrow(y)
y=d.box(y+32,'2  Represent each cell',['Separate X / y encoders','Column code: rank 48 → width 192','Missing value → mask embedding']);d.arrow(y)
y=d.box(y+32,'3  Repeat 12 dual-axis blocks',['Mix columns within each row','Mix rows within each column','2 feature passes : 1 sample pass','Attention + feedforward + residual'],'#dcece4');d.arrow(y)
y=d.box(y+32,'4  Decode hidden quantities',['Feature head → missing X values','Target head → y predictions']);d.arrow(y)
d.box(y+32,'5  Score the selected cells',['Table 23: normalized feature RMSE','Truth is used only by the scorer','Historical packet: source-gated'],'#e8eafa');d.save('limix16m')
d=Diagram('LimiX-2 · separate pathways',1060)
y=d.box(54,'1  Encode the observed table',['Feature cells: N × F × 256','Task slots: N × 4 × 256','Query targets replaced by MASK','Missing X + column code retained']);d.arrow(y)
y=d.box(y+32,'2  Attend across rows',['X: each column attends over rows','Y: join 4 slots → width 1024','Both read support keys only','Residual preserves own query X'],'#dcece4');d.arrow(y)
y=d.box(y+32,'3  Gate each pathway',['Independent feature / task SwiGLU','gate = SiLU(Wg z + bg)','output = Wo(gate ⊙ (Wv z + bv))']);d.arrow(y)
y=d.box(y+32,'4  Attend within each row',['Feature queries → X and task slots','Task queries → X only','Separate feature / task projections','Repeat steps 2–4 across 24 blocks'],'#dcece4');d.arrow(y)
y=d.box(y+32,'5  Read out at two depths',['Shallow X → adapter → reconstruct','Final task slots → adapter → target','Class logits or 5000 regression bins']);d.arrow(y)
d.box(y+32,'Course mechanism differs',['2 blocks, width 16, scalar MSE heads','Same access rules; no weight parity'],'#e8eafa');d.save('limix2')
# Portable masks: rows read columns, not the reverse.
fig,axes=plt.subplots(2,1,figsize=(4.4,7),layout='constrained');fig.patch.set_facecolor('#f2f5f1')
for ax,mat,labels,title in [(axes[0],np.array([[1,1,0,0]]*4),['S1','S2','Q1','Q2'],'Across rows'),(axes[1],np.array([[1,1,1],[1,1,1],[1,1,0]]),['X1','X2','T'],'Within one row')]:
 ax.imshow(mat,cmap=matplotlib.colors.ListedColormap(['#eee7e1','#7eaf98']),vmin=0,vmax=1)
 ax.set_xticks(range(len(labels)),labels);ax.set_yticks(range(len(labels)),labels);ax.set_title(title);ax.set_xlabel('Key read');ax.set_ylabel('Query reads')
 for (i,j),v in np.ndenumerate(mat):ax.text(j,i,'yes' if v else '—',ha='center',va='center',fontsize=12)
fig.suptitle('Access rules · T summarizes task slots',fontsize=12);fig.savefig(F/'masks.png',dpi=160);plt.close(fig)
r=json.loads((P/'evidence/b08/course-audit.json').read_text());colors=['#2b735d','#ab693a','#5365ab'];arms=['target','feature','combined']
fig,axes=plt.subplots(2,1,figsize=(4.4,6.2),layout='constrained');fig.patch.set_facecolor('#f2f5f1')
for ax,metric,title in zip(axes,['target_mse','feature_mse'],['Target prediction','Masked-feature reconstruction']):
 for seed in range(3):
  vals=[next(x[metric] for x in r['rows'] if x['seed']==seed and x['objective']==a) for a in arms]
  ax.plot(range(3),vals,'o-',alpha=.8,label=f'Seed {seed}',color=colors[seed])
 base=np.mean([x[metric] for x in r['baselines']]);ax.axhline(base,color='#555',ls='--',label='Support-mean baseline')
 ax.set_xticks(range(3),['Target only','Feature only','Combined']);ax.set_ylabel('MSE ↓');ax.set_ylim(bottom=0);ax.set_title(title);ax.grid(axis='y',alpha=.2)
axes[0].legend(fontsize=9);fig.suptitle('Measured course results · same synthetic family',fontsize=11);fig.savefig(F/'results.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(4.4,3.5),layout='constrained');fig.patch.set_facecolor('#f2f5f1')
for p in r['paired']:ax.scatter(p['seed'],p['combined_minus_target'],s=70,color=colors[p['seed']])
ax.axhline(0,color='#555',ls='--');ax.set_xticks([0,1,2],['Seed 0','Seed 1','Seed 2']);ax.set_ylabel('Combined − target-only MSE');ax.set_title('Paired target effect · below zero helps');ax.grid(axis='y',alpha=.2);fig.savefig(F/'paired.png',dpi=170);plt.close(fig)
print('Architecture geometry and five figure exports PASS')
