"""Model-specific native vector maps and portable counterparts."""
from pathlib import Path
import html,json
import cairosvg
P=Path(__file__).resolve().parent;F=P/'figures/b07a';F.mkdir(parents=True,exist_ok=True)
class Diagram:
 def __init__(self,title,h):
  self.h=h;self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="460" height="{h}" viewBox="0 0 460 {h}" role="img"><title>{html.escape(title)}</title><rect width="460" height="{h}" rx="16" fill="#f3f5f0"/><style>text{{font-family:DejaVu Sans,sans-serif;font-size:19px;fill:#213d39}}.title{{font-size:22px;font-weight:bold}}.arrow{{stroke:#287a68;stroke-width:2;fill:none}}</style><text x="20" y="36" class="title">{html.escape(title)}</text>']
 def box(self,y,title,lines,fill='white'):
  h=55+27*len(lines);self.parts.append(f'<rect x="18" y="{y}" width="424" height="{h}" rx="10" fill="{fill}" stroke="#a7bdb4"/><text class="title" x="30" y="{y+29}">{html.escape(title)}</text>')
  for i,line in enumerate(lines):self.parts.append(f'<text x="30" y="{y+59+i*27}">{html.escape(line)}</text>')
  return y+h
 def arrow(self,y):self.parts.append(f'<path class="arrow" d="M230 {y+3} v22 m-6 -7 l6 7 l6 -7"/>');return y+32
 def save(self,name):
  s=''.join(self.parts)+'</svg>';(F/(name+'.svg')).write_text(s);cairosvg.svg2png(bytestring=s.encode(),write_to=str(F/(name+'.png')))
d=Diagram('MotherNet · compact child MLP',960)
y=d.box(60,'A labeled table becomes tokens',['n rows; pad features to 100','feature projection + class encoding']);y=d.arrow(y)
y=d.box(y,'12 transformer layers',['Support rows interact at width 512.','Pool by class → 10 × 512 summary.'],'#deebe4');y=d.arrow(y)
y=d.box(y,'Decode variable weight factors',['Task-specific P: 512 × 32','Meta-learned fixed F: 32 × 100']);y=d.arrow(y)
y=d.box(y,'One generated layer, unpacked',['x[100] → F x[32] → P(F x)[512]','Add bias, then ReLU.','Repeat with second 512-wide layer.'],'#e7eafa');y=d.arrow(y)
y=d.box(y,'Query → child → probabilities',['Two hidden layers; rank 32 factors.','Keep factors, biases and preprocessing.','The query does not reread support.']);y=d.arrow(y)
d.box(y,'Pretraining is a separate loop',['Synthetic tasks → query loss → θ update','B07a: architecture only, NOT_RUN.'],'#fff0d6');d.save('mothernet')
d=Diagram('HyperFast · generate, then serve',1140)
y=d.box(60,'Support: labeled rows',['Train-only scaler; sample 512 rows.','Repeat to 1024 for the 784-d PCA.']);y=d.arrow(y)
y=d.box(y,'Random features → fitted PCA',['X[1024,d] → ReLU(X R)[1024,32768]','Center → project → Z[1024,784]']);y=d.arrow(y)
y=d.box(y,'Condition on table and class',['[zᵢ, mean(z), mean(z | yᵢ), onehot(yᵢ)]','First hypernetwork input: 2398 values.'],'#deebe4');y=d.arrow(y)
y=d.box(y,'Generate hidden layers',['Shared MLP → mean over support','Linear decoder → matrix 784 × 784','Then next layer sees prior activations.']);y=d.arrow(y)
y=d.box(y,'Generate each output column',['Mean generated vector for class k','+ mean hidden vector for class k','Stack → output matrix 784 × K + bias'],'#e7eafa');y=d.arrow(y)
y=d.box(y,'Query → RF/PCA → main MLP',['Hidden layers 784 → 784 + residual','Output logits[K] → softmax']);y=d.arrow(y)
y=d.box(y,'Optional two-space 1-NN',['Keep support inputs and labels.','Add learned bias at each nearest class.','B07a varies only this correction.'],'#fff0d6');d.save('hyperfast')
d=Diagram('iLTM · weights plus context',1080)
y=d.box(60,'Fit the task representation',['GBDT leaves and/or robust features','One-hot leaf path: leaf 2 → [0,1,0]']);y=d.arrow(y)
y=d.box(y,'Project to a fixed width',['Randomized embedding → width 512','Keep fitted trees and projection state.']);y=d.arrow(y)
y=d.box(y,'Sequential hypernetwork',['Generation rows + labels + summaries','Main MLP: width 512, three layers.'],'#deebe4');y=d.arrow(y)
y=d.box(y,'Two paths, one representation',['Query → MLP → main logits','Context → hidden vectors + known labels']);y=d.arrow(y)
y=d.box(y,'Retrieve a second score',['Cosine similarity S[q,c]','S × context one-hot labels / τ','Mix: (1−α) main + α retrieval'],'#e7eafa');y=d.arrow(y)
y=d.box(y,'Serve the complete model',['α > 0 retains contextual dependence.','Optional fine-tuning/ensemble add cost.']);y=d.arrow(y)
d.box(y,'Evidence boundary',['Real-table meta-training in the paper.','B07a: architecture only, NOT_RUN.'],'#fff0d6');d.save('iltm')
# Results are added after the full run, on aligned scales.
if (P/'evidence/b07a/course-audit.json').exists():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 r=json.loads((P/'evidence/b07a/course-audit.json').read_text())
 fig,axes=plt.subplots(3,1,figsize=(5,8),sharey=True)
 for ax,name in zip(axes,['banknote','phoneme','diabetes']):
  for seed in [0,1,2]:
   vals=[next(x['balanced_accuracy'] for x in r['rows'] if x['dataset']==name and x['seed']==seed and x['arm']==a) for a in ['weights_only','retrieval']]
   ax.plot([0,1],vals,marker='o',label=f'split {seed}',alpha=.8)
  ax.set(title=name,ylabel='Balanced accuracy',xticks=[0,1],xticklabels=['Weights only','+ retrieval'],ylim=(.45,1.025));ax.grid(axis='y',alpha=.2)
 axes[0].legend(fontsize=9);fig.suptitle('Same generated weights, two query paths');fig.tight_layout();fig.savefig(F/'quality.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(3,1,figsize=(5,8))
 for ax,name in zip(axes,['banknote','phoneme','diabetes']):
  for arm,color in [('weights_only','#287a68'),('retrieval','#b27533')]:
   vals=[x for x in r['timings'] if x['dataset']==name and x['arm']==arm];ax.plot([x['query_rows'] for x in vals],[1000*x['median_seconds'] for x in vals],marker='o',label=arm,color=color)
  ax.set(title=name,xlabel='Rows per prediction call',ylabel='Median wall time (ms)',xscale='log');ax.grid(alpha=.2);ax.legend(fontsize=9)
 fig.suptitle('Measured CPU query cost\n9 measurements per point; construction excluded');fig.tight_layout();fig.savefig(F/'timing.png',dpi=150);plt.close(fig)
print('B07a diagrams generated')
