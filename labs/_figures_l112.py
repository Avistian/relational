"""Deterministic portable SVG/PNG figures for the reproduction lesson."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
P=Path(__file__).resolve().parent;out=P/'figures/l112';out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.hashsalt':'l112','savefig.facecolor':'#fafaf7'})
def save(fig,name):
 fig.savefig(out/f'{name}.svg',metadata={'Date':None},bbox_inches='tight');fig.savefig(out/f'{name}.png',dpi=160,bbox_inches='tight',metadata={'Software':'L112'});plt.close(fig)
def box(ax,x,y,w,h,text,color='#e3eeec'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.015',facecolor=color,edgecolor='#52706c'));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
fig,ax=plt.subplots(figsize=(11,7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title('GCN on ogbn-arxiv · the complete prediction path',loc='left',fontweight='bold',pad=18)
box(ax,.02,.78,.40,.15,'All paper features\n169,343 × 128')
box(ax,.57,.78,.40,.15,'Directed citations → binary undirected\nAdd loops; cache S = D⁻½(A + I)D⁻½')
box(ax,.16,.54,.68,.16,'S X W₁ᵀ + b₁ → BatchNorm → ReLU → Dropout(0.5)\n169,343 × 256')
box(ax,.16,.31,.68,.16,'S H₁ W₂ᵀ + b₂ → BatchNorm → ReLU → Dropout(0.5)\n169,343 × 256')
box(ax,.16,.08,.68,.16,'S H₂ W₃ᵀ + b₃ → log-softmax → argmax class\n169,343 × 40 → 169,343 predictions')
for x,y,xx,yy in [(.22,.78,.35,.70),(.75,.78,.65,.70),(.5,.54,.5,.47),(.5,.31,.5,.24)]:ax.annotate('',(xx,yy),(x,y),arrowprops={'arrowstyle':'->','lw':2,'color':'#21766e'})
ax.text(.5,-.025,'Train: NLL on 90,941 labels + Adam  ·  Select: validation  ·  Predict: saved weights AND BN buffers',ha='center',fontsize=10)
save(fig,'architecture')
fig,ax=plt.subplots(figsize=(11,5));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1));ax.set_title('Visibility contract · graph inputs and label roles',loc='left',fontweight='bold')
for x,title,count,years,col in [(.015,'TRAIN','90,941','≤ 2017','#e3eeec'),(.35,'VALIDATION','29,799','2018','#f4e6ca'),(.685,'TEST','48,603','≥ 2019','#e3e8f0')]:
 box(ax,x,.58,.30,.27,f'{title}\n{count} nodes · {years}',col)
box(ax,.015,.30,.97,.17,'Features + citation structure from ALL three groups → full-graph propagation + training BatchNorm')
for x,text in [(.165,'Train labels\nNLL gradients'),(.5,'Validation labels\nfirst best epoch'),(.835,'Test labels\nreport selected predictor')]:ax.text(x,.12,text,ha='center',va='center');ax.annotate('',(x,.49),(x,.58),arrowprops={'arrowstyle':'->'})
save(fig,'visibility')
fig,ax=plt.subplots(figsize=(10,5));summary=P/'evidence/l112/summary.json'
if summary.exists():
 r=json.loads(summary.read_text());scores=np.array([x['test_percent'] for x in r['seeds']]);ax.scatter(np.arange(10),scores,color='#21766e',s=55,label='Fresh selected checkpoints');mean=scores.mean();sd=scores.std(ddof=1);ax.errorbar([11],[mean],yerr=[sd],fmt='o',color='#af6333',capsize=8,label='Mean ± sample seed SD');ax.axhline(71.74,color='#384b6a',ls='--',label='Published test mean 71.74%');ax.axhspan(71.24,72.24,color='#384b6a',alpha=.07,label='Predeclared ±0.5 pp band');ax.set_xticks(list(range(10))+[11],list(map(str,range(10)))+['Mean']);ax.set_ylabel('Test accuracy (%)');ax.set_xlabel('Seed · each 500 epochs, selected by validation');ax.legend(fontsize=9)
else:ax.text(.5,.5,'Full ten-run experiment pending\nNo measured mean claimed',ha='center',va='center',transform=ax.transAxes);ax.set_axis_off()
ax.set_title('One fixed graph, ten fresh fits',loc='left',fontweight='bold');save(fig,'results')
