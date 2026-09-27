"""Portable computation diagrams and measured results; no illustrative fake measurements."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).parent;D=P/'figures/l134';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l134','axes.spines.top':False,'axes.spines.right':False})
navy='#17324d';teal='#007f82';amber='#b85d20'
def save(fig,name):
 fig.savefig(D/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});fig.savefig(D/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
def box(ax,x,y,w,h,title,body,color=navy):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015',facecolor='#f1f6f8',edgecolor=color,lw=1.7));ax.text(x+.02,y+h-.055,title,color=color,weight='bold',va='top');ax.text(x+.02,y+h-.14,body,va='top',fontsize=10,linespacing=1.5,color=navy)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=teal,lw=2))
f,ax=plt.subplots(figsize=(12,6));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title('One temporal mini-batch: preserve identity while shrinking computation',loc='left',weight='bold',pad=20)
box(ax,.02,.57,.27,.35,'HOST · full relational graph','All rows + typed FK edges\nNode timestamps / row features\nAdjacency stays on host')
box(ax,.37,.57,.27,.35,'CPU · temporal sampler','B × (entity, cutoff, label)\nIncoming relations; [f₁, f₂]\nOne root cutoff at every hop',teal)
box(ax,.72,.57,.26,.35,'SAMPLED COORDINATES','n_id → original row\nbatch → query owner\ninput_id → task row',teal)
box(ax,.02,.08,.27,.34,'DEVICE · encode rows','F1: Frame + time encoders\nScale: constant + log age\nNₜ × width per node type')
box(ax,.37,.08,.27,.34,'DEVICE · two typed layers','Relation sum-SAGE → sum\nNode LayerNorm → ReLU\nL133 arithmetic on sampled rows')
box(ax,.72,.08,.26,.34,'SEEDS · head and update','First B entity embeddings\nB predictions ↔ B query labels\nLoss → backward → optimizer',amber)
arrow(ax,(.30,.75),(.36,.75));arrow(ax,(.65,.75),(.71,.75));arrow(ax,(.85,.55),(.85,.46));arrow(ax,(.85,.46),(.16,.46));ax.text(.5,.50,'Sampled tensors cross to GPU; synchronize to measure transfer',ha='center',fontsize=10,color=teal);arrow(ax,(.16,.46),(.16,.43));arrow(ax,(.30,.25),(.36,.25));arrow(ax,(.65,.25),(.71,.25));save(f,'batch')
f,ax=plt.subplots(figsize=(10,3.8));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1));ax.set_title('Incoming relations create a typed expansion tree',loc='left',weight='bold',pad=15)
box(ax,.02,.3,.25,.47,'HOP 0 · 2 user queries','2 root occurrences\nQuery identities stay separate')
box(ax,.37,.3,.25,.47,'HOP 1 · fanout 3','2 × 3 = 6 orders\norders → users',teal)
box(ax,.72,.3,.26,.47,'HOP 2 · fanout 2','6 × 2 = 12 items\n6 × 2 = 12 users',teal)
arrow(ax,(.28,.55),(.36,.55));arrow(ax,(.63,.55),(.71,.55));ax.text(.5,.12,'2 + 6 + 12 + 12 = 32 occurrences · 30 edges · one width-128 matrix = 16 KiB',ha='center',color=navy);save(f,'frontier')
s=json.loads((P/'evidence/l134/summary.json').read_text());f,axes=plt.subplots(1,2,figsize=(10,3.8))
for ax,k in zip(axes,['val','test']):
 values=[r[k] for r in s['seeds']];ax.scatter(range(5),values,color=teal,s=55,label='Fresh full fit');ax.axhline(s['metrics'][k]['target'],color=amber,linestyle='--',label='Published mean');ax.set(title=k.title()+' MAE',xlabel='Seed',ylabel='MAE');ax.set_xticks(range(5));ax.legend(fontsize=9)
f.suptitle('F1 selected reproduction · five complete ten-epoch fits',weight='bold');f.tight_layout();save(f,'scores')
scale_path=P/'evidence/l134/scale/scale.json'
if scale_path.exists():
 s=json.loads(scale_path.read_text());rows=s['configurations'];f,axes=plt.subplots(1,3,figsize=(12,4));labels=[f"B={r['batch_size']}\n{r['fanouts']}" for r in rows]
 axes[0].bar(labels,[r['summary']['queries_per_second'] for r in rows],color=teal);axes[0].set(title='Core throughput',ylabel='Seed queries / second')
 axes[1].bar(labels,[r['summary']['peak_allocated_bytes']/2**20 for r in rows],color=navy);axes[1].set(title='Peak allocated GPU memory',ylabel='MiB')
 bottom=[0.]*len(rows)
 for key,label,color in [('sample_s','Sample',navy),('transfer_s','Transfer',amber),('step_s','Train step',teal)]:
  v=[r['summary'][key] for r in rows];axes[2].bar(labels,v,bottom=bottom,label=label,color=color);bottom=[x+y for x,y in zip(bottom,v)]
 axes[2].set(title='Measured pass time',ylabel='Seconds');axes[2].legend(fontsize=8)
 f.suptitle('Full rel-stack topology · simplified width-32 predictor · 2,048 fixed queries',weight='bold');f.tight_layout();save(f,'measurements')
