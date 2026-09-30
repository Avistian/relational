"""Deterministic model-specific computation diagrams and measured paired results."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;D=P/'figures/l142';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l142'})
INK='#16324a';TEAL='#087e83';ORANGE='#ae541e'
def canvas(w,h):
 f,a=plt.subplots(figsize=(w,h));a.set(xlim=(0,w),ylim=(0,h));a.axis('off');return f,a
def box(a,x,y,w,h,text,color=TEAL,size=11):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.03',facecolor='#f0f7f7',edgecolor=color,linewidth=1.5);a.add_patch(p)
 t=a.text(x+w/2,y+h/2,text,ha='center',va='center',color=INK,fontsize=size);a._pairs=getattr(a,'_pairs',[])+[(p,t)]
def arrow(a,x,y,u,v):a.annotate('',(u,v),(x,y),arrowprops={'arrowstyle':'->','color':INK,'lw':1.6})
def save(f,name):
 f.canvas.draw();renderer=f.canvas.get_renderer()
 for a in f.axes:
  for p,t in getattr(a,'_pairs',[]):
   b=p.get_window_extent(renderer);q=t.get_window_extent(renderer)
   assert b.x0<=q.x0 and b.x1>=q.x1 and b.y0<=q.y0 and b.y1>=q.y1,(name,t.get_text())
 for ext in ['svg','png']:f.savefig(D/f'{name}.{ext}',dpi=145,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else None)
 plt.close(f)
f,a=canvas(10,6.8);a.text(.15,6.4,'COUNT THE WALKS BEFORE LEARNING WEIGHTS',weight='bold',fontsize=15,color=INK)
for x,label in [(.2,'SOURCE s = 2'),(3.6,'FACT f = 1'),(7,'DESTINATION d = 3')]:box(a,x,5.2,2.8,.65,label)
arrow(a,3,5.52,3.55,5.52);arrow(a,6.4,5.52,6.95,5.52)
box(a,.2,3.6,4.4,.95,'Ordinary layer 1\nf₁ = s + f + d = 6; d₁ = d + f = 4',ORANGE)
box(a,5.2,3.6,4.5,.95,'Ordinary layer 2\nd₂ = d₁ + f₁ = s + 2f + 2d = 10',ORANGE)
arrow(a,4.6,4.07,5.15,4.07)
a.text(.25,2.6,'Source: s → f → d (once)\nFact: f → f → d and f → d → d (twice)\nDestination: d → d → d and d → f → d (twice)',color=INK,linespacing=1.6)
box(a,.2,.75,9.5,.9,'Composite diagnostic: fuse u = s + f = 3; then d′ = d + u = 6\nOne source contribution, one fact contribution, one destination root',TEAL)
a.text(.2,.12,'Identity weights + self sums; a mechanism fixture, not the trained attention network.',fontsize=10,color=INK);save(f,'walks')
f,a=canvas(10,6.4);a.text(.2,6,'SAME SHARED SUM · DIFFERENT SOURCE ROLE',weight='bold',fontsize=15,color=INK)
box(a,.2,4.4,4.4,.95,'World A\nsource s = 2; third role n = 8')
box(a,5.2,4.4,4.5,.95,'World B\nsource s = 8; third role n = 2')
box(a,.2,2.7,9.5,.85,'Ordinary: s + 2f + 2d + n = 18 in both worlds\nf = 1, d = 3; the shared scalar cannot distinguish A from B',ORANGE)
arrow(a,2.4,4.4,2.4,3.58);arrow(a,7.45,4.4,7.45,3.58)
box(a,.2,1.3,4.4,.7,'Source-specific: d + f + s = 6')
box(a,5.2,1.3,4.5,.7,'Source-specific: d + f + s = 12')
a.text(.25,.38,'If target = s: a harmful collision. If target = s + n: no target information is lost.\nSeparate learned role channels can also distinguish these worlds in an ordinary GNN.',fontsize=11,color=INK);save(f,'collision')
f,a=canvas(10,12);a.text(.2,11.6,'TWO-EDGE REACH · TWO DIFFERENT COMPUTATIONS',weight='bold',fontsize=15,color=INK)
box(a,.3,10.15,9.3,.9,'Full F1 tables + query (driver, cutoff)\nTemporal sampling: 128 / 64 neighbors; 512 query roots')
arrow(a,4.95,10.15,4.95,9.7)
box(a,.3,8.75,9.3,.9,'Same row-encoder design → [sampled rows, 128]\nAdd per-query relative-time embeddings; preserve owner cutoff')
arrow(a,2.45,8.75,2.45,8.1);arrow(a,7.4,8.75,7.4,8.1)
box(a,.3,6.35,4.3,1.7,'COMPOSITE · one layer\nsource + fact → route-specific fusion\ndestination query → attention\n4 heads × 128 → 512 → 128',TEAL,10.5)
box(a,5.25,6.35,4.3,1.7,'ORDINARY · layer 1\nattention on every directed edge type\nfact combines all adjacent roles\n4 heads × 128 → 512 → 128',ORANGE,10.5)
arrow(a,7.4,6.35,7.4,5.95)
box(a,5.25,4.45,4.3,1.45,'Sum relations → LayerNorm → ReLU\nORDINARY · layer 2\nupdated fact → driver attention\nsource now reaches the destination',ORANGE,10.5)
arrow(a,2.45,6.35,2.45,3.75);arrow(a,7.4,4.45,7.4,3.75)
box(a,.3,2.8,9.3,.9,'Sum route/relation outputs → per-node LayerNorm → ReLU\nSelect query driver roots [B,128] → scalar head [B,1]')
arrow(a,4.95,2.8,4.95,2.3)
box(a,.3,1.2,9.3,1.05,'Train: unclipped absolute error, Adam 0.005, 10 full epochs\nSelect: first validation minimum · Evaluate: train-target 2/98% clipping\nSame data/protocol; different parameter counts and random trajectories')
a.text(.25,.35,'The ordinary arm is a course comparator. Equal graph reach is not equal model capacity.\nThe full code retains RelGNN fact updates and per-edge destination skip terms.',fontsize=11,color=INK);save(f,'architecture')
summary=json.loads((P/'evidence/l142/training.json').read_text());fig,axs=plt.subplots(1,2,figsize=(10,4.5))
for i,arm in enumerate(['composite','ordinary']):
 values=[json.loads((P/f'evidence/l142/{arm}-{s}/result.json').read_text())['scores']['test']['mae'] for s in range(5)]
 axs[0].scatter([i]*5,values,color=TEAL if i==0 else ORANGE,s=45,zorder=3)
 for seed,v in enumerate(values):axs[0].annotate(str(seed),(i+.035,v),fontsize=9)
axs[0].axhline(3.798,color=INK,ls='--',label='Published RelGNN mean 3.798');axs[0].set(xticks=[0,1],xticklabels=['Composite','Ordinary'],ylabel='Test MAE · lower is better',xlim=(-.5,1.5));axs[0].legend(fontsize=9)
gaps=summary['paired']['test']['seed_gaps'];axs[1].bar(range(5),gaps,color=[TEAL if v>=0 else ORANGE for v in gaps]);axs[1].axhline(0,color=INK);axs[1].set(xlabel='Paired seed label',ylabel='Ordinary − composite MAE',title='Positive favors composite')
fig.suptitle('Five complete fresh fits per arm · descriptive course comparison',fontsize=13);fig.tight_layout();save(fig,'results')
print('Four deterministic figures built')
