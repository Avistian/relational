"""Deterministic checkpoint figures, tied to fresh L140 evidence only."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;D=P/'figures/l140';D.mkdir(parents=True,exist_ok=True);E=P/'evidence/l140'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l140'})
blue='#17324d';teal='#007f82';orange='#b85d20'
def save(f,n):
 f.canvas.draw();renderer=f.canvas.get_renderer()
 for ax in f.axes:
  for patch,label in getattr(ax,'_l140_pairs',[]):
   a=patch.get_window_extent(renderer);b=label.get_window_extent(renderer)
   assert a.x0<=b.x0 and b.x1<=a.x1 and a.y0<=b.y0 and b.y1<=a.y1,(n,label.get_text())
 for ext in ['svg','png']:f.savefig(D/f'{n}.{ext}',dpi=145,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else None)
 for path in D.glob(n+'.svg'):
  path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
 plt.close(f)
def box(a,x,y,w,h,text,color=teal):
 patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02',edgecolor=color,facecolor='#eff7f6',linewidth=1.5);a.add_patch(patch);label=a.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11,color=blue)
 if not hasattr(a,'_l140_pairs'):a._l140_pairs=[]
 a._l140_pairs.append((patch,label))
def arrow(a,x,y,u,v):a.annotate('',xy=(u,v),xytext=(x,y),arrowprops=dict(arrowstyle='->',color=teal,lw=2))
f,a=plt.subplots(figsize=(9,7));a.set(xlim=(0,10),ylim=(0,8));a.axis('off');a.set_title('One RDL stack; two explicitly different contracts',loc='left',fontweight='bold',color=blue)
box(a,.1,6.5,4.5,1,'AMAZON · product → review → customer\nsum neighbors · fanout 128/64')
box(a,5.2,6.5,4.5,1,'TRIAL · facility → association → study\nmean neighbors · fanout 64/32')
arrow(a,2.4,6.45,2.4,5.9);arrow(a,7.4,6.45,7.4,5.9)
box(a,.1,4.8,9.6,1.05,'Typed columns + pinned GloVe [N_text, 300]\nper-table ResNet encoders → h_type [N_type, 128] + owner-relative time')
arrow(a,4.9,4.75,4.9,4.25)
box(a,.1,3.1,9.6,1.1,'Scalar illustration before affine map: neighbors [2, 6] → sum 8 / mean 4\nGraphSAGE relation outputs SUM → LayerNorm → ReLU (two layers)\nEach timestamp ≤ its own query cutoff; vector width 128')
arrow(a,4.9,3.05,4.9,2.6)
box(a,.1,1.65,4.5,.9,'Query roots [B,128]\nlinear head → logits [B,1]');box(a,5.2,1.65,4.5,.9,'Sigmoid → probabilities [B]\nvalidation AUROC → first best epoch')
arrow(a,4.65,2.1,5.15,2.1)
box(a,.1,.1,9.6,1,'BCE gradients train row encoders + GNN + head\nFresh weights per seed · Adam .005 / .0001 · 10 / 20 source epochs',orange)
arrow(a,2.35,1.6,2.35,1.15);save(f,'architecture')
f,a=plt.subplots(figsize=(8.5,4.4));a.axis('off');a.set(xlim=(0,10),ylim=(0,5));a.set_title('Three questions; three independent answers',loc='left',fontweight='bold',color=blue)
for y,title,value,color in [(3.55,'All planned runs completed?','5 / 5 → COMPLETE',teal),(2.1,'Mean inside the frozen\n±1pp band?','Gap 0.2pp → CLOSE',teal),(.65,'Does available protocol\nevidence match?','24,172 fewer training rows\nGAPPED',orange)]:
 box(a,.1,y,4.5,.85,title,color);box(a,5.2,y,4.5,.85,value,color);arrow(a,4.65,y+.425,5.15,y+.425)
a.text(.15,.12,'Illustrative score gap; actual Amazon count discrepancy. Historical identity stays unestablished.',fontsize=10,color=blue);save(f,'evidence')
if (E/'training.json').exists():
 summary=json.loads((E/'training.json').read_text())
 f,axes=plt.subplots(1,2,figsize=(9,4.4),sharey=False)
 for ax,task in zip(axes,['amazon','trial']):
  for split,x,c in [('val',0,teal),('test',1,orange)]:
   r=summary['tasks'][task]['metrics'][split];values=np.array(list(r['values'].values()))*100
   ax.scatter(x+np.linspace(-.12,.12,5),values,color=c,label=split,zorder=3)
   ax.errorbar(x,100*r['mean'],yerr=100*r['sample_sd'],color=c,fmt='D',capsize=7)
   ax.plot([x-.28,x+.28],[100*r['paper_target']]*2,'--',color=blue)
  ax.set(title=task.capitalize(),xticks=[0,1],xticklabels=['Validation','Test'],ylabel='AUROC (%)');ax.grid(axis='y',alpha=.2)
 f.suptitle('Fresh runs: dots = seeds; diamond ± sample SD; dashed = paper mean',fontsize=12)
 f.text(.1,.005,'Amazon: training-population gap. Separate y scales show seed spread; not an equivalence test.',fontsize=10);f.tight_layout(rect=(0,.04,1,.93));save(f,'scores')
 f,axes=plt.subplots(2,1,figsize=(9,6.5))
 for ax,task in zip(axes,['amazon','trial']):
  for seed in range(5):
   r=json.loads((E/f'{task}/seed-{seed}/result.json').read_text());ys=[100*h['val']['roc_auc'] for h in r['history']];xs=np.arange(1,len(ys)+1);line,=ax.plot(xs,ys,alpha=.8,label=f'seed{seed}')
   ax.scatter([r['best_epoch']],[ys[r['best_epoch']-1]],color=line.get_color(),s=40)
  ax.set(title=task.capitalize()+' — measured validation histories',xlabel='Epoch',ylabel='AUROC (%)');ax.grid(alpha=.2);ax.legend(ncol=5,fontsize=9)
 f.tight_layout();save(f,'selection')
print('Built L140 figures')
