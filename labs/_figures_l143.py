"""Model-specific diagrams and measured seed plot, deterministic exports."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;D=P/'figures/l143';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l143'})
INK='#17324d';TEAL='#007f82';GOLD='#a35919'
def box(ax,x,y,w,h,text,color=TEAL,size=11):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02',facecolor='#f0f7f6',edgecolor=color,linewidth=1.4);ax.add_patch(p)
 t=ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color=INK);ax._pairs=getattr(ax,'_pairs',[])+[(p,t)]
def arrow(ax,x,y,u,v,label=None):
 ax.annotate('',(u,v),(x,y),arrowprops={'arrowstyle':'->','color':INK,'lw':1.5})
 if label:ax.text((x+u)/2+.06,(y+v)/2,label,fontsize=10,color=INK)
def canvas(w,h):
 f,a=plt.subplots(figsize=(w,h));a.set(xlim=(0,w),ylim=(0,h));a.axis('off');return f,a
def save(f,name):
 f.canvas.draw();r=f.canvas.get_renderer()
 for a in f.axes:
  for p,t in getattr(a,'_pairs',[]):
   b=p.get_window_extent(r);q=t.get_window_extent(r)
   assert b.x0<=q.x0 and b.x1>=q.x1 and b.y0<=q.y0 and b.y1>=q.y1,(name,t.get_text())
 for ext in ['png','svg']:f.savefig(D/f'{name}.{ext}',dpi=145,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else None)
 plt.close(f)
f,a=canvas(9,6.3);a.text(.2,6,'ONE TASK · TWO EXECUTION LANES',color=INK,weight='bold',fontsize=15)
box(a,.3,4.55,8.3,.9,'Pinned full F1 tables + task queries + source commit\nFreeze the selected claim and reconstruction choices')
arrow(a,2.3,4.55,2.3,3.9);arrow(a,6.6,4.55,6.6,3.9)
box(a,.3,2.7,3.9,1.2,'Released weights\nCompatibility reconstruction\nOne checkpoint replay',size=10)
box(a,4.7,2.7,3.9,1.2,'Fresh initialization × five seeds\nTen complete epochs each\nValidation selects each checkpoint',size=10)
arrow(a,2.3,2.7,2.3,2.1);arrow(a,6.6,2.7,6.6,2.1)
box(a,.3,.9,8.3,1.2,'Keyed predictions + original-model parity + label audit\nReport each lane separately; aggregate only complete fresh fits\nHistorical training identity remains NOT_ESTABLISHED',size=11)
a.text(.3,.15,'A close number is one comparison, not a substitute for this evidence chain.',color=TEAL,fontsize=11);save(f,'contract')
f,a=canvas(9,6.6);a.text(.2,6.3,'WHAT DOES NUMERICAL COLUMN 2 MEAN?',color=INK,weight='bold',fontsize=15)
box(a,.3,4.8,3.9,.95,'Fresh feature proposal\nnumber: numerical\nposition: categorical',size=10)
box(a,4.7,4.8,3.9,.95,'Released checkpoint\nTwo numerical columns\nSaved mean and SD vectors',size=10)
arrow(a,2.3,4.8,4.45,4.1);arrow(a,6.6,4.8,4.45,4.1)
box(a,.3,2.95,8.3,1.15,'Document candidate reconstruction: [number, position]\nCompare ordered raw-table means and population SD\nReject count, order, moment or finite-value mismatch',size=11)
arrow(a,4.45,2.95,4.45,2.35)
box(a,.3,1.25,8.3,1.1,'Supported compatibility → evaluate the released weights\nMissing historical stypes cache → keep identity unestablished',size=11)
a.text(.3,.4,'Same shape can hide different feature meanings. Equal moments are not unique identities.',color=TEAL,fontsize=10);save(f,'compatibility')
f,a=canvas(9,12);a.text(.2,11.65,'RELGNN · F1 DRIVER-POSITION',weight='bold',fontsize=16,color=INK)
a.text(.2,11.2,'One composite layer • four heads • 128 channels per head',fontsize=12,color=TEAL)
box(a,.3,9.8,8.3,.9,'Query (driver, cutoff) + full relational tables\nUniform temporal sample: 128 / 64 neighbors; bidirectional edges')
arrow(a,4.45,9.8,4.45,9.35)
box(a,.3,8.25,8.3,1.1,'Per-table feature encoders → X_table [N_table, 128]\nAdd projected positional encoding of (owner cutoff − row time) / day')
arrow(a,4.45,8.25,4.45,7.8)
box(a,.3,5.55,8.3,2.25,'ONE ORDERED ROUTE: constructor → result → driver\n\nu = W_source × constructor + bias + W_fact × result\nQuery from driver; keys and values from u\nq, k, v: [nodes, 4 heads, 128 channels]',size=11)
arrow(a,4.45,5.55,4.45,5.1)
box(a,.3,3.85,8.3,1.25,'Per destination / head: softmax(q · k / √128) → weighted values\nConcat heads [N_driver, 512] + destination skip\nFinal linear projection → route output [N_driver, 128]')
arrow(a,4.45,3.85,4.45,3.4)
box(a,.3,2.25,8.3,1.15,'Sum route outputs → per-node LayerNorm → ReLU\nFact outputs also accumulate route-specific fused intermediates\nSelect driver roots [batch, 128]')
arrow(a,4.45,2.25,4.45,1.8)
box(a,.3,.65,8.3,1.15,'One-layer head → predicted position [batch, 1]\nTraining: unclipped L1 loss → gradients through all learned blocks\nEvaluation: clip at training-target percentiles 2 / 98')
a.text(.3,.05,'Two sampled graph edges ≠ two model layers. Every node occurrence keeps its query cutoff.',fontsize=10,color=INK);save(f,'architecture')
summary=P/'evidence/l143/training.json'
if summary.exists():
 s=json.loads(summary.read_text());f,a=plt.subplots(figsize=(8.5,4.8));vals=[json.loads((P/f'evidence/l143/seed-{i}/result.json').read_text())['scores']['test']['mae'] for i in range(5)]
 a.scatter(range(5),vals,color=TEAL,s=65,label='Fresh reconstructed training')
 a.axhline(3.798,color=INK,ls='--',label='Paper five-seed mean: 3.798')
 a.axhline(s['replay']['test']['mae'],color=GOLD,ls=':',label='Released checkpoint replay')
 a.errorbar(5.1,np.mean(vals),yerr=np.std(vals,ddof=1),fmt='D',color=TEAL,capsize=5,label='Fresh mean ± sample SD')
 a.set(xticks=list(range(5))+[5.1],xticklabels=['0','1','2','3','4','Mean'],ylabel='Test MAE · lower is better',xlabel='Fresh seed',title='Same task; different evidence populations')
 a.legend(fontsize=9,loc='best');a.grid(axis='y',alpha=.2);f.tight_layout();save(f,'results')

# Use the same model-specific operation map in the mechanism and replay lessons.
from _relgnn_architecture import draw
draw(Path(__file__).resolve().parent / "figures/l143")
