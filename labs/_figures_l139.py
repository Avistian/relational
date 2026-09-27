"""Deterministic clinical computation diagrams; PNGs travel with notebooks."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D=Path(__file__).resolve().parent/'figures/l139';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l139'})
blue='#17324d';teal='#007f82';orange='#b85d20'
def save(f,n):
 for ext in ['svg','png']:f.savefig(D/f'{n}.{ext}',dpi=160,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else None)
 plt.close(f)
def box(a,x,y,w,h,text,color=teal):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02',edgecolor=color,facecolor='#eff7f6',linewidth=1.5));a.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11,color=blue)
def arrow(a,x,y,u,v):a.annotate('',xy=(u,v),xytext=(x,y),arrowprops=dict(arrowstyle='->',color=teal,lw=2))
f,a=plt.subplots(figsize=(9,5));a.set(xlim=(0,10),ylim=(0,6));a.axis('off');a.set_title('First select the query; then define its label',loc='left',color=blue,fontweight='bold')
box(a,.1,4.7,9.5,.8,'Study started by cutoff t → inspect analyses dated in (t, t + 365 days]')
arrow(a,4.8,4.68,4.8,4.2);box(a,.1,3.35,9.5,.8,'Primary outcome + p in [0,1] + modifier is not “>”')
arrow(a,2.5,3.33,2.5,2.75);arrow(a,7.3,3.33,7.3,2.75)
box(a,.1,1.75,4.4,1,'No qualifying analysis\nNO QUERY',orange);box(a,5.1,1.75,4.5,1,'At least one qualifying analysis\nELIGIBLE')
arrow(a,7.35,1.73,7.35,1.2);box(a,5.1,.1,4.5,1.05,'min(p) ≤ .05 → 1\nmin(p) > .05 → 0')
a.text(.25,.8,'Missing observation ≠ negative label\nThe evaluation population is conditional.',color=orange,fontsize=11);save(f,'eligibility')
f,a=plt.subplots(figsize=(9.5,7.5));a.set(xlim=(0,10),ylim=(0,9));a.axis('off');a.set_title('A clinical study score, from rows to prediction',loc='left',color=blue,fontweight='bold')
box(a,.1,7.65,2.4,.9,'Facility f\nlocation/category');box(a,3.1,7.65,3.1,.9,'facilities_studies a\ndate ≤ query cutoff');box(a,7,7.65,2.7,.9,'Study s\ntext + categories')
arrow(a,2.55,8.1,3.05,8.1);arrow(a,6.25,8.1,6.95,8.1)
a.text(.2,7.13,'Highlighted path in the full 15-table graph; outcomes and analyses are also timestamped.',fontsize=10,color=blue)
box(a,.1,5.65,9.6,1.05,'Per-table column encoders → ResNet row encoder → [N_type, 128]\nGloVe text [N_text, 300] → projection; owner-specific relative-time encoding')
arrow(a,4.85,6.99,4.85,6.74)
box(a,.1,4.15,4.4,.95,'Layer 1: facility → association\nmean neighbor aggregation');box(a,5.3,4.15,4.4,.95,'Layer 2: association → study\nmean neighbor aggregation')
arrow(a,2.3,5.63,2.3,5.13);arrow(a,4.55,4.6,5.25,4.6)
a.text(.2,3.6,'Within each layer: relation outputs SUM → LayerNorm → ReLU',fontsize=12,color=teal)
a.text(.2,3.13,'Disjoint queries · batch512 · fanout64/32 · preserve cutoff at every hop',fontsize=11,color=blue)
box(a,.1,1.7,4.4,.95,'Study roots [B,128]\nlinear head → logits [B,1]');box(a,5.3,1.7,4.4,.95,'Sigmoid probabilities [B]\nvalidation AUROC selects')
a.plot([9.85,9.85,2.3],[4.1,2.95,2.95],color=teal,lw=2);arrow(a,2.3,2.95,2.3,2.7);arrow(a,4.55,2.17,5.25,2.17)
box(a,.1,.12,9.6,.9,'Training: BCE(logits, observed targets) → gradients through head, GNN and encoders\nAdam .0001 · 20 epochs · fresh weights per seed')
arrow(a,2.3,1.68,2.3,1.05);save(f,'architecture')
f,a=plt.subplots(figsize=(8.5,4.8));a.set(xlim=(0,10),ylim=(0,6));a.axis('off');a.set_title('What stays; what must be re-audited',loc='left',color=blue,fontweight='bold')
box(a,.1,4.35,9.5,1,'SHARED COMPUTATION\nrow features → encoders → temporal neighborhoods → GNN → prediction')
box(a,.1,2.35,4.4,1.2,'AMAZON\ncustomer / review / product\nabsence of reviews over91days');box(a,5.2,2.35,4.4,1.2,'TRIAL\nstudy / association / facility\nqualifying analysis over365days')
arrow(a,2.3,4.3,2.3,3.6);arrow(a,7.4,4.3,7.4,3.6)
box(a,.1,.15,9.5,1.15,'CHANGE AND VERIFY\nquery population · target SQL · timestamp meaning · schema · paper settings\nFRESH MODEL FITS; no transfer of Amazon weights',orange);arrow(a,2.3,2.3,2.3,1.35);arrow(a,7.4,2.3,7.4,1.35);save(f,'transfer')
# Measured validation histories; test is never a selection curve.
E=D.parents[1]/'evidence/l139'
if all((E/f'full/seed-{i}/result.json').exists() for i in range(5)):
 import json,numpy as np
 f,a=plt.subplots(figsize=(8.5,4.8))
 for seed in range(5):
  r=json.loads((E/f'full/seed-{seed}/result.json').read_text());y=[100*x['val']['roc_auc'] for x in r['history']];epochs=np.arange(1,21);line,=a.plot(epochs,y,lw=1.6,alpha=.8,label=f"Seed {seed} · selected epoch {r['best_epoch']}");k=r['best_epoch'];a.scatter([k],[y[k-1]],color=line.get_color(),s=45,zorder=4)
 a.set(xlabel='Completed training epoch',ylabel='Validation AUROC (%)',xticks=[1,5,10,15,20]);a.set_title('Train for 20 epochs; retain the validation-selected state',loc='left',fontweight='bold',color=blue);a.grid(alpha=.2);a.legend(fontsize=9,loc='lower right');save(f,'selection')
