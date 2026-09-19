"""Computation-specific static notebook figures; all quantities are declared or measured."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
LAB=Path(__file__).resolve().parent;OUT=LAB/'figures/l083';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fafaf7','axes.facecolor':'#fafaf7'})
navy='#18354b';teal='#167d8d';orange='#b75d2a'
def canvas(title,size=(11,6)):
 f,a=plt.subplots(figsize=size);a.set(xlim=(0,11),ylim=(0,6));a.axis('off');a.text(.3,5.65,title,fontsize=17,fontweight='bold',color=navy);return f,a
def box(a,x,y,w,h,text,color=teal):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',fc='white',ec=color,lw=1.6));a.text(x+w/2,y+h/2,text,ha='center',va='center',color=navy,fontsize=11)
def arrow(a,p,q,label=None):
 a.annotate('',xy=q,xytext=p,arrowprops={'arrowstyle':'->','lw':1.6,'color':navy})
 if label:a.text((p[0]+q[0])/2,(p[1]+q[1])/2+.12,label,ha='center',fontsize=10)
def save(f,name):f.savefig(OUT/f'{name}.png',dpi=160,bbox_inches='tight');plt.close(f)
f,a=canvas('Two graph views; one frozen encoder')
box(a,.3,3.35,3.0,1.5,'TRAIN tissues\nfeatures + edges + labels\nfit scaler and weights')
box(a,4.0,3.35,3.0,1.5,'VALIDATION tissues\nnew features + edges\nlabels select learning rate',orange)
box(a,7.7,3.35,3.0,1.5,'TEST tissues\nnew features + edges\nlabels only score',orange)
for x in [.8,4.5,8.2]:
 for dx,dy in [(0,0),(.8,.4),(1.6,0)]:a.plot(x+dx,2.3+dy,'o',color=teal if x<2 else orange,ms=13)
 a.plot([x,x+.8,x+1.6],[2.3,2.7,2.3],color=navy,zorder=0)
a.text(5.5,1.45,'No edges across splits in PPI. Held-out nodes never enter training support.',ha='center',color=navy)
a.text(5.5,.65,'Scaler: fit TRAIN → transform all. Weights: fit TRAIN → freeze for evaluation.',ha='center',fontsize=12,color=navy)
save(f,'boundary')
f,a=canvas('Trace one receiver: preserve self and neighbor paths')
box(a,.4,3.55,2.3,1.1,'self\n[2, 4]')
box(a,.4,1.6,2.3,1.3,'neighbors\n[1, 3], [5, 1]')
box(a,3.7,3.55,2.5,1.1,'× identity\n[2, 4]')
box(a,3.7,1.6,2.5,1.3,'mean → × identity\n[3, 2]')
arrow(a,(2.8,4.1),(3.6,4.1));arrow(a,(2.8,2.25),(3.6,2.25))
box(a,7.45,2.4,2.95,1.5,'concatenate\n[2, 4 | 3, 2]\n4 coordinates')
arrow(a,(6.3,4.1),(7.35,3.45));arrow(a,(6.3,2.25),(7.35,2.75))
a.text(.4,.65,'Final normalization (if this is the last layer): divide by √33 → [.348, .696, .522, .348]',color=navy)
save(f,'trace')
f,a=canvas('Model architecture · released supervised GraphSAGE-mean',(12,7))
box(a,.25,4.35,2.7,.8,'roots: B × 50')
box(a,4.05,4.35,2.7,.8,'hop 1: 10B × 50')
box(a,7.9,4.35,2.7,.8,'hop 2: 250B × 50')
arrow(a,(3.05,4.75),(3.95,4.75),'×10');arrow(a,(6.85,4.75),(7.8,4.75),'×25')
box(a,.25,2.6,2.7,1.0,'self × W₁ | mean × U₁\nReLU → B × 256')
box(a,4.05,2.6,3.0,1.0,'self × W₁ | mean × U₁\nReLU → 10B × 256')
arrow(a,(1.6,4.25),(1.6,3.7));arrow(a,(5.4,4.25),(2.85,3.7));arrow(a,(5.4,4.25),(5.4,3.7));arrow(a,(9.2,4.25),(6.6,3.7))
a.text(8.7,3.0,'W₁, U₁ shared\nacross both calls',ha='center',color=teal)
box(a,.25,.85,3.55,1.0,'self × W₂ | mean × U₂\nB × 256 → final L2 norm')
arrow(a,(1.6,2.5),(1.6,1.95));arrow(a,(5.4,2.5),(3.45,1.95))
box(a,4.65,.85,2.35,1.0,'linear head + bias\nB × 121 logits')
box(a,7.85,.85,2.75,1.0,'train: BCE, root labels\ninfer: sigmoid > .5',orange)
arrow(a,(3.9,1.35),(4.55,1.35));arrow(a,(7.1,1.35),(7.75,1.35))
a.text(.3,.15,'Sample outward (top); aggregate inward. Only root labels enter the supervised loss.',fontsize=11,color=navy)
save(f,'architecture')
r=json.loads((LAB/'_paper_l083_results.json').read_text())
f,a=plt.subplots(figsize=(10,4.8))
x=list(range(len(r['runs'])));y=[v['test']['micro_f1'] for v in r['runs']]
a.scatter(x,y,s=85,c=teal,zorder=3);a.axhline(.598,color=orange,ls='--',label='Paper Table 1: .598 (protocol gaps remain)')
a.set_xticks(x,[f'Seed {v["seed"]}\nlr={v["selected_lr"]}' for v in r['runs']]);a.set_ylim(0,1);a.set_ylabel('Micro-F1 (0–1)');a.set_title('Full PPI port · each seed selects from three rates on validation',loc='left',color=navy)
for i,v in zip(x,y):a.annotate(f'{v:.4f}',(i,v),xytext=(0,12),textcoords='offset points',ha='center')
a.legend(loc='lower right');a.grid(axis='y',alpha=.2);f.tight_layout();save(f,'results')
