"""Task-specific diagrams with separate input and supervision paths."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent/'figures/l124';P.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'l124','font.size':11})
INK='#19384a';GREEN='#dceee4';GOLD='#f5e9c7';BLUE='#dfeaf2'
def save(f,name):
 f.tight_layout();f.savefig(P/(name+'.svg'),metadata={'Date':None});f.savefig(P/(name+'.png'),dpi=150,metadata={'Software':'L124'});plt.close(f)
def setup(title):
 f,a=plt.subplots(figsize=(11,6));a.axis('off');a.set(xlim=(0,11),ylim=(0,6));f.patch.set_facecolor('#faf9f6');a.set_title(title,loc='left',fontsize=18,weight='bold',color=INK);return f,a

def box(a,x,y,w,h,text,color=GREEN):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',fc=color,ec=INK));a.text(x+w/2,y+h/2,text,ha='center',va='center',color=INK,fontsize=11)
def arrow(a,x,y,xx,yy,label=None):
 a.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':INK,'lw':2})
 if label:a.text((x+xx)/2,(y+yy)/2+.15,label,ha='center',fontsize=10,color=INK)
f,a=setup('One entity is not one prediction')
box(a,.2,2.5,2.5,1.2,'ENTITY TABLE\nDriver 0\nOne reusable graph node')
box(a,4,4,3,1.3,'TASK ROW A\n(Driver 0, Jan 1)\nFuture mean: 4',GOLD)
box(a,4,1.5,3,1.3,'TASK ROW B\n(Driver 0, Mar 1)\nA different future mean',GOLD)
arrow(a,2.8,3.2,3.9,4.5);arrow(a,2.8,3,3.9,2.2)
box(a,8,4,2.6,1.3,'Input A\nOnly rows known\nby Jan 1',BLUE);box(a,8,1.5,2.6,1.3,'Input B\nOnly rows known\nby Mar 1',BLUE)
arrow(a,7.1,4.6,7.9,4.6);arrow(a,7.1,2.1,7.9,2.1)
a.text(.2,.5,'Join predictions by (entity, cutoff). Labels supervise the output; they are not node features.',color=INK,fontsize=12);save(f,'identity')
f,a=setup('From a question to a loss: keep the label on its own path')
box(a,.2,4.35,2.5,1,'Task input\nB × (driver ID, cutoff)',GOLD)
box(a,3.2,4.35,3.1,1,'Temporal neighborhoods\nRows and FK edges at t',BLUE)
box(a,7,4.35,3.6,1,'Typed row + time encoders\nN[type] × 128',BLUE)
arrow(a,2.8,4.85,3.1,4.85);arrow(a,6.4,4.85,6.9,4.85)
box(a,7,2.25,3.6,1,'Two typed GraphSAGE layers\nRead B seed vectors: B × 128',BLUE);arrow(a,8.8,4.2,8.8,3.4)
box(a,3.2,2.25,3.1,1,'Scalar head\nB × 1 predictions',BLUE);arrow(a,6.9,2.75,6.4,2.75)
box(a,.2,2.25,2.5,1,'Task target\nB future means',GOLD)
box(a,3.2,.4,3.1,1,'L1 loss over B queries\nTrain: Adam update',GREEN);arrow(a,4.75,2.1,4.75,1.5);arrow(a,1.45,2.1,3.1,.95)
a.text(7,.65,'Validation selects the checkpoint.\nTest scores the frozen selection.\nContext nodes have no direct label loss.',color=INK,fontsize=11);save(f,'forward')
s=json.loads((P.parents[1]/'evidence/l124/summary.json').read_text());f,axs=plt.subplots(1,2,figsize=(10,4.5));f.patch.set_facecolor('#faf9f6')
for ax,split,title in zip(axs,['val','test'],['Validation','Test']):
 m=s['metrics'][split];ax.scatter(range(5),[r[split] for r in s['seeds']],s=60,color='#176b79',label='Fresh seed');ax.axhline(m['target'],color='#a04c3e',ls='--',label='Published mean');ax.errorbar(5.4,m['mean'],yerr=m['sample_sd'],fmt='D',color=INK,capsize=6,label='Mean ± sample SD');ax.set(xticks=list(range(5))+[5.4],xticklabels=['0','1','2','3','4','Mean'],ylabel='MAE (position units)',title=title+' · detail scale');ax.grid(axis='y',alpha=.2)
axs[0].legend(fontsize=9);f.suptitle('Five fresh complete fits · seed SD is not a confidence interval',fontsize=13);save(f,'scores')
