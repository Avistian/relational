"""Portable model-specific diagrams and measured seed results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;OUT=P/'figures/l133';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l133'})
INK='#183947';BLUE='#e1edf4';GOLD='#faeccb';GREEN='#e2f1e6'
def canvas(title,h):
 f,a=plt.subplots(figsize=(11,h));f.patch.set_facecolor('#faf9f6');a.set(xlim=(0,11),ylim=(0,h));a.axis('off');a.set_title(title,loc='left',fontsize=17,weight='bold',pad=18);return f,a
def box(a,x,y,w,h,text,c=BLUE):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',fc=c,ec=INK));a.text(x+w/2,y+h/2,text,ha='center',va='center',color=INK,fontsize=11)
def arrow(a,x,y,u,v):a.annotate('',xy=(u,v),xytext=(x,y),arrowprops={'arrowstyle':'->','lw':1.8,'color':INK})
def save(f,n):
 f.tight_layout();f.savefig(OUT/(n+'.svg'),metadata={'Date':None});f.savefig(OUT/(n+'.png'),dpi=140,metadata={'Software':'L133'});plt.close(f)
f,a=canvas('A relational layer inside the full driver-position predictor',8)
box(a,.2,6.6,3,1,'Driver + query time\nFuture position is the label',GOLD);box(a,3.9,6.6,3,1,'Past sampled context\nTyped FK + reverse edges');box(a,7.6,6.6,3,1,'Per-table row encoders\n+ relative-time vectors')
arrow(a,3.3,7.1,3.8,7.1);arrow(a,7,7.1,7.5,7.1)
box(a,.3,4.65,4.3,1.1,'Relation r: source rows [Ns,128]\nGather → sum at destination\nNeighbor matrix + bias');box(a,6.1,4.65,4.3,1.1,'Same destination rows [Nd,128]\nRelation-specific root matrix\nAdd inside EACH relation',GOLD)
arrow(a,8.8,6.5,8.3,5.85);arrow(a,8,6.5,2.5,5.85)
box(a,2.6,2.8,5.8,1.1,'Sum relation outputs by destination type\nNode LayerNorm → ReLU\nRepeat with new weights for layer 2',GREEN);arrow(a,2.5,4.55,4.5,4);arrow(a,8.3,4.55,6.5,4)
box(a,2.6,.9,5.8,1.1,'First B driver vectors → scalar head [B,1]\nMean L1 loss → gradients through the whole stack',GOLD);arrow(a,5.5,2.7,5.5,2.1)
a.text(.3,.15,'Nine tables • width 128 • two layers • only query roots receive task labels',color=INK);save(f,'architecture')
f,a=canvas('One customer, two relations: the root transform appears twice',6)
box(a,.25,3.85,3,1.3,'Orders: 1 and 2\nNeighbor sum = 3\n2 × 3 + 1 + 3 × 4 = 19');box(a,7.75,3.85,3,1.3,'Tickets: 5\nNeighbor sum = 5\n0.5 × 5 − 1 − 2 × 4 = −6.5')
box(a,4,4,3,1,'Customer vector = 4\nShared input, distinct weights',GOLD);arrow(a,4,4.5,3.35,4.5);arrow(a,7,4.5,7.65,4.5)
box(a,3.25,1.65,4.5,1.2,'Relation sum = 19 − 6.5 = 12.5\nThen LayerNorm → ReLU',GREEN);arrow(a,1.75,3.75,4,3);arrow(a,9.25,3.75,7,3)
a.text(.5,.6,'Scalar teaching trace stops before normalization. Real features have 128 channels.\nBias belongs to the neighbor transform; the destination transform has no bias.',fontsize=11,color=INK);save(f,'arithmetic')
f,a=canvas('An empty relation still transforms its destination',5)
rows=[['Present, one edge','5','−6.5','12.5'],['Present, zero edges','0','−9','10'],['Absent dictionary key','Not executed','Skipped','19']]
t=a.table(cellText=rows,colLabels=['Tickets relation','Neighbor sum','Tickets output','Total with orders'],cellLoc='center',bbox=[.01,.27,.98,.53]);t.auto_set_font_size(False);t.set_fontsize(11)
for (r,c),cell in t.get_celld().items():cell.set_facecolor(BLUE if r==0 else '#ffffff');cell.set_edgecolor('#bbcbd0')
a.text(.1,.5,'Fixed customer = 4; fixed orders output = 19.\nWith zero ticket edges: 0.5 × 0 − 1 − 2 × 4 = −9.',fontsize=12,color=INK);save(f,'presence')
s=json.loads((P/'evidence/l133/summary.json').read_text())
f,axes=plt.subplots(1,2,figsize=(11,4.7));f.patch.set_facecolor('#faf9f6')
for ax,k,label in zip(axes,['val','test'],['Validation','Test']):
 vals=[v[k] for v in s['seeds']];m=s['metrics'][k];ax.scatter(range(5),vals,color=INK,s=45);ax.axhline(m['target'],ls='--',color='#a44731',label='Published mean');ax.errorbar([5.2],[m['mean']],yerr=[m['sample_sd']],fmt='D',color='#276754',capsize=6,label='Mean ± seed SD');ax.set(xticks=list(range(5))+[5.2],xticklabels=['0','1','2','3','4','Mean'],ylabel='MAE (lower is better)',title=label+' · five fresh full-data fits');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=9)
f.suptitle('Selected RelBench v1 experiment — distinct scales for the two splits',fontsize=15);save(f,'scores')
