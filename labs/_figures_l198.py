"""Portable experiment diagrams; every score is explicitly illustrative."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.hashsalt':'l198'})
F=Path(__file__).resolve().parent/'figures/l198';F.mkdir(parents=True,exist_ok=True)
ink='#173c44';green='#166f64';orange='#a46132'
def setup(title,subtitle):
    fig,ax=plt.subplots(figsize=(10,6.3));fig.patch.set_facecolor('#f3f7f4');ax.set_facecolor('#f3f7f4');ax.set_xlim(0,10);ax.set_ylim(0,6.3);ax.axis('off')
    ax.text(.35,5.85,title,fontsize=21,fontweight='bold',color=ink);ax.text(.35,5.42,subtitle,fontsize=11.5,color=ink)
    return fig,ax
def box(ax,x,y,w,h,title,text,color='#e0ebe7'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.09',facecolor=color,edgecolor='#9bb8af'))
    ax.text(x+.15,y+h-.30,title,fontsize=12,fontweight='bold',color=ink,va='top')
    ax.text(x+.15,y+h-.76,text,fontsize=13,color=ink,va='top',linespacing=1.6)
def save(fig,name):
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0);fig.savefig(F/(name+'.png'),dpi=150,metadata={'Software':'L198 experiment design'});fig.savefig(F/(name+'.svg'),metadata={'Date':None});plt.close(fig)
fig,ax=setup('01  Change the availability policy','Invented AUROC values · identical task, query keys, targets and paired support schedule')
box(ax,.45,2.7,4.3,2.25,'Timestamp-only inclusion','Tree  A = 0.70      PFN  B = 0.74\nPFN advantage: B − A = +0.04')
box(ax,5.2,2.7,4.3,2.25,'Explicit availability inclusion','Tree  C = 0.71      PFN  D = 0.72\nPFN advantage: D − C = +0.01')
ax.annotate('',xy=(7.3,2.15),xytext=(2.6,2.15),arrowprops={'arrowstyle':'->','lw':2,'color':green})
ax.text(5,1.65,'Change in model contrast = +0.01 − (+0.04) = −0.03',ha='center',fontsize=15,fontweight='bold',color=green)
ax.text(.5,.95,'Full proposed matrix: 2 tasks × 2 models × 2 policies × 3 seeds = 24 evaluations',color=ink)
ax.text(.5,.45,'Interpretation: sensitivity to assumptions. Historical leakage needs arrival-time evidence.',fontsize=11.5,color=orange)
save(fig,'temporal')
fig,ax=setup('02  Cross the prior with the encoder','Invented AUROC values · matched head, information, selection and compute accounting')
ax.text(3,4.8,'Conventional encoder',ha='center',fontweight='bold',color=ink);ax.text(6.2,4.8,'Composite encoder',ha='center',fontweight='bold',color=ink)
for y,label,a,b in [(3.55,'Flat prior','A = 0.70','B = 0.74'),(2.1,'Relational prior','C = 0.72','D = 0.73')]:
    ax.text(.4,y+.35,label,fontweight='bold',color=ink)
    for x,value in [(2.15,a),(5.35,b)]:
        ax.add_patch(FancyBboxPatch((x,y),1.7,.9,boxstyle='round,pad=.06',facecolor='#e0ebe7',edgecolor='#9bb8af'));ax.text(x+.85,y+.45,value,ha='center',va='center',fontsize=16,color=ink)
    ax.annotate('',xy=(5.15,y+.45),xytext=(4.05,y+.45),arrowprops={'arrowstyle':'->','lw':2,'color':green})
ax.text(7.8,4,'Gain +0.04',fontweight='bold',color=green);ax.text(7.8,2.55,'Gain +0.01',fontweight='bold',color=green)
ax.text(5,1.4,'Interaction = +0.01 − (+0.04) = −0.03',ha='center',fontsize=17,fontweight='bold',color=orange)
ax.text(.5,.85,'12 checkpoints × 2 tasks × 3 support draws = 72 primary prediction batches',color=ink)
ax.text(.5,.35,'Positive conditional gain does not establish positive interaction. All model runs: NOT_RUN.',fontsize=11.3,color=ink)
save(fig,'composite')
fig,ax=setup('03  Match cost and hold out the database','Invented AUROC values · compare each temporal objective against extra-compute scratch')
box(ax,.45,3.5,4.3,1.5,'Source-only pretraining','Exclude the whole target database',color='#e8e4ef')
box(ax,5.2,3.5,4.3,1.5,'Target adaptation + evaluation','Source + target cost = same budget')
for y,label,value,color in [(2.95,'Ordinary scratch',.70,'#93aaa2'),(2.3,'Extra-compute scratch',.73,green),(1.65,'Temporal-pretrained',.74,orange)]:
    ax.text(.5,y,label,va='center',color=ink);ax.barh(y,(value-.65)*50,left=4,height=.3,color=color);ax.text(4+(value-.65)*50+.12,y,f'{value:.2f}',va='center',color=ink)
ax.text(5,.95,'Decisive gain = 0.74 − 0.73 = +0.01',ha='center',fontsize=16,fontweight='bold',color=green)
ax.text(.5,.43,'12 source fits + 24 target fits · bars show AUROC above 0.65, not cost',fontsize=12,color=ink)
save(fig,'transfer')
print('Generated three annotated PNG/SVG experimental designs')
