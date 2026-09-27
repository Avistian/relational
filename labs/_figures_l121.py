"""Portable historical map and original numeric ownership trace."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
P=Path(__file__).resolve().parent/'figures/l121';P.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l121'})
def save(fig,name):
 fig.savefig(P/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});fig.savefig(P/(name+'.png'),dpi=160,bbox_inches='tight',metadata={'Software':'L121'});plt.close(fig)
def box(ax,x,y,w,h,text,color='#e8f0ed',size=11):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.03',facecolor=color,edgecolor='#91aaa8'))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color='#203448')
def arrow(ax,a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color='#216b91',lw=1.5))
fig,ax=plt.subplots(figsize=(9,8));ax.set(xlim=(0,10),ylim=(0,10));ax.axis('off')
ax.text(0,9.6,'Where does the representation come from?',fontsize=17,fontweight='bold',color='#203448')
items=[('1991 anchor · ILP','Learn rules over facts + background knowledge'),('2001 anchor · Propositionalization','Construct relational attributes → tabular learner'),('2015 · Deep Feature Synthesis','Compose primitives along keys → feature matrix'),('2018 · Neural relational features','Learn representations from related records'),('2019 / 2020 · Cvitkovic','Extract applicant graph → GNN → pooled prediction'),('2024 · RDL + RelBench','Temporal typed graphs + shared evaluation tasks')]
for i,(title,desc) in enumerate(items):
 y=8.15-i*1.3;box(ax,.15,y,9.6,1.0,title+'\n'+desc,color='#e1edf4' if i in [3,4,5] else '#e8f0ed')
 if i<5:arrow(ax,(.55,y-.06),(.55,y-.26))
ax.text(.15,.05,'Arrows indicate reading order, not universal replacement or proven superiority.',fontsize=10)
save(fig,'lineage')
fig,axes=plt.subplots(1,2,figsize=(10,7))
for ax,merged,title in zip(axes,[False,True],['Customer A · separate orders','Customer B · shared order']):
 ax.set(xlim=(0,5),ylim=(0,8));ax.axis('off');ax.set_title(title,fontsize=13,pad=15,color='#203448')
 box(ax,.15,6.4,2, .8,'Line: 2');box(ax,2.85,6.4,2,.8,'Line: 8')
 if not merged:
  arrow(ax,(1.15,6.3),(1.15,5.45));arrow(ax,(3.85,6.3),(3.85,5.45));box(ax,.15,4.6,2,.8,'Order sum: 2');box(ax,2.85,4.6,2,.8,'Order sum: 8')
  box(ax,.15,3.2,2,.75,'Square: 4');box(ax,2.85,3.2,2,.75,'Square: 64');arrow(ax,(1.15,4.55),(1.15,4.0));arrow(ax,(3.85,4.55),(3.85,4.0));arrow(ax,(1.15,3.1),(2.1,2.5));arrow(ax,(3.85,3.1),(2.9,2.5))
 else:
  arrow(ax,(1.15,6.3),(2,5.45));arrow(ax,(3.85,6.3),(3,5.45));box(ax,1,4.6,3,.8,'Order sum: 10');arrow(ax,(2.5,4.55),(2.5,4));box(ax,1,3.2,3,.75,'Square: 100');arrow(ax,(2.5,3.1),(2.5,2.5))
 box(ax,1,1.55,3,.9,'Root sum: '+('68' if not merged else '100'),color='#c7e9df',size=14)
 ax.text(2.5,.5,'Flat leaf summary: [2, 10, 8]\n[count, sum, maximum]',ha='center',va='center',fontsize=11)
fig.text(.5,.015,'Same leaf values. Different ownership. A nested engineered feature can preserve this too.',ha='center',fontsize=11)
fig.tight_layout(rect=(0,.05,1,1));save(fig,'grouping')
print('Built two L121 computation figures; Cvitkovic architecture reused from L118')
