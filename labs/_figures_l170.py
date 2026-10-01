"""Generate portable paradigm flows, paired evidence and claim boundaries."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
P=Path(__file__).resolve().parent;O=P/'figures/l170';O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'svg.hashsalt':'l170'})
ink='#193952';teal='#087f82';orange='#a95823';purple='#6653a2'
def save(fig,name):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for ax in fig.axes:
        for text in ax.texts:
            b=text.get_window_extent(renderer)
            assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,(name,text.get_text())
    fig.savefig(O/(name+'.svg'),metadata={'Date':None});fig.savefig(O/(name+'.png'),dpi=150,metadata={'Software':'L170'});plt.close(fig)
    svg=O/(name+'.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
fig,ax=plt.subplots(figsize=(9,10));fig.patch.set_facecolor('#f5f8fa');ax.set(xlim=(0,10),ylim=(0,12));ax.axis('off')
ax.text(.15,11.65,'Same prediction goal. Different learned routes.',fontsize=20,weight='bold',color=ink)
ax.text(.15,11.13,'Trace where relational information and target labels enter.',fontsize=13,color=ink)
rows=[(8.2,'1  Griffin · supervised graph transfer',teal,
       ['Linked rows\n+ key edges','Shared cell attention\n+ relation messages','Frozen decoder\non adapted state'],
       'Before target: source-task supervision learns reusable weights.',
       'At target: labels → loss → shared-weight updates; decoder stays frozen.'),
      (4.8,'2  RDB-PFN · a synthetic relational prior',purple,
       ['Linked records\n→ DFS features','K labeled rows\n+ query features','Frozen predictor\n→ probability'],
       'Before target: synthetic relational tasks → predictor weights.',
       'At target: support labels enter the input; weights stay fixed.'),
      (1.4,'3  RDBLearn · featurize, then use tabular ICL',orange,
       ['Linked records\n→ aggregates','Materialized table\n+ support labels','Existing tabular FM\n→ probability'],
       'Before target: the tabular FM has already been pretrained.',
       'At target: relational features + labels; no predictor weight updates.')]
for y,title,color,boxes,before,after in rows:
    ax.text(.15,y+2.1,title,fontsize=17,weight='bold',color=color)
    ax.text(.15,y+1.52,before,fontsize=12.5,color=ink)
    for i,label in enumerate(boxes):
        x=.15+i*3.35
        ax.add_patch(FancyBboxPatch((x,y),2.9,1.05,boxstyle='round,pad=.06',facecolor='white',edgecolor=color,lw=1.5))
        ax.text(x+1.45,y+.53,label,ha='center',va='center',fontsize=12.5,color=ink)
        if i<2:ax.annotate('',xy=(x+3.22,y+.52),xytext=(x+2.97,y+.52),arrowprops=dict(arrowstyle='->',color=color,lw=2))
    ax.text(.15,y-.52,after,fontsize=12,color=ink)
ax.text(.15,.18,'Conceptual routes, not matched performance results. Griffin ≠ graph-native ICL.',fontsize=11,color=ink)
fig.subplots_adjust(left=.02,right=.99,top=.99,bottom=.02);save(fig,'paradigms')
r=json.loads((P/'evidence/l170/report.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10,6),sharex=True,sharey=True);fig.patch.set_facecolor('#f5f8fa')
for ax,db,title in zip(axes,['rel-f1','rel-trial'],['F1 · 702 queries','Trial · 825 queries']):
    for i,k in enumerate([64,128,256,512,1024]):
        v=r['paired']['by_task_context'][db+'/'+str(k)];d=np.array(v['per_seed']);offset=np.linspace(-.16,.16,10)
        ax.scatter(d,i+offset,c=np.where(d>=0,teal,orange),s=34,alpha=.8,zorder=3)
        ax.scatter([v['mean']],[i],marker='D',facecolor='white',edgecolor=ink,s=100,lw=1.5,zorder=5)
    ax.axvline(0,color=ink,lw=1);ax.set(yticks=range(5),yticklabels=['64','128','256','512','1024'],xlabel='Paired AUROC difference',title=title,xlim=(-.10,.10),ylim=(-.6,4.6));ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Labeled context K');axes[1].tick_params(labelleft=True)
fig.suptitle('RDB-PFN minus TabICL: retain every support draw',fontsize=18,color=ink)
fig.text(.09,.075,'Each dot = one of 10 paired support draws. Diamond = their mean. Same scales on both tasks.',fontsize=11)
fig.text(.09,.03,'Left of zero favors TabICL. These are reused predictions, not new L170 inference.',fontsize=11)
fig.tight_layout(rect=[0,.12,1,.92]);save(fig,'paired')
fig,ax=plt.subplots(figsize=(9,6.8));fig.patch.set_facecolor('#f5f8fa');ax.set(xlim=(0,10),ylim=(0,8));ax.axis('off')
ax.text(.15,7.6,'A complete replay does not complete every claim',fontsize=18,weight='bold',color=ink)
columns=[(.15,teal,'CHECKED IN THE REPLAY',['949 pinned input files','300 complete experiment cells','229,050 keyed predictions','Paired scores and context curves']),
         (5.2,orange,'STILL MISSING / FAILED',['Full DFS + temporal lineage','Fresh predictor pretraining','Matched three-paradigm study','Exact TabICL repeatability'])]
for x,color,title,items in columns:
    ax.text(x,6.85,title,fontsize=13,weight='bold',color=color)
    for i,item in enumerate(items):
        y=5.6-i*1.05
        ax.add_patch(FancyBboxPatch((x,y),4.5,.76,boxstyle='round,pad=.04',facecolor='white',edgecolor=color,lw=1.2))
        ax.text(x+.2,y+.38,item,fontsize=12,va='center',color=ink)
ax.text(.15,1.55,'Replay result: computational evidence is traceable.',fontsize=15,color=teal,weight='bold')
ax.text(.15,.93,'Design result: justify a next experiment; retain the missing evidence.',fontsize=12.5,color=ink)
ax.text(.15,.35,'Personal result: written defense remains pending teacher review.',fontsize=12.5,color=ink)
fig.subplots_adjust(left=.02,right=.99,top=.99,bottom=.02);save(fig,'gates')
print('Built 3 portable L170 figures')
