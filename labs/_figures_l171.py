"""Portable, data-derived corpus and full-snapshot figures."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
P=Path(__file__).resolve().parent;D=P/'figures/l171';D.mkdir(parents=True,exist_ok=True)
r=json.loads((P/'evidence/l171/report.json').read_text());a=r['full_snapshot']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.hashsalt':'l171','axes.spines.top':False,'axes.spines.right':False})
ink='#17324d';teal='#157f78';red='#a34832';muted='#526477';cream='#faf8f2'
def save(fig,name):
    fig.savefig(D/(name+'.svg'),bbox_inches='tight',facecolor=cream,metadata={'Date':None})
    fig.savefig(D/(name+'.png'),dpi=150,bbox_inches='tight',facecolor=cream,metadata={'Software':'L171'})
    plt.close(fig)
fig,ax=plt.subplots(figsize=(11.6,4.9));ax.set(xlim=(0,12),ylim=(0,5));ax.axis('off')
ax.text(.1,4.7,'A database name is not an isolation boundary',fontsize=19,weight='bold',color=ink)
ax.text(.1,4.22,'Illustrative contamination: follow identity links until no new source is reached.',color=muted)
labels=[('F1 held out','start',.1,red),('Renamed copy','same archive',3.15,red),('Bridge corpus','shared family',6.2,red),('Tail corpus','shared family',9.25,red)]
for title,sub,x,color in labels:
    ax.add_patch(FancyBboxPatch((x,2.55),2.65,1.03,boxstyle='round,pad=.08',facecolor='#f7e5dd',edgecolor=color,lw=1.5))
    ax.text(x+1.325,3.21,title,ha='center',color=ink,weight='bold');ax.text(x+1.325,2.82,sub,ha='center',color=color)
for x in [2.85,5.90,8.95]:ax.add_patch(FancyArrowPatch((x,3.05),(x+.23,3.05),arrowstyle='-|>',mutation_scale=18,color=red))
ax.text(.1,1.95,'Pass 0: F1',color=red);ax.text(3.15,1.95,'Pass 1: + copy',color=red);ax.text(6.2,1.95,'Pass 2: + bridge',color=red);ax.text(9.25,1.95,'Pass 3: + tail',color=red)
ax.add_patch(FancyBboxPatch((.18,.48),11.55,.85,boxstyle='round,pad=.08',facecolor='#e4f2ed',edgecolor=teal))
ax.text(.45,1.0,'Training candidates: the other six declared original database families',color=teal,weight='bold')
ax.text(.45,.65,'Excluded: held-out F1 + three synthetic relatives. Unknown lineage is still unknown.',color=ink)
save(fig,'holdout')
# Cell area encodes FK population; every declared FK column contributes its non-null references.
names=sorted(a['tables']);matrix=np.zeros((9,9),dtype=int);edges=np.zeros((9,9),dtype=int)
for fk in a['foreign_keys']:
    i,j=names.index(fk['table']),names.index(fk['target']);matrix[i,j]+=fk['nonnull'];edges[i,j]+=1
fig,ax=plt.subplots(figsize=(11.6,7.5));ax.set_facecolor(cream)
for i in range(9):
 for j in range(9):
    ax.scatter(j,i,s=1350 if matrix[i,j] else 14,marker='s',c=teal if matrix[i,j] else '#ccd4d9',alpha=.16 if matrix[i,j] else .5)
    if matrix[i,j]:ax.text(j,i,f'{matrix[i,j]:,}',ha='center',va='center',fontsize=10,color=ink)
short=[x.replace('constructor_','ctor_') for x in names]
ax.set_xticks(range(9),short,rotation=38,ha='right');ax.set_yticks(range(9),short);ax.invert_yaxis()
ax.set(xlim=(-.65,8.65),ylim=(8.65,-.65),xlabel='Referenced primary-key table',ylabel='Table containing the foreign key')
ax.set_title('Every declared F1 relationship, over every row',loc='left',fontsize=18,weight='bold',pad=25,color=ink)
fig.text(.08,.015,f"{a['foreign_key_columns']} FK columns · {sum(f['nonnull'] for f in a['foreign_keys']):,} non-null references · 0 dangling references\nCounts are references, not distinct entities. Empty cells mean no declared FK; they do not prove independence.",fontsize=11,color=muted)
fig.subplots_adjust(left=.23,bottom=.25,top=.88,right=.97);save(fig,'foreign-keys')
fig,ax=plt.subplots(figsize=(11.6,5.5));timed=[(n,t) for n,t in a['tables'].items() if t['time_windows']]
left=np.zeros(len(timed));colors=[teal,'#d6a347','#7b6f9b'];keys=['before_val','val_to_test','at_or_after_test'];labels=['Before 2005','2005 to before 2010','2010 onward']
for key,color,label in zip(keys,colors,labels):
    nums=np.array([t['time_windows'][key] for n,t in timed]);values=nums/np.array([t['rows'] for n,t in timed])*100
    ax.barh(range(len(timed)),values,left=left,color=color,label=label,height=.64)
    for i,(v,n) in enumerate(zip(values,nums)):
        if v>9:ax.text(left[i]+v/2,i,f'{n:,}',ha='center',va='center',color='white',fontsize=11)
    left+=values
ax.set_yticks(range(len(timed)),[n.replace('constructor_','ctor_')+'\n'+f"{t['rows']:,} rows" for n,t in timed]);ax.invert_yaxis()
ax.set_xlim(0,100);ax.set_xlabel('Share of each full snapshot table (%) — labels show row counts')
ax.set_title('Snapshot time is not historical availability',loc='left',fontsize=18,weight='bold',pad=24,color=ink)
ax.legend(loc='lower left',bbox_to_anchor=(-.02,1.01),ncol=3,frameon=False,fontsize=10)
fig.text(.07,.025,'Circuits, constructors and drivers have no time column (1,145 rows).\nThese are descriptive windows, not task splits or a training-ready temporal filter.',fontsize=11,color=muted)
fig.subplots_adjust(left=.24,bottom=.22,top=.78,right=.97);save(fig,'time-windows')
print('Built 3 portable figures from the frozen report')
