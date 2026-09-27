"""Portable, deterministic figures: lost distinctions, MPNN limit, fresh results."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent;D=P/'figures/l119';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none','svg.hashsalt':'l119','savefig.facecolor':'#fcfbf8'})
TEAL='#176c68';GOLD='#a85c21';INK='#233846'
def save(fig,name):
    fig.savefig(D/f'{name}.svg',bbox_inches='tight',metadata={'Date':None})
    fig.savefig(D/f'{name}.png',bbox_inches='tight',dpi=150,metadata={'Software':'L119'})
    plt.close(fig)
    p=D/f'{name}.svg';p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')

fig,axes=plt.subplots(3,1,figsize=(9,9),gridspec_kw={'height_ratios':[1.4,1,1]})
ax=axes[0]
ax.plot([1,2,3],[10,30,50],'o-',color=TEAL,label='Ada: rising')
ax.plot([1,2,3],[50,30,10],'s--',color=GOLD,label='Bo: falling')
ax.set(xticks=[1,2,3],xlabel='Order time (course units)',ylabel='Amount',ylim=(0,60))
ax.legend(frameon=False,ncol=2);ax.set_title('Same amounts, different order',loc='left',weight='bold');ax.spines[['top','right']].set_visible(False)
for ax in axes[1:]:ax.axis('off')
axes[1].text(.02,.88,'COARSE SUMMARY → COLLISION',color=TEAL,weight='bold',transform=axes[1].transAxes)
t=axes[1].table(cellText=[['Ada',3,90,30,50],['Bo',3,90,30,50]],colLabels=['Customer','Count','Total','Mean','Max'],cellLoc='center',bbox=[0,.02,1,.67]);t.auto_set_font_size(False);t.set_fontsize(12)
axes[2].text(.02,.83,'ADD A TIME–AMOUNT INTERACTION',weight='bold',color=GOLD)
axes[2].text(.02,.50,'Ada: 1×10 + 2×30 + 3×50 = 220\nBo:   1×50 + 2×30 + 3×10 = 140',linespacing=1.7,fontsize=14)
axes[2].text(.02,.06,'An engineered column OR a suitable message sum separates the pair.\nThis is an exact two-row construction; no predictive model was trained.',fontsize=11)
fig.tight_layout(h_pad=2);save(fig,'collision')

fig,axes=plt.subplots(2,2,figsize=(9,8),gridspec_kw={'height_ratios':[1.6,1]})
cycle=np.array([(np.cos(t),np.sin(t)) for t in np.linspace(0,2*np.pi,6,endpoint=False)])
tri=np.concatenate([.55*np.array([(np.cos(t),np.sin(t)) for t in np.linspace(0,2*np.pi,3,endpoint=False)])+offset for offset in [(-.9,0),(.9,0)]])
graphs=[(cycle,[(i,(i+1)%6) for i in range(6)],'One cycle · connected'),(tri,[(0,1),(1,2),(2,0),(3,4),(4,5),(5,3)],'Two triangles · disconnected')]
for ax,(pos,edges,title) in zip(axes[0],graphs):
    for u,v in edges:ax.plot(pos[[u,v],0],pos[[u,v],1],color='#8ca5a5',lw=2,zorder=1)
    ax.scatter(pos[:,0],pos[:,1],s=800,c='#e4f0ed',edgecolors=TEAL,zorder=2)
    for x,y in pos:ax.text(x,y,'1',ha='center',va='center',fontsize=16,zorder=3)
    ax.set(xlim=(-1.7,1.7),ylim=(-1.3,1.3),aspect='equal');ax.axis('off');ax.set_title(title,fontsize=12)
axes[1,1].axis('off');axes[1,0].axis('off')
axes[1,0].text(0,.9,'Every node has two neighbors.',weight='bold',color=TEAL)
axes[1,0].text(0,.6,'h(next) = h + h + h = 3h\nSame features, types, and update.\nNo node IDs or positional features.',linespacing=1.8,fontsize=11)
t=axes[1,1].table(cellText=[[0,1,6],[1,3,18],[2,9,54],[3,27,162],[4,81,486]],colLabels=['Round','Each node','Graph sum'],cellLoc='center',bbox=[0,.05,1,.85]);t.auto_set_font_size(False);t.set_fontsize(11)
fig.suptitle('Local equality persists in both graphs',x=.08,ha='left',weight='bold',color=INK)
fig.tight_layout(rect=[0,0,1,.94]);save(fig,'expressiveness')

summary=P/'evidence/l119/summary.json'
if summary.exists():
    s=json.loads(summary.read_text());fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=False)
    for ax,split,label in zip(axes,['val','test'],['Validation','Test']):
        scores=[r[split] for r in s['seeds']];mean=s['metrics'][split]['mean'];sd=s['metrics'][split]['sample_sd'];target=s['metrics'][split]['target']
        ax.scatter(scores,np.arange(5),color=TEAL,label='Fresh L119 seed',s=50)
        ax.axvline(target,color=GOLD,ls='--',label='Published RDL mean')
        ax.errorbar(mean,5,xerr=sd,fmt='D',color=INK,capsize=5,label='Fresh mean ± seed SD')
        ax.set(yticks=list(range(6)),yticklabels=['seed '+str(i) for i in range(5)]+['mean ± SD'],xlabel='MAE (finishing-position units; lower is better)')
        ax.set_title(label+' · complete selected experiment',loc='left',fontsize=12,weight='bold');ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,fontsize=9,frameon=False)
    fig.tight_layout(h_pad=2,rect=[0,0,1,.94]);save(fig,'results')
print('Built L119 figures; results require completed evidence')
