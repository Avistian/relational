"""Model-specific pathways, information access and contrasts; deterministic SVG/PNG."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;F=P/'figures/l183';F.mkdir(parents=True,exist_ok=True)
r=json.loads((P/'evidence/l183/report.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.hashsalt':'l183'})
navy='#19364b';teal='#087f83';red='#b0413e';gold='#95600e';pale='#f2f7f7'
def save(fig,name):
    fig.canvas.draw()
    for ax in fig.axes:
        items=getattr(ax,'_l183_boxes',[])
        for i,(x,y,w,h,label) in enumerate(items):
            assert x>=0 and y>=0 and x+w<=12 and y+h<=ax.get_ylim()[1], (name,label.get_text())
            lo=ax.transData.transform((x,y));hi=ax.transData.transform((x+w,y+h));bb=label.get_window_extent(fig.canvas.get_renderer())
            assert bb.x0>=lo[0]-2 and bb.x1<=hi[0]+2 and bb.y0>=lo[1]-2 and bb.y1<=hi[1]+2, (name,'text overflow',label.get_text())
            for xx,yy,ww,hh,_ in items[:i]:assert not (x<xx+ww and xx<x+w and y<yy+hh and yy<y+h), (name,'box overlap')
    fig.savefig(F/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});fig.savefig(F/(name+'.png'),dpi=150,bbox_inches='tight',metadata={'Software':'L183'});plt.close(fig)
def panel(title,subtitle,height=7):
    fig,ax=plt.subplots(figsize=(12,height));ax.set(xlim=(0,12),ylim=(0,height));ax.axis('off');ax.text(.12,height-.42,title,fontsize=20,weight='bold',color=navy);ax.text(.12,height-.9,subtitle,fontsize=11,color=red);return fig,ax
def box(ax,x,y,w,h,text,color=teal,size=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',edgecolor=color,facecolor=pale,lw=1.5));label=ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color=navy)
    if not hasattr(ax,'_l183_boxes'):ax._l183_boxes=[]
    ax._l183_boxes.append((x,y,w,h,label))
def arrow(ax,a,b,color=teal):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=color,lw=1.7))
fig,ax=panel('RelGT: build a token before attending','Published architecture; F1 full-search dimensions. No new model execution in L183.',8)
box(ax,.15,5.75,3.25,.95,'(entity, cutoff) query\nTemporal FK neighborhood')
box(ax,4.05,5.75,3.45,.95,'K = 300 sampled rows\nPer-table feature encoders')
box(ax,8.15,5.75,3.55,.95,'Five d = 512 vectors / row\nB × K × 5d before mixing')
arrow(ax,(3.5,6.23),(3.93,6.23));arrow(ax,(7.6,6.23),(8.03,6.23))
for i,t in enumerate(['Type','Hop','Time','Features','Local structure']):
    box(ax,.15+i*2.35,4.3,2.15,.65,t,size=10.5)
ax.text(6,3.87,'Concatenate 5 × 512 → learned projection → 512 coordinates per token',ha='center',color=navy)
arrow(ax,(9.9,5.65),(9.9,5.07))
box(ax,.15,2.05,3.3,1.1,'Local token attention\nRows attend within K slots\nL = 1 / 4 / 8 blocks')
box(ax,4.05,2.05,3.45,1.1,'Parallel global attention\n4,096 learned centroids\nFuse local/global + feed-forward')
box(ax,8.15,2.05,3.55,1.1,'Task head → prediction\nL1 loss for F1 position\nValidation chooses weights')
arrow(ax,(1.8,3.65),(1.8,3.25));arrow(ax,(3.55,2.6),(3.93,2.6));arrow(ax,(7.6,2.6),(8.03,2.6))
ax.text(.15,1.22,'Proposed transfer adapter: shared cell/name encoder + schema-semantic type encoder',weight='bold',color=gold,fontsize=12)
ax.text(.15,.75,'It would replace schema-bound inputs above. Shape agreement alone cannot preserve feature meanings.',color=navy,fontsize=11)
ax.text(.15,.29,'Then pretrain this new model on source tasks → fine-tune on held-out F1. This proposed path is NOT_RUN.',color=red,fontsize=11)
save(fig,'relgt')
fig,ax=panel('Griffin: task-conditioned cells become graph messages','Published/released design; selected transfer lane d = 512, 4 message layers, 8 attention heads.',8)
box(ax,.15,5.75,3.3,.95,'Rows, cells, metadata\nQuery target withheld\nTask column → task vector')
box(ax,4.05,5.75,3.45,.95,'Text / categories → text vectors\nNumbers → normalize + ENC\nFixed float ENC / DEC')
box(ax,8.15,5.75,3.55,.95,'Row i: Lᵢ cells × 512\nCell values X; metadata M\nTask vector t: width 512')
arrow(ax,(3.55,6.23),(3.93,6.23));arrow(ax,(7.6,6.23),(8.03,6.23))
box(ax,.15,3.6,5.4,1.28,'Task-conditioned cell attention\nQuery: current row state + task vector\nKeys: metadata M; values: encoded cells X\nWeighted cell sum updates the row state')
box(ax,6.2,3.6,5.5,1.28,'Relation-aware message passing\nAggregate neighbors within each relation\nCombine relations; update row/task state\nRepeat four layers')
ax.plot([9.9,9.9,2.8],[5.64,5.27,5.27],color=teal);arrow(ax,(2.8,5.27),(2.8,5.0));arrow(ax,(5.65,4.25),(6.08,4.25))
box(ax,.15,1.65,3.3,1.1,'Shared task decoder\nTarget-row representation\n→ class / numeric prediction')
box(ax,4.05,1.65,3.45,1.1,'Source pretraining\nCompletion and supervised\nstages depend on release lane',gold)
box(ax,8.15,1.65,3.55,1.1,'Others-2 transfer lane\nSame Griffin architecture\n→ target-task fine-tuning',gold)
ax.plot([9.,9.,1.8],[3.48,3.12,3.12],color=teal);arrow(ax,(1.8,3.12),(1.8,2.87));arrow(ax,(7.6,2.2),(8.03,2.2),gold)
ax.text(.15,.87,'A Griffin state dict initializes Griffin. RelGT has different modules and parameter meanings.',color=red,fontsize=12,weight='bold')
ax.text(.15,.35,'Transfer a training idea by building and training the new architecture; do not relabel copied weights.',color=navy,fontsize=11)
save(fig,'griffin')
fig,ax=plt.subplots(figsize=(10.5,4.7));labels=['Past feature','Future event','Unfinished label','Unknown arrival','Query answer'];events=[5,12,5,5,5]
for i,(label,event) in enumerate(zip(labels,events)):
    color=teal if i==0 else red;ax.plot([event,event],[i-.17,i+.17],color=color,lw=4);ax.text(13.1,i,'VISIBLE' if i==0 else 'WITHHOLD',va='center',color=color,fontsize=10,weight='bold')
ax.plot([5,12],[2,2],color=gold,lw=2);ax.scatter([12],[2],color=gold,marker='s',zorder=3)
ax.axvline(10,color=navy,linestyle='--',label='Query cutoff = 10');ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.set_xlim(0,16);ax.set_xticks([0,5,10,12]);ax.set_xlabel('Illustrative time units; access policy is stricter than event-time filtering')
ax.set_title('Masking the answer is only one access check',loc='left',fontsize=18,pad=18);ax.set_ylim(4.4,-.95);ax.text(10,-.58,'cutoff = 10',ha='center',color=navy,fontsize=10);ax.spines[['top','right']].set_visible(False)
fig.text(.03,.015,'Synthetic fixture: past feature arrives at 6; label window ends at 12; unknown arrival is not presumed safe.',fontsize=10,color=navy);fig.tight_layout(rect=(0,.065,1,1));save(fig,'access')
fig,ax=plt.subplots(figsize=(9.7,5.2))
ax.plot([0,1],[4.,3.6],'-o',color=navy,label='Message passing: gain 0.4',lw=2,markersize=8)
ax.plot([0,1],[3.9,3.2],'-s',color=teal,label='Graph transformer: gain 0.7',lw=2,markersize=8)
for x,y,t in [(0,4.,'4.0'),(1,3.6,'3.6'),(0,3.9,'3.9'),(1,3.2,'3.2')]:ax.annotate(t,(x,y),xytext=(-26,8) if x==0 else (10,0),textcoords='offset points',fontsize=12)
ax.set(xticks=[0,1],xticklabels=['Scratch','Pretrained'],xlim=(-.18,1.38),ylim=(3.05,4.18),ylabel='Synthetic MAE (lower is better)');ax.set_title('Extra pretraining benefit = 0.7 − 0.4 = +0.3',fontsize=17,loc='left',pad=16);ax.legend(loc='upper right');ax.spines[['top','right']].set_visible(False)
fig.text(.08,.015,'Invented teaching values, not fits. Positive interaction measures an extra benefit, not universal superiority.',fontsize=10,color=red);fig.tight_layout(rect=(0,.07,1,1));save(fig,'factorial')
fig,axes=plt.subplots(1,2,figsize=(11.5,4.8),sharey=True)
for ax,split in zip(axes,['val','test']):
    for j,arm in enumerate(['gnn','relgt']):
        values=r['summary'][arm][split]['values'];mean=r['summary'][arm][split]['mean'];sd=r['summary'][arm][split]['sample_sd'];color=navy if j==0 else teal
        ax.scatter([j-.065,j,j+.065],values,color=color,s=48,zorder=3);ax.errorbar(j,mean,yerr=sd,fmt='_',color=color,markersize=22,capsize=7)
    ax.set(xticks=[0,1],xticklabels=['Course GNN','Reduced RelGT'],xlim=(-.4,1.4),ylim=(2.75,5.25),title='Validation' if split=='val' else 'Test');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('F1 driver-position MAE (lower is better)');fig.suptitle('Saved L146 evidence: the validation advantage reverses',fontsize=18,color=navy)
fig.text(.07,.02,'All six saved fits rescored. Dots: seeds 0–2; bars: mean ± sample seed SD, not confidence intervals. No pretraining arms.',fontsize=10,color=red);fig.tight_layout(rect=(0,.08,1,.94));save(fig,'replay')
print('Built five figures')
