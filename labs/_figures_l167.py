"""Model-specific transfer routes and independently audited paired evidence."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;O=P/'figures/l167';O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l167'})
def canvas(title,subtitle,height=8):
    fig,ax=plt.subplots(figsize=(12,height));fig.subplots_adjust(left=.025,right=.975,bottom=.035,top=.85)
    fig.patch.set_facecolor('#f4f7fa');ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    fig.text(.04,.96,title,fontsize=20,weight='bold',color='#193952')
    fig.text(.04,.915,subtitle,fontsize=11,color='#4b6172');return fig,ax
def box(ax,x,y,w,h,title,body,color='#e5edf8'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008',facecolor=color,edgecolor='#9ab0c4'))
    ax.text(x+.012,y+h-.023,title,va='top',weight='bold',fontsize=12,color='#193952')
    ax.text(x+.012,y+h-.078,body,va='top',fontsize=10.5,color='#2b4357',linespacing=1.5)
def arrow(ax,x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='->',color='#417e86',lw=2))
def save(fig,name):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for ax in fig.axes:
        for t in ax.texts:
            b=t.get_window_extent(renderer)
            assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,(name,t.get_text())
    fig.savefig(O/(name+'.svg'),metadata={'Date':None})
    fig.savefig(O/(name+'.png'),dpi=140,metadata={'Software':'L167'})
    plt.close(fig)
fig,ax=canvas('Three predictors; locate the relational handoff','S = support rows · Q = query rows · F = constructed features · all inference weights frozen',10)
box(ax,.015,.835,.97,.145,'Database → query-owned preparation','PK/FK join paths + strict owner cutoff → parent attributes / child aggregates → numeric task matrix\nXsupport [S,F], ysupport [S], Xquery [Q,F]   |   Same preparation for a fair comparator.','#e1f0e9')
for x in [.165,.50,.835]:arrow(ax,x,.825,x,.77)
lanes=[(.015,'TabPFN v2','Weights from tabular tasks','Cell / feature-group tokens\nFeature attention within rows\nSample attention across rows','Query reads support context\nDecode target representation\nEnsemble → [Q,K]','Conceptual comparison only'),
       (.35,'TabICL v1.1','Weights from tabular tasks','Column-conditioned cell encoding\nWithin-row attention → row vector\nLabeled-context ICL across rows','Query row → class head\n32 estimators in audited run\nBinary probabilities → [Q,2]','Measured: S=512, Q=702, F=72'),
       (.685,'RDB-PFN','Weights: tabular + RDB tasks','Scalar + label tokens, width 96\n6 feature / support-row blocks\nSupport keys/values only','Query target token → decoder\nBinary softmax → [Q,2]\nSame DFS input as TabICL','Measured: S=512, Q=702, F=72')]
for x,title,prior,body,out,foot in lanes:
    box(ax,x,.62,.30,.14,title,prior)
    arrow(ax,x+.15,.61,x+.15,.555)
    box(ax,x,.32,.30,.22,'Inside the predictor',body)
    arrow(ax,x+.15,.31,x+.15,.255)
    box(ax,x,.03,.30,.21,'Prediction',out,'#fff0d9')
    ax.text(x,.005,foot,fontsize=9.5,color='#4b6172')
save(fig,'architecture')
fig,ax=canvas('Two clocks must agree before a row becomes context','Course fixture: immediate event availability; labels have a separate readiness time',6.8)
box(ax,.02,.55,.44,.40,'Feature clock · query (customer 7, day 10)','day 2 → value 4     INCLUDED\nday 8 → value 8     INCLUDED\nday 10 → value 100  EXCLUDED\nday 14 → value 12   EXCLUDED','#e1f0e9')
box(ax,.55,.55,.43,.40,'Label clock · prediction cutoff 10','support A: query 2, label ready 9   YES\nsupport B: query 5, label ready 12  NO\nsupport C: query 10, label ready 10 NO\nPolicy: both times strictly before cutoff.','#e5edf8')
arrow(ax,.24,.53,.24,.38);arrow(ax,.76,.53,.76,.38)
box(ax,.02,.12,.44,.24,'Construct the features','Count = 2; mean = (4 + 8) / 2 = 6\nChanging day 14 cannot change this row.','#e1f0e9')
box(ax,.55,.12,.43,.24,'Construct the context','Only support A supplies a label.\nAttention masking cannot repair a late label.','#fff0d9')
ax.text(.025,.025,'At cutoff 20: count = 4, mean = 31, supports A/B/C all eligible. Each query owns its cutoff.',fontsize=11)
save(fig,'cutoff')
r=json.loads((P/'evidence/l167/report.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12,5.4),sharey=True);fig.patch.set_facecolor('#f4f7fa')
for ax,arm,title in zip(axes,['RDBPFN_single','TabICLv1.1'],['RDB-PFN − single-table checkpoint','RDB-PFN − TabICLv1.1']):
    d=r['paired'][arm];ax.axvline(0,color='#6c7984',lw=1);ax.scatter(d['per_seed'],range(10),s=55,color='#167d80',zorder=3)
    ax.axvline(d['mean'],color='#be7632',ls='--',label=f"Mean {d['mean']:+.5f}")
    ax.set(title=title,xlabel='Paired AUROC difference',yticks=range(10));ax.grid(axis='y',alpha=.15);ax.legend(loc='lower left',bbox_to_anchor=(0,-.36),frameon=False)
    ax.spines[['right','top']].set_visible(False)
axes[0].set_ylabel('Support draw seed');axes[0].invert_yaxis()
fig.suptitle('Same task, same test rows, matched support draws',fontsize=17,color='#193952')
fig.text(.06,.025,'Ten support draws are not ten databases. Fresh L167 metric audit; original L166 checkpoint predictions.',fontsize=10,color='#4b6172')
fig.tight_layout(rect=[0,.04,1,.93]);save(fig,'paired')
print('Built three L167 figures')
