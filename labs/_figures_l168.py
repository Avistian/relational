"""Two-database information flow, adaptation contracts, and measured paired gains."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;O=P/'figures/l168';O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l168'})
def canvas(title,subtitle,height=8):
    fig,ax=plt.subplots(figsize=(12,height));fig.subplots_adjust(left=.025,right=.975,bottom=.04,top=.85)
    fig.patch.set_facecolor('#f4f7fa');ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    fig.text(.04,.96,title,fontsize=20,weight='bold',color='#193952')
    fig.text(.04,.915,subtitle,fontsize=11,color='#4b6172');return fig,ax

def box(ax,x,y,w,h,title,body,color='#e5edf8'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008',facecolor=color,edgecolor='#9ab0c4'))
    ax.text(x+.012,y+h-.023,title,va='top',weight='bold',fontsize=12,color='#193952')
    ax.text(x+.012,y+h-.08,body,va='top',fontsize=10.5,color='#2b4357',linespacing=1.5)

def arrow(ax,x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='->',color='#417e86',lw=2))

def save(fig,name):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for ax in fig.axes:
        for t in ax.texts:
            b=t.get_window_extent(renderer)
            assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,(name,t.get_text())
    fig.savefig(O/(name+'.svg'),metadata={'Date':None});fig.savefig(O/(name+'.png'),dpi=140,metadata={'Software':'L168'});plt.close(fig)

fig,ax=canvas('One frozen predictor; two database interfaces','The target databases supply features and 512 labels. Their test targets never enter context.',10.2)
box(ax,.025,.80,.95,.18,'Before target inference · reported RDB-PFN training route','Spider / BIRD schemas → LayerDAG → synthetic schema / keys / cells → DFS tasks → pretrained weights\nSynthetic predictor data is reported. Exact target-schema exclusion and checkpoint lineage are not independently established.','#fff0d9')
box(ax,.025,.52,.44,.22,'F1 · reused measured evidence','drivers ← results → races\nReleased DFS → 72 features; 3 MAX clocks checked\nSupport [512,72] + labels; queries [702,72]','#e1f0e9')
box(ax,.535,.52,.44,.22,'Clinical trials · fresh measured evidence','studies ← associations → conditions / sponsors\nReleased DFS → 176 features; 12 MAX clocks checked\nSupport [512,176] + labels; queries [825,176]','#e1f0e9')
arrow(ax,.5,.79,.5,.45);arrow(ax,.245,.50,.245,.44);arrow(ax,.755,.50,.755,.44)
box(ax,.025,.22,.95,.21,'Same checkpoint architecture · variable feature axis','Scalar + target tokens [512 + Q, F + 1, 96] → 6 feature / support-row attention blocks\nQuery target token → 2-class decoder → probabilities [Q,2]; no target optimizer updates\nMatched comparators: single-table RDB-PFN weights; TabICLv1.1 with 32 estimators. Same DFS and support.','#e5edf8')
arrow(ax,.5,.20,.5,.14)
box(ax,.025,.015,.95,.115,'Evaluate by task, then by database','Complete entity/date keys → paired AUROC by seed → one mean gain per selected database','#fff0d9')
save(fig,'architecture')

fig,ax=canvas('Two independent questions: exclusion and adaptation','Worked target: trial. A complete source-data inventory {shop, forum} is a stipulated teaching example.',8.4)
box(ax,.02,.65,.45,.30,'Question 1 · what did pretraining see?','Target absent + complete inventory → HELD_OUT\nTarget present → SEEN\nTarget absent + incomplete inventory\n→ NOT_ESTABLISHED','#e1f0e9')
box(ax,.54,.65,.44,.30,'Question 2 · how is the target used?','512 labels + frozen weights → FEW_SHOT_ICL\n512 labels + 5 updates → supervised adaptation\n0 labels + 0 updates → zero-label category\n(not a measured RDB-PFN execution mode)','#e5edf8')
ax.plot([.24,.24,.76,.76],[.63,.58,.58,.63],color='#417e86',lw=2)
arrow(ax,.24,.58,.24,.52);arrow(ax,.76,.58,.76,.52)
box(ax,.02,.23,.45,.27,'Griffin · graph route','Unified row encoders → graph message passing\n→ task decoder → supervised target adaptation\nDeclare target labels + optimizer/selection budget.\nL164 fresh experiment: budget-blocked.','#e5edf8')
box(ax,.54,.23,.44,.27,'RDB-PFN · context route','DFS matrix → feature / support attention\n→ class decoder; fixed pretrained weights\n512 target labels enter the context.\nL168: complete selected inference experiment.','#e5edf8')
ax.text(.025,.11,'A score cannot certify a missing pretraining inventory. Changing the evidence can change the permitted claim',fontsize=11)
ax.text(.025,.065,'while leaving the prediction unchanged. No fresh Griffin score is mixed into the measured comparison.',fontsize=11)
save(fig,'protocol')

r=json.loads((P/'evidence/l168/report.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12,5.8),sharex=True,sharey=True);fig.patch.set_facecolor('#f4f7fa')
for ax,db,title in zip(axes,['rel-f1','rel-trial'],['F1 / driver-dnf · reused','Trial / study-outcome · fresh']):
    d=r['paired'][db];ax.axvline(0,color='#6c7984',lw=1);ax.scatter(d['per_seed'],range(10),s=55,color='#167d80',zorder=3)
    ax.axvline(d['mean'],color='#be7632',ls='--',label=f"Mean {d['mean']:+.5f}")
    ax.set(title=title,xlabel='RDB-PFN − TabICLv1.1 · AUROC',yticks=range(10));ax.grid(axis='y',alpha=.15)
    ax.legend(loc='lower left',bbox_to_anchor=(0,-.33),frameon=False);ax.spines[['right','top']].set_visible(False)
axes[0].set_ylabel('Support draw seed');axes[0].invert_yaxis()
fig.suptitle('A second database adds a second transfer observation',fontsize=17,color='#193952')
fig.text(.065,.04,f"Equal-database mean gain: {r['macro']['macro_gain']:+.5f}. Two selected tasks; support seeds do not measure database uncertainty.",fontsize=10,color='#4b6172')
fig.tight_layout(rect=[0,.06,1,.93]);save(fig,'results')
print('Built three L168 figures')
