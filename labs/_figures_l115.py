"""Model-specific computation maps and measured results, exported as SVG and PNG."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;D=P/'figures/l115';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none','svg.hashsalt':'l115','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'#fcfbf8'})
TEAL='#176c68';GOLD='#ae6625';INK='#213743'
def save(fig,name):
 fig.savefig(D/f'{name}.svg',bbox_inches='tight',metadata={'Date':None});fig.savefig(D/f'{name}.png',bbox_inches='tight',dpi=155,metadata={'Software':'L115'});plt.close(fig)
def box(ax,x,y,w,h,title,detail,color=TEAL):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',facecolor='#eef5f3',edgecolor=color,linewidth=1.4));ax.text(x+w/2,y+h*.67,title,ha='center',va='center',weight='bold',color=INK,fontsize=12);ax.text(x+w/2,y+h*.26,detail,ha='center',va='center',fontsize=10,color=INK)
def arrow(ax,x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':INK,'lw':1.5})
fig,ax=plt.subplots(figsize=(8.4,9));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title('One GCN, three module responsibilities',loc='left',weight='bold',pad=18)
box(ax,.03,.85,.64,.10,'Encoder = identity','X: 169,343 papers × 128 numeric features')
box(ax,.74,.63,.23,.31,'Graph S','Undirected\n+ deduplicate\n+ one self-loop\n+ symmetric\nnormalization')
box(ax,.03,.59,.64,.19,'Processor: two hidden GCN blocks','128 → 256 → 256\nEach: S(HWᵀ) + b → BN → ReLU → dropout .5')
box(ax,.03,.37,.64,.15,'Head: final GCN + log-softmax','256 → 40\nS(HW₃ᵀ) + b₃ still aggregates neighbors',GOLD)
arrow(ax,.35,.85,.35,.79);arrow(ax,.35,.59,.35,.53);arrow(ax,.74,.71,.67,.71);arrow(ax,.85,.63,.65,.46)
box(ax,.03,.14,.94,.15,'Train → select → predict','Train-label mean NLL → Adam .01 → 500 epochs\nFirst best validation state, including BN buffers → class argmax')
arrow(ax,.35,.37,.35,.30);ax.text(.03,.04,'All node features enter the forward pass; only training labels enter loss.',fontsize=11,color=INK)
save(fig,'architecture')
fig,ax=plt.subplots(figsize=(8.4,7.4));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title('The prediction unit determines the readout',loc='left',weight='bold',pad=16)
box(ax,.03,.83,.94,.12,'Fixed illustrative node vectors Z','z₀=[1,2]     z₁=[3,5]     z₂=[7,11]')
box(ax,.03,.60,.94,.13,'Node 0: select one row','[1,2] → node classifier → one class-score vector')
box(ax,.03,.35,.94,.17,'Candidate 0 → 1: combine endpoints','Undirected product: [3,10]     Reversed: [3,10]\nDirected concatenation: [1,2,3,5]     Reversed: [3,5,1,2]',GOLD)
box(ax,.03,.09,.94,.17,'Graph IDs [0,1,0]: pool within examples','Graph 0 mean: ([1,2]+[7,11])/2 = [4,6.5]\nGraph 1 mean: [3,5] → one prediction per graph')
# These are alternative readouts: no arrows connecting the alternatives.
ax.text(.03,.01,'Each box is an alternative readout of Z, not a sequence of learned layers.',fontsize=10);save(fig,'readouts')
fig,ax=plt.subplots(figsize=(8.4,7.2));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title('Shape is only one part of the contract',loc='left',weight='bold',pad=16)
for y,title,detail,c in [(.76,'Identity and type','Which row is customer 42?  Which coordinates encode product attributes?',TEAL),(.54,'Direction and membership','Does u → v mean the same as v → u?  Which graph owns each node?',TEAL),(.32,'Event time and availability','A day-4 event arriving on day 11 is excluded from a day-10 prediction.',GOLD),(.10,'Supervision and selection','Mature training labels only; validation selects; frozen test evaluates.',TEAL)]:
 box(ax,.03,y,.94,.16,title,detail,c)
save(fig,'boundaries')
s=json.loads((P/'evidence/l115/summary.json').read_text());fig,ax=plt.subplots(figsize=(8.4,4.8))
for i,(pop,color) in enumerate([('valid',TEAL),('test',GOLD)]):
 r=s['summary'][pop];ax.scatter([i+(j-4.5)*.035 for j in range(10)],[v[pop+'_percent'] for v in s['seeds']],color=color,alpha=.55,label=pop+' seeds');ax.errorbar(i+.24,r['mean_percent'],yerr=r['sample_sd_pp'],fmt='D',capsize=5,color=color);ax.hlines(r['target_percent'],i-.28,i+.32,color=color,linestyle='--')
ax.set_xticks([0,1],['Validation','Test']);ax.set_ylabel('Accuracy (%)');ax.set_title('Fresh modular GCN: ten complete 500-epoch fits',loc='left',weight='bold');ax.grid(axis='y',alpha=.2)
fig.text(.07,.02,'Dots: seeds. Diamond: mean ± sample seed SD. Dashed: OGB Table 6 mean.',fontsize=10);fig.tight_layout(rect=(0,.07,1,1));save(fig,'results');print('Four computation/results figures built')
