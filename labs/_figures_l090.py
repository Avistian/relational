"""Generate explanatory snapshots and measured seed distributions."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R=Path(__file__).resolve().parent;out=R/'figures/l090';out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#faf8f2','axes.facecolor':'#faf8f2','font.family':'DejaVu Sans'})
fig,ax=plt.subplots(figsize=(11,4));ax.axis('off');ax.set(xlim=(0,11),ylim=(0,4))
ax.text(.2,3.65,'ONE GRAPH · TWO INFORMATION CONTRACTS',fontsize=17,weight='bold')
for x,title,degree,expression,value in [(.3,'Full path 0 — 1 — 2','d = [2, 3, 2]','2/√6 + 4/3 + 8/√6','5.415816'),(5.9,'Training edge 0 — 1','d = [2, 2]','2/2 + 4/2','3.000000')]:
 ax.text(x,2.9,title,fontsize=16,color='#176e79',weight='bold');ax.text(x,2.3,'Features [2, 4, 8]; W = 1' if x<1 else 'Node 2 absent before normalization')
 ax.text(x,1.8,degree);ax.text(x,1.15,expression+' =',fontsize=16);ax.text(x,.55,value,fontsize=24,color='#ad5e2a',weight='bold')
fig.savefig(out/'trace.png',dpi=150,bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(2,1,figsize=(12,6.8))
for ax in axs:ax.axis('off');ax.set(xlim=(-.2,12),ylim=(0,3))
def box(ax,x,y,w,h,text,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor=color,edgecolor='#587070'));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
def arrow(ax,x,y,x2,y2):ax.annotate('',xy=(x2,y2),xytext=(x,y),arrowprops={'arrowstyle':'->','lw':1.6})
axs[0].text(0,2.7,'A · PUBLISHED GCN PROTOCOL PORT',fontsize=16,weight='bold')
for x,t in [(0,'All Cora nodes\nX: 2708 × 1433\nrow-normalized'),(3.2,'input dropout\nS X W₀ → ReLU\n2708 × 16'),(6.4,'hidden dropout\nS H W₁\n2708 × 7')]:box(axs[0],x,1.1,2.6,1,t,'#e4efeb')
arrow(axs[0],2.65,1.6,3.1,1.6);arrow(axs[0],5.85,1.6,6.3,1.6)
axs[0].text(9.5,1.6,'140 labels → loss\n+ L2 on W₀ only\nAdam update',va='center');arrow(axs[0],9.05,1.6,9.4,1.6)
axs[0].text(0,.35,'S = D⁻¹ᐟ²(A + I)D⁻¹ᐟ²    •    Validation stops training    •    Dropout off → test argmax',fontsize=12)
axs[1].text(0,2.7,'B · INDUCTIVE MINI-BATCH EXTENSION',fontsize=16,weight='bold')
box(axs[1],0,1.05,2.6,1.1,'Eligible nodes only\n64 roots → 640 draws\n→ 6400 outer draws','#f4e5d5')
box(axs[1],3.2,1.05,3.2,1.1,'Shared W₁ for roots + draws\n[self ‖ neighbor mean]\n→ ReLU → L2 norm: 32','#f4e5d5')
box(axs[1],7,1.05,2.6,1.1,'[root h ‖ mean h]\n64 coordinates → 7\nroot labels → loss','#f4e5d5')
arrow(axs[1],2.65,1.6,3.1,1.6);arrow(axs[1],6.45,1.6,6.9,1.6)
axs[1].text(10,1.6,'Three updates\nper 140-label\ntraining pass',va='center',fontsize=11)
axs[1].text(0,.3,'Validation / test add their own nodes separately. Inference uses full eligible-neighbor means.',fontsize=12)
fig.tight_layout();fig.savefig(out/'architecture.png',dpi=150,bbox_inches='tight');plt.close(fig)
r=json.loads((R/'_paper_l090_results.json').read_text());q=json.loads((R/'_inductive_l090_results.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(11,4),gridspec_kw={'width_ratios':[3,1]})
for ax,d,title in [(axs[0],r,'GCN · 100 initializations'),(axs[1],q,'Inductive · 3 seeds')]:
 scores=np.array([v['test_accuracy'] for v in d['runs']])*100
 ax.scatter(np.arange(len(scores)),scores,s=22,color='#176e79');ax.axhline(scores.mean(),color='#176e79',label=f'Mean {scores.mean():.3f}%')
 ax.set(title=title,xlabel='Declared seed',ylabel='Test accuracy (%)');ax.legend(fontsize=10)
axs[0].axhspan(80.5,82.5,color='#cfc6a3',alpha=.25);axs[0].axhline(81.5,ls='--',color='#ad5e2a');axs[0].text(.02,.025,'Shaded: course tolerance, not a confidence interval',transform=axs[0].transAxes,fontsize=9,bbox={'facecolor':'#faf8f2','alpha':.85,'edgecolor':'none'})
fig.suptitle('Separate protocols — these panels are not a model ranking',fontsize=15);fig.tight_layout();fig.savefig(out/'results.png',dpi=150,bbox_inches='tight');plt.close(fig)
