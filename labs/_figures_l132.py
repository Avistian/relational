"""Computation-specific portable figures; all displayed results come from measured fixture."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;D=P/'figures/l132';D.mkdir(parents=True,exist_ok=True)
r=json.loads((P/'evidence/l132/fixture.json').read_text())
plt.rcParams.update({'font.size':12,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'l132'})
blue='#2563a6';orange='#b45309';green='#087f72';ink='#19334b'
def save(fig,name):
 fig.savefig(D/(name+'.svg'),bbox_inches='tight',facecolor='white',metadata={'Date':None});fig.savefig(D/(name+'.png'),dpi=160,bbox_inches='tight',facecolor='white');plt.close(fig)
def box(ax,x,y,w,h,label,color=blue):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.03',facecolor=color+'12',edgecolor=color,lw=1.5));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=11,color=ink)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':ink,'lw':1.5})
fig,ax=plt.subplots(figsize=(12,7));ax.set(xlim=(0,12),ylim=(0,7));ax.axis('off')
ax.text(.1,6.7,'Two complete predictors — different questions inside the network',fontsize=17,weight='bold',color=ink)
ax.text(.1,6.15,'TWO-TOWER GRAPHSAGE  ·  2 layers  ·  shared network weights',color=blue,weight='bold')
box(ax,.15,4.6,2,1.1,'Source + clock\nTyped features');box(ax,2.7,4.6,3,1.1,'Table ResNets → time addition\n2 graph layers → root head\nB × 128')
box(ax,.15,3,2,1.1,'Sponsor + clock\nTyped features');box(ax,2.7,3,3,1.1,'Row + time + ID embedding\n2 graph layers → root head\nD × 128')
box(ax,6.4,3.8,2.3,1.3,'U @ Vᵀ\nB × D scores\nrank sponsors')
box(ax,9.35,3.8,2.2,1.3,'Training: BPR\nsoftplus(s− − s+)\nshared negatives')
for a,b in [((2.15,5.15),(2.65,5.15)),((2.15,3.55),(2.65,3.55)),((5.7,5.15),(6.4,4.65)),((5.7,3.55),(6.4,4.05)),((8.7,4.45),(9.35,4.45))]:arrow(ax,a,b)
ax.text(.1,2.55,'IDENTITY-AWARE RELBENCH  ·  4 layers  ·  source-conditioned sponsor vectors',color=orange,weight='bold')
box(ax,.15,.7,2.15,1.35,'One disjoint graph\nper condition query\nroot is occurrence q',orange)
box(ax,2.8,.7,2.35,1.35,'ResNet row vectors\nroot += marker\n[1,128] + time',orange)
box(ax,5.65,.7,2.35,1.35,'4 typed graph layers\ncandidate readout\nNcand × 1 logits',orange)
box(ax,8.55,.7,3,1.35,'BCE on (owner, sponsor)\nSigmoid → dense B × D\nUnseen sponsors score 0',orange)
for a,b in [((2.3,1.35),(2.8,1.35)),((5.15,1.35),(5.65,1.35)),((8,1.35),(8.55,1.35))]:arrow(ax,a,b)
ax.text(.2,.15,'Keys define graph connectivity. The marker is one shared role vector, not a condition-ID lookup table.',fontsize=11)
save(fig,'architecture')
fig,axs=plt.subplots(1,3,figsize=(12,3.8),gridspec_kw={'width_ratios':[1,1,1.2]})
for ax,row in zip(axs[:2],r['cycle_witness']):
 coords=np.c_[np.cos(np.arange(6)*np.pi/3),np.sin(np.arange(6)*np.pi/3)]
 edges=[(i,(i+1)%6) for i in range(6)] if row['graph']=='six-cycle' else [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3)]
 for u,v in edges:ax.plot(coords[[u,v],0],coords[[u,v],1],color='#bccad4',zorder=0)
 ax.scatter(coords[:,0],coords[:,1],s=330,c=[orange]+[blue]*5)
 for i,(x,y) in enumerate(coords):ax.text(x,y,str(i),ha='center',va='center',color='white')
 ax.set_title(row['graph']);ax.axis('off');ax.set_aspect('equal')
 ax.text(0,-1.5,f"Plain after 3 sums: all 8\nMarked root return: {row['root_return']:.0f}",ha='center')
ax=axs[2];ax.axis('off');ax.text(0,.92,'Follow one marked signal',weight='bold',fontsize=14)
ax.text(0,.7,'m⁰ = [1, 0, 0, 0, 0, 0]\nmˡ⁺¹ = A mˡ\n\nAt step 3, a triangle returns\ntwo walks to its marked root.\nThe six-cycle returns none.\n\nNo node IDs are learned.',va='top',linespacing=1.5)
fig.suptitle('A degree-only collision becomes a root-relative computation',fontsize=16);fig.tight_layout();save(fig,'collision')
fig,ax=plt.subplots(figsize=(10,4));ax.axis('off')
columns=['Occurrence','Query owner','Sponsor ID','Is positive?','Global-ID bug']
rows=[['0','0','7','1','1'],['1','0','8','0','1'],['2','1','7','0','1'],['3','1','8','1','1']]
t=ax.table(cellText=rows,colLabels=columns,loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(13);t.scale(1,2)
for i in [2,3]:t[i,4].set_facecolor('#ffe5d0')
ax.set_title('The same sponsor can have different labels in different queries',pad=20,fontsize=16)
ax.text(.01,.02,'Positive pairs: {(query 0, sponsor 7), (query 1, sponsor 8)}.\nCompare owner + B × sponsor_id; ID alone merges two different prediction questions.',transform=ax.transAxes)
save(fig,'ownership')
fig,ax=plt.subplots(figsize=(12,3.8));ax.set(xlim=(0,12),ylim=(0,4));ax.axis('off')
labels=['condition\nroot','conditions_studies\nassociation row','study\ntrial','sponsors_studies\nassociation row','sponsor\ncandidate']
for i,label in enumerate(labels):
 x=.15+i*2.4;box(ax,x,1.35,2.0,1,label,orange if i in [0,4] else blue);ax.text(x+1,1.0,f'{i} hops',ha='center')
 if i<4:arrow(ax,(x+2,1.85),(x+2.4,1.85))
ax.text(.15,3.2,'Why the selected ID-GNN uses four layers',fontsize=17,weight='bold')
ax.text(.15,.15,'Two hops reach the study, not the sponsor. Four hops permit a sponsor candidate; sampling may still miss it.\nEvery dated row must remain at or before the query cutoff. Static-row availability is an assumption.',fontsize=12)
save(fig,'reach')
fig,axs=plt.subplots(1,2,figsize=(11,4))
for name,color in [('unmarked',blue),('marked',orange)]:axs[0].plot(np.arange(1,81),r['course_fits'][name]['loss_trace'],label=name,color=color)
axs[0].set(xlabel='Training epoch',ylabel='Binary cross-entropy',title='Same fixture, initialization and objective');axs[0].legend()
x=np.arange(4)
axs[1].bar(x-.18,r['course_fits']['unmarked']['probabilities'],.35,label='unmarked',color=blue);axs[1].bar(x+.18,r['course_fits']['marked']['probabilities'],.35,label='marked',color=orange)
axs[1].scatter(x,r['targets'],marker='x',c='black',label='label',zorder=5);axs[1].set(xticks=x,xticklabels=['q0 / 7','q0 / 8','q1 / 7','q1 / 8'],ylim=(-.05,1.1),ylabel='Probability',title='Query-specific predictions');axs[1].legend()
fig.suptitle('Measured synthetic ablation — not a RelBench benchmark result',fontsize=15);fig.tight_layout();save(fig,'fixture')
print('Wrote five SVG/PNG figures')
