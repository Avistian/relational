"""Portable computation figures, shared by HTML and both notebooks."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
LAB=Path(__file__).resolve().parent;OUT=LAB/'figures/l088';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
BLUE='#2563eb';TEAL='#059669';ORANGE='#d97706';INK='#172554'
def save(fig,name):fig.savefig(OUT/f'{name}.png',dpi=170,bbox_inches='tight',facecolor='white');plt.close(fig)
def box(ax,x,y,w,h,title,body,color=BLUE):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.006',edgecolor=color,facecolor='#f8fafc',lw=1.4))
 ax.text(x+w/2,y+h-.035,title,ha='center',va='top',weight='bold',color=color,fontsize=11)
 ax.text(x+w/2,y+h/2-.02,body,ha='center',va='center',fontsize=10,linespacing=1.5)
def arrow(ax,a,b):ax.annotate('',b,a,arrowprops={'arrowstyle':'->','color':INK,'lw':1.5})
fig,ax=plt.subplots(figsize=(11.5,7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(0,1.02,'GIN-0 · four learned updates, five graph readouts',fontsize=18,weight='bold',color=INK)
box(ax,.02,.72,.24,.23,'MOLECULE BATCH','7-way atom tags X: [N,7]\nDisjoint edges: [2,E]\nGraph IDs: [N]')
box(ax,.37,.72,.27,.23,'ONE NODE UPDATE','center [1,0] + neighbors [1,1]\n= [2,1] → two-layer MLP\n→ outer BN → ReLU',TEAL)
box(ax,.74,.72,.23,.23,'REPEAT × 4','H¹ … H⁴: [N,d]\nNew weights at each depth\nShared across nodes',TEAL)
arrow(ax,(.26,.84),(.37,.84));arrow(ax,(.64,.84),(.74,.84))
ax.text(.5,.645,'Retain X and every hidden state — do not discard the shallow branch',ha='center',fontsize=12,color=INK)
for i,label in enumerate(['X [N,7]','H¹ [N,d]','H² [N,d]','H³ [N,d]','H⁴ [N,d]']):
 x=.02+i*.196
 box(ax,x,.34,.176,.21,label,'sum within graph\n→ [B,7] or [B,d]\n→ affine → [B,2]')
 arrow(ax,(x+.088,.61),(x+.088,.55));arrow(ax,(x+.088,.34),(x+.088,.285))
ax.plot([.108,.892],[.285,.285],color=INK,lw=1.5)
box(ax,.24,.075,.52,.15,'SUM DEPTH LOGITS → CROSS-ENTROPY','Train: per-head logit dropout + graph labels\nPredict: dropout off; running BN statistics; argmax',ORANGE)
arrow(ax,(.5,.285),(.5,.225))
ax.text(.02,.0,'N = total nodes · B = graphs · d ∈ {16,32} · ε = 0 · no bond-label input',fontsize=10,color='#475569')
save(fig,'architecture')
fig,ax=plt.subplots(figsize=(10,3.4));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
ax.text(0,1.02,'Which information survives graph readout?',fontsize=17,weight='bold',color=INK)
rows=[['{A,A,B}','[2,1]','[2/3,1/3]','[1,1]'],['{A,A,A,A,B,B}','[4,2]','[2/3,1/3]','[1,1]']]
t=ax.table(cellText=rows,colLabels=['Node multiset','Sum: counts','Mean: proportions','Max: presence'],cellLoc='center',bbox=[0,.22,1,.64]);t.auto_set_font_size(False);t.set_fontsize(12)
for (r,c),cell in t.get_celld().items():cell.set_edgecolor('#cbd5e1');cell.set_facecolor('#eff6ff' if r==0 else '#ecfdf5' if c==1 else 'white')
ax.text(0,.06,'A = [1,0], B = [0,1]. Synthetic encoded categories; raw scalar sums can still collide.',fontsize=11)
save(fig,'pooling')
fig,axes=plt.subplots(1,4,figsize=(11.5,3.3))
spec=[('Path · round 1',4,[(0,1),(1,2),(2,3)],[0,1,1,0],'1-neighbor ×2; 2-neighbor ×2'),('Star · round 1',4,[(0,1),(0,2),(0,3)],[2,0,0,0],'1-neighbor ×3; 3-neighbor ×1'),('Cycle · every round',6,[(i,(i+1)%6) for i in range(6)],[0]*6,'One color ×6'),('Triangles · every round',6,[(0,1),(1,2),(2,0),(3,4),(4,5),(5,3)],[0]*6,'One color ×6')]
for ax,(title,n,edges,cs,caption) in zip(axes,spec):
 pos=np.array([[np.cos(2*np.pi*i/n),np.sin(2*np.pi*i/n)] for i in range(n)])
 if title.startswith('Triangles'):pos=np.array([[-.65,.7],[-1.,-.3],[-.3,-.3],[.65,.7],[.3,-.3],[1.,-.3]])
 for a,b in edges:ax.plot(*pos[[a,b]].T,color='#94a3b8',zorder=1,lw=2)
 ax.scatter(*pos.T,s=450,c=[[BLUE,ORANGE,TEAL][c] for c in cs],zorder=2)
 ax.set_title(title,fontsize=11,weight='bold');ax.set(xlim=(-1.4,1.4),ylim=(-1.5,1.4));ax.axis('off');ax.text(0,-1.4,caption,ha='center',fontsize=9)
fig.suptitle('WL counts separate some graphs — equal counts do not prove isomorphism',fontsize=15,weight='bold');fig.tight_layout();save(fig,'wl')
p=LAB/'_paper_l088_results.json'
if p.exists():
 r=json.loads(p.read_text());fig,ax=plt.subplots(figsize=(10,4.5))
 for i,c in enumerate(r['candidates']):
  values=100*np.array(c['fold_values']);ax.scatter(i+np.linspace(-.07,.07,10),values,color=BLUE,alpha=.5,s=24)
  ax.scatter(i,100*c['mean'],marker='_',s=350,color=INK,zorder=3)
 ax.axhline(89.4,color=ORANGE,ls='--',label='Paper target 89.4%; historical parity INCOMPARABLE')
 ax.set_xticks(range(len(r['candidates'])),[f"d{c['config']['hidden']} / b{c['config']['batch_size']}\np{c['config']['dropout']} · e{c['epoch']}" for c in r['candidates']],fontsize=8)
 ax.set(xlim=(-.5,len(r['candidates'])-.5),ylabel='Selected CV accuracy (%)',ylim=(40,103),title='Completed configurations · dots are validation folds, bars are means')
 if len(r['candidates'])==1:
  ax.clear();c=r['candidates'][0];a=c['config']
  ax.scatter(range(1,11),100*np.array(c['fold_values']),s=55,color=BLUE,label='Held-out fold at the common selected epoch')
  ax.axhline(100*c['mean'],color=INK,label=f"Mean {100*c['mean']:.2f}%; sample SD {100*c['sample_sd']:.2f}pp")
  ax.axhline(89.4,color=ORANGE,ls='--',label='Paper target 89.4%; historical parity INCOMPARABLE')
  ax.set_xticks(range(1,11));ax.set(xlabel='Validation fold (overlapping training sets)',ylabel='Selected CV accuracy (%)',ylim=(40,103),title=f"One fixed configuration · width {a['hidden']}, batch {a['batch_size']}, dropout {a['dropout']} · epoch {c['epoch']}")
 ax.legend(fontsize=9,loc='lower left');fig.tight_layout();save(fig,'results')
