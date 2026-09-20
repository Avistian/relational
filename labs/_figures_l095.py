"""Generate computation-first figures; numbers in results come from executed full folds."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;D=P/'figures/l095';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'#fcfcf9'})
teal='#087f8c';blue='#264c78';red='#ae4637';gold='#976c15'
def save(fig,name):
    fig.savefig(D/f'{name}.png',dpi=150,bbox_inches='tight');fig.savefig(D/f'{name}.svg',bbox_inches='tight');plt.close(fig)
def box(ax,xy,w,h,text,color=teal):
    x,y=xy;ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',ec=color,fc='white',lw=1.8));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=12,color='#17324a')
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.8,'color':blue})
fig,ax=plt.subplots(figsize=(12,6.3));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title('From interaction rows to a recommendation\nFull ML-100K experiment · course protocol',loc='left',fontsize=20,pad=20)
box(ax,(.02,.74),.24,.18,'Official fold\n80,000 base rows\n20,000 test rows',blue)
box(ax,(.38,.74),.24,.18,'Inner split of base\n72,000 fit + 8,000 val\nTest stays sealed',blue)
box(ax,(.74,.74),.24,.18,'Typed fitting graph\n943 users × 1,682 items\nB[u,i] = 1 if rating ≥ 4')
arrow(ax,(.27,.83),(.37,.83));arrow(ax,(.63,.83),(.73,.83))
box(ax,(.74,.40),.24,.20,'Three typed steps\nP = row_normalize(B)\nQ = row_normalize(Bᵀ)\nS = P Q P')
arrow(ax,(.86,.73),(.86,.62))
box(ax,(.38,.40),.24,.20,'Score each candidate\ns = αS + (1−α)pop\nα ∈ {0, ½, 1}\nSelect on val NDCG@10',gold)
arrow(ax,(.73,.50),(.63,.50))
box(ax,(.02,.40),.24,.20,'Refit full 80,000 base\nRecompute degrees\nand walk probabilities\nNo test-label access',blue)
arrow(ax,(.37,.50),(.27,.50))
box(ax,(.02,.06),.38,.20,'Prediction handoff: 943 × 1,682 scores\nMask ALL seen base ratings\nSort remaining items; keep top 10',teal)
box(ax,(.57,.06),.41,.20,'Final test: held-out ratings ≥ 4\nRecall@10 and NDCG@10 per eligible user\nAverage users, then five folds',gold)
arrow(ax,(.14,.39),(.14,.28));arrow(ax,(.41,.16),(.56,.16));save(fig,'pipeline')
fig,axes=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1,1.2]});a=axes[0];a.axis('off');a.set(xlim=(0,1),ylim=(0,1));a.set_title('1 · The same typed IDs stay distinct',loc='left',fontsize=15)
for u,y in [(0,.72),(1,.27)]:a.scatter(.12,y,s=900,color=blue);a.text(.12,y,f'u{u}',ha='center',va='center',color='white',fontsize=15)
for i,y in [(0,.88),(1,.5),(2,.12)]:a.scatter(.84,y,s=900,marker='s',color=teal);a.text(.84,y,f'i{i}',ha='center',va='center',color='white',fontsize=15)
for u,i in [(0,0),(0,1),(1,1),(1,2)]:a.plot([.2,.75],[{0:.72,1:.27}[u],{0:.88,1:.5,2:.12}[i]],color=teal,lw=2)
a.text(.02,.01,'Train likes: u0–i0, u0–i1, u1–i1, u1–i2',fontsize=10)
a=axes[1];a.axis('off');a.set_title('2 · Walk from u0 to the unseen item i2',loc='left',fontsize=15)
a.text(.02,.84,'u0 → i1 → u1 → i2',fontsize=24,color=teal)
a.text(.02,.66,'Probability:  ½ × ½ × ½ = ⅛',fontsize=21,color=blue)
a.text(.02,.45,'All three-step endpoints:\nS[u0] = [0.375, 0.500, 0.125]',fontsize=17,linespacing=1.7)
a.text(.02,.14,'Mask seen i0 and i1 → recommend i2.\nA low raw probability can still rank first\namong eligible items.',fontsize=13,linespacing=1.7)
fig.suptitle('A bipartite walk alternates types at every step',fontsize=20,y=1.04);fig.tight_layout();save(fig,'walk')
fig,axes=plt.subplots(1,3,figsize=(12,4));mats=[np.array([[1,1,0],[0,1,1]]),np.array([[2,1],[1,2]]),np.array([[1,1,0],[1,2,1],[0,1,1]])];names=['B: users × items','BBᵀ: users × users','BᵀB: items × items']
for ax,m,title in zip(axes,mats,names):
    ax.imshow(m,cmap='Blues',vmin=0,vmax=2);ax.set_title(title,fontsize=15)
    for (y,x),v in np.ndenumerate(m):ax.text(x,y,str(v),ha='center',va='center',fontsize=22,color='white' if v==2 else blue)
    ax.set_xticks(range(m.shape[1]));ax.set_yticks(range(m.shape[0]))
fig.suptitle('Projection keeps shared counts but loses the intermediate item identity',fontsize=17,y=1.05);fig.tight_layout();save(fig,'projection')
if (P/'_experiment_l095_results.json').exists():
    r=json.loads((P/'_experiment_l095_results.json').read_text());fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,metric in zip(axes,['recall','ndcg']):
        for j,(name,color) in enumerate([('popularity',blue),('walk',teal),('validation_selected',gold)]):
            ys=[x['test'][name][metric] for x in r['runs']];ax.scatter(np.arange(1,6)+(j-1)*.08,ys,label=name.replace('_',' '),color=color,s=45);ax.plot(range(1,6),ys,color=color,alpha=.4)
        ax.set(xlabel='Official fold (shared dataset)',ylabel=metric.upper()+'@10',xticks=range(1,6),ylim=(0,.5));ax.grid(alpha=.2)
    axes[1].legend(fontsize=10);fig.suptitle('Measured on all five complete folds · descriptive variation, not five datasets',fontsize=15);fig.tight_layout();save(fig,'results')
