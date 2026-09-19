"""Rebuild all figures from the actual experiment output."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures/l085';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fbfaf6','axes.facecolor':'#fbfaf6','savefig.facecolor':'#fbfaf6'})
def save(fig,name):fig.savefig(OUT/(name+'.png'),dpi=170,bbox_inches='tight');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(10,3.7));s=np.array([[.5,6**-.5,0],[6**-.5,1/3,6**-.5],[0,6**-.5,.5]])
axs[0].imshow(s,cmap='Blues',vmin=0,vmax=.6)
for i in range(3):
 for j in range(3):axs[0].text(j,i,f'{s[i,j]:.3f}',ha='center',va='center',color='white' if s[i,j]>.35 else '#142e45')
axs[0].set(xticks=range(3),xticklabels=['sender A','sender B','sender C'],yticks=range(3),yticklabels=['receiver A','receiver B','receiver C'],title='A—B—C + one self-loop each')
v=np.array([2,4,8]);axs[1].bar(np.arange(3)-.18,v,.36,label='input',color='#aabcc8');axs[1].bar(np.arange(3)+.18,s@v,.36,label='after S × input',color='#187c91');axs[1].set(xticks=range(3),xticklabels=['A','B','C'],ylabel='Scalar feature',title='B = 2/√6 + 4/3 + 8/√6 = 5.416');axs[1].legend();fig.tight_layout();save(fig,'trace')

fig,ax=plt.subplots(figsize=(10,4.4));ax.axis('off')
texts=[('INPUT\n34 nodes · 78 edges\nX = I₃₄ (one-hot features)',.02,.61,.27),('MIX + TRANSFORM\nH′ = ReLU(S H W)\n34 × 16 hidden states',.36,.61,.28),('OUTPUT\nS H W → 34 × 2\ntwo coordinates per node',.71,.61,.27)]
for t,x,y,w in texts:ax.add_patch(plt.Rectangle((x,y),w,.31,facecolor='#e3eef0',edgecolor='#187c91',lw=1.5));ax.text(x+w/2,y+.155,t,ha='center',va='center',linespacing=1.7)
for x,y in [(.3,.35),(.65,.7)]:ax.annotate('',xy=(y,.76),xytext=(x,.76),arrowprops={'arrowstyle':'->','lw':2,'color':'#187c91'})
ax.text(.5,.46,'Depth 1: input → output  |  Depth 5: four hidden layers → output',ha='center',fontsize=12)
ax.text(.5,.29,'Sᵢⱼ = (A + I)ᵢⱼ / √(dᵢ dⱼ)   ·   each W is shared across nodes\nDifferent layers own different W · Glorot uniform · hidden width 16',ha='center',linespacing=1.7)
ax.text(.5,.08,'NO TRAINING: no loss, optimizer, split or checkpoint\nClub labels only color points; they do not enter the model.',ha='center',color='#935329',linespacing=1.7);save(fig,'architecture')

r=json.loads((ROOT/'_paper_l085_results.json').read_text());labels=json.loads((ROOT/'sources/l085/karate.json').read_text())['labels'];rows=[x for x in r['runs'] if x['seed']==0]
fig,axs=plt.subplots(1,5,figsize=(13,3.2),sharex=True,sharey=True)
coords=np.array([x['coordinates'] for x in rows]);lim=float(np.abs(coords).max())*1.15
for ax,row in zip(axs,rows):
 z=np.array(row['coordinates'])
 for label,color,marker in [(0,'#177d91','o'),(1,'#be633a','^')]:
  mask=np.array(labels)==label;ax.scatter(z[mask,0],z[mask,1],c=color,marker=marker,s=23,alpha=.85,label=['Mr. Hi','Officer'][label])
 ax.set(title=f"Depth {row['depth']}",xlim=(-lim,lim),ylim=(-lim,lim),xlabel='Coordinate 1');ax.axhline(0,color='#ccc',lw=.5);ax.axvline(0,color='#ccc',lw=.5)
axs[0].set_ylabel('Coordinate 2');axs[-1].legend(fontsize=8);fig.suptitle('Full karate setup · untrained seed 0 · identical axes · classes shown only by color/shape',fontsize=12);fig.tight_layout();save(fig,'karate')

fig,axs=plt.subplots(1,2,figsize=(10,3.7));pure=r['pure_propagation_extension'];axs[0].semilogy([x['depth'] for x in pure],[max(x['degree_variance'],1e-30) for x in pure],'o-',color='#177d91');axs[0].set(xlabel='Fixed propagation steps',ylabel='Degree-corrected variance',title='Sᵏ I: no weights or activations')
ds=range(1,6);v=np.array([[x['mean_cosine'] for x in r['runs'] if x['depth']==d and x['mean_cosine'] is not None] for d in ds]);axs[1].plot(ds,np.median(v,axis=1),'o-',color='#be633a');axs[1].fill_between(ds,np.quantile(v,.1,axis=1),np.quantile(v,.9,axis=1),color='#be633a',alpha=.18);axs[1].set(xlabel='Untrained GCN depth',ylabel='Mean off-diagonal cosine',ylim=(-.1,1.05),xticks=list(ds),title='100 seeds: median + 10–90% range');fig.tight_layout();save(fig,'mixing')

p=ROOT/'_depth_l085_results.json'
if p.exists():
 r=json.loads(p.read_text());fig,axs=plt.subplots(1,2,figsize=(10,3.7));ds=r['depths']
 for key,color,label in [('test_accuracy','#177d91','test'),('train_accuracy','#be633a','train')]:
  v=np.array([[x[key] for x in r['runs'] if x['depth']==d] for d in ds]);axs[0].errorbar(ds,v.mean(1)*100,yerr=v.std(1,ddof=1)*100,marker='o',color=color,label=label,capsize=3)
  for d,vals in zip(ds,v):axs[0].scatter([d]*len(vals),vals*100,color=color,s=8,alpha=.4)
 axs[0].set(xlabel='GCN layers',ylabel='Accuracy (%)',title='Full Cora: means ± sample SD');axs[0].legend()
 for d in ds:
  vals=[x['mean_cosine'] for x in r['runs'] if x['depth']==d];axs[1].scatter([d]*len(vals),vals,s=18,color='#177d91');axs[1].plot(d,np.mean(vals),'_',ms=18,color='black')
 axs[1].set(xlabel='GCN layers (depth 1 uses logits)',ylabel='Mean off-diagonal cosine',title='Each dot is one initialization',ylim=(-.05,1.05))
 for ax in axs:ax.set_xscale('log',base=2);ax.set_xticks(ds,ds)
 fig.tight_layout();save(fig,'depth')
