"""Portable computational figures generated from recorded data and explicit fixtures."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import t
LAB=Path(__file__).resolve().parent;OUT=LAB/'figures/l087';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':12,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
ink='#153b4e';teal='#267d78';orange='#c26729'
def save(fig,name):fig.savefig(OUT/(name+'.png'),dpi=150,bbox_inches='tight',facecolor=fig.get_facecolor());plt.close(fig)
def box(ax,x,y,w,h,txt,color=teal):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',facecolor='white',edgecolor=color,lw=2));ax.text(x+w/2,y+h/2,txt,ha='center',va='center',color=ink,linespacing=1.5)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':orange,'lw':2})
fig,axs=plt.subplots(1,2,figsize=(10,3.6),facecolor='#f7f5ef')
for ax,leak in zip(axs,[False,True]):
 ax.axis('off');ax.set(xlim=(-.5,2.5),ylim=(-.9,1.1));ax.set_title('Reverse edge absent' if not leak else 'Forbidden A → C still present',color=ink,weight='bold')
 for i in range(2):ax.plot([i,i+1],[0,0],color=teal,lw=3,zorder=0)
 if leak:ax.annotate('',xy=(2,.12),xytext=(0,.12),arrowprops={'arrowstyle':'->','color':orange,'lw':3,'connectionstyle':'arc3,rad=-.4'})
 for i,label in enumerate('ABC'):
  ax.scatter(i,0,s=1000,color='white',edgecolor=ink,lw=2,zorder=2);ax.text(i,0,label,ha='center',va='center',weight='bold');ax.text(i,-.4,'X='+str([1,0,0][i]),ha='center')
 ax.text(1,-.75,'sum-neighbor H = '+('[0,1,1]' if leak else '[0,1,0]'),ha='center',weight='bold',color=orange if leak else teal)
fig.suptitle('Hide the input relationship, not just its loss label',fontsize=16,weight='bold',color=ink);fig.tight_layout();save(fig,'split')
fig,ax=plt.subplots(figsize=(11,5),facecolor='#f7f5ef');ax.axis('off');ax.set(xlim=(0,11),ylim=(0,5))
ax.text(.1,4.65,'TEACHING GCN → DOT-PRODUCT EDGE SCORE',weight='bold',fontsize=17,color=ink)
box(ax,.15,2.6,2.4,1.3,'Context edges only\nS = D⁻½(A+I)D⁻½\nTrainable IDs E [N,32]')
box(ax,3.35,2.6,3.1,1.3,'H = ReLU(S E W₁)\nZ = S H W₂  [N,16]\nShared across queries')
box(ax,7.25,2.6,3.4,1.3,'Gather zᵤ, zᵥ [16]\ns(u,v) = Σⱼ zᵤⱼ zᵥⱼ\nOne logit per pair [B]')
arrow(ax,(2.7,3.25),(3.2,3.25));arrow(ax,(6.6,3.25),(7.1,3.25))
ax.text(.25,1.8,'TRACE   z_A=[1,2], z_C=[3,−1] → products [3,−2] → logit 1 → sigmoid .731',color=teal,weight='bold')
ax.text(.25,1.2,'TRAIN   Context graph is fixed; BCE on disjoint supervision pairs updates E, W₁ and W₂.',fontsize=11)
ax.text(.25,.72,'SELECT   Highest validation AUC, earliest tie; restore weights before any test scoring.',fontsize=11)
ax.text(.25,.24,'INFER   Encode once → score eligible pairs. Symmetric decoder; no unseen-node ID support.',fontsize=11)
save(fig,'architecture')
fig,axs=plt.subplots(1,2,figsize=(10,3.6),gridspec_kw={'width_ratios':[1.15,1]},facecolor='#f7f5ef')
ax=axs[0];ax.barh(['negative .1','positive .5','negative .5','negative .6'],[.1,.5,.5,.6],color=['#b8c3c5',orange,'#b8c3c5','#b8c3c5']);ax.set_xlim(0,.8);ax.set_xlabel('score (fixed synthetic fixture)');ax.set_title('One strictly higher · one tie',color=ink)
axs[1].axis('off');axs[1].text(.05,.82,'Optimistic rank = 2\nPessimistic rank = 3\nAverage rank = 2.5\nReciprocal = 1/2.5 = .4',va='top',linespacing=1.8,fontsize=14,color=ink);axs[1].text(.05,.1,'Together with a rank-1 query:\nMRR = (.4+1)/2 = .7',color=teal,weight='bold')
fig.tight_layout();save(fig,'ranking')
fig,axs=plt.subplots(1,2,figsize=(11,4.5),gridspec_kw={'width_ratios':[1.1,1.3]},facecolor='#f7f5ef')
ax=axs[0];ax.axis('off');pos={'A':(0,1),'B':(1,2),'C':(2,1),'D':(1,0),'E':(3,1)}
for u,v in [('A','B'),('B','C'),('A','D'),('D','C'),('C','E')]:ax.plot([pos[u][0],pos[v][0]],[pos[u][1],pos[v][1]],color=teal,lw=2,zorder=0)
ax.plot([0,2],[1,1],ls='--',color=orange);ax.text(.9,1.13,'remove A—C',ha='center',fontsize=10,color=orange)
for u,(x,y) in pos.items():
 lab={'A':1,'C':1,'B':2,'D':2,'E':0}[u];ax.scatter(x,y,s=1050,facecolor='white',edgecolor=ink,zorder=2,lw=2);ax.text(x,y,f'{u}\nz={lab}',ha='center',va='center',fontsize=11)
ax.set(xlim=(-.6,3.6),ylim=(-.55,2.5));ax.set_title('Candidate roots A,C · h=1',color=ink,weight='bold')
ax=axs[1];ax.axis('off');ax.text(.03,.96,'ROOT-RELATIVE LABELS',va='top',fontsize=15,weight='bold',color=ink);ax.text(.03,.82,'B,D: distances (1,1) → label 2\nE: unreachable with C deleted → label 0\nA,C: roots → label 1',va='top',linespacing=1.8)
ax.text(.03,.45,'SEAL CLASSIFIER PATH (not trained here)',color=orange,fontsize=11,weight='bold');ax.text(.03,.36,'Labeled enclosing subgraph\n↓ shared graph convolutions\n↓ SortPooling → conv/dense layers\nLink-existence prediction',va='top',linespacing=1.65,color=ink)
fig.tight_layout();save(fig,'seal')
r=json.loads((LAB/'_paper_l087_results.json').read_text());names=list(r['summary']);fig,axs=plt.subplots(2,4,figsize=(12,6.8),sharey=True,facecolor='#f7f5ef')
for ax,name in zip(axs.flat,names):
 for j,m in enumerate(['CN','AA','RA']):
  a=np.array([x['auc'][m] for x in r['runs'] if x['dataset']==name])*100;ax.scatter(j+np.linspace(-.10,.10,len(a)),a,color=teal,s=13,alpha=.6);ci=t.ppf(.975,9)*a.std(ddof=1)/np.sqrt(10);ax.errorbar(j,a.mean(),yerr=ci,color=orange,fmt='D',capsize=4,markersize=5)
 ax.set_title(name);ax.set_xticks(range(3),['CN','AA','RA']);ax.set_ylim(45,101);ax.grid(axis='y',alpha=.18)
axs[0,0].set_ylabel('Test ROC AUC (%)');axs[1,0].set_ylabel('Test ROC AUC (%)')
fig.suptitle('Full-data reconstruction · points: splits · diamonds: means ± conditional t95% CI',fontsize=14,color=ink);fig.tight_layout();save(fig,'results')
print('Built five computational/measurement figures')
# Critical-difference plot: one rank per dataset, not per seed.
s=r['statistics'];rank=dict(zip(s['methods'],s['mean_ranks']));cd=s['nemenyi_cd_05']
fig,ax=plt.subplots(figsize=(9,3.3),facecolor='#f7f5ef');ax.set_xlim(.8,3.25);ax.set_ylim(-.2,1.25);ax.axis('off')
ax.plot([1,3],[.7,.7],color=ink)
for j in [1,2,3]:ax.plot([j,j],[.66,.74],color=ink);ax.text(j,.8,str(j),ha='center')
for m,y in [('RA',.36),('AA',.12),('CN',-.12)]:
 x=rank[m];ax.plot([x,x],[y+.1,.7],color=teal);ax.scatter(x,.7,color=teal);ax.text(x,y,f'{m}  {x:.2f}',ha='center',color=ink)
ax.plot([1,1+cd],[1.02,1.02],color=orange,lw=3);ax.text(1+cd/2,1.09,f'CD(.05) = {cd:.3f}',ha='center',color=orange)
ax.plot([rank['RA'],rank['AA']],[.58,.58],color=ink,lw=5)
fig.suptitle('Mean ranks across 8 datasets · lower is better · bar joins RA–AA (gap ≤ CD)',color=ink,fontsize=13)
save(fig,'cd')
