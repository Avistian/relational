"""Portable computation and evidence figures, generated from the audit."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/relational-matplotlib')
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures/l056';OUT.mkdir(parents=True,exist_ok=True)
R=json.loads((ROOT/'_verify_l056_results.json').read_text())
ARMS=['CatBoost','LightGBM','RealMLP','TabM']
COLORS=['#a45b24','#61703a','#7958a2','#176e8a']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fffefb','axes.facecolor':'#fffefb'})

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(9,4.5))
x=np.zeros((8,24));x[:,16:]=2
for i in range(8):x[i,2*i:2*i+2]=1
ax.imshow(x,cmap=ListedColormap(['#cfdfdf','#e7ad53','#dedede']),aspect='auto',vmin=0,vmax=2)
for i in range(8):
    ax.text(2*i+.5,i,'V',ha='center',va='center',weight='bold')
ax.text(19.5,3.5,'Outer test\nuntouched\nuntil selection',ha='center',va='center',fontsize=12)
ax.set(yticks=range(8),yticklabels=[f'Inner fit {i+1}' for i in range(8)],xticks=[3.5,11.5,19.5],xticklabels=['Outer training rows','Outer training rows','Outer test rows'],title='Which rows may select the configuration?')
ax.legend(handles=[Patch(color='#cfdfdf',label='Fit weights'),Patch(color='#e7ad53',label='Validate / select'),Patch(color='#dedede',label='Final test')],loc='upper center',bbox_to_anchor=(.5,-.13),ncol=3,frameon=False)
fig.text(.03,.01,'Illustration: 24 rows, one outer split. Collect V predictions across 8 fits; select by validation.\nBagged prediction averages the fitted members on the untouched outer test rows.',fontsize=10)
fig.subplots_adjust(bottom=.26);save(fig,'protocol')

fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
for ax,values,title in zip(axes,[[1.5,1.5],[1.75,1.25]],['Each dataset gets equal weight','Each split gets equal weight']):
    ax.bar(['A','B'],values,color=['#176e8a','#a45b24'],width=.5)
    for i,v in enumerate(values):ax.text(i,v+.06,f'{v:.2f}',ha='center')
    ax.set(title=title,ylim=(0,2.15),ylabel='Mean rank · lower is better')
fig.text(.02,.02,'Synthetic trace: large dataset → A=1, B=2 once; small dataset → A=2, B=1 three times.\nDataset average ties. Pooling splits gives the small dataset 3× influence; B appears ahead.',fontsize=10)
fig.subplots_adjust(bottom=.25,wspace=.25);save(fig,'weighting')

fig,ax=plt.subplots(figsize=(8,4.8))
regimes=['default','tuned','tuned_ensemble']
for a,c in zip(ARMS,COLORS):
    vals=[R['summary'][k]['mean_ranks'][a] for k in regimes]
    ax.plot(range(3),vals,'o-',label=a,color=c,lw=2)
    for i,v in enumerate(vals):ax.annotate(f'{v:.3f}',(i,v),xytext=(5,6),textcoords='offset points',fontsize=9,color=c)
ax.set(xticks=range(3),xticklabels=['Default','Tuned','Tuned + ensemble'],ylabel='Dataset-balanced mean split rank',ylim=(3.15,1.65),xlim=(-.15,2.35),title='Same 51 datasets · different fitted procedures')
ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.13),frameon=False)
fig.text(.02,.01,'Frozen 2025-06-12 result artifacts; four-method pool. Lower ranks appear higher.\nThese are reanalysed measurements, not the paper’s full-pool Elo ratings or retrained models.',fontsize=10)
fig.subplots_adjust(bottom=.25);save(fig,'ranks')

fig,ax=plt.subplots(figsize=(8,4))
for i,k in enumerate(regimes):
    v,lo,hi=R['summary'][k]['tabm_minus_catboost_rank_gap']
    ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color='#176e8a',capsize=5)
    ax.annotate(f'{v:+.3f} [{lo:+.3f}, {hi:+.3f}]',(v,i),xytext=(0,12),textcoords='offset points',ha='center',fontsize=10)
ax.axvline(0,color='#666',ls='--');ax.set(yticks=range(3),yticklabels=['Default','Tuned','Tuned + ensemble'],xlim=(-1.1,1.05),ylim=(-.5,2.6),xlabel='TabM rank − CatBoost rank  (negative favors TabM)',title='Paired dataset bootstrap · 2,000 draws · seed 56')
fig.text(.02,.01,'Resample 51 whole datasets with replacement, keeping all their splits paired.\nPercentile 95% intervals describe this benchmark mix under resampling; they do not cover temporal shift.',fontsize=10)
fig.subplots_adjust(left=.23,bottom=.25);save(fig,'uncertainty')

fig,ax=plt.subplots(figsize=(8,3.9))
k='tuned_ensemble';s=R['summary'][k];ordered=sorted(ARMS,key=lambda a:s['mean_error_ranks'][a])
for i,a in enumerate(ordered):
    v=s['mean_error_ranks'][a];ax.plot(v,i,'o',color=COLORS[ARMS.index(a)]);ax.text(v+.04,i,f'{v:.3f}',va='center')
cd=s['nemenyi_cd'];ax.plot([1,1+cd],[4,4],color='#333',lw=2);ax.text(1+cd/2,4.15,f'CD = {cd:.3f}',ha='center')
ax.set(yticks=range(4),yticklabels=ordered,xlim=(.9,4.1),ylim=(-.5,4.6),xlabel='Rank of each dataset’s mean error · lower is better',title=f'Supplementary Nemenyi comparison · Friedman p = {s["friedman_p"]:.4g}')
fig.text(.02,.01,'51 dataset units; 4 methods. Rank mean errors within datasets before averaging.\nThis order of operations differs from the split-rank plot; a CD is a pairwise threshold, not an interval.',fontsize=10)
fig.subplots_adjust(left=.18,bottom=.25);save(fig,'critical-difference')
