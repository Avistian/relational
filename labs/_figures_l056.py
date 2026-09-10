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

# Separate analytical mechanism and newly measured extension; old score JSON is untouched.
fig,axes=plt.subplots(1,3,figsize=(10,3.7))
axes[0].bar(['A wins','B wins'],[3,1],color=COLORS[:2]);axes[0].set(ylim=(0,4),ylabel='Contests',title='1. Observed outcomes')
axes[1].axis('off');axes[1].text(.5,.6,'Odds = 3 / 1\n\nGap = 400 log10(3)\n= 190.85 Elo points',ha='center',va='center',fontsize=12);axes[1].set_title('2. Fit log-strength gap')
axes[2].bar(['Observed','Fitted'],[.75,.75],color=[COLORS[0],COLORS[3]]);axes[2].set(ylim=(0,1),ylabel='A win probability',title='3. Check the prediction')
fig.text(.03,.01,'Synthetic two-method example. Three wins and one loss; no draws.\nCurrent-source tiny ridge changes the analytical gap by less than .001; a multi-method fit need not match every pair.',fontsize=10)
fig.subplots_adjust(bottom=.25,wspace=.45);save(fig,'rating-trace')

new=json.loads((ROOT/'_verify_l056_elo_results.json').read_text())['results']['tuned_ensemble']
fig,axes=plt.subplots(1,2,figsize=(10,4.3))
axes[0].scatter(new['ratings'],ARMS,c=COLORS,s=65);axes[0].set(xlim=(900,1100),xlabel='Elo · local field mean 1000',title='Tuned + ensemble · four methods')
for j,value in enumerate(new['ratings']):axes[0].text(value+2,j,f'{value:.1f}',va='center',fontsize=10)
for j in [1,2,3]:
 v=new['catboost_contrast'][j];lo,hi=new['catboost_contrast_interval'][j]
 axes[1].plot([lo,hi],[j,j],color=COLORS[j],lw=2);axes[1].plot(v,j,'o',color=COLORS[j])
axes[1].axvline(0,color='#555',ls='--');axes[1].set(yticks=[1,2,3],yticklabels=ARMS[1:],xlabel='Elo − CatBoost Elo',title='Paired dataset-refit contrasts')
fig.text(.03,.01,'Frozen 51 datasets; current-source solver. 100 bootstrap draws, seed 56: coarse 95% percentile intervals.\nDifferent from the 2,000-draw mean-rank analysis and the initial paper roster, anchor and solver.',fontsize=10)
fig.subplots_adjust(bottom=.26,left=.12,wspace=.55);save(fig,'rating-evidence')

from matplotlib.patches import FancyBboxPatch
fig,ax=plt.subplots(figsize=(9,7));ax.set(xlim=(0,10),ylim=(0,10));ax.axis('off')
boxes=[(.3,8,4.3,1.4,'Outer training: N rows\n8 inner fits / configuration\nEach fit excludes its validation rows'),
       (5.4,8,4.3,1.4,'Outer test: Q rows\nLabels sealed until final score\nPredictions allowed before selection'),
       (.3,5.6,4.3,1.5,'Scatter held-out predictions\nOOF array: [N, C]\nOne excluded-fit prediction per row'),
       (5.4,5.6,4.3,1.5,'Average 8 test prediction arrays\nBagged prediction: [Q, C]\nAverage probabilities before loss'),
       (.3,3.2,4.3,1.5,'Use OOF predictions + N labels\nSelect candidate or ensemble weights\nNo outer-test labels enter here'),
       (5.4,3.2,4.3,1.5,'Apply selected candidate / weights\nFinal prediction: [Q, C]\nConfigurations may be combined'),
       (5.4,.8,4.3,1.4,'Unseal Q test labels → one error\nRepeat 9 or 30 outer splits\n51 dataset blocks → audit ratings')]
for x,y,w,h,text in boxes:
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',facecolor='#edf3f2',edgecolor='#176e8a'))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
for start,end in [((4.66,8.12),(5.34,6.9)),((2.45,8),(2.45,7.1)),((7.55,8),(7.55,7.1)),((2.45,5.6),(2.45,4.7)),((7.55,5.6),(7.55,4.7)),((4.65,3.95),(5.35,3.95)),((7.55,3.2),(7.55,2.2))]:
 ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#176e8a','lw':2})
ax.text(.4,1.5,'Pictured: bagged non-foundation methods.\nC = classes; regression uses C=1.\nFoundation models use a refit/context branch\nin place of eight-fold prediction bagging.',fontsize=10,va='center')
ax.set_title('One outer split: which predictions select, which predictions score?',fontsize=13,pad=10)
fig.tight_layout();save(fig,'protocol-flow')
