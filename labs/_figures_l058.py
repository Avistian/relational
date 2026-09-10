"""Portable numerical protocol and paired evidence figures for L058."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures/l058';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11,5),gridspec_kw={'width_ratios':[1.05,1]})
for ax in axes:ax.axis('off')
axes[0].set_title('Compute ranks AFTER reserving methods',loc='left',weight='bold')
rows=[[1,2,1,2],[1,2,2,1],[2,1,1,2],[2,1,2,1],[1,2,1,2],[2,1,2,1]]
t=axes[0].table(cellText=rows,colLabels=['Seen A','Seen B','Held C','Held D'],rowLabels=[f'Task {i}' for i in range(6)],loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(13);t.scale(1,1.8)
for i in range(1,7):
 for j in [2,3]:t[i,j].set_facecolor('#f7e3d1')
axes[0].text(0,-.02,'Full mean = [1.5, 1.5] in EACH pool\nRanks within seen / held-out pools independently',transform=axes[0].transAxes,fontsize=11)
axes[1].set_title('Only seen ranks select the task IDs',loc='left',weight='bold')
steps=[('1  Candidate [0, 1]','Seen mean [1, 2] → MAE 0.50\nHeld-out mean [1.5, 1.5] → MAE 0.00'),('2  Candidate [0, 2]','Seen mean [1.5, 1.5] → MAE 0.00\nFreeze these task IDs before evaluation'),('3  Evaluate frozen [0, 2]','Held-out mean [1, 2] → MAE 0.50\nBetter seen match; worse unseen match')]
for i,(title,body) in enumerate(steps):
 y=.87-i*.30
 axes[1].text(.02,y,title,weight='bold',transform=axes[1].transAxes)
 axes[1].text(.02,y-.065,body,va='top',transform=axes[1].transAxes,fontsize=12)
fig.suptitle('Synthetic protocol trace • no model is trained',fontsize=16)
fig.tight_layout(rect=[.02,.04,1,.94]);fig.savefig(OUT/'protocol.png',dpi=170);plt.close(fig)
r=json.loads((ROOT/'_verify_l058_v2_results.json').read_text())
fig,ax=plt.subplots(figsize=(9,4.5));arms=r['arms'];mu=np.array(list(r['mean_ranks'].values()));ci=np.array(r['percentile95'])
y=np.arange(len(arms));ax.errorbar(mu,y,xerr=[mu-ci[0],ci[1]-mu],fmt='o',capsize=4,color='#275e76');ax.set_yticks(y,arms);ax.invert_yaxis();ax.set_xlabel('Mean rank in six-method pool (lower is better)');ax.set_xlim(1,6);ax.grid(axis='x',alpha=.2);ax.set_title('300 frozen tasks • paired dataset bootstrap, 2,000 draws\nIntervals condition on rounded published means',fontsize=14)
fig.tight_layout();fig.savefig(OUT/'rank-intervals.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,4.5),sharey=True)
for ax,task in zip(axes,['binary','multiclass','regression']):
 rows=[v for v in r['tiny'] if v['task']==task]
 for row in rows:
  ax.plot([0,1],[row['random']['unseen_mae'],row['selected']['unseen_mae']],'-o',alpha=.45,lw=1,ms=4,color='#b26d36')
 ax.set_xticks([0,1],['First random','Best seen']);ax.set_title(task.capitalize());ax.set_xlim(-.2,1.2);ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Held-out-pool rank MAE (lower is better)')
fig.suptitle('Does selecting by seen methods improve unseen fidelity?\n1,000 candidates • each line = one declared method rotation / search seed',fontsize=13)
fig.tight_layout(rect=[0,0,1,.88]);fig.savefig(OUT/'tiny.png',dpi=170);plt.close(fig)
