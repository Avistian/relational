"""Portable explanatory snapshots and measured evidence for L055."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/relational-matplotlib')
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from relkit.temporal_experiment import ARMS, summarize
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures/l055';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fcfbf8','axes.facecolor':'#fcfbf8','savefig.facecolor':'#fcfbf8'})
colors=['#286b96','#bc7026','#51835b']
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(9,3.4))
parts=[['T','V','T','E','T','V','T','T','E','T','V','T'],['T']*7+['V']*3+['E']*2]
for j,row in enumerate(parts):
    y=1-j
    for i,part in enumerate(row):
        c=dict(T=colors[0],V=colors[1],E=colors[2])[part]
        ax.scatter(i+1,y,s=420,c=c,marker='s' if part=='T' else 'o' if part=='V' else 'D')
        ax.text(i+1,y,part,ha='center',va='center',color='white',fontsize=10)
ax.set(yticks=[0,1],yticklabels=['Chronological','Shuffled'],xticks=range(1,13),xlabel='Event time (illustrative units)',ylim=(-.7,1.7),xlim=(.4,12.6))
ax.set_title('Same 12 events and counts; different information in training',loc='left',pad=20)
ax.text(.02,.02,'T = train (7)    V = validation (3)    E = test (2)',transform=ax.transAxes,fontsize=10)
fig.tight_layout();save(fig,'splits')

fig,ax=plt.subplots(figsize=(9,4.8))
for t in range(1,11):
    deadline=8 if t<8 else 11; ok=t+3<=deadline
    ax.plot([t,t+3],[t,t],color=colors[0] if ok else colors[1],lw=2)
    ax.scatter(t,t,color=colors[0],marker='o',s=35)
    ax.scatter(t+3,t,color=colors[2] if ok else colors[1],marker='s',s=45)
ax.axvline(8,ls='--',color=colors[0]);ax.axvline(11,ls='--',color=colors[2])
ax.set(yticks=range(1,11),yticklabels=[f'Train {t}' if t<8 else f'Valid {t}' for t in range(1,11)],xlabel='Time: circle = event; square = label arrival',xlim=(.5,14),ylim=(10.8,.2))
ax.set_title('Delay = 3: past events can still have unavailable labels',loc='left',pad=22)
ax.text(8,.45,'Fit at 8',ha='center',fontsize=10,backgroundcolor='#fcfbf8');ax.text(11,.45,'Select at 11',ha='center',fontsize=10,backgroundcolor='#fcfbf8')
fig.tight_layout();save(fig,'availability')

fig,axs=plt.subplots(1,2,figsize=(9,3.7),sharey=True)
for ax,values,title in zip(axs,[[.21,.18],[.20,.25]],['1. Select using validation','2. Score selected B on test']):
    ax.bar(['A','B'],values,color=[colors[0],colors[1]])
    for i,v in enumerate(values):ax.text(i,v+.008,f'{v:.2f}',ha='center')
    ax.set(ylim=(0,.31),title=title,ylabel='Error (lower is better)')
axs[1].text(.5,.29,'Keep B even though A looks better',ha='center',fontsize=10)
fig.tight_layout();save(fig,'selection')

if (ROOT/'_verify_l055_results.json').exists():
    r=json.loads((ROOT/'_verify_l055_results.json').read_text());summary=summarize(r)
    fig,axs=plt.subplots(3,1,figsize=(9,10))
    for ax,(name,task) in zip(axs,r['tasks'].items()):
        for i,arm in enumerate(ARMS):
            means=[]
            for j,strategy in enumerate(['random','temporal']):
                es=np.array([v['error'] for v in task[strategy][0]['arms'][arm]])
                x=j+(i-1)*.13;means.append(es.mean())
                ax.errorbar(x,es.mean(),yerr=4.30265273*es.std(ddof=1)/np.sqrt(3),color=colors[i],capsize=5,fmt='s')
                ax.scatter(x+np.array([-.018,0,.018]),es,s=24,color=colors[i],alpha=.75)
            ax.plot(np.array([0,1])+(i-1)*.13,means,label=arm,color=colors[i],lw=1)
        ax.set(xticks=[0,1],xticklabels=['Random-0','Sliding-window-0'],xlim=(-.4,1.4),ylabel=task['random'][0]['metric'],title=name)
        ax.grid(axis='y',alpha=.2)
    axs[0].legend(ncol=3,fontsize=10);fig.suptitle('Measured errors: fixed row caps and candidate budgets\nPoints = seeds; bars = conditional 95% t intervals',y=.99)
    fig.tight_layout(rect=(0,0,1,.96));save(fig,'results')
    fig,axs=plt.subplots(1,2,figsize=(9,4),sharey=True)
    for ax,strategy in zip(axs,['random','temporal']):
        s=summary[strategy]
        for i,(arm,v) in enumerate(zip(ARMS,s['mean_ranks'])):
            ax.scatter(v,i,color=colors[i],s=65);ax.text(v+.06,i,f'{v:.2f}',va='center',fontsize=10)
        cd=s['nemenyi_cd'];ax.plot([1,1+cd],[2.65,2.65],color='#333',marker='|')
        ax.text(1,2.85,f'CD={cd:.2f}',fontsize=10)
        ax.set(yticks=range(3),yticklabels=ARMS,xlim=(.8,3.3),ylim=(-.5,3.2),xlabel='Mean rank (lower is better)',title=f'{strategy}: Friedman p={s["friedman_p"]:.3f}')
    fig.tight_layout();save(fig,'ranks')
print(OUT)
