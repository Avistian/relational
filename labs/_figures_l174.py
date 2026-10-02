"""Deterministic portable architecture and measured adaptation figures."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;E=P/'evidence/l174/runs';F=P/'figures/l174';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l174','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fbfaf6','axes.facecolor':'#fbfaf6'})
r=json.loads((E/'report.json').read_text());b=json.loads((E/'baselines.json').read_text());colors={'freeze':'#326b86','full':'#ad553c','adapter':'#6d58a4','scratch':'#267c61'}
def save(fig,name):
 fig.savefig(F/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});fig.savefig(F/(name+'.png'),dpi=145,bbox_inches='tight',metadata={'Software':'L174'});plt.close(fig)
fig,ax=plt.subplots(figsize=(13,6));ax.set(xlim=(0,13),ylim=(0,6));ax.axis('off')
def box(x,y,w,h,title,body,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',linewidth=1.4,edgecolor=color,facecolor='white'))
 ax.text(x+.15,y+h-.2,title,weight='bold',va='top',color=color,fontsize=12);ax.text(x+.15,y+h-.65,body,va='top',fontsize=10,linespacing=1.6)
box(.15,3.25,2.7,1.9,'01  Masked context','B × 8 cell identities\n4 row + 4 dated FK slots\nTarget absent on every path','#326b86')
box(3.35,3.25,2.7,1.9,'02  Inherited encoder','Typed tokens: B × 8 × 32\nTwo pools + query: B × 96\nShared MLP: 96 → 64 → 32','#326b86')
box(6.55,3.25,2.7,1.9,'03  Optional adapter','h′ = h + U ReLU(Dh + b) + c\n32 → 8 → 32; 552 parameters\nU = 0, c = 0 at initialization','#6d58a4')
box(9.75,3.25,2.95,1.9,'04  Original task heads','B × 32 → numeric / class logits\n21 heads; 9,834 parameters\n2006 equal-task objective','#ad553c')
for x in [2.9,6.1,9.3]:ax.annotate('',xy=(x+.4,4.15),xytext=(x,4.15),arrowprops=dict(arrowstyle='->',color='#555',lw=2))
ax.text(.1,5.65,'Where may the gradient change the model?',fontsize=21,weight='bold')
for i,(arm,label) in enumerate([('freeze','FREEZE   encoder fixed · heads learn'),('full','FULL   encoder + heads learn'),('adapter','ADAPTER   encoder fixed · adapter + heads learn'),('scratch','SCRATCH   encoder + heads learn from random weights')]):
 ax.text(.25,2.6-i*.5,label,color=colors[arm],weight='bold',fontsize=12)
ax.text(.25,.1,'Source: pre-2005 training → 2005 checkpoint selection  |  Adapt: 2006 → select: 2007 → score: 2008+',fontsize=11)
save(fig,'architecture')
fig,axes=plt.subplots(1,2,figsize=(12,4.2))
for arm in colors:
 runs=[x for x in r['runs'] if x['arm']==arm];vals=np.array([[e['validation']['macro_loss'] for e in x['epochs']] for x in runs]);mean=vals.mean(0);sd=vals.std(0,ddof=1)
 axes[0].plot(range(1,11),mean,label=arm,color=colors[arm]);axes[0].fill_between(range(1,11),mean-sd,mean+sd,alpha=.12,color=colors[arm])
 for x in runs:axes[0].scatter(x['selected_epoch'],x['epochs'][x['selected_epoch']-1]['validation']['macro_loss'],facecolors='none',edgecolors=colors[arm])
 axes[1].plot(range(3),[x['test']['macro_loss'] for x in runs],marker='o',label=arm,color=colors[arm])
axes[0].set(title='2007 validation: selection happens here',xlabel='Epoch',ylabel='Macro task loss');axes[0].legend(ncol=2,fontsize=9)
axes[1].axhline(b[-1]['test']['macro_loss'],color='#777',linestyle='--',label='2006 constant baseline')
axes[1].plot(range(3),[x['test']['macro_loss'] for x in b[:3]],color='#999',linestyle=':',marker='x',label='unchanged pretrained')
axes[1].set(title='2008+ test: complete population',xlabel='Paired seed',ylabel='Macro task loss',xticks=[0,1,2]);axes[1].legend(fontsize=8)
for ax in axes:ax.grid(alpha=.15)
fig.tight_layout();save(fig,'curves')
fig,ax=plt.subplots(figsize=(10,3.5));names=list(colors);x=np.arange(4)
train=[next(v for v in r['runs'] if v['arm']==arm)['trainable_parameters'] for arm in names];total=[next(v for v in r['runs'] if v['arm']==arm)['total_parameters'] for arm in names]
ax.barh(x,total,color='#e3e0d9',label='Total stored model');ax.barh(x,train,color=[colors[a] for a in names],label='Trainable parameters')
for i,(a,z) in enumerate(zip(train,total)):ax.text(z+300,i,f'{a:,} / {z:,}',va='center',fontsize=10)
ax.set(yticks=x,yticklabels=names,xlabel='Parameters (not measured runtime)',xlim=(0,37000),title='Heads dominate the adaptation budget');ax.invert_yaxis();ax.legend(loc='lower right',fontsize=9);fig.tight_layout();save(fig,'parameters')
fig,axes=plt.subplots(7,3,figsize=(12,20))
for i,ax in enumerate(axes.flat):
 for arm in colors:
  vals=np.array([[e['validation']['tasks'][i]['loss'] for e in x['epochs']] for x in r['runs'] if x['arm']==arm]);ax.plot(range(1,11),vals.mean(0),color=colors[arm],label=arm)
 ax.set_title(r['runs'][0]['test']['tasks'][i]['task'],fontsize=10);ax.set_xlabel('Epoch');ax.set_ylabel('Task loss');ax.grid(alpha=.15)
axes.flat[0].legend(fontsize=8);fig.suptitle('All 21 tasks · validation loss · mean across three seeds',fontsize=17);fig.tight_layout(rect=(0,0,1,.98));save(fig,'per-task-curves')
print('Four portable figure pairs built')
