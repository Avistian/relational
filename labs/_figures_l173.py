"""Model-specific computation trace, loss weights, and measured learning curves."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;D=P/'figures/l173';D.mkdir(exist_ok=True,parents=True)
r=json.loads((P/'evidence/l173/report.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l173'})
ink='#17324d';teal='#157f78';rust='#a34832';muted='#526477';cream='#faf8f2'
def save(fig,name):
    fig.savefig(D/(name+'.svg'),bbox_inches='tight',facecolor=cream,metadata={'Date':None})
    fig.savefig(D/(name+'.png'),dpi=140,bbox_inches='tight',facecolor=cream,metadata={'Software':'L173'});plt.close(fig)
def box(ax,x,y,w,h,title,body,color=teal):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',facecolor='#e4f2ed' if color==teal else '#f7e5dd',edgecolor=color))
    ax.text(x+.13,y+h-.3,title,fontsize=11,weight='bold',color=color)
    ax.text(x+.13,y+h-.65,body,fontsize=10,color=ink,va='top',linespacing=1.55)
def arrow(ax,x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','lw':1.6,'color':muted})
fig,ax=plt.subplots(figsize=(12,7));ax.set(xlim=(0,12),ylim=(0,7));ax.axis('off')
ax.text(.1,6.6,'A hidden cell, two context pools, one shared encoder',fontsize=18,weight='bold',color=ink)
ax.text(.1,6.15,'Course model · example: results.points · no attention or cross-sample support',color=muted)
box(ax,.15,3.65,2.65,1.8,'1  INPUT CONTRACT','Target: points → separate y\nRow: ≤4 other cells\nFK: ≤4 dated parent cells')
box(ax,3.2,3.65,2.65,1.8,'2  TYPE → TOKEN','Column embedding\n+ numeric projection OR\n  categorical embedding\nB × 8 × 32')
box(ax,6.25,3.65,2.5,1.8,'3  TWO MEANS','Row pool: B × 32\nParent pool: B × 32\nPadding excluded')
box(ax,9.15,3.65,2.65,1.8,'4  SHARED MLP','+ query embedding (32)\nConcatenate → B × 96\n96 → 64 → 32')
for x in [2.85,5.9,8.8]:arrow(ax,x,4.55,x+.3,4.55)
box(ax,6.25,1.25,5.55,1.65,'5  ROUTE BY TARGET COLUMN','Numerical: 32 → 1 → Huber(normalized y)\nCategory: 32 → K → cross-entropy(class y)\nAll 21 heads train the same encoder',rust)
arrow(ax,10.45,3.58,10.45,3.0)
box(ax,.15,1.25,5.55,1.65,'6  AGGREGATE → BACKPROPAGATE','cell: mean(loss)       task: mean(weight × loss)\nweight = N / (21 × training_count_of_task)\nValidation macro loss selects a saved checkpoint',rust)
arrow(ax,6.16,2.1,5.82,2.1)
ax.text(.15,.45,'Keys choose links; they are not model features. Hidden target values never enter either context pool.',color=ink)
save(fig,'architecture')
fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
for ax,title,weights,loss in [(axes[0],'Cell mean: 1.4',[.9,.1],1.4),(axes[1],'Equal-task mean: 3.0',[.5,.5],3.)]:
    ax.bar(['A: 9 cells','B: 1 cell'],weights,color=[teal,rust],width=.6)
    for i,(w,l) in enumerate(zip(weights,[1,5])):ax.text(i,w+.035,f'{w:.1f} × loss {l}\n= {w*l:.1f}',ha='center',color=ink)
    ax.set_ylim(0,1.2);ax.set_title(title,color=ink,weight='bold');ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Task share of the scalar objective')
fig.suptitle('Same ten errors; a different meaning of “average”',fontsize=17,color=ink)
fig.text(.5,.01,'Illustrative losses stay fixed: A = 1, B = 5. Equal-task weighting is not a guarantee of equal gradient norms.',ha='center',fontsize=10,color=muted)
fig.tight_layout(rect=(0,.08,1,.92));save(fig,'weights')
fig,axes=plt.subplots(1,2,figsize=(11,4.6))
for run in r['runs']:
    color=teal if run['arm']=='cell' else rust;label=run['arm']+' weighted' if run['seed']==0 else None
    axes[0].plot([1,2,3],[e['validation']['macro_loss'] for e in run['epochs']],marker='o',color=color,alpha=.7,label=label)
    chosen=run['selected_epoch'];value=run['epochs'][chosen-1]['validation']['macro_loss'];axes[0].scatter([chosen],[value],s=115,facecolors='none',edgecolors=color)
    axes[1].scatter(run['seed']+(-.07 if run['arm']=='cell' else .07),run['test']['macro_loss'],color=color,label=label,s=55)
for seed in [0,1,2]:
    pair=[next(x for x in r['runs'] if x['seed']==seed and x['arm']==arm) for arm in ['cell','task']]
    axes[1].plot([seed-.07,seed+.07],[x['test']['macro_loss'] for x in pair],color=muted,lw=1)
baseline=json.loads((P/'evidence/l173/baseline.json').read_text())
axes[1].axhline(baseline['macro_loss'],color=ink,linestyle='--',label='constant baseline')
axes[0].set(title='Validation curves choose the epoch',xlabel='Epoch',ylabel='Macro task loss',xticks=[1,2,3]);axes[0].legend(fontsize=10)
axes[1].set(title='Selected checkpoint: full test population',xlabel='Paired initialization seed',ylabel='Macro task loss',xticks=[0,1,2]);axes[1].legend(fontsize=10)
for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
fig.suptitle('Measured F1 autocomplete: six fresh runs',fontsize=17,color=ink)
fig.text(.5,.015,'Rings mark validation selection. Lower is better. Seeds vary optimization, not the database population.',ha='center',fontsize=10,color=muted)
fig.tight_layout(rect=(0,.07,1,.93));save(fig,'curves')
print('Built three portable figures')
# Complete task-level validation curves: common epochs, separate labeled loss scales.
import numpy as np
tasks=json.loads((P/'evidence/l173/tasks.json').read_text())
fig,axes=plt.subplots(7,3,figsize=(13,19))
for i,(spec,ax) in enumerate(zip(tasks,axes.flat)):
    for arm,color in [('cell',teal),('task',rust)]:
        ys=np.array([[e['validation']['tasks'][i]['loss'] for e in run['epochs']] for run in r['runs'] if run['arm']==arm])
        ax.errorbar([1,2,3],ys.mean(0),yerr=ys.std(0,ddof=1),marker='o',color=color,label=arm,capsize=3,lw=1.2)
    ax.set(title=spec['name'],xticks=[1,2,3],xlabel='Epoch',ylabel='Validation loss')
    ax.title.set_fontsize(10);ax.tick_params(labelsize=9);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
axes.flat[0].legend(fontsize=9)
fig.suptitle('Every task: validation loss across three full epochs',fontsize=19,color=ink,y=.997)
fig.text(.5,.002,'Mean ± sample SD over three initialization seeds. Huber/CE scales differ across tasks; axes are not shared.',ha='center',fontsize=11,color=muted)
fig.tight_layout(rect=(0,.015,1,.985));save(fig,'per-task-curves')
