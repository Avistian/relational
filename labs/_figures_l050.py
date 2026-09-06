"""Original, portable checkpoint figures: explanatory interventions and measured evidence."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/relational-matplotlib')
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE=Path(__file__).resolve().parent
OUT=HERE/'figures/l050';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
COLORS=['#216b57','#b36b19','#426d9f']
def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=170,bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(6.2,9.3));ax.set(xlim=(0,10),ylim=(0,16));ax.axis('off')
boxes=[(14.7,'Numeric rows  x  [B,C]','Training-only median / mean / SD'),
       (12.4,'Feature tokens + CLS  [B,C+1,32]','xⱼwⱼ + bⱼ; prepend learned CLS'),
       (9.6,'Attention update  ·  4 heads × 8','QKᵀ / √8 → softmax → weights × V\nconcat heads → project → ADD residual'),
       (6.7,'ReGLU update  ·  hidden h = 42','LayerNorm → Linear 32→84 → split a,b\na × ReLU(b) → dropout → Linear 42→32\nADD residual · repeat block twice'),
       (3.6,'Read the final CLS  [B,32]','LayerNorm → ReLU → Linear 32→1'),
       (1.1,'One logit per row  [B]','Sigmoid → probability; BCE trains logits')]
for i,(y,title,body) in enumerate(boxes):
    height=2 if i in (2,3) else 1.65
    ax.add_patch(FancyBboxPatch((.25,y-height/2),9.5,height,boxstyle='round,pad=.12',facecolor='#edf5f2' if i in (2,3) else '#f4f5f6',edgecolor='#8ba69e'))
    ax.text(5,y+(.65 if i==3 else .38),title,ha='center',va='center',weight='bold',fontsize=11)
    ax.text(5,y-.3,body,ha='center',va='center',fontsize=9.7,linespacing=1.5)
    if i<len(boxes)-1:
        next_y=boxes[i+1][0];nh=2 if i+1 in (2,3) else 1.65
        ax.annotate('',(5,next_y+nh/2+.12),(5,y-height/2-.08),arrowprops={'arrowstyle':'->','color':'#426d63','lw':1.6})
ax.text(5,11.05,'Block 1: skip attention LayerNorm\nBlock 2: normalize before attention',ha='center',va='center',fontsize=9.5,color='#216b57',bbox={'facecolor':'white','edgecolor':'none','pad':1})
ax.set_title('FT-Transformer · numeric checkpoint variant',fontsize=13,pad=12)
save(fig,'architecture')

fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
held=np.linspace(2,200,100)
for ax,what in zip(axes,['Fitted mean','Centered training value 2']):
    leak=(2+held)/3
    ax.plot(held,np.ones_like(held),label='Fit on train [0,2]',color=COLORS[0],lw=2.5)
    ax.plot(held,leak if what=='Fitted mean' else 2-leak,label='Fit on all three rows',color=COLORS[1],lw=2.5)
    ax.axvline(100,ls=':',color='#777');ax.set(title=what,xlabel='Held-out value (synthetic)',ylabel='Value')
    ax.legend(fontsize=8)
fig.suptitle('Held-out intervention · training rows stay fixed',weight='bold');save(fig,'fit-boundary')

fig,axes=plt.subplots(1,2,figsize=(8,3.6),layout='constrained')
for ax,values,title in zip(axes,[[.80,.86],[.91,.82]],['Validation selects B','Test evaluates B; do not select A']):
    ax.bar(['A','B'],values,color=[COLORS[1],COLORS[0]],width=.5)
    ax.set(ylim=(0,1),ylabel='Synthetic AUROC',title=title)
    for i,v in enumerate(values):ax.text(i,v+.02,f'{v:.2f}',ha='center')
fig.suptitle('Candidate selection · a higher test score can be an invalid result',fontsize=12);save(fig,'selection')

result_path=HERE/'_verify_l050_results.json'
if result_path.exists():
    result=json.loads(result_path.read_text());models=result['models']
    fig,axes=plt.subplots(3,1,figsize=(7.5,8.8),layout='constrained')
    for ax,(name,row) in zip(axes,result['datasets'].items()):
        for i,m in enumerate(models):
            vals=[r['auc'] for r in row['runs'][m]];s=row['summary'][m]
            ax.scatter(np.arange(3)*.055+i-.055,vals,color=COLORS[i],s=27,zorder=3)
            ax.errorbar(i,s['mean'],yerr=s['sd'],color=COLORS[i],fmt='D',capsize=6,markersize=6)
        ax.set(xticks=range(3),xticklabels=models,ylabel='Test AUROC',title=f'{name} · n={row["n"]} · detail scale')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Measured scores · 3 seeds per model · whiskers = sample SD',fontsize=12);save(fig,'scores')
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for i,(name,row) in enumerate(result['datasets'].items()):
        s=row['ft_minus_xgb'];delta=np.array([r['auc'] for r in row['runs'][models[0]]])-np.array([r['auc'] for r in row['runs'][models[1]]])
        ax.scatter(delta,i+np.array([-.08,0,.08]),color=COLORS[0],s=30)
        ax.errorbar(s['mean'],i,xerr=[[s['mean']-s['ci95'][0]],[s['ci95'][1]-s['mean']]],fmt='D',color=COLORS[0],capsize=5)
    ax.axvline(0,color='#333',ls='--');ax.set(yticks=range(3),yticklabels=list(result['datasets']),xlabel='FT-Transformer minus XGBoost · test AUROC',title='Paired differences · 95% t intervals over training seeds only')
    ax.grid(axis='x',alpha=.2);save(fig,'paired')
print('Figures:',[p.name for p in OUT.glob('*.png')])
