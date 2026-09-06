"""Computed explanatory snapshots and measured effects, at notebook reading width."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/relational-matplotlib')
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from relkit.bias_interventions import smooth_targets
ROOT=Path(__file__).parent;OUT=ROOT/'figures/l051';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
colors=['#276e58','#276899','#ad4c35']
def save(fig,name):fig.savefig(OUT/f'{name}.png',dpi=160,bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(3,1,figsize=(7,8),layout='constrained')
x=np.arange(3);y=np.array([0,1,0]);w=np.exp(-.5*(x-1)**2);p=w@y/w.sum()
ax[0].scatter(x,y,s=120,color=colors[1]);ax[0].set(xticks=x,ylim=(-.2,1.25),ylabel='Original label',title='1  Fixed training rows: x = (0, 1, 2), y = (0, 1, 0)')
ax[1].bar(x,w,color=['#bed0d8',colors[1],'#bed0d8']);ax[1].set(xticks=x,ylim=(0,1.25),ylabel='Weight into x = 1',title='2  Covariance C = 1, lengthscale h = 1; self-weight = 1')
for i,a in enumerate(w):ax[1].text(i,a+.04,f'{a:.3f} × {y[i]}',ha='center')
hs=np.linspace(0,1.5,101);ps=[smooth_targets(x[:,None],y,h,np.eye(1))[0][1] for h in hs]
ax[2].plot(hs,ps,color=colors[1],lw=2);ax[2].axhline(.5,color=colors[2],ls='--');ax[2].scatter([1],[p],color=colors[2],zorder=4)
ax[2].annotate(f'h=1: 1 / {w.sum():.3f} = {p:.3f}\nHard label becomes 0',(1,p),(.65,.78),arrowprops=dict(arrowstyle='->'),fontsize=10)
ax[2].set(xlabel='Lengthscale h',ylabel='Smoothed probability',title='3  Threshold p > 0.5; validation/test labels never change')
fig.suptitle('Training-target smoothing · synthetic worked example',weight='bold');save(fig,'smoothing')

fig,axes=plt.subplots(2,1,figsize=(6.5,9),layout='constrained')
points=np.array([(a,b) for a in range(-2,3) for b in range(-2,3)])
for ax,deg in zip(axes,[0,45]):
 a=np.deg2rad(deg);r=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
 z=points@r;boundary=np.column_stack([np.full(100,.4),np.linspace(-2.5,2.5,100)])@r
 ax.scatter(z[:,0],z[:,1],c=np.where(points[:,0]>.4,colors[0],colors[1]),s=35)
 ax.plot(boundary[:,0],boundary[:,1],color=colors[2],lw=2,label='Same decision boundary')
 ax.axhline(0,lw=.5,color='#bbb');ax.axvline(0,lw=.5,color='#bbb');ax.set(aspect='equal',xlim=(-3,3),ylim=(-3,3),xlabel='Coordinate 0',ylabel='Coordinate 1',title=f'{deg}°: '+('one-coordinate threshold x₀ > 0.4' if deg==0 else '0.707 x′₀ − 0.707 x′₁ > 0.4'))
 ax.legend(loc='upper left',fontsize=9)
fig.suptitle('Rotation preserves rows, labels and distances\nA diagonal boundary changes a finite tree’s job',weight='bold');save(fig,'rotation')

# The browser arithmetic is also exported by _viz_check_l051.js.
g=json.loads((ROOT/'_noise_l051_example.json').read_text());a=np.array(g['all'])
fig,axes=plt.subplots(2,1,figsize=(7,6.5),layout='constrained')
axes[0].bar(np.arange(1,41),a,color='#91acb6');axes[0].plot(np.arange(1,41),np.maximum.accumulate(a),color=colors[2],lw=2,label='Largest gain seen so far')
axes[0].set(xlabel='Independent noise-column candidate',ylabel='Best training Gini gain',title='Same 32 balanced labels; search all thresholds in each column');axes[0].legend(fontsize=9)
axes[1].plot(np.arange(41),64*np.arange(41),color=colors[1],lw=2);axes[1].set(xlabel='Extra input columns',ylabel='Extra first-layer weights',title='Width-64 MLP: 64 additional weights per input')
fig.suptitle('Irrelevant does not mean harmless\nSynthetic search arithmetic; no test-score claim',weight='bold');save(fig,'noise')

fig,ax=plt.subplots(figsize=(7,5.8));ax.axis('off')
rows=[['Original','All numeric columns','Original labels','Original labels'],['Rotated','One common X @ R','Original labels','Original labels'],['Noise','Original + 2d junk','Original labels','Original labels'],['Top five','Train-only RF selection','Original labels','Original labels'],['Smoothed','Same top five','Gaussian → > .5','Original labels']]
tab=ax.table(cellText=rows,colLabels=['Condition','Features','Training target','Validation / test'],loc='center',cellLoc='left',colWidths=[.14,.33,.25,.28]);tab.auto_set_font_size(False);tab.set_fontsize(9.3);tab.scale(1,2.5)
for (r,c),cell in tab.get_celld().items():cell.set_edgecolor('#ccd7d8');cell.set_facecolor('#e8f2ef' if r==0 else ('#f0f5f7' if r in [4,5] else 'white'))
ax.text(.02,.1,'Smoothing must be compared with TOP FIVE, not ORIGINAL.\nModel seeds, split, recipe and evaluation targets are paired.',transform=ax.transAxes,fontsize=11,linespacing=1.7)
ax.set_title('Five conditions; three questions; one fixed test set',pad=10,weight='bold');save(fig,'protocol')

r=json.loads((ROOT/'_verify_l051_results.json').read_text())
fig,axes=plt.subplots(3,1,figsize=(8,11),layout='constrained')
for ax,condition in zip(axes,['rotated','noise','smoothed']):
 for i,(name,row) in enumerate(r['datasets'].items()):
  for j,m in enumerate(['MLP','FT-T','XGBoost']):
   e=row['effects'][condition][m];pos=i+(j-1)*.2
   baseline='top5' if condition=='smoothed' else 'original'
   delta=np.array([a['accuracy']-b['accuracy'] for a,b in zip(row['runs'][condition][m],row['runs'][baseline][m])])*100
   ax.scatter(delta,np.full(3,pos),s=17,color=colors[j],alpha=.7)
   ax.errorbar(e['mean']*100,pos,xerr=[[100*(e['mean']-e['ci95'][0])],[100*(e['ci95'][1]-e['mean'])]],fmt='D',markersize=4,color=colors[j],capsize=3,label=m if i==0 else None)
 ax.axvline(0,color='#667',ls='--');ax.set(yticks=range(3),yticklabels=list(r['datasets']),xlabel='Accuracy change (percentage points); ← worse | better →',title=condition.capitalize()+' − '+('top-five baseline' if condition=='smoothed' else 'original baseline'));ax.legend(ncol=3,fontsize=9)
fig.suptitle('Measured local effects · 3 seeds, one split per task\nDots: paired seeds. Whiskers: conditional 95% t interval.',weight='bold');save(fig,'effects')

s=r['statistics']['original'];fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained')
for i,(m,rank) in enumerate(s['mean_ranks'].items()):ax.scatter(rank,i,s=70,color=colors[i]);ax.text(rank+.035,i+.08,f'{rank:.3f}',fontsize=10)
ax.plot([1,1+s['cd']],[-.7,-.7],color='#555',lw=2);ax.text(1,-.95,f"Nemenyi CD = {s['cd']:.3f}",fontsize=10)
ax.set(xlim=(.9,3.15),ylim=(2.5,-1.1),yticks=range(3),yticklabels=list(s['mean_ranks']),xlabel='Mean rank (smaller is better)',title=f"Original conditions: N=3 tasks, Friedman p={s['friedman_p']:.3f}\nNo pair exceeds CD; this is not evidence of equivalence")
save(fig,'ranks')
print('Saved six L051 figures')
