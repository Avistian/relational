"""Current L055 figures: protocol and aggregation expose intermediate quantities."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/relational-matplotlib')
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures/l055'
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fcfbf8','axes.facecolor':'#fcfbf8','savefig.facecolor':'#fcfbf8'})
C=['#286b96','#bc7026','#51835b']
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(9,10));ax.set(xlim=(0,10),ylim=(0,12));ax.axis('off')
layout_records=[]
def box(x,y,w,h,title,detail,c=C[0]):
 patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',ec=c,fc='white',lw=1.5);ax.add_patch(patch)
 title_text=ax.text(x+w/2,y+h-.22,title,ha='center',va='top',weight='bold',fontsize=11.5,color=c)
 detail_text=ax.text(x+w/2,y+h-.58,detail,ha='center',va='top',fontsize=10.5,linespacing=1.25)
 layout_records.append((patch,title_text,detail_text))
def arrow(a,b): ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.6,color='#444'))
box(1,10,8,1.5,'Released temporal window: preserve row identity','Cap once without labels → train 1500 / valid 600 / test 600\nUnion = the same 2700 original row IDs in both arms')
box(.3,7.7,4.3,1.5,'Temporal membership','Early train → later valid → test\nReleased order; disclose time ties')
box(5.4,7.7,4.3,1.5,'Random reassignment','Shuffle those exact 2700 IDs\nCut at 1500 and 2100\nKeep x, y and time linked',C[1])
arrow((3,10),(2.5,9.2));arrow((7,10),(7.5,9.2))
box(.3,5.1,4.3,1.7,'Fit preprocessing separately','Train [1500,d] → median / mean / scale\nApply frozen state to valid [600,d]\nand test [600,d]')
box(5.4,5.1,4.3,1.7,'Fit on the reassigned train','New train [1500,d] → its own statistics\nNo validation or test values in fit\nIdentical model candidate budget',C[1])
arrow((2.5,7.7),(2.5,6.8));arrow((7.5,7.7),(7.5,6.8))
box(1,2.7,8,1.55,'For each model and seed: fit → validate → freeze','Train two candidates; neural checkpoints selected on validation\nValidation [.21, .18] selects B; freeze B before test predictions',C[2])
arrow((2.5,5.1),(3,4.25));arrow((7.5,5.1),(7,4.25))
box(1,.3,8,1.55,'Score selected predictions; retain the whole audit','B test error .25 stays .25 even if A could score .20\nSave selected recipe, test predictions, targets, IDs and source hashes',C[2])
arrow((5,2.7),(5,1.85))
fig.suptitle('End-to-end protocol · current local v2',y=.98,weight='bold')
fig.canvas.draw();renderer=fig.canvas.get_renderer()
for patch,*labels in layout_records:
 bound=patch.get_window_extent(renderer)
 for label in labels:
  extent=label.get_window_extent(renderer)
  assert bound.x0 <= extent.x0 and extent.x1 <= bound.x1 and bound.y0 <= extent.y0 and extent.y1 <= bound.y1, label.get_text()
print('PASS: all protocol labels fit their boxes at export size')
save(fig,'protocol')

fig,ax=plt.subplots(figsize=(9,4.6));ax.set(xlim=(0,16),ylim=(-.9,3.5));ax.set_yticks([2,1,0],['Purchase A: 20','Purchase B: 30','Purchase C: 70']);ax.set_xlabel('Day · circle = event; square = amount available')
for y,event,arrival,amount in [(2,4,4,20),(1,9,9,30),(0,10,14,70)]:
 c=C[0] if arrival<=12 else C[1]
 ax.plot([event,arrival],[y,y],color=c,lw=3);ax.scatter(event,y,marker='o',s=110,color=c);ax.scatter(arrival,y,marker='s',s=45,facecolor='white',edgecolor=c,zorder=4)
 ax.text(15.7,y,'USE' if arrival<=12 else 'WAIT',ha='right',va='center',color=c,weight='bold')
ax.axvline(12,ls='--',color=C[2]);ax.text(12,2.6,'Predict at 12',ha='center',color=C[2]);ax.text(.4,-.7,'As-of spend = 20 + 30 = 50; count = 2; mean = 25',fontsize=12,weight='bold')
ax.set_title('Historical event ≠ available value · synthetic trace',pad=14,loc='left');save(fig,'feature-history')

r=json.loads((ROOT/'_paper_l055_results.json').read_text())
fig,axs=plt.subplots(4,2,figsize=(10,11))
for ax,task in zip(axs.flat,dict.fromkeys(v['task'] for v in r['summary'])):
 rows={(v['model'],v['protocol']):v for v in r['summary'] if v['task']==task};metric=rows['XGBoost','random']['metric'];sign=1 if metric=='roc-auc' else -1
 vals=[sign*(rows['XGBoost',p]['mean']-rows['MLP-PLR',p]['mean']) for p in ('random','temporal')]
 ax.axhline(0,color='#888',lw=1);ax.plot([0,1],vals,color=C[0],marker='o');ax.set(xticks=[0,1],xticklabels=['Random','Temporal'],xlim=(-.35,1.35),title=task,ylabel='Advantage ('+('AUROC' if metric=='roc-auc' else 'RMSE')+')')
 pad=max(abs(v) for v in vals)*.7+.0004;ax.set_ylim(min(0,*vals)-pad,max(0,*vals)+pad)
 for x,v in enumerate(vals):ax.annotate(f'{v:+.5f}',(x,v),xytext=(0,10),textcoords='offset points',ha='center',fontsize=11)
fig.suptitle('Author-report reanalysis · XGBoost advantage over MLP-PLR\nPositive = XGBoost better; panels have different metric scales',y=.99,fontsize=14)
fig.tight_layout(rect=(0,0,1,.94));save(fig,'paper-margins')
# Existing measured figure code, now driven by corrected operator/evidence; historical JSON unchanged.
source=(ROOT/'_figures_l055.py').read_text().replace('relkit.temporal_experiment import','relkit.temporal_experiment_v2 import').replace('_verify_l055_results.json','_verify_l055_v2_results.json').replace("'Random-0','Sliding-window-0'","'Same-pool random','Temporal window 0'")
exec(compile(source,str(ROOT/'_figures_l055.py'),'exec'))
