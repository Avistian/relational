"""Portable survey synthesis, real paired errors and explicit planning assumptions."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;D=P/'figures/l147';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'svg.hashsalt':'l147','axes.spines.top':False,'axes.spines.right':False})
ink='#163b36';green='#247e70';orange='#ae5c29'
def save(fig,name):
 fig.savefig(D/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});fig.savefig(D/(name+'.png'),bbox_inches='tight',dpi=140);plt.close(fig)
fig,ax=plt.subplots(figsize=(11,8));ax.set(xlim=(0,11),ylim=(0,8));ax.axis('off')
ax.text(.1,7.7,'Follow the information; locate the unanswered question',fontsize=19,weight='bold',color=ink)
ax.text(.1,7.22,'Course synthesis of the survey + Lessons 141–146. Not a performance ranking.',fontsize=12)
rows=[('01 · ACCESS','Rows + owner cutoff → legal context','Scale / time: is a needed row absent or illegal?'),('02 · REPRESENT','Typed columns → row vectors → routed messages','Heterogeneity: which role distinction survives?'),('03 · PREDICT','Root / candidate representation → prediction','Compare complete procedures on identical query keys.'),('04 · TRANSFER','Pretrained weights → adaptation → unseen database','Foundation-model claim: hold out entire databases.')]
for i,(title,flow,gap) in enumerate(rows):
 y=5.6-i*1.62
 ax.add_patch(FancyBboxPatch((.15,y),10.6,1.25,boxstyle='round,pad=.08',fc='#f0f6f4',ec='#afc8c1'))
 ax.text(.4,y+.91,title,weight='bold',color=green,fontsize=12)
 ax.text(.4,y+.51,flow,fontsize=15,color=ink)
 ax.text(.4,y+.12,gap,fontsize=12)
 if i<3:ax.annotate('',xy=(5.4,y-.3),xytext=(5.4,y-.09),arrowprops=dict(arrowstyle='->',color=green,lw=2))
save(fig,'map')
s=json.loads((P/'evidence/l147/summary.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
for ax,split,title in zip(axes,['val','test'],['Validation: reduced RelGT lower','Test: GNN lower']):
 d=s['paired'][split];ax.axvline(0,color='#999',lw=1);ax.scatter(d['differences'],[0,1,2],s=90,color=green if split=='val' else orange)
 for i,v in enumerate(d['differences']):ax.annotate(f'{v:+.3f}',(v,i),xytext=(7,8),textcoords='offset points',fontsize=11)
 ax.set(xlim=(-.72,.46),ylim=(-.5,2.6),yticks=[0,1,2],yticklabels=['seed 0','seed 1','seed 2'],xlabel='GNN MAE − reduced RelGT MAE',title=title)
 ax.text(.03,-.24,f"Mean {d['mean']:+.4f}; sample SD {d['sample_sd']:.4f}",transform=ax.transAxes,fontsize=12)
fig.suptitle('Same six saved fits, freshly rescored by full query key',fontsize=17,y=1.05)
fig.text(.5,-.23,'7,554 held-out predictions • Reanalysis, no training • One task does not test database transfer',ha='center',fontsize=11)
save(fig,'evidence')
fig,ax=plt.subplots(figsize=(10,5.8));ax.axis('off');ax.set(xlim=(0,10),ylim=(0,6))
ax.text(.1,5.6,'A transparent planning rule—not an empirical leaderboard',fontsize=17,weight='bold',color=ink)
for x,title,body in [(0.2,'FILTER','Ready inputs?\nHours ≤ allowance?\nDollars ≤ $10?'),(3.6,'ORDER','Evidence level ↓\nEstimated hours ↑\nStable ID breaks ties'),(7,'CHALLENGE','Change a judgment.\nRecompute the order.\nDefend the difference.')]:
 ax.add_patch(FancyBboxPatch((x,2.5),2.8,2.3,boxstyle='round,pad=.1',fc='#f0f6f4',ec='#afc8c1'));ax.text(x+.15,4.35,title,color=green,weight='bold');ax.text(x+.15,3.8,body,va='top',fontsize=12,linespacing=1.7)
ax.text(.2,1.7,'1 hour: selection',fontsize=15)
ax.text(.2,1.1,'4 hours: selection → ownership → routes',fontsize=15)
ax.text(.2,.5,'8 hours: add coverage. Transfer remains blocked by missing inputs.',fontsize=13)
save(fig,'ranking')
print('Built three portable figures')
