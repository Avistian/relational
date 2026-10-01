import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;F=P/'figures/l157';F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l157','savefig.facecolor':'#fcfbf7'})
fig,ax=plt.subplots(figsize=(10,6));fig.patch.set_facecolor('#fcfbf7');ax.set(xlim=(0,10),ylim=(0,6));ax.axis('off')
ax.text(.2,5.7,'From a pinned experiment to a reviewable release',fontsize=17,weight='bold')
boxes=[(.3,3.8,'BEFORE RUNNING','Source + data hashes\nProtocol + budget\nExperiment manifest'),(3.65,3.8,'FRESH COMPUTATION','Prepare all original data\nTrain 5 seeds × 2 policies\nSelect by validation only'),(7,3.8,'SAVED EVIDENCE','Checkpoints + predictions\nPer-batch audits\nRun identity + cost'),(.3,1.1,'FINAL PACKAGE','Code + notices + evidence\nREADME + contribution draft\nRelease manifest'),(3.65,1.1,'INDEPENDENT REPLAY','Check bytes + all run keys\nRescore 12,590 predictions\nReject missing seed'),(7,1.1,'BOUNDED CLAIM','Selected experiment: supported\nAvailability: not established\nPublication: still pending')]
for x,y,title,body in boxes:
 ax.add_patch(FancyBboxPatch((x,y),2.7,1.5,boxstyle='round,pad=.15',facecolor='#e8f0eb',edgecolor='#277364'))
 ax.text(x+.05,y+1.18,title,fontsize=10,weight='bold',color='#174f45');ax.text(x+.05,y+.88,body,fontsize=9.3,va='top',linespacing=1.6)
for x1,y1,x2,y2 in [(3.05,4.5,3.45,4.5),(6.4,4.5,6.8,4.5),(7,3.4,1.65,2.85),(3.05,1.8,3.45,1.8),(6.4,1.8,6.8,1.8)]:ax.annotate('',(x2,y2),(x1,y1),arrowprops={'arrowstyle':'->','color':'#476861','lw':1.6})
ax.text(.25,.35,'Hashes attest byte consistency. A fresh run, a review, and a public URL are different evidence.',fontsize=10)
for ext in ['svg','png']:fig.savefig(F/('flow.'+ext),dpi=140,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else {})
plt.close(fig)
r=json.loads((P/'evidence/l157/report.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,4));fig.patch.set_facecolor('#fcfbf7')
for ax,split in zip(axes,['val','test']):
 for i,(name,color) in enumerate([('paper','#226657'),('fit_horizon','#b46a20')]):
  lane=r['lanes'][name];ys=[x[split] for x in lane['records']];m=lane['metrics'][split]
  ax.scatter([i+(s-2)*.04 for s in range(5)],ys,color=color,s=40,zorder=3)
  ax.errorbar(i+.2,m['mean'],yerr=m['sd'],fmt='D',color=color,capsize=4)
 ax.set_xticks([0,1],['Released','Fit by 2005']);ax.set_ylabel('MAE · finishing positions');ax.set_title('Validation' if split=='val' else 'Test');ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.suptitle('L157 · Fresh package-only runs',weight='bold');fig.text(.5,.01,'Dots: five seeds. Diamonds/bars: mean ± sample seed SD. Separate vertical scales.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,.94))
for ext in ['svg','png']:fig.savefig(F/('results.'+ext),dpi=140,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else {})
plt.close(fig)
