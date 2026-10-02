"""Portable figures: a worked reasoning trace and the complete rubric sweep."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;D=P/'figures/l189';D.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l189','axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(9,6));ax.set(xlim=(0,10),ylim=(0,8));ax.axis('off')
items=[('READ THE LIMITATION','Temporal pretraining: within-database study\nUnseen databases + matched cost left open'),('NARROW THE QUESTION','Hold out the entire target database\nCompare two objectives at equal all-in cost'),('DECLARE THE TEST','Pretrained − extra-compute scratch\nUseful gain: 0.01 AUROC per target task'),('SEPARATE THE CLAIMS','Research candidate: yes  ·  Novelty: unknown\nFull run: NOT_RUN  ·  Cost bound: unknown')]
for i,(title,body) in enumerate(items):
 y=6.2-i*1.7
 ax.add_patch(FancyBboxPatch((.6,y),8.8,1.3,boxstyle='round,pad=0.12',facecolor=['#eaf4f1','#eaf4f1','#edf2fa','#fff3df'][i],edgecolor='#aac6c0'))
 ax.text(1,y+.95,title,color='#146b64',fontsize=10,weight='bold');ax.text(1,y+.44,body,fontsize=12,va='center',linespacing=1.4)
 if i<3:ax.annotate('',xy=(5,y-.27),xytext=(5,y-.04),arrowprops={'arrowstyle':'->','color':'#146b64','lw':2})
fig.tight_layout()
for ext in ['svg','png']:fig.savefig(D/f'trace.{ext}',dpi=150,metadata={'Date':None} if ext=='svg' else None)
plt.close(fig)
r=json.loads((P/'evidence/l189/report.json').read_text());fig,ax=plt.subplots(figsize=(9,4.5))
for i,(key,label) in enumerate([('temporal','Temporal validity'),('composite','Composite structure'),('transfer','Cross-database pretraining')]):
 values=[g['scores'][key] for g in r['weight_grid']];y=2-i
 ax.scatter(values,[y+((j%5)-2)*.035 for j in range(27)],s=25,color='#146b64',alpha=.5)
 base=next(x['score'] for x in r['ranking'] if x['id']==key);ax.scatter([base],[y],s=95,marker='D',color='#b36c22',zorder=3)
 ax.text(24.5,y,f'{base:.2f}',ha='right',va='center',weight='bold')
ax.set(yticks=[2,1,0],yticklabels=['Temporal validity','Composite structure','Cross-database pretraining'],xlim=(0,25),ylim=(-.6,2.6),xlabel='Authored impact × weighted feasibility (0–25)')
ax.set_title('27 weight settings · same authored scores',loc='left',pad=16,weight='bold');ax.grid(axis='x',alpha=.2)
fig.text(.5,.02,'Dots: full weight sweep. Diamonds: equal weights. These are judgments, not performance measurements.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.07,1,1))
for ext in ['svg','png']:fig.savefig(D/f'ranking.{ext}',dpi=150,metadata={'Date':None} if ext=='svg' else None)
plt.close(fig)
# Portrait trace keeps the operation readable without panning on a phone.
fig,ax=plt.subplots(figsize=(5,7.5));ax.set(xlim=(0,5),ylim=(0,10));ax.axis('off')
mobile=[('READ THE LIMITATION','Within-database pretraining\nTransfer + equal cost left open'),('NARROW THE QUESTION','Hold out the whole database\nCompare objectives at equal cost'),('DECLARE THE TEST','Pretrained − extra-compute scratch\nUseful gain: 0.01 AUROC per task'),('SEPARATE THE CLAIMS','Candidate: yes · Novelty: unknown\nFull run + cost bound: unresolved')]
for i,(title,body) in enumerate(mobile):
 y=7.9-i*2.35
 ax.add_patch(FancyBboxPatch((.15,y),4.7,1.85,boxstyle='round,pad=.06',facecolor='#fff3df' if i==3 else '#eaf4f1',edgecolor='#aac6c0'))
 ax.text(.35,y+1.4,title,fontsize=11,color='#146b64',weight='bold');ax.text(.35,y+.7,body,fontsize=12,va='center',linespacing=1.6)
 if i<3:ax.annotate('',xy=(2.5,y-.4),xytext=(2.5,y-.08),arrowprops={'arrowstyle':'->','lw':2,'color':'#146b64'})
fig.subplots_adjust(left=.02,right=.98,top=.99,bottom=.01)
fig.savefig(D/'trace-mobile.svg',metadata={'Date':None});plt.close(fig)
