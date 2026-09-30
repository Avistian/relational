"""Intervention computation, owner windows, paired evidence and interaction."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;D=P/'figures/l148';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.hashsalt':'l148','axes.spines.top':False,'axes.spines.right':False})
ink='#183b35';green='#247e70';orange='#ae5c29'
def save(fig,name):
 fig.savefig(D/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});fig.savefig(D/(name+'.png'),bbox_inches='tight',dpi=145);plt.close(fig)
fig,ax=plt.subplots(figsize=(11,7));ax.axis('off');ax.set(xlim=(0,11),ylim=(0,7))
ax.text(.15,6.65,'Locate the intervention before interpreting its score',fontsize=19,weight='bold',color=ink)
rows=[('ACCESS','Rows + owner cutoff → sampled context','History: prune old dated non-roots; do not refill slots.'),('ENCODE','Columns → embeddings → 4 residual blocks → 128','Encoder: keep column embeddings + block 1 + decoder.'),('PROPAGATE','h′ = neighbor map(Σ hneighbor) + root map(hroot)','Messages: empty edges; root maps and biases remain.'),('PREDICT','2 propagation layers → driver vector → scalar position','All arms: same labels, loss, schedule and selection rule.')]
for i,(label,operation,change) in enumerate(rows):
 y=4.95-i*1.42
 ax.add_patch(FancyBboxPatch((.2,y),10.4,1.08,boxstyle='round,pad=.12',fc='#eff6f2',ec='#acc8bb'))
 ax.text(.4,y+.77,label,weight='bold',color=green,fontsize=11);ax.text(.4,y+.42,operation,fontsize=14);ax.text(.4,y+.07,change,fontsize=11,color=orange)
 if i<3:ax.annotate('',xy=(5.5,y-.28),xytext=(5.5,y-.12),arrowprops=dict(arrowstyle='->',color=green,lw=2))
save(fig,'interventions')
fig,ax=plt.subplots(figsize=(10,4.8))
for y,c,name in [(1,400,'Query A'),(0,500,'Query B')]:
 ax.plot([c-365,c],[y,y],lw=14,color='#b5d8c6',solid_capstyle='butt');ax.scatter([c],[y],marker='|',s=500,color=ink)
 for t in [34,100,399,450,501]:
  keep=c-365<=t<=c;ax.scatter(t,y,s=65,color=green if keep else orange,zorder=4)
  ax.annotate(str(t),(t,y),xytext=(0,15 if t not in [399,501] else -22),textcoords='offset points',ha='center',fontsize=11)
 ax.text(-25,y,name,ha='right',va='center');ax.text(c-365,y+.32,f'lower = {c-365}',fontsize=10)
ax.set(xlim=(-30,560),ylim=(-.65,1.75),yticks=[],xlabel='Illustrative day; green segment is the inclusive 365-day window')
ax.spines['left'].set_visible(False)
ax.set_title('Same row day 100: legal for A, too old for B',fontsize=17,pad=18)
ax.text(.02,-.3,'Dated non-root example. Query roots and undated rows have explicit exceptions.\nFuture rows are always excluded, regardless of window length.',transform=ax.transAxes,fontsize=11)
save(fig,'history')
s=json.loads((P/'evidence/l148/summary.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,5.2),sharex=True,sharey=True)
labels=['Encoder','Messages','History','Combined'];arms=['encoder','messages','history','combined']
allv=[v for split in ['val','test'] for a in arms for v in s['paired'][split][a]['differences']];lo=min(0,min(allv))-.15;hi=max(0,max(allv))+.15
for ax,split in zip(axes,['val','test']):
 ax.axvline(0,color='#888',lw=1)
 for i,a in enumerate(arms):
  d=s['paired'][split][a]
  ax.scatter(d['differences'],[i+(j-2)*.075 for j in range(5)],s=42,color=green,alpha=.8)
  ax.scatter(d['mean'],i,marker='D',color=orange,s=55,zorder=5)
 ax.set(yticks=range(4),yticklabels=labels,xlim=(lo,hi),title=split.capitalize(),xlabel='Arm MAE − full MAE (positive is worse)');ax.invert_yaxis()
fig.suptitle('Fresh full-data fits: five paired seed differences per arm',fontsize=17,y=1.01)
fig.text(.5,-.03,'Dots = seeds; diamond = mean. Shared scale. No confidence interval or cross-database inference.',ha='center',fontsize=11)
save(fig,'results')
fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
for ax,sp in zip(axes,['val','test']):
 d=s['interaction'][sp];ax.axhline(0,color='#888',lw=1);ax.scatter(range(5),d['differences'],color=green,s=65)
 ax.set(xticks=range(5),xlabel='Paired seed',title=sp.capitalize());ax.text(.03,.96,f"Mean {d['mean']:+.3f}\nSample SD {d['sample_sd']:.3f}",transform=ax.transAxes,va='top')
axes[0].set_ylabel('EM − E − M + F (MAE units)');fig.suptitle('Does removing two components add their separate penalties?',fontsize=16,y=1.04)
fig.text(.5,-.06,'Negative: combined penalty is smaller than additive prediction. Data interactions are unmeasured.',ha='center',fontsize=11)
save(fig,'interaction')
