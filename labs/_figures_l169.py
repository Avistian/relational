"""Scale axes, full observed curves and non-monotone doubling gains."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent;O=P/'figures/l169';O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l169'})
def save(fig,name):
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 for ax in fig.axes:
  for t in ax.texts:
   b=t.get_window_extent(renderer)
   assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,(name,t.get_text())
 fig.savefig(O/(name+'.svg'),metadata={'Date':None});fig.savefig(O/(name+'.png'),dpi=140,metadata={'Software':'L169'});plt.close(fig)
fig,ax=plt.subplots(figsize=(12,4.8));fig.patch.set_facecolor('#f4f7fa');ax.axis('off')
fig.suptitle('Which quantity changed?',fontsize=20,color='#193952',y=.97)
cols=['Experiment','Context K','Parameters\nP','Pretrain\ndata D','Schema mix\nS','What is\nmeasured?']
rows=[['L169 observed sweep','64 → 1024','FIXED','FIXED','FIXED','Target-context response'],['Parameter proposal','FIXED','VARIED','FIXED','FIXED','Capacity response*'],['Data proposal','FIXED','FIXED','VARIED','FIXED','Data response*'],['Diversity proposal','FIXED','FIXED','FIXED','VARIED','Diversity response*']]
table=ax.table(cellText=rows,colLabels=cols,cellLoc='center',loc='center',colWidths=[.2,.13,.13,.16,.13,.25]);table.auto_set_font_size(False);table.set_fontsize(13);table.scale(1,2.8)
for (i,j),c in table.get_celld().items():
 c.set_edgecolor('#c5d2dc')
 if i==0:c.set_facecolor('#193952');c.set_text_props(color='white',weight='bold')
 elif i==1:c.set_facecolor('#e0f0e8')
 elif c.get_text().get_text()=='VARIED':c.set_facecolor('#fff0d9')
 else:c.set_facecolor('#f4f7fa')
fig.text(.045,.12,'*Proposed controls, not completed pretraining experiments. Training compute allocation must also be declared.',fontsize=10)
fig.text(.045,.065,'Trace: the same frozen θ reads [64,F] → [128,F] → … → [1024,F] labeled rows; the full query set stays fixed.',fontsize=11,color='#193952')
fig.subplots_adjust(left=.03,right=.97,top=.90,bottom=.18);save(fig,'protocol')
r=json.loads((P/'evidence/l169/report.json').read_text());arms=['RDBPFN','RDBPFN_single','TabICLv1.1'];colors=['#167d80','#a85a26','#695bad'];ks=[64,128,256,512,1024]
fig,axes=plt.subplots(1,2,figsize=(12,6.5),sharey=True);fig.patch.set_facecolor('#f4f7fa')
for ax,db in zip(axes,['rel-f1','rel-trial']):
 for arm,color in zip(arms,colors):
  levels=r['curves'][db+'/'+arm]['levels'];mean=np.array([a['mean'] for a in levels]);sd=np.array([a['sample_sd'] for a in levels]);target=[next(x['paper'] for x in r['comparisons'] if x['database']==db and x['arm']==arm and x['context']==k) for k in ks]
  ax.plot(ks,mean,'o-',label=arm,color=color,lw=2);ax.fill_between(ks,mean-sd,mean+sd,color=color,alpha=.10);ax.plot(ks,target,':D',color=color,alpha=.7,ms=4)
  ax.scatter([512],[mean[3]],facecolors='white',edgecolors=color,s=65,zorder=5)
 ax.set_xscale('log',base=2);ax.set(xticks=ks,xticklabels=ks,ylim=(.45,.85),title='F1 / driver-dnf' if db=='rel-f1' else 'Trial / study-outcome',xlabel='Labeled context K · log₂ spacing');ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('AUROC · same scale on both tasks');axes[0].legend(frameon=False,loc='lower right',fontsize=9)
fig.suptitle('Five sizes expose what one 512-context result cannot',fontsize=18,color='#193952')
fig.text(.055,.06,'Solid circles / bands: measured mean ± sample SD over 10 draws. Dotted diamonds: published means.',fontsize=10)
fig.text(.055,.025,'Hollow 512 marker: reused L166/L168 evidence. Other contexts: fresh L169. Bands are not database-level confidence.',fontsize=10)
fig.tight_layout(rect=[0,.10,1,.94]);save(fig,'curves')
fig,axes=plt.subplots(1,2,figsize=(12,6),sharey=True);fig.patch.set_facecolor('#f4f7fa')
for ax,db in zip(axes,['rel-f1','rel-trial']):
 for i,(arm,color) in enumerate(zip(arms,colors)):
  d=r['curves'][db+'/'+arm]['doublings'];x=np.arange(4)+(i-1)*.20
  ax.errorbar(x,[a['mean_gain'] for a in d],yerr=[a['sample_sd'] for a in d],fmt='o',capsize=3,color=color,label=arm)
 ax.axhline(0,color='#475b6b',lw=1);ax.set(xticks=range(4),xticklabels=['64→128','128→256','256→512','512→1024'],title=db,xlabel='Context doubling');ax.tick_params(axis='x',labelsize=10);ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('AUROC change · mean ± sample SD');axes[0].legend(frameon=False,fontsize=9)
fig.suptitle('Averaging must retain negative doubling responses',fontsize=18,color='#193952')
fig.text(.045,.055,'Seed-indexed contrasts under the released sampler. Draws at different sizes are not nested support sets.',fontsize=10)
fig.tight_layout(rect=[0,.10,1,.93]);save(fig,'gains');print('Built3 portable figures')
