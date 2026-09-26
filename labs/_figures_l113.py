"""Portable computation diagrams; no generated artwork or hidden numerical claims."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
P=Path(__file__).resolve().parent;out=P/'figures/l113';out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.hashsalt':'l113','savefig.facecolor':'#fafaf7'})
def save(fig,name):
 fig.savefig(out/f'{name}.svg',metadata={'Date':None},bbox_inches='tight');fig.savefig(out/f'{name}.png',dpi=150,bbox_inches='tight',metadata={'Software':'L113'});plt.close(fig)
def box(ax,x,y,w,h,s,color='#e3eeec'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',facecolor=color,edgecolor='#52706c'));ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=11)
def canvas(title,h=7):
 fig,ax=plt.subplots(figsize=(9,h));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title(title,loc='left',fontweight='bold',pad=20);return fig,ax
def arrow(ax,a,b):ax.annotate('',b,a,arrowprops={'arrowstyle':'->','lw':2,'color':'#21766e'})
fig,ax=canvas('Two decisions: which nodes? Which aggregation?',8)
box(ax,.07,.84,.86,.12,'Full products graph: 2,449,029 nodes × 100 features\nPartition once with METIS → 15,000 disjoint groups')
box(ax,.07,.65,.86,.12,'Shuffle groups → union of 32 groups → induced edges\nRetain edges between selected groups; cut edges to excluded nodes')
box(ax,.07,.38,.4,.19,'Neighbor branch\nmean of incoming features\n→ learned Wneighbor + bias')
box(ax,.53,.38,.4,.19,'Root branch\nnode’s own features\n→ separate learned Wroot')
box(ax,.07,.18,.86,.12,'Add branches → ReLU → dropout(0.5)\nRepeat: 100 → 256 → 256 → 47 (no activation on last logits)')
box(ax,.07,.015,.86,.095,'Training labels only → NLL → Adam(0.001)\nInference: ALL neighbors, one complete layer at a time','#f4e6ca')
for a,b in [((.5,.84),(.5,.77)),((.5,.65),(.27,.57)),((.5,.65),(.73,.57)),((.27,.38),(.4,.30)),((.73,.38),(.6,.30)),((.5,.18),(.5,.11))]:arrow(ax,a,b)
save(fig,'architecture')
fig,axes=plt.subplots(1,2,figsize=(10,4.8))
for ax,q,title in zip(axes,[1,2],['C0 only: cut the boundary','C0 + C1: restore the boundary']):
 ax.set(xlim=(-.4,3.4),ylim=(-.8,1.1));ax.axis('off');ax.set_title(title,fontsize=13,fontweight='bold')
 for i in range(3):ax.plot([i,i+1],[.5,.5],color='#21766e' if i<2*q-1 else '#aaaaaa',ls='-' if i<2*q-1 else '--',lw=3)
 for i in range(4):ax.scatter([i],[.5],s=1000,color='#21766e' if i<2*q else '#cccccc',zorder=3);ax.text(i,.5,str(i),ha='center',va='center',color='white',zorder=4);ax.text(i,.12,'x='+str(2**(i+1)),ha='center')
 ax.text(.5,.9,'C0',ha='center');ax.text(2.5,.9,'C1',ha='center');ax.text(1.5,-.42,'node 1: 2 + 4 = 6' if q==1 else 'node 1: (2 + 8)/2 + 4 = 9',ha='center',fontsize=13)
fig.suptitle('Same weights; changed neighborhood → changed computation',fontweight='bold');save(fig,'boundary')
fig,ax=canvas('Exact inference: finish each layer before starting the next',6)
for y,tag,shape in [(.78,'Read all input features','N × 100'),(.48,'Complete hidden layer 1','N × 256'),(.18,'Complete hidden layer 2','N × 256')]:
 box(ax,.025,y,.385,.13,tag+'\n'+shape)
 box(ax,.56,y,.415,.13,'For each receiver chunk:\nALL incoming neighbors → output rows','#f4e6ca');arrow(ax,(.41,y+.065),(.56,y+.065))
arrow(ax,(.77,.78),(.21,.61));arrow(ax,(.77,.48),(.21,.31));ax.text(.5,.05,'Final layer: N × 47 logits → predictions in original global node order',ha='center',fontsize=11)
save(fig,'inference')
fig,ax=plt.subplots(figsize=(9,4.8));batches=np.array([256,512,1024]);f=10;n=batches*(1+f+f*f+f**3)
ax.bar(['256 roots','512 roots','1024 roots'],n*256*4/2**30,color='#21766e');ax.axhline(2449029*256*4/2**30,color='#b06b34',ls='--',label='Full-products hidden buffer')
ax.set_ylabel('GiB · one float32, width-256 buffer');ax.set_title('Neighbor expansion: a sizing bound, not measured peak memory',loc='left',fontweight='bold',fontsize=12);ax.legend();ax.text(.02,.96,'Three hops, fanout 10; no overlap assumed.\nEdges, gradients, optimizer and workspace excluded.',transform=ax.transAxes,va='top',fontsize=10);save(fig,'memory')
path=P/'evidence/l113/summary.json'
r=json.loads(path.read_text()) if path.exists() else {};rows=r.get('seeds',[])
if rows:
 fig,axes=plt.subplots(2,1,figsize=(9,8),sharex=True)
 for ax,key,target in zip(axes,['valid','test'],[92.12,78.97]):
  vals=np.array([a[key+'_percent'] for a in rows]);ax.scatter([a['seed'] for a in rows],vals,color='#21766e',s=55,label='Selected checkpoint')
  ax.axhline(target,color='#a65a24',ls='--',label='Published ten-run mean')
  if len(rows)==10:
   ax.axhspan(target-.5,target+.5,color='#a65a24',alpha=.08,label='Tolerance for mean, not individual seeds')
   ax.errorbar([11],[vals.mean()],yerr=[vals.std(ddof=1)],fmt='o',capsize=6,color='#344b7d',label='Mean ± sample seed SD')
  ax.set_ylabel(('Validation' if key=='valid' else 'Test')+' accuracy (%)');ax.legend(fontsize=8,loc='best')
 axes[-1].set_xticks(list(range(10))+[11],list(map(str,range(10)))+['Mean']);axes[-1].set_xlabel('Seed · each 50 epochs, checkpoint selected by validation')
 fig.suptitle(f"{len(rows)}/10 complete runs · {r.get('status','INCOMPLETE')}",fontweight='bold')
else:
 fig,ax=plt.subplots(figsize=(9,4.8));ax.axis('off');ax.text(.5,.55,'Ten-run reproduction: no complete seed recorded yet',ha='center',va='center',fontsize=14);ax.text(.5,.35,'A pilot validates execution and estimates cost.\nIt does not reproduce the published result.',ha='center',va='center')
save(fig,'results')
