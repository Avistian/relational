"""Editable vector plus portable raster figures for a survey, not a new model."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent/'figures/l094';P.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(12,7));fig.patch.set_facecolor('#f8fafc');ax.axis('off');ax.set_xlim(0,12);ax.set_ylim(0,7)
ax.text(.2,6.6,'Where does heterogeneity enter?',fontsize=23,weight='bold',color='#16324f')
ax.text(.2,6.15,'Read each row left → right. Route design and encoder family are separate axes.',fontsize=12)
rows=[('Fixed route counts','Typed adjacency matrices','A_AP · A_PA → path counts','Pair features → chosen task', '#8b5cf6'),('metapath2vec','Typed random-walk contexts','Learn a vector for each node ID','Embedding → downstream model','#8b5cf6'),('R-GCN · L091','One-hop typed neighbors','Σᵣ mean(Wᵣ h_source) + self','Repeated layers → entity head','#0f766e'),('HAN · L092','Meta-path neighbor graphs','Neighbor attention → path fusion','Node embedding → task head','#0369a1'),('HGT · L093','Typed, timed sampled edges','Typed Q/K/V → relation attention','Repeated layers → ranking head','#b45309')]
for i,(name,inp,op,out,col) in enumerate(rows):
 y=5.35-i*1.05
 ax.text(.2,y+.13,name,weight='bold',color=col)
 for x,w,text in [(2.05,2.65,inp),(5,3.35,op),(8.7,3.1,out)]:
  ax.add_patch(plt.Rectangle((x,y-.2),w,.67,facecolor='white',edgecolor=col,lw=1.3))
  # wrap deliberately to retain readable lines
  import textwrap
  ax.text(x+.12,y+.13,'\n'.join(textwrap.wrap(text,31)),va='center',fontsize=10)
 for x in [4.75,8.4]:ax.annotate('',xy=(x+.22,y+.12),xytext=(x,y+.12),arrowprops={'arrowstyle':'->','color':'#64748b'})
ax.text(.2,.1,'HAN occupies both “explicit meta-path” and “GNN”. This map does not rank predictive quality.',fontsize=11,color='#334155')
fig.tight_layout();fig.savefig(P/'family-map.png',dpi=160);fig.savefig(P/'family-map.svg');plt.close(fig)
ap=np.array([[1,1,0,0],[0,1,1,0],[0,0,0,1],[0,0,0,0]]);pv=np.array([[1,0],[0,1],[0,1],[1,0]])
fig,axs=plt.subplots(1,3,figsize=(12,4.4));fig.patch.set_facecolor('#f8fafc')
for ax,mat,title in zip(axs,[ap,ap@ap.T,ap@pv@pv.T@ap.T],['A→P adjacency (4 × 4)','A→P→A path counts','A→P→V→P→A counts']):
 ax.imshow(mat,cmap='Blues',vmin=0,vmax=4)
 for (i,j),v in np.ndenumerate(mat):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>=3 else '#16324f',fontsize=16)
 ax.set_title(title,fontsize=12,pad=12);ax.set_yticks(range(4),['Ada','Bo','Cy','Dee']);ax.set_xticks(range(4),['P0','P1','P2','P3'] if ax is axs[0] else ['Ada','Bo','Cy','Dee'])
fig.suptitle('Same four authors: changing the route changes the meaning',fontsize=18,weight='bold')
fig.text(.5,.03,'Synthetic illustration. Ada–Cy: 0 shared-paper paths, 1 shared-venue path. Dee stays isolated.',ha='center',fontsize=11)
fig.tight_layout(rect=(0,.08,1,.92));fig.savefig(P/'route-trace.png',dpi=160);fig.savefig(P/'route-trace.svg');plt.close(fig)
print('Generated family map and exact route trace')
