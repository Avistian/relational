"""Whole computation map: schema -> query -> dependency closure -> loss/inference."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent/'figures/l098';P.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(14,9),facecolor='#faf8f2');ax.set(xlim=(0,14),ylim=(0,9));ax.axis('off')
ink='#193248'
ax.text(.4,8.55,'Heterogeneous mini-batching: a query defines its computation',fontsize=18,weight='bold',color=ink)
def box(x,y,w,h,title,body,c='#e1efec'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',facecolor=c,edgecolor='#74928e'))
 ax.text(x+.15,y+h-.35,title,fontsize=12,weight='bold',color=ink)
 ax.text(x+.15,y+h-.75,body,fontsize=11,va='top',linespacing=1.6,color=ink)
def arrow(a,b,label=''):
 ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#087f8c','lw':2})
 if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.1,label,fontsize=10,ha='center',color=ink)
box(.4,6,3.4,1.9,'1  Typed database','customer [24,4] · static\norders [144,4] · time 1…6\nproduct [12,4] · static')
box(5.0,6,3.6,1.9,'2  Query / legal inputs','4 customer seeds; cutoff = 5\nFilter forward + reverse edges\nLabels remain outside features','#f6e7cb')
box(9.7,6,3.8,1.9,'3  Incoming dependencies','Hop 1: customer ← orders\nHop 2: orders ← product / customer\nFanout: per relation, per hop')
arrow((3.95,7),(4.85,7));arrow((8.75,7),(9.55,7))
box(.4,2.85,3.4,2.3,'4  Batch-local tensors','x_dict: [N_type,4]\nedge_index_dict: [2,E_relation]\nn_id: local → typed global ID\ne_id: original relation-edge ID')
box(5.0,2.85,3.6,2.3,'5  Shared typed encoder','Layer 1: means + root → ReLU\nLayer 2: means + root → ReLU\nWidths: 4 → 16 → 16\nCustomer head → [N_customer]')
box(9.7,2.85,3.8,2.3,'6  Seeds own the loss','First B customer logits → BCE\nOnly seed labels → objective\nBackward + Adam update\nContext supplies messages','#f6e7cb')
arrow((11.6,5.85),(11.6,5.55));ax.plot([11.6,2.1],[5.55,5.55],color='#087f8c',lw=2);arrow((2.1,5.55),(2.1,5.25))
arrow((3.95,4),(4.85,4));arrow((8.75,4),(9.55,4))
box(.4,.35,6.2,1.65,'AUDIT · freeze weights, then compare','All neighbors: logits + gradients ≈ dense oracle\nSame customer, two cutoffs: two disjoint query histories','#e8e8f2')
box(7.15,.35,6.35,1.65,'INFERENCE · freeze the selected checkpoint','Full legal context → validation / test predictions\nTraining fanout varies; evaluation stays fixed','#e8e8f2')
fig.savefig(P/'pipeline.svg',bbox_inches='tight');fig.savefig(P/'pipeline.png',dpi=145,bbox_inches='tight');plt.close(fig)
