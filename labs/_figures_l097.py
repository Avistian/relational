"""A computation map for conditional sampling and shared-parameter pairwise training."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent/'figures/l097';P.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans'})
fig,ax=plt.subplots(figsize=(13,7.3));fig.patch.set_facecolor('#f7faf9');ax.set(xlim=(0,13),ylim=(0,7.3));ax.axis('off')
ax.text(.3,6.92,'WHAT THE SAMPLER CHANGES',fontsize=19,weight='bold',color='#173c36')
ax.text(.3,6.52,'One typed relation • two shared score branches • one fixed evaluation catalog',fontsize=12,color='#46635e')
def box(x,y,w,h,title,body,color='#e4f2ec'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',facecolor=color,edgecolor='#789b90'))
 ax.text(x+.12,y+h-.3,title,fontsize=12,weight='bold',color='#173c36',va='top')
 ax.text(x+.12,y+h-.73,body,fontsize=10.5,linespacing=1.6,va='top',color='#243b37')
def arrow(a,b,label=None):
 ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#2c7464','lw':1.8})
 if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,fontsize=9,ha='center',color='#245e52')
box(.3,4.15,3.05,1.85,'TRAIN BASE ONLY','Observed ratings → block mask\nLikes ≥4 → positive pairs [N,2]\nLike-degrees → proposal weights')
box(4.0,4.15,3.65,1.85,'CONDITIONAL SAMPLER','Fix user u and relation likes\nUniform / (degree+1)^0.75\nOr highest score of 4 uniform draws')
box(8.3,4.15,4.1,1.85,'SHARED EMBEDDINGS','User table U [943,32]\nItem table V [1682,32] + bias\nSame U,V score positive i and negative j')
arrow((3.48,5.0),(3.85,5.0));arrow((7.78,5.0),(8.15,5.0))
box(.3,1.95,3.05,1.4,'LABEL BOUNDARY','No test labels enter fit()\nNo future-positive blacklist','#f9ebd5')
box(4.,1.95,3.65,1.4,'PAIRWISE GRADIENT','softplus(s_neg − s_pos) + L2\nAdam updates U, V and item bias')
box(8.3,1.95,4.1,1.4,'TWO SCORES [B]','s_pos = U[u] · V[i] + b[i]\ns_neg = U[u] · V[j] + b[j]')
arrow((10.35,4.0),(10.35,3.49));arrow((8.15,2.65),(7.79,2.65));arrow((5.8,3.49),(5.8,4.0),'next update')
box(.3,.22,12.1,1.0,'INFERENCE: FREEZE PARAMETERS, SCORE EVERY ELIGIBLE ITEM','All catalog items minus base observations → Recall@10 / NDCG@10. Separately: same scores, test likes +99 distractors.','#e4edf8')
arrow((10.35,1.81),(10.35,1.36))
fig.savefig(P/'pipeline.svg',bbox_inches='tight');fig.savefig(P/'pipeline.png',dpi=140,bbox_inches='tight');plt.close(fig)
