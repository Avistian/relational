"""Portable model-specific architecture with the two summary directions exposed."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent/'figures/b09';P.mkdir(exist_ok=True)
fig,ax=plt.subplots(figsize=(11,10));fig.patch.set_facecolor('#f8faf9');ax.set(xlim=(0,11),ylim=(0,10));ax.axis('off')
def box(x,y,w,h,title,detail,color='#e2efeb'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor=color,edgecolor='#2e7068'))
 ax.text(x+.15,y+h-.22,title,fontsize=12,weight='bold',va='top',color='#173c37');ax.text(x+.15,y+h-.65,detail,fontsize=10,va='top',linespacing=1.5)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#27695f',lw=2))
ax.text(.2,9.7,'EXAONE release · where do the summaries travel?',fontsize=18,weight='bold',color='#153e36')
box(.2,7.7,10.5,1.5,'1  Encode known evidence','353 support rows × 10 features + observed targets; 89 query rows × 10 features.\nCell vectors: [batch, items, feature slots, 192]. Query target slot is hidden.')
arrow((5.5,7.65),(5.5,7.3))
box(.2,5.4,5.05,1.8,'2a  Along each row','Feature attention connects cells within a row.\n3 item-summary tokens collect feature evidence.\n[batch, items, 3, 192]','#e3edf8')
box(5.7,5.4,5,1.8,'2b  Along each column','Item attention connects support examples.\n32 feature-summary tokens collect row evidence.\n[batch, 32, feature slots, 192]','#f7ecdc')
arrow((5.25,6.3),(5.65,6.3));arrow((5.7,5.9),(5.3,5.9))
box(.2,3.5,10.5,1.4,'3  Cross-axis exchange · repeated 12 layers','Before axis-wise attention, cells read summaries attached to the opposite axis.\nFeature attention repeats twice per layer; feed-forward computation also costs time.\nQuery access obeys the support boundary; summaries do not create query labels.')
arrow((2.7,5.3),(2.7,5.0));arrow((8.2,5.3),(8.2,5.0));arrow((5.5,3.4),(5.5,3.05))
box(.2,1.55,10.5,1.4,'4  Read query states → 999 regression quantiles → point estimate','The wrapper reverses target transforms and combines preprocessing/ensemble views.\nTimed calls include these operations. No gradient updates occur in this inference lane.')
ax.text(.2,.9,'A visibility micro-trace (illustrative equal-score attention, not SSMax):',fontsize=11,weight='bold')
ax.text(.2,.45,'Q1 ← support values [2, 6]  →  weights [½, ½]  →  readout 4.\nAdding Q2=30 to the key set would change the computation and violate this boundary.',fontsize=11)
fig.savefig(P/'exaone-architecture.png',dpi=160,bbox_inches='tight');plt.close(fig)
# Distinct model paths: expose where the cell grid collapses (TabFM) or persists (Nori).
fig,axes=plt.subplots(1,2,figsize=(11,7.5));fig.patch.set_facecolor('#f8faf9')
paths=[('TabFM · current regression release',[
('Numeric groups','353 support +89 query rows\n3-column groups → Fourier features\nCell vectors have width 256'),
('Compress the table','Column induced attention ↔ row attention\n256 inducing points; 8 row-summary slots\nPool each row: 8×256=2048 coordinates'),
('Mix row representations','Observed support targets condition the context\n24-layer in-context predictor\nQueries read the permitted support evidence'),
('Predict one scalar','MLP regression head → target transform inverse\nOur base wrapper: 1 estimator; no extra views\nCurrent file: 1.65B stored tensor elements')]),('Nori-6M · current release',[
('Encode feature pairs','353 support +89 query rows\nPair features → radial-basis embeddings\nVectors have width 128'),
('Keep feature and sample axes','16 layers; 2 attention heads\nFeature attention ↔ sample attention\nSupport target evidence conditions queries'),
('Decode uncertainty','999 quantile-level predictions\nA quantile is a value at a probability level\nPretraining uses pinball loss; inference is frozen'),
('Aggregate the system','Current loader selects EMA weights\nNative transformed views → point estimate\n5.87M elements per weight-state dictionary')])]
for ax,(title,stages) in zip(axes,paths):
 ax.set(xlim=(0,5),ylim=(0,8));ax.axis('off');ax.text(.05,7.65,title,fontsize=13,weight='bold',color='#163d37')
 for i,(title,detail) in enumerate(stages):
  y=5.9-i*1.75;ax.add_patch(FancyBboxPatch((.05,y),4.85,1.4,boxstyle='round,pad=.04',facecolor='#e4efec',edgecolor='#39756a'))
  ax.text(.18,y+1.2,title,fontsize=11,weight='bold');ax.text(.18,y+.9,detail,fontsize=9,va='top',linespacing=1.6)
  if i<3:ax.annotate('',xy=(2.5,y-.25),xytext=(2.5,y-.04),arrowprops=dict(arrowstyle='->',lw=2,color='#39756a'))
fig.tight_layout();fig.savefig(P/'other-architectures.png',dpi=180,bbox_inches='tight');plt.close(fig)
