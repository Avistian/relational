"""Portable scientific figures for the architecture and state dependencies."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;F=P/'figures/b18a';F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':11,'figure.facecolor':'#fafcf9','axes.facecolor':'#fafcf9'})
fig,ax=plt.subplots(figsize=(8,10));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
def box(x,y,w,h,title,body,color='#e6f1ec'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',fc=color,ec='#648c77'))
 ax.text(x+.018,y+h-.018,title,weight='bold',fontsize=12,va='top');ax.text(x+.018,y+h-.053,body,fontsize=10.5,va='top',linespacing=1.45)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.8,color='#345c48'))
ax.text(.02,.985,'TACO: learned context → repeated prediction',fontsize=17,weight='bold',va='top')
box(.04,.80,.92,.12,'1 · Versioned, legal labeled support','N rows × (M features + target); fit preprocessing on support.\nEmbed each cell into width L. Query labels never enter support.')
box(.04,.59,.43,.13,'2 · Dummy rows','K initial support-derived rows\nTarget column is masked.\nThese become latent cells.')
box(.53,.59,.43,.13,'3 · Compressor gφ','Row + column attention\n12 layers · 6 heads · L=192\nJoint pretraining with fθ')
arrow((.26,.80),(.26,.73));arrow((.75,.80),(.75,.73));arrow((.47,.655),(.52,.655))
box(.18,.41,.64,.105,'4 · Latent context + residual MLP','K × (M+1) × L latent cells\nMaterialize once; reuse while support state is valid.','#fbefda')
arrow((.75,.59),(.55,.525))
box(.04,.19,.58,.14,'5 · Predictor fθ + optional KV cache','Compressed context attends with query cells.\n12 layers · 6 heads · width192\nOutput: class probabilities for query rows')
box(.69,.225,.27,.095,'Query input','M features\nEmbedded cells','#f0edf8')
arrow((.40,.41),(.34,.34));arrow((.69,.27),(.63,.27))
ax.text(.04,.09,'Training: task query loss updates BOTH compressor and predictor.\nServing: weights fixed; support changes require rebuilt derived state.\nRelease detail: compressed-context reuse exists even without predictor KV.',fontsize=11,linespacing=1.6)
fig.savefig(F/'architecture.png',dpi=150,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,7));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
ax.text(.02,.97,'Three inputs; three different questions',fontsize=18,weight='bold',va='top')
for y,title,body,color in [(.70,'Predictive support','Changes the in-context predictor; rebuild support-derived state.','#e6f1ec'),(.49,'Query features','Changes the row being predicted; support cache can remain valid.','#f0edf8'),(.28,'Explanation background','Changes the comparison reference; predictor can remain identical.','#fbefda')]:box(.04,y,.92,.16,title,body,color)
box(.04,.035,.92,.16,'Separate branch: teacher → student',
    'Versioned teacher → out-of-fold targets → student fit → predictions\nSupport changes require teacher rebuild and student revalidation.','#e8eef8')
fig.savefig(F/'explanation-inputs.png',dpi=150,bbox_inches='tight');plt.close(fig)
print('Built two architecture/estimand figures')
