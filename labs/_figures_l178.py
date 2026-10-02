"""Three portable figures: actual route internals, failure trace and evidence boundary."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;F=P/'figures/l178';F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l178'})
navy='#17324d';teal='#087f83';gold='#a66412';red='#a93d36';muted='#536575'
def save(fig,name):
 fig.savefig(F/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});fig.savefig(F/(name+'.png'),dpi=150,bbox_inches='tight',metadata={'Software':'L178'});plt.close(fig)
def box(ax,x,y,w,h,text,color=teal):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.07',facecolor='#f2f6f6',edgecolor=color));ax.text(x+w/2,y+h/2,text,ha='center',va='center',color=navy,fontsize=10.4)
def arrow(ax,a,b,color=teal):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':color,'lw':1.8})
fig,ax=plt.subplots(figsize=(13,8));ax.set(xlim=(0,13),ylim=(0,8));ax.axis('off')
ax.text(.1,7.6,'One information contract, three computational routes',fontsize=20,weight='bold',color=navy)
ax.text(.1,7.16,'Planned: 1,024 shared support keys / seed · 566 validation queries · 702 test queries',color=muted)
rows=[(5.25,'RDB-PFN · released relational prior', ['Relational features\n[1,024 + Q, F]','Support normalization\n+ scalar / label embeddings','6 attention blocks\nwidth 96 · 4 heads','Frozen classifier\nQ DNF probabilities'], 'No target gradient updates. Full feature regeneration was not reached.'),
(3.12,'RelGNN · task-specific learning', ['Typed FK graph\nowner-cutoff neighborhoods','Table ResNets: 128\n+ relative-time encoding','Composite route attention\n4 heads · sum fusion','BCE training / validation\n2 rates × 3 seeds × 10 epochs'],'Training-health gate failed in the numerical row encoder, before graph training.'),
(.99,'RDBLearn + TabICL · deterministic features, pretrained predictor', ['Same relational database\nexplicit query cutoffs','Depth-2 DFS → SQL\nno target-history injection','Fit preprocessing on support\nTabICL v1.1 · 32 estimators','Frozen ICL predictor\nQ DNF probabilities'],'One-query FastDFS intervention passed. No full RDBLearn model evaluation.')]
for y,title,texts,caption in rows:
 ax.text(.1,y+1.31,title,color=navy,weight='bold',fontsize=12)
 for i,t in enumerate(texts):
  x=.1+i*3.26;box(ax,x,y,2.9,.98,t)
  if i<3:arrow(ax,(x+2.99,y+.49),(x+3.19,y+.49))
 ax.text(.1,y-.33,caption,fontsize=10.3,color=red if 'RelGNN' in title else muted)
ax.text(.1,.1,'Shared label access and spending ceilings do not imply equal model capacity, search effort or pretraining cost.',fontsize=10.2,color=muted)
save(fig,'routes')
fig,ax=plt.subplots(figsize=(12,6.2));ax.set(xlim=(0,12),ylim=(0,6.2));ax.axis('off')
ax.text(.1,5.8,'Finite output does not guarantee a usable gradient',fontsize=20,weight='bold',color=navy)
ax.text(.1,5.28,'Actual support history: 58 rows × 8 numeric columns · two columns contain missing cells',color=muted)
for i,(text,color) in enumerate([('Missing raw input\nx = NaN',gold),('Affine encoding\nz = w × x',teal),('Output fallback\nnan_to_num(z) = 0',teal)]):
 x=.15+4*i;box(ax,x,3.8,3.4,.95,text,color)
 if i<2:arrow(ax,(x+3.5,4.28),(x+3.9,4.28))
box(ax,.15,2.03,5.2,1.13,'Backward through the masked output:\n∂L/∂w contains 0 × NaN = NaN',red)
box(ax,6.08,2.03,5.6,1.13,'Observed numerical weight gradient:\n2 affected columns × 128 channels = 256',red)
arrow(ax,(9.8,3.65),(9.8,3.28),red)
ax.text(.2,1.38,'Independent checks',color=navy,weight='bold');ax.text(.2,.98,'NumPy arithmetic and the original LinearEncoder both reproduce 256 nonfinite entries.',color=muted)
ax.text(.2,.55,'Early input imputation gives zero in a diagnostic control. Repaired full-model training remains NOT_RUN.',color=muted,fontsize=10.5)
ax.text(.2,.1,'Local CPU Torch 2.13.0 + PyTorch Frame 0.2.3. No historical CUDA or whole-model failure claim.',color=muted,fontsize=10)
save(fig,'gradient')
fig,ax=plt.subplots(figsize=(12,5.5));ax.set(xlim=(0,12),ylim=(0,5.5));ax.axis('off')
ax.text(.1,5.02,'Two evidence lanes; keep their conclusions separate',fontsize=20,weight='bold',color=navy)
box(ax,.2,1.6,5.4,2.72,'COMPLETE · published-result replay\n\nRDBPFN / RDBPFN_single / TabICL\n512 supports · 10 seeds · 702 queries\n30 saved runs · 21,060 probabilities\n\nAuthenticates and rescores original evidence',teal)
box(ax,6.35,1.6,5.4,2.72,'INCOMPLETE · fresh matched comparison\n\nRDB-PFN / RelGNN / RDBLearn\n1,024 supports · 3 seeds planned\n0 model runs · training-health stop\n\nNo three-model winner can be reported',red)
ax.text(.25,.98,'Reused predictions are not independent replications. No fresh model inference or training was executed.',color=muted,fontsize=10.8)
ax.text(.25,.48,'New cloud/API spend: $0. Whole-paper reproduction: NOT_RUN. Learner: PENDING_WRITTEN_DEFENSE.',color=muted,fontsize=10.8)
save(fig,'evidence')
print('Built 3 portable model/evidence figures')
