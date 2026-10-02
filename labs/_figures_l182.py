"""Model-specific portable figures, with measured and proposed quantities separate."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
P=Path(__file__).resolve().parent;F=P/'figures/l182';F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l182'})
ink='#17324d';teal='#087f83';gold='#ad6416';red='#ac4940'
def box(ax,x,y,w,h,text,color=teal):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',facecolor='#f1f7f7',edgecolor=color,lw=1.4));ax.text(x+w/2,y+h/2,text,ha='center',va='center',color=ink,fontsize=10.5)
def arrow(ax,a,b,color=teal):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.7,color=color))
def save(fig,name):
 fig.savefig(F/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});fig.savefig(F/(name+'.png'),bbox_inches='tight',dpi=145,metadata={'Software':'L182'});plt.close(fig)
fig,ax=plt.subplots(figsize=(12,8));ax.set(xlim=(0,12),ylim=(0,8));ax.axis('off')
ax.text(.1,7.65,'RDB-PFN: generation is upstream of the predictor',fontsize=18,color=ink,weight='bold')
box(ax,.2,5.9,2.1,1,'Schema + FK graph\n+ synthetic content')
box(ax,3,5.9,2,1,'DFS linearization\ncounts, means, …')
box(ax,5.7,5.9,2.5,1,'Labeled support/query\n→ Transformer')
box(ax,9,5.9,2.6,1,'Query-label loss\nupdates predictor')
for a,b in [((2.35,6.4),(2.95,6.4)),((5.05,6.4),(5.65,6.4)),((8.25,6.4),(8.95,6.4))]:arrow(ax,a,b)
ax.text(.25,5.28,'GENERATOR INSERTION: change the task distribution → retrain the existing predictor',color=gold,fontsize=11)
box(ax,.2,3.6,2.1,1,'Released DFS input\n512 support rows\n702 query rows\nF numeric features')
box(ax,3,3.6,2.0,1,'Support-only scaling\nscalar → width 96\n+ label token')
box(ax,5.7,3.35,2.5,1.5,'6 attention blocks\n1 · features within row\n2 · rows read support\n3 · MLP + residuals')
box(ax,9,3.6,2.6,1,'Query label token\n96 → 192 → 2 logits\n702 probabilities')
for a,b in [((2.35,4.1),(2.95,4.1)),((5.05,4.1),(5.65,4.1)),((8.25,4.1),(8.95,4.1))]:arrow(ax,a,b)
ax.text(.25,2.92,'PUBLISHED EVALUATION: frozen weights · no target-task gradients · complete test population',color=teal,fontsize=11)
box(ax,.2,.7,2.1,1.1,'Relational graph\nlegal neighborhoods\n+ support labels',gold)
box(ax,3,.7,2.0,1.1,'Graph encoder\nordinary two-hop OR\ncomposite messages',gold)
box(ax,5.7,.7,2.5,1.1,'Shared ICL head\nnewly pretrained\nmatched task loss',gold)
box(ax,9,.7,2.6,1.1,'Held-out schema\nquery probabilities\nNOT_RUN',gold)
for a,b in [((2.35,1.25),(2.95,1.25)),((5.05,1.25),(5.65,1.25)),((8.25,1.25),(8.95,1.25))]:arrow(ax,a,b,gold)
ax.text(.25,.14,'PREDICTOR INSERTION: new interface + cross-schema weight sharing; released weights cannot be assumed compatible',color=gold,fontsize=10.5)
save(fig,'architecture')
fig,ax=plt.subplots(figsize=(11,5.8));ax.set(xlim=(0,11),ylim=(0,5.8));ax.axis('off')
ax.text(.1,5.38,'Follow the product → purchase → customer route',fontsize=17,color=ink,weight='bold')
box(ax,.2,3.7,2.2,.9,'Product [1, 0]\nPurchase [0, 1]');box(ax,.2,2.1,2.2,.9,'Product [0, 1]\nPurchase [2, 0]')
box(ax,3.1,3.7,1.9,.9,'Fused [1, 1]\nscore 0.7071');box(ax,3.1,2.1,1.9,.9,'Fused [2, 1]\nscore 1.4142')
box(ax,5.8,2.65,1.9,1.45,'Softmax\nweights\n0.3302 / 0.6698')
box(ax,8.5,2.65,2.15,1.45,'Customer query [1, 0]\nweighted output\n[1.6698, 1.0000]')
for a,b in [((2.45,4.15),(3.05,4.15)),((2.45,2.55),(3.05,2.55)),((5.05,4.15),(5.75,3.7)),((5.05,2.55),(5.75,3.0)),((7.75,3.35),(8.45,3.35))]:arrow(ax,a,b)
box(ax,.2,.45,4.8,.85,'Purchase at time 11 → excluded before fusion\nCustomer cutoff = 10; retain event time < 10',red)
ax.text(5.8,.8,'One head · identity projections\nZero destination skip · synthetic arithmetic',color=ink,va='center')
save(fig,'trace')
r=json.loads((P/'evidence/l182/report.json').read_text());fig,axes=plt.subplots(1,2,figsize=(12,4.8))
labels=['Relational PFN','Single-table PFN','TabICL v1.1'];arms=['RDBPFN','RDBPFN_single','TabICLv1.1']
for i,(arm,color) in enumerate(zip(arms,[teal,gold,'#536cad'])):
 v=r['models'][arm]['per_seed'];axes[0].scatter(np.linspace(i-.12,i+.12,10),v,s=27,color=color,alpha=.8)
 axes[0].errorbar(i,r['models'][arm]['mean'],yerr=r['models'][arm]['sample_sd'],fmt='D',color=ink,capsize=7)
axes[0].set_xticks(range(3),labels,fontsize=10);axes[0].set(ylabel='Test AUROC',ylim=(.58,.78),title='Fresh published comparison: 30 evaluations');axes[0].grid(axis='y',alpha=.2)
v=r['paired']['TabICLv1.1']['per_seed'];axes[1].bar(range(10),v,color=[teal if x>0 else red for x in v]);axes[1].axhline(0,color=ink,lw=1);axes[1].set(xticks=range(10),xlabel='Support seed',ylabel='RDB-PFN minus TabICL AUROC',title='Paired differences: 6 positive, 4 negative')
fig.text(.5,.01,'Same 702 test queries throughout · diamonds show mean ± sample SD over support draws · no hybrid arm',ha='center',fontsize=10,color=ink);fig.tight_layout(rect=[0,.055,1,1]);save(fig,'results')
fig,ax=plt.subplots(figsize=(10,5.6));ax.set(xlim=(0,10),ylim=(0,5.6));ax.axis('off')
ax.text(.1,5.15,'A better combined model can have a negative interaction',fontsize=16,color=ink,weight='bold')
ax.text(3.8,4.5,'Conventional encoder',ha='center',color=ink);ax.text(7.55,4.5,'Composite encoder',ha='center',color=ink)
ax.text(.1,3.5,'Single-table\nprior',va='center',color=ink);ax.text(.1,2,'Relational\nprior',va='center',color=ink)
for x,y,t in [(2.2,3.0,'A = 0.60'),(6,3.0,'B = 0.65'),(2.2,1.5,'C = 0.64'),(6,1.5,'D = 0.66')]:box(ax,x,y,3.1,.95,t,gold)
ax.text(.15,.75,'Composite benefit: +0.05 with single-table prior, +0.02 with relational prior',color=ink,fontsize=12)
ax.text(.15,.25,'Interaction = +0.02 − +0.05 = −0.03. Fabricated AUROC values; all four proposed arms NOT_RUN.',color=red,fontsize=11)
save(fig,'factorial')
print('4 SVG + 4 portable PNG figures')
