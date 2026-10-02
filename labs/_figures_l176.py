"""Model-specific portable support/attention flow and measured nested-context curves."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle
P=Path(__file__).resolve().parent;D=P/'figures/l176';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l176'})
blue='#2563eb';teal='#0d9488';ink='#172554';red='#b91c1c'
def save(fig,name):
 fig.savefig(D/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});fig.savefig(D/(name+'.png'),dpi=160,bbox_inches='tight',metadata={'Software':'L176'});plt.close(fig)
def box(ax,x,y,w,h,title,body,col=blue):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.006',facecolor='#f8fafc',edgecolor=col,lw=1.5))
 ax.text(x+.016,y+h-.035,title,color=col,weight='bold',va='top',fontsize=11.5)
 ax.text(x+.016,y+h-.10,body,va='top',fontsize=10.4,linespacing=1.65)
fig,ax=plt.subplots(figsize=(12.5,7.8));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(.015,.98,'RDB-PFN: labels enter through context, not gradient updates',fontsize=17,weight='bold',color=ink,va='top')
box(ax,.02,.66,.26,.24,'1  Shared DFS inputs','DFS support X: [k, F]\nKnown support y: [k]\nQuery X: [Q, F]')
box(ax,.345,.66,.29,.24,'2  Encode features + labels','Support statistics → normalize\nScalar values → width 96\nQuery y placeholder: mean(y)',teal)
box(ax,.70,.66,.28,.24,'3  Six frozen blocks','Feature attention within rows\nRow attention to support only\nGELU feed-forward updates')
for x1,x2 in [(.28,.345),(.634,.70)]:ax.annotate('',(x2,.78),(x1,.78),arrowprops=dict(arrowstyle='->',color=blue,lw=2))
ax.text(.03,.575,'Inside row attention: which rows supply keys and values?',fontsize=13,weight='bold',color=ink)
# A compact explicit attention access matrix.
x0,y0=.075,.26;cw=.075;ch=.055
for row in range(4):
 for col in range(4):
  ax.add_patch(Rectangle((x0+col*cw,y0+(3-row)*ch),cw-.003,ch-.003,facecolor=blue if col<2 else '#e2e8f0'))
for i,name in enumerate(['S1','S2','Q1','Q2']):
 ax.text(x0+(i+.5)*cw,y0+4*ch+.015,name,ha='center',fontsize=10)
 ax.text(x0-.012,y0+(3-i+.5)*ch,name,ha='right',va='center',fontsize=10)
ax.text(.065,.19,'Blue: allowed. Gray: excluded.\nS = support; Q = query. Rows issue queries.',fontsize=10,color=ink,linespacing=1.6)
box(ax,.45,.27,.25,.23,'4  Query label token','Queries read support rows\nNo query-to-query row links\nFinal target-column state',teal)
box(ax,.76,.27,.22,.23,'5  Decode','96 → 192 → 2 logits\nWrapper → probability\nScore all Q queries')
ax.plot([.84,.84,.575],[.65,.535,.535],color=blue,lw=1.7)
ax.annotate('',(.575,.50),(.575,.535),arrowprops=dict(arrowstyle='->',color=blue,lw=1.7))
ax.annotate('',(.76,.385),(.70,.385),arrowprops=dict(arrowstyle='->',color=blue,lw=2))
ax.text(.03,.105,'F = 72 (F1) or 176 (trial); Q = 702 or 825. Four attention heads; FFN width 192.',fontsize=11,color=ink)
ax.text(.03,.045,'Fixed weights. Changing k can also change imputation, normalization and target-mean placeholders.',fontsize=11,color=red,weight='bold')
save(fig,'architecture')
fig,ax=plt.subplots(figsize=(12,5));ax.set(xlim=(0,12),ylim=(0,5));ax.axis('off');ax.text(.1,4.7,'Nested support preserves the information already shown',fontsize=17,color=ink,weight='bold')
labels=[('k = 2',2,3.4),('k = 4',4,2.35),('k = 8',8,1.3)];ids=[7,2,9,4,8,1,6,3]
for label,n,y in labels:
 ax.text(.15,y+.3,label,weight='bold',fontsize=13)
 for i in range(n):
  col=blue if i<2 else teal
  ax.add_patch(FancyBboxPatch((1.5+i*1.05,y),.8,.65,boxstyle='round,pad=.025',facecolor=col,edgecolor=col))
  ax.text(1.9+i*1.05,y+.325,str(ids[i]),ha='center',va='center',color='white',weight='bold',fontsize=13)
 ax.text(10.2,y+.3,f'{min(2,n)}/2 retained',color=blue,fontsize=11)
ax.text(.2,.55,'Synthetic row IDs. One ordered draw; smaller supports are prefixes. Blue marks the fixed baseline.',fontsize=11,color=ink)
ax.text(.2,.10,'Same seed with separate draws is not a nesting guarantee. More examples do not guarantee a higher AUROC.',fontsize=11,color=red)
save(fig,'supports')
report=P/'evidence/l176/report.json'
if report.exists():
 r=json.loads(report.read_text());fig,axes=plt.subplots(1,2,figsize=(12,5.2),sharey=True)
 for ax,db in zip(axes,['rel-f1','rel-trial']):
  for arm,col in zip(['RDBPFN','RDBPFN_single','TabICLv1.1'],[blue,teal,'#9333ea']):
   ls=r['curves'][db+'/'+arm]['levels'];x=[a['context'] for a in ls];y=[a['mean'] for a in ls];sd=[a['sample_sd'] for a in ls]
   ax.plot(x,y,'o-',label=arm,color=col,lw=2);ax.fill_between(x,[a-b for a,b in zip(y,sd)],[a+b for a,b in zip(y,sd)],color=col,alpha=.10)
  ax.set_xscale('log',base=2);ax.set_xticks([64,128,256,512,1024],[64,128,256,512,1024]);ax.set_xlabel('Labeled support rows (nested prefixes)');ax.set_title(db,loc='left',weight='bold');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_ylim(.4,.90)
 axes[0].set_ylabel('AUROC');axes[1].legend(frameon=False,fontsize=10)
 fig.suptitle('Complete fresh nested-support experiment · 300 evaluations',x=.06,ha='left',fontsize=16,weight='bold');fig.tight_layout(rect=(0,.09,1,.91));fig.text(.08,.025,'Bands: ±1 sample SD over ten support draws on each fixed task. No test-based choice of k or model.',fontsize=10,color=ink);save(fig,'curves')
