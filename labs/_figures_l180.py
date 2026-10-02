"""Portable RT-specific model, temporal witness and full-run cost figures."""
import json
from datetime import datetime,timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;F=P/'figures/l180';F.mkdir(exist_ok=True,parents=True)
r=json.loads((P/'evidence/l180/report.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l180'})
navy='#17324d';teal='#087f83';red='#a93d36';gold='#a66412'
def save(fig,name):
 fig.savefig(F/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});fig.savefig(F/(name+'.png'),dpi=145,bbox_inches='tight',metadata={'Software':'L180'});plt.close(fig)
def box(ax,x,y,w,h,label,color=teal):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',facecolor='#f2f6f6',edgecolor=color));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10.5,color=navy)
def arrow(ax,a,b,color=teal):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=color,lw=1.7))
fig,ax=plt.subplots(figsize=(13,8));ax.set(xlim=(0,13),ylim=(0,8));ax.axis('off')
ax.text(.1,7.6,'RT-v1: a public relational encoder, a supervised target',fontsize=19,weight='bold',color=navy)
ax.text(.1,7.12,'Archived architecture and planned training path. No L180 checkpoint loading or model execution.',color=red)
box(ax,.1,5.25,3.55,1.3,'F1 database + task table\n(driverId, cutoff) → DNF\nQuery label masked; dated FK context')
box(ax,4.35,5.25,3.6,1.3,'Native BFS, width 256\n1,024 cells per query\nValue + column/table semantics')
box(ax,8.65,5.25,4.1,1.3,'Frozen 384-wide MiniLM vectors\n+ trainable datatype encoders\n+ learned masked-cell vectors')
arrow(ax,(3.73,5.9),(4.25,5.9));arrow(ax,(8.03,5.9),(8.55,5.9));ax.plot([10.7,10.7,1.55],[5.16,4.7,4.7],color=teal,lw=1.7);arrow(ax,(1.55,4.7),(1.55,4.43))
ax.text(.15,2.65,'Public weights initialize the same 12 relational blocks · [B, 1,024, 256] · 8 attention heads',color=navy,weight='bold')
for i,(title,desc) in enumerate([('Column','same column'),('Feature','same row / F→P'),('Neighbor','P→F relations'),('Full','all unpadded cells')]):
 x=.15+i*3.2;box(ax,x,3.4,2.8,.95,title+' attention\n'+desc)
 if i<3:arrow(ax,(x+2.9,3.88),(x+3.1,3.88))
ax.text(.2,2.99,'Each attention: RMSNorm → attention → residual. Then RMSNorm → gated FFN (width 1,024) → residual.',fontsize=10.5,color=navy)
arrow(ax,(11.15,3.32),(11.15,2.5));box(ax,8.65,1.3,4.1,1.12,'Boolean decoder → query logit\nBCE on masked targets\nInference: sigmoid → DNF probability')
box(ax,4.35,1.3,3.6,1.12,'Supervised gradient updates\nTrainable RT parameters\nAdamW · LR 0.0001 · WD 0',gold)
box(ax,.1,1.3,3.55,1.12,'Validation AUROC selects weights\nComplete keyed test evaluation\nFresh fit + independent audit',gold)
arrow(ax,(8.57,1.86),(8.04,1.86),gold);arrow(ax,(4.27,1.86),(3.74,1.86),gold)
ax.text(.1,.53,'Training gates: temporal legality · checkpoint identity · finite gradients · complete evaluation · total cost',color=red,fontsize=11)
ax.text(.1,.08,'B = 32 per GPU in the released example; eight workers give global batch 256. All arrows describe the model, not an executed run.',fontsize=9.8,color=navy)
save(fig,'architecture')
w=r['witness'];date=lambda t:datetime.fromtimestamp(t,timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
fig,ax=plt.subplots(figsize=(12,5.6));ax.set(xlim=(0,12),ylim=(0,5.6));ax.axis('off')
ax.text(.1,5.12,'One real context cell explains the temporal stop',fontsize=20,weight='bold',color=navy)
ax.text(.1,4.62,f"Seed {w['seed']} · driver {w['driver_id']} · {w['table']} / {w['column']}",fontsize=11,color=navy)
box(ax,.15,2.75,4.5,1.25,'Query cutoff\n'+date(w['query_cutoff'])+'\nTarget is masked')
box(ax,7.05,2.75,4.5,1.25,'Sampled cell event date\n'+date(w['context_time'])+'\nLater than this query cutoff',red)
arrow(ax,(4.8,3.37),(6.9,3.37),red)
ax.text(6,2.14,'385 future-dated cells in 77 of 2,106 contexts',ha='center',fontsize=16,weight='bold',color=red)
ax.text(6,1.5,'All belong to race schedules. Future event date does not establish when a schedule was published.',ha='center',fontsize=11,color=navy)
ax.text(6,.99,'The declared event-time contract fails; outcome leakage is not established by this witness.',ha='center',fontsize=11,color=navy)
ax.text(6,.38,'Replay also finds 0 exposed query targets, 0 unfinished-window labels and 432,050 unknown-time cell slots.',ha='center',fontsize=10.5,color=navy)
save(fig,'temporal')
fig,ax=plt.subplots(figsize=(10.5,4.7));costs=[float(x) for x in r['costs_gpu_only_usd'].values()]
ax.barh(['A100 40 GB','A100 80 GB'],costs,color=[teal,navy],height=.48)
ax.axvline(10,color=red,lw=2,linestyle='--',label='Entire lesson ceiling: $10')
for i,c in enumerate(costs):ax.text(c+.3,i,f'${c:.2f}',va='center',fontsize=12)
ax.set_xlim(0,35);ax.set_xlabel('USD, GPU-only scenario for one reported full fine-tuning run');ax.invert_yaxis()
ax.set_title('8 GPUs × 1.5 hours × per-GPU rate',loc='left',fontsize=18,pad=18);ax.legend(loc='lower right');ax.spines[['top','right']].set_visible(False)
fig.text(.1,.015,'Excludes CPU, RAM, setup, evaluation overhead and retries. Not a pilot measurement or an invoice.',fontsize=10,color=navy);fig.tight_layout(rect=(0,.07,1,1));save(fig,'cost')
print('Built three RT checkpoint figures')
