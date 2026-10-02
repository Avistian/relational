"""Portable original figures; measured clocks and scope-specific compute routes."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;F=P/'figures/l177';F.mkdir(parents=True,exist_ok=True)
r=json.loads((P/'evidence/l177/report.json').read_text());v=r['l176']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l177','axes.spines.top':False,'axes.spines.right':False})
navy='#17324d';teal='#087f83';gold='#b87718';muted='#536575'
def save(fig,name):
 fig.savefig(F/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});fig.savefig(F/(name+'.png'),dpi=150,bbox_inches='tight',metadata={'Software':'L177'});plt.close(fig)
fig,ax=plt.subplots(figsize=(12,6.7));ax.set(xlim=(0,12),ylim=(0,6.7));ax.axis('off')
ax.text(.1,6.35,'A compute budget buys a complete route',fontsize=21,color=navy,weight='bold')
ax.text(.1,5.91,'Place the data preparation and evidence checks around the model operation.',color=muted)
rows=[('L173 · CPU course pretraining',4.55,['Dated F1 cells\nmask the target','Small shared MLP\nupdate all weights','6 complete fits\n3 epochs each'],'Build a reusable initialization; foundation-model transfer not established.'),('L174 · CPU course adaptation',2.8,['Selected L173 weights\n2006 target data','Freeze / full / adapter\nor scratch control','12 complete fits\n10 epochs each'],'Measure adaptation on the same tasks; no unseen-database claim.'),('L176 · GPU checkpoint inference',1.05,['Prepared DFS features\n+ labeled support','Frozen RDB-PFN\nor TabICL predictor','300 complete evaluations\n2 tasks × 3 models × 5 k × 10 seeds'],'No weight updates; feature regeneration and original pretraining excluded.')]
for title,y,boxes,caption in rows:
 ax.text(.1,y+.92,title,color=navy,weight='bold',fontsize=12)
 for i,text in enumerate(boxes):
  x=.1+4*i
  ax.add_patch(FancyBboxPatch((x,y),3.55,.7,boxstyle='round,pad=.08',facecolor='#edf5f5',edgecolor='#aec6ca'))
  ax.text(x+1.775,y+.35,text,ha='center',va='center',fontsize=10.5,color=navy)
  if i<2:ax.annotate('',xy=(x+3.94,y+.35),xytext=(x+3.65,y+.35),arrowprops={'arrowstyle':'->','color':teal,'lw':2})
 ax.text(.1,y-.35,caption,fontsize=10.5,color=muted)
ax.text(.1,.03,'Each route: preparation → computation → verification → written defense. Different workloads are not a speed ranking.',fontsize=10,color=muted)
save(fig,'routes')
fig,(ax,bx)=plt.subplots(2,1,figsize=(11,6.7),gridspec_kw={'height_ratios':[1,1.25]})
fig.suptitle('One worker clock; three money questions',x=.07,ha='left',fontsize=21,color=navy)
parts=[v['evaluation_seconds'],v['distinct_load_seconds'],v['worker_seconds']-v['evaluation_seconds']-v['distinct_load_seconds']]
left=0
for val,c in zip(parts,[teal,gold,'#a7b9c7']):ax.barh([0],val,left=left,color=c,height=.38);left+=val
ax.set_xlim(0,520);ax.set_yticks([]);ax.set_xlabel('Seconds inside the worker body (491.471 s total)')
ax.text(0,.55,'300 evaluations: 474.434 s',color=teal,weight='bold')
fig.text(.07,.46,'Distinct loads: 0.939 s  ·  Residual worker work: 16.098 s  ·  Cloud startup not measured here',fontsize=10,color=muted)
ax.set_ylim(-.4,.5)
bx.axis('off')
for x,title,value,caption,c in [(.01,'WORKER ESTIMATE','$0.139440','491.471 s × resource rate\nExcludes unitemized costs',teal),(.36,'PLANNING RESERVATION','$5.230039','Timeouts + lifecycle allowances\n+ $3 overhead reserve',navy),(.71,'PROVIDER INVOICE','NOT ITEMIZED','No exact billed total\ncan be claimed',gold)]:
 bx.text(x,.87,title,transform=bx.transAxes,fontsize=10,color=c,weight='bold');bx.text(x,.55,value,transform=bx.transAxes,fontsize=21,color=c);bx.text(x,.25,caption,transform=bx.transAxes,fontsize=10,color=muted)
fig.subplots_adjust(hspace=.65,top=.83,bottom=.1,left=.07,right=.98);save(fig,'accounting')
fig,axes=plt.subplots(1,2,figsize=(12,5.8));fig.suptitle('Context changes time and memory, even with frozen weights',x=.07,ha='left',fontsize=18,color=navy)
colors={'RDBPFN':teal,'RDBPFN_single':navy,'TabICLv1.1':gold}
for db,style in [('rel-f1','-'),('rel-trial','--')]:
 for arm,c in colors.items():
  rows=[x for x in r['timing_by_task_model_context'] if x['database']==db and x['arm']==arm]
  for ax,key in zip(axes,['mean_seconds','max_allocator_gib']):ax.plot([x['context'] for x in rows],[x[key] for x in rows],style,marker='o',color=c,label=arm+' · '+db,markersize=4)
for ax in axes:
 ax.set_xscale('log',base=2);ax.set_xticks([64,128,256,512,1024],labels=['64','128','256','512','1024']);ax.set_xlabel('Labeled context rows');ax.set_ylim(bottom=0);ax.grid(alpha=.15)
axes[0].set_ylabel('Mean timed evaluation (seconds)');axes[1].set_ylabel('Maximum allocated tensors (GiB)')
axes[0].set_title('Ten runs per point; source pipeline timer',fontsize=11);axes[1].set_title('Across ten runs; not total device usage',fontsize=11)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=9)
fig.subplots_adjust(top=.79,bottom=.24,wspace=.27,left=.07,right=.98);save(fig,'context')
print('Built 3 portable figures')
