"""Portable operation-specific diagrams; figures derived from declared computations."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
P=Path(__file__).resolve().parent;O=P/'figures/l108';O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none','svg.hashsalt':'l108','axes.spines.top':False,'axes.spines.right':False})
TEAL='#087e82';ORANGE='#b65b2c';INK='#233b45';GRAY='#667879'
def canvas(title,sub,h=4.5):
 f,ax=plt.subplots(figsize=(10,h));f.patch.set_facecolor('#fbfcf9');ax.set_xlim(0,10);ax.set_ylim(0,h);ax.axis('off');ax.text(.15,h-.35,title,fontsize=18,weight='bold',color=INK);ax.text(.15,h-.72,sub,fontsize=11,color=GRAY);return f,ax
def box(ax,x,y,w,h,text,color=TEAL):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.07',facecolor='white',edgecolor=color,lw=1.5));ax.text(x+.12,y+h-.13,text,va='top',fontsize=11,color=INK,linespacing=1.7)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':TEAL,'lw':2})
def save(f,name):
 f.savefig(O/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});f.savefig(O/(name+'.png'),dpi=150,bbox_inches='tight',metadata={'Software':'L108'});plt.close(f)
f,ax=canvas('Two searches define one legal slice','Synthetic example · cutoff 8, width 5 · include 3, exclude 8')
for i,t in enumerate([1,3,3,8,11]):
 x=.5+i*1.85;color=TEAL if i in [1,2] else GRAY
 box(ax,x,2,1.3,.8,f'e{i+1}  |  t={t}',color);ax.text(x+.65,1.65,f'index {i}',ha='center',fontsize=10,color=GRAY)
ax.annotate('start = 1',xy=(2.35,2.95),xytext=(2.35,3.4),arrowprops={'arrowstyle':'->','color':TEAL},ha='center',color=TEAL)
ax.annotate('end = 3',xy=(6.05,2.95),xytext=(6.05,3.4),arrowprops={'arrowstyle':'->','color':ORANGE},ha='center',color=ORANGE)
box(ax,.5,.3,8.7,.65,'times[1:3] → [3, 3]      recent k=2 → [e2, e3]      no event at cutoff')
save(f,'interval')
f,ax=canvas('One index, many different historical questions','Course sampler → frozen TGAT → candidate probabilities',h=6)
box(ax,.3,3.75,3.4,1.1,'Flat records: node | event | time\nOffsets delimit each node’s row\nSort once by (time, event ID)')
box(ax,5.2,3.75,4.25,1.1,'B queries: node[B], cutoff[B]\nTwo binary searches per query\nEligible [start, end) positions')
arrow(ax,(3.8,4.25),(5.1,4.25))
box(ax,5.2,1.75,4.25,1.2,'Policy + fanout k → B × k slots\nGather node IDs, event IDs, times\nPadding has ID 0; features via IDs')
arrow(ax,(7.3,3.65),(7.3,3.05))
box(ax,.3,1.75,3.4,1.2,'Recursive TGAT + time encoding\nMasked attention and root merge\nFrozen weights; no fitting here')
arrow(ax,(5.1,2.35),(3.8,2.35))
box(ax,.3,.25,9.15,.65,'Shared pair head → p(positive), p(negative) → paired AP + synchronized timing')
arrow(ax,(2,1.65),(2,.98));save(f,'pipeline')
f,ax=canvas('The edge carries the next cutoff','Synthetic two-hop trace · do not reuse the root’s time',h=4.8)
box(ax,.4,2.5,2.1,.8,'Root query\nA at cutoff 8')
box(ax,3.9,2.5,2.1,.8,'Child query\nB at cutoff 3')
box(ax,7.4,2.5,2.1,.8,'Candidate edge\nB–C at time 6',ORANGE)
arrow(ax,(2.6,2.9),(3.8,2.9));ax.text(3.2,3.6,'A–B at 3',ha='center',color=TEAL)
arrow(ax,(6.1,2.9),(7.3,2.9));ax.text(6.8,3.6,'6 < 3? No',ha='center',color=ORANGE)
box(ax,.4,.45,9.1,1.1,'B at 3 and B at 8 are different queries. A node-only cache can mix their histories.\nNeighbor-tree slots at depth 2: k=20 → 421; k=5 → 31.\nThese are slot bounds; TGAT also recursively computes the source branch.')
save(f,'recursion')
path=P/'_analysis_l108_results.json'
if path.exists():
 r=json.loads(path.read_text());f,axes=plt.subplots(1,2,figsize=(10,4.5),sharex=True,sharey=True);f.patch.set_facecolor('#fbfcf9')
 for ax,lane,title in zip(axes,['all','new'],['All-event questions','New-node questions']):
  for arm,color in zip(['uniform20','uniform5','recent20','day20'],[TEAL,ORANGE,'#5369ad','#96713c']):
   row=r['summary'][arm][lane];x=row['questions_per_second'];y=row['ap']
   ax.scatter(x['values'],y['values'],s=16,alpha=.35,color=color)
   ax.errorbar(x['mean'],y['mean'],xerr=x['sd'],yerr=y['sd'],fmt='o',capsize=3,color=color,label=arm)
  ax.set_title(title,fontsize=13);ax.set_xlabel('Positive questions / second');ax.set_ylabel('Pooled AP (%)');ax.grid(alpha=.15)
 axes[0].legend(fontsize=9);f.suptitle('Measured frozen-checkpoint sampling interventions',fontsize=16)
 f.text(.5,.01,'10 seeds · mean ± sample SD · same questions and negatives · one inference timing pass per seed',ha='center',fontsize=9);f.tight_layout(rect=(0,.05,1,.94));save(f,'results')
print('Figures generated')
