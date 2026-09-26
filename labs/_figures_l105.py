"""Portable mechanism and measured-result figures, with text retained in SVG."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;O=P/'figures/l105';O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','svg.hashsalt':'l105'})
TEAL='#087e82';INK='#233d43';GOLD='#b56d25';MUTED='#5c7075';BG='#fbfcf9'
def save(fig,name):
 fig.savefig(O/f'{name}.svg',facecolor=BG,metadata={'Date':None});fig.savefig(O/f'{name}.png',dpi=160,facecolor=BG,metadata={'Software':'L105'});plt.close(fig)
def canvas(title,subtitle,h=4.5):
 fig=plt.figure(figsize=(10,h));fig.text(.055,.92,title,fontsize=18,color=INK,weight='bold');fig.text(.055,.855,subtitle,fontsize=10,color=MUTED);return fig
fig=canvas('Six events → five weighted edges','Worked fixture · 10-second half-open windows · time in seconds',4.6)
a=fig.add_axes([.10,.45,.84,.30]);a.set_xlim(0,20);a.set_ylim(0,2.3);a.set_yticks([]);a.set_xticks([0,5,10,15,20]);a.tick_params(colors=MUTED)
for b in [0,10]:a.axvspan(b,b+10,color=TEAL,alpha=.055 if b==0 else .11)
for t,label,y in [(1,'AX',1),(3,'AX',1),(3,'BY',1.8),(8,'AY',1),(12,'BX',1),(19,'AX',1)]:
 a.scatter([t],[y],s=75,color=TEAL,zorder=3);a.text(t,y+.22,label,ha='center',fontsize=11,color=INK)
a.axvline(10,color=GOLD,ls='--');a.set_xlabel('Event timestamp · the two records at 3 s are tied',fontsize=10,color=MUTED)
for sp in ['top','right','left']:a.spines[sp].set_visible(False)
fig.text(.10,.27,'[0, 10)  →  AX: 2     BY: 1     AY: 1',color=TEAL,fontsize=13,weight='bold')
fig.text(.10,.18,'[10, 20) →  BX: 1     AX: 1',color=TEAL,fontsize=13,weight='bold')
fig.text(.10,.065,'Binary: 5 edge records     Weighted: 6 total interactions     Exact ordering: lost',fontsize=11,color=INK)
save(fig,'aggregation')
fig=canvas('Same snapshot. Different temporal path.','Synthetic directed contacts · no waiting backward in time · both edges in [0, 10)',4.5)
ax=fig.add_axes([.03,.07,.94,.68]);ax.set_xlim(0,10);ax.set_ylim(0,4);ax.axis('off')
for y,t1,t2,label in [(3,1,2,'History I: A can reach C'),(1.5,2,1,'History II: A cannot reach C')]:
 for x,node in [(1,'A'),(4,'B'),(7,'C')]:
  ax.text(x,y,node,ha='center',va='center',fontsize=16,color=TEAL,bbox={'boxstyle':'circle,pad=.4','fc':'white','ec':TEAL})
 for x1,x2,t in [(1.4,3.6,t1),(4.4,6.6,t2)]:
  ax.annotate('',(x2,y),(x1,y),arrowprops={'arrowstyle':'->','color':TEAL,'lw':2});ax.text((x1+x2)/2,y+.35,f't = {t} s',ha='center',color=INK)
 ax.text(8,y,label.replace(': ',':\n'),fontsize=10,color=INK,va='center')
ax.text(.4,.25,'Both aggregate to AB: 1, BC: 1. No decoder can infer which order from those counts alone.',fontsize=10,color=INK)
save(fig,'order')
fig=canvas('Closed windows trade freshness for completeness','Worked fixture · query just before 6 s · arrival equals event time',4.6)
ax=fig.add_axes([.07,.18,.88,.58]);ax.set_xlim(0,20);ax.set_ylim(-.6,2.8);ax.set_yticks([0,1,2],['Completed current window','Released snapshots','Strict event history']);ax.tick_params(axis='y',labelsize=10);ax.set_xticks([0,3,6,10,20]);ax.axvline(6,color=GOLD,lw=2);ax.axvline(10,color=TEAL,ls='--');ax.text(6,2.7,'query 6',ha='center',color=GOLD);ax.text(10,2.7,'release 10',ha='center',color=TEAL)
for t,offset in [(1,0),(3,-.10),(3,.10)]:ax.scatter(t,2+offset,s=65,color=TEAL,alpha=.8)
ax.text(12,2,'3 eligible records',color=TEAL,va='center');ax.text(.8,1,'WAIT: window [0,10) is unfinished',color=MUTED,va='center')
for t,offset in [(1,0),(3,-.10),(3,.10),(8,0)]:ax.scatter(t,offset,s=65,color=GOLD if t==8 else TEAL,alpha=.8)
ax.text(12,0,'t = 8 is nonpast → illegal',color=GOLD,va='center')
for sp in ['top','right','left']:ax.spines[sp].set_visible(False)
fig.subplots_adjust(left=.2);ax.set_position([.27,.18,.67,.58]);fig.text(.07,.055,'At query 10, [0,10) is legal: its four events are strictly earlier. A late arrival can delay release.',fontsize=10,color=INK)
save(fig,'release')
r=json.loads((P/'_analysis_l105_results.json').read_text())['records']
fig,axes=plt.subplots(1,2,figsize=(10,4.7));fig.patch.set_facecolor(BG)
labels=['Hourly','Daily','Weekly'];vals=[x['snapshot_edges'] for x in r]
axes[0].bar(labels,vals,color=TEAL,width=.55);axes[0].axhline(157474,color=GOLD,ls='--');axes[0].set_ylim(0,185000);axes[0].text(-.35,163000,'157,474 original events',fontsize=10,color=GOLD)
for i,v in enumerate(vals):axes[0].text(i,v+4000,f'{v:,}',ha='center',fontsize=11,color=INK)
axes[0].set_title('Binary window edges',loc='left',fontsize=13,color=INK);axes[0].set_ylabel('Number of records');axes[0].ticklabel_format(axis='y',style='plain')
delay=[x['delay_mean_seconds']/3600 for x in r];axes[1].bar(labels,delay,color=TEAL,width=.55)
for i,v in enumerate(delay):axes[1].text(i,v+2,f'{v:.2f} h',ha='center',color=INK)
axes[1].set_ylim(0,110);axes[1].set_title('Mean wait for window close',loc='left',fontsize=13,color=INK);axes[1].set_ylabel('Hours per event')
for a in axes:
 a.set_facecolor(BG);a.spines[['top','right']].set_visible(False);a.tick_params(labelsize=10)
fig.suptitle('Full Wikipedia census · three predetermined widths',fontsize=16,color=INK,weight='bold',x=.06,ha='left');fig.text(.06,.035,'Origin = 0. Weighted counts sum to 157,474 in every condition. No model score or confidence interval.',fontsize=10,color=MUTED)
fig.subplots_adjust(left=.10,right=.97,top=.80,bottom=.19,wspace=.38);save(fig,'results')
print('Built four portable mechanism/results figures')
