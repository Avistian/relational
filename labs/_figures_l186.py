"""Original portable serving diagrams and complete-grid plots."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;F=P/'figures/l186';F.mkdir(exist_ok=True)
r=json.loads((P/'evidence/l186/report.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l186','axes.spines.top':False,'axes.spines.right':False})
navy='#17324d';teal='#087f83';gold='#b87718';red='#b35d48';muted='#536575'
def save(fig,name):
 fig.savefig(F/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});fig.savefig(F/(name+'.png'),dpi=140,bbox_inches='tight',metadata={'Software':'L186'});plt.close(fig)
def box(ax,x,y,w,h,text,color=teal):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor='#f0f6f6',edgecolor=color,lw=1.4));ax.text(x+w/2,y+h/2,text,ha='center',va='center',color=navy,fontsize=11)
def arrow(ax,a,b,color=teal):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=color,lw=1.7))
fig,ax=plt.subplots(figsize=(12,7));ax.set(xlim=(0,12),ylim=(0,7));ax.axis('off')
ax.text(.1,6.65,'A relational request carries several clocks',fontsize=22,color=navy,weight='bold')
ax.text(.1,6.22,'One customer → orders + payments → legal snapshot → serving policy → response',color=muted)
box(ax,.15,4.6,2.1,1,'Orders\nevent → arrival');box(ax,.15,2.9,2.1,1,'Payments\nevent → arrival',gold)
box(ax,3,3.55,2.15,1.05,'Observed state\nby customer ID')
arrow(ax,(2.3,5.1),(2.94,4.3));arrow(ax,(2.3,3.4),(2.94,3.85),gold)
for y,title,service,refresh in [(4.9,'Precomputed',1,'5,000 ms'),(3.25,'Cached features',5,'1,000 ms'),(1.6,'Request-time',15,'read at start')]:
 box(ax,6,y,2.5,1.15,f'{title}\nrefresh: {refresh}\nservice: {service} ms')
 arrow(ax,(5.23,4.05),(5.93,y+.55))
 arrow(ax,(8.58,y+.55),(9.15,3.84))
box(ax,9.2,3.25,2.5,1.15,'Response monitor\nlatency + source age\nlabels arrive later')
ax.text(.15,1.75,'Late payments can make\nthe joined feature stale\neven if orders are current.',color=gold,fontsize=12)
ax.text(.15,.75,'Frozen course assumptions: one FIFO worker per policy; independent background refresh.',color=muted)
ax.text(.15,.32,'These times are hypothetical. No model executes here; real network / refresh contention is unmeasured.',color=muted,fontsize=10.5)
save(fig,'serving')
fig,axes=plt.subplots(1,2,figsize=(12,5.4));fig.suptitle('Freshness and latency answer different questions',fontsize=20,color=navy,x=.07,ha='left')
colors={'precomputed':gold,'cached':teal,'request':navy}
for policy,color in colors.items():
 for seed in [0,1,2]:
  rows=[x for x in r['results'] if x['policy']==policy and x['condition']=='normal' and x['seed']==seed]
  rows.sort(key=lambda x:x['rate']);rates=[x['rate'] for x in rows]
  axes[0].plot(rates,[x['p99_ms'] for x in rows],marker='o',color=color,alpha=1 if seed==0 else .35,label=policy if seed==0 else None)
  axes[1].plot(rates,[x['stale_responses']/100 for x in rows],marker='o',color=color,alpha=1 if seed==0 else .35)
axes[0].set_yscale('log');axes[0].set_ylabel('p99 request latency (ms; log scale)');axes[0].axhline(100,ls='--',color=red,label='100 ms deadline');axes[0].legend(fontsize=9)
axes[1].set_ylabel('Responses with source age > 2,000 ms (%)');axes[1].set_ylim(-3,100)
for ax in axes:ax.set_xlabel('Offered requests / second');ax.set_xticks([10,50,100]);ax.grid(alpha=.15)
fig.text(.07,.03,'All three seeds shown; normal arrivals. Fresh source snapshots do not undo time already spent waiting in a queue.',color=muted,fontsize=10)
fig.subplots_adjust(top=.83,bottom=.2,wspace=.33);save(fig,'tradeoffs')
fig,ax=plt.subplots(figsize=(12,5.6));ax.axis('off');ax.set(xlim=(0,12),ylim=(0,5.6))
w=r['witness'][0]
ax.text(.1,5.25,'A 1 ms response can carry a 4.827 s old dependency',fontsize=21,color=navy,weight='bold')
ax.text(.1,4.78,'Actual simulated witness · seed 0 · 50 requests/s · interrupted payments · customer 6',color=muted)
labels=[('64,501','Orders snapshot',gold),('64,572','Payments snapshot',gold),('65,000','Refresh reads',teal),('65,020','Refresh ready',teal),('69,327','Request arrives',navy),('69,328','Response sent',navy)]
for i,(time,label,color) in enumerate(labels):
 x=.3+i*1.97;ax.plot([x,x],[2.55,3.05],color=color,lw=2);ax.scatter([x],[2.8],s=50,color=color)
 ax.text(x,3.45,time+' ms',ha='center',color=color,weight='bold',fontsize=11)
 ax.text(x,2.05,label.replace(' ','\n',1),ha='center',color=navy,fontsize=10)
ax.plot([.3,10.15],[2.8,2.8],color='#a8b9c7',lw=1,zorder=0)
ax.text(.1,1.1,'Response latency = 69,328 − 69,327 = 1 ms',color=navy,fontsize=13)
ax.text(6.1,1.1,'Source age = 69,328 − 64,501 = 4,827 ms',color=gold,fontsize=13)
ax.text(.1,.48,'Schematic spacing; clocks are milliseconds. Refresh age is 4,308 ms — a third, different measurement.',color=muted,fontsize=11)
save(fig,'clocks')
print('Three portable figures built')
