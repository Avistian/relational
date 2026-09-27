"""Portable temporal decision trace and measured uncertainty; deterministic."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent/'figures/l123';P.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'l123','font.size':12})
INK='#19384a';GREEN='#dceee4';RED='#f4ddd9'
def save(f,name):
 f.tight_layout();f.savefig(P/(name+'.svg'),metadata={'Date':None});f.savefig(P/(name+'.png'),dpi=150,metadata={'Software':'L123'});plt.close(f)
f,a=plt.subplots(figsize=(10,6));a.axis('off');a.set(xlim=(0,10),ylim=(0,6));f.patch.set_facecolor('#faf9f6')
a.set_title('What can Person0 know at day 8?',loc='left',fontsize=18,weight='bold',color=INK)
def box(x,y,text,color=GREEN,w=2.7):
 a.add_patch(FancyBboxPatch((x,y),w,1.05,boxstyle='round,pad=.08',fc=color,ec=INK));a.text(x+w/2,y+.525,text,ha='center',va='center',color=INK,fontsize=12)
def arrow(x,y,xx,yy):a.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':INK,'lw':2})
box(.15,3.9,'Memo1 · day 7 / 7\n7 ≤ 8: INCLUDE');box(3.55,3.9,'Transfer0 · day 4 / 4\n4 ≤ 8: INCLUDE');box(7,3.9,'Person0\nquery t = 8');arrow(2.98,4.43,3.4,4.43);arrow(6.4,4.43,6.85,4.43)
a.text(.15,3.32,'Message path: Memo1 → Transfer0 → Person0',color=INK,weight='bold')
box(.15,1.75,'Memo0 · 12 / 12\nFuture at hop 2',RED);box(3.55,1.75,'Transfer1 · 9 / 9\nFuture at hop 1',RED);box(7,1.75,'Transfer2 · 5 / 11\nArrives after query',RED)
a.text(.15,.9,'Each pair is event day / available day. Both must be ≤ root day 8.',color=INK)
a.text(.15,.3,'Result: 3 nodes, 2 edges. Parent day 4 does not replace query day 8.',color=INK,weight='bold');save(f,'cutoff')
for ext in ['svg','png']:(P/('paper.'+ext)).write_bytes((P.parent/'l117'/('architecture.'+ext)).read_bytes())
s=json.loads((P.parents[1]/'evidence/l123/summary.json').read_text())
f,axs=plt.subplots(1,2,figsize=(10,4.5));f.patch.set_facecolor('#faf9f6')
for ax,split,label in zip(axs,['val','test'],['Validation','Test']):
 m=s['metrics'][split];values=[r[split] for r in s['seeds']]
 ax.scatter(range(5),values,s=60,color='#176b79',label='Fresh seed')
 ax.axhline(m['target'],color='#a04c3e',ls='--',label='Published mean')
 ax.errorbar(5.4,m['mean'],yerr=m['sample_sd'],fmt='D',color='#19384a',capsize=6,label='Mean ± sample SD')
 ax.set(xticks=list(range(5))+[5.4],xticklabels=['0','1','2','3','4','Mean'],ylabel='MAE (position units)',title=label+' · detail scale');ax.grid(axis='y',alpha=.2)
 axs[0].legend(fontsize=9,loc='best')
f.suptitle('Five fresh full-data RDL fits · SD is seed variability, not a confidence interval',fontsize=13);save(f,'scores')
