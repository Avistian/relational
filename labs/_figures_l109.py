"""Four portable diagrams: clocks, versioned joins, label maturity and real evidence."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;O=P/'figures/l109';O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none','svg.hashsalt':'l109','axes.spines.top':False,'axes.spines.right':False})
INK='#233b45';TEAL='#087e82';RUST='#b65b2c';GRAY='#647579'
def canvas(title,subtitle,h=4.8):
 f,ax=plt.subplots(figsize=(10,h));f.patch.set_facecolor('#fbfcf9');ax.set(xlim=(0,10),ylim=(0,h));ax.axis('off');ax.text(.2,h-.35,title,color=INK,fontsize=18,weight='bold');ax.text(.2,h-.72,subtitle,color=GRAY,fontsize=11);return f,ax

def box(ax,x,y,w,h,s,color=TEAL):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',facecolor='white',edgecolor=color,lw=1.5));ax.text(x+.12,y+h-.13,s,va='top',color=INK,fontsize=11,linespacing=1.65)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':2,'color':TEAL})
def save(f,name):
 f.savefig(O/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});f.savefig(O/(name+'.png'),dpi=150,bbox_inches='tight',metadata={'Software':'L109'});plt.close(f)
f,ax=canvas('A past event can still be future knowledge','Synthetic immutable event · event day 2, arrival day 7, prediction day 5')
x=lambda t:.65+t*.76
ax.plot([x(0),x(11)],[2.8,2.8],color=GRAY)
for t,label,col in [(2,'Event occurs',TEAL),(5,'Predict',INK),(7,'Row arrives',RUST)]:
 ax.plot(x(t),2.8,'o',color=col,ms=10);ax.text(x(t),3.15,label,ha='center',color=col);ax.text(x(t),2.42,f'day {t}',ha='center',color=col)
ax.axvline(x(5),ymin=.42,ymax=.77,color=INK,linestyle='--')
box(ax,.4,.45,4.2,1.1,'Event-only filter: 2 ≤ 5 → yes\nIt incorrectly admits this record.',RUST)
box(ax,5,.45,4.5,1.1,'Two-clock filter: 2 ≤ 5 AND 7 ≤ 5\nFalse → exclude the record.',TEAL)
save(f,'clocks')
f,ax=canvas('Select both histories, then follow the key','Synthetic correction · fact r changes parent A → B when revision arrives on day 8',h=6.2)
box(ax,.3,3.4,4.25,1.65,'Stored versions of fact r\nvalid 2 | observed 3 | parent A\nvalid 2 | observed 8 | parent B\nvalid 2 | observed 10 | deleted')
box(ax,5.2,3.4,4.25,1.65,'Stored parent versions\nA: valid 0 | observed 0 | value 1\nA: valid 0 | observed 9 | value 9\nB: valid 0 | observed 7 | value 2')
for x0,t,line in [(.35,6,'r → A(value 1)\nedge ready = max(2,3,0,0) = 3'),(5.25,8,'r → B(value 2)\nedge ready = max(2,8,0,7) = 8')]:
 box(ax,x0,1.15,4.2,1.5,f'As known on day {t}\n'+line);arrow(ax,(x0+2.1,3.3),(x0+2.1,2.8))
ax.text(.4,.45,'Day 10: select the deletion revision → no fact node and no outgoing edge.',color=RUST,fontsize=12)
save(f,'versions')
f,ax=canvas('A future target is legal only on the label side','Synthetic query day 5 · horizon 4 days · late outcome arrives on day 12',h=4.8)
ax.plot([.7,9.3],[2.6,2.6],color=GRAY,lw=2)
for t,label,x0,col in [(5,'Query',1,INK),(9,'Window ends',5,TEAL),(12,'Last outcome arrives',8,RUST)]:
 ax.plot(x0,2.6,'o',color=col,ms=10);ax.text(x0,3.0,label,ha='center',color=col);ax.text(x0,2.25,f'day {t}',ha='center',color=col)
ax.annotate('',xy=(5,2.6),xytext=(1,2.6),arrowprops={'arrowstyle':'->','color':TEAL,'lw':4});ax.text(3,1.85,'label event interval (5, 9]',ha='center',color=TEAL)
box(ax,.4,.25,9.0,1.05,'Features: occurred AND observed by 5.  Fit label: no earlier than max(9,12) = 12.\nA completeness certificate is also needed; unseen arrivals cannot certify themselves.')
save(f,'maturity')
r=json.loads((P/'evidence/l109/reproduction.json').read_text())
f,axs=plt.subplots(1,2,figsize=(10,4.6));f.patch.set_facecolor('#fbfcf9')
for ax,split in zip(axs,['val','test']):
 rows=[a for a in r['results'] if a['split']==split];positions=list(range(5))
 ax.barh(positions,[a['mae'] for a in rows],color=TEAL,alpha=.75,label='Fresh reconstruction')
 ax.plot([a['paper_mae'] for a in rows],positions,'|',color=INK,ms=15,mew=2,label='Published rounded value')
 ax.set_yticks(positions,['Zero','Global mean','Global median','Entity mean','Entity median']);ax.invert_yaxis();ax.set_xlim(0,13);ax.set_xlabel('Mean absolute error (positions)');ax.set_title(split.upper()+' · '+str(r['labels'][split]['rows'])+' queries');ax.grid(axis='x',alpha=.15)
axs[1].legend(loc='lower right',fontsize=9);f.suptitle('All ten selected paper cells match within 0.0005',fontsize=16,color=INK);f.tight_layout(rect=(0,0,1,.93));save(f,'results')
print('Four SVG + PNG figure pairs')
