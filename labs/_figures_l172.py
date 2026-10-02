"""Portable worked traces: role routing, measured scalar token and fit boundary."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;D=P/'figures/l172';D.mkdir(parents=True,exist_ok=True)
r=json.loads((P/'evidence/l172/report.json').read_text());w=r['worked_trace']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.hashsalt':'l172'})
ink='#17324d';teal='#157f78';rust='#a34832';muted='#526477';cream='#faf8f2'
def canvas(title,subtitle,height=4.8):
 fig,ax=plt.subplots(figsize=(11,height));ax.set(xlim=(0,11),ylim=(0,height));ax.axis('off')
 ax.text(.1,height-.35,title,fontsize=19,weight='bold',color=ink)
 ax.text(.1,height-.78,subtitle,color=muted,fontsize=11)
 return fig,ax
def box(ax,x,y,width,height,title,body,color=teal):
 ax.add_patch(FancyBboxPatch((x,y),width,height,boxstyle='round,pad=.08',facecolor='#e4f2ed' if color==teal else '#f7e5dd',edgecolor=color))
 ax.text(x+.15,y+height-.32,title,color=color,weight='bold',fontsize=12)
 ax.text(x+.15,y+height-.72,body,color=ink,fontsize=12,va='top',linespacing=1.6)
def save(fig,name):
 fig.savefig(D/(name+'.svg'),bbox_inches='tight',facecolor=cream,metadata={'Date':None})
 fig.savefig(D/(name+'.png'),dpi=150,bbox_inches='tight',facecolor=cream,metadata={'Software':'L172'});plt.close(fig)
fig,ax=canvas('Same storage type, different operations','Illustrative integers. A schema policy determines meaning; a dtype cannot decide.')
for y,label,operation,output in [(2.75,'driverId = 7','foreign key → identity','"7" → drivers.driverId'),(1.65,'laps = 7','quantity → normalize','(7 − mean) / scale'),(.55,'statusId = 7','category → vocabulary','local code, not distance')]:
 ax.text(.2,y+.3,label,color=ink,weight='bold');ax.annotate('',xy=(4.0,y+.36),xytext=(2.5,y+.36),arrowprops={'arrowstyle':'->','color':teal});ax.text(4.1,y+.3,operation,color=teal);ax.text(7.5,y+.3,output,color=ink,fontsize=11)
save(fig,'roles')
fig,ax=canvas('A real F1 cell becomes a typed payload',f"Measured results.points · resultId {w['resultId']} · event date {w['date'][:10]}",5.6)
box(ax,.2,2.3,3.1,1.8,'1  READ',f"raw points = {w['raw']:g}\nstate = VALUE\nschema = points of results")
box(ax,3.95,2.3,3.1,1.8,'2  APPLY FROZEN FIT',f"mean = {w['mean']:.6f}\nscale = {w['scale']:.6f}\n{w['fit_rows']:,} admitted rows")
box(ax,7.7,2.3,2.9,1.8,'3  EMIT',f"({w['raw']:g} − {w['mean']:.4f})\n÷ {w['scale']:.4f}\n= {w['payload']:.6f}")
for x in [3.4,7.15]:ax.annotate('',xy=(x+.43,3.2),xytext=(x,3.2),arrowprops={'arrowstyle':'->','color':teal,'lw':2})
box(ax,.2,.3,10.4,1.25,'INTERVENTION: MASK THIS CELL','MASKED + payload 0. The points value is erased; the schema descriptor stays.',rust)
save(fig,'trace')
fig,ax=canvas('Transform all rows; fit only admitted rows','Measured row counts. Full-snapshot coverage does not authorize query-time access.',6)
names=list(r['tables']);total=sum(t['admitted_rows'] for t in r['tables'].values())
for i,name in enumerate(names):
 t=r['tables'][name];y=4.5-i*.42
 ax.text(.1,y,name,fontsize=11,color=ink)
 width=4.7;fraction=t['admitted_rows']/t['rows'];ax.barh(y,width,left=3.2,height=.24,color='#dddfe2');ax.barh(y,width*fraction,left=3.2,height=.24,color=teal)
 ax.text(8.1,y,f"{t['admitted_rows']:,} / {t['rows']:,}",va='center',fontsize=11,color=ink)
ax.text(3.2,4.94,'teal: fit admitted     gray: transform only',color=teal,fontsize=11)
ax.text(.1,.18,f'{total:,} fit-admitted rows; 1,145 untimed rows receive no fitted statistics. No task split is claimed.',color=muted,fontsize=11)
save(fig,'fit-boundary')
print('Built 3 portable figures')
