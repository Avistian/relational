"""Portable RT architecture and evaluation-boundary figures."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;D=P/'figures/l175';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l175'})
blue='#2563eb';green='#047857';red='#b91c1c';ink='#172554'
def save(fig,name):
 fig.savefig(D/(name+'.svg'),bbox_inches='tight',metadata={'Date':None});fig.savefig(D/(name+'.png'),dpi=160,bbox_inches='tight',metadata={'Software':'L175'});plt.close(fig)
def box(ax,x,y,w,h,title,body,color=blue):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.006',facecolor='#f8fafc',edgecolor=color,linewidth=1.6));ax.text(x+.02,y+h-.045,title,color=color,weight='bold',va='top',fontsize=12);ax.text(x+.02,y+h-.115,body,va='top',linespacing=1.6,fontsize=10.5)
fig,ax=plt.subplots(figsize=(12,6.7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.text(.02,.98,'RT-v1: the same frozen network reads a different database',fontsize=17,weight='bold',color=ink,va='top')
box(ax,.02,.65,.215,.24,'1  Sample cells','1,024 cells per query\nValues + column names\nRow IDs and FK links')
box(ax,.27,.65,.215,.24,'2  Encode cells','Name: 384 → 256\nTyped value → 256\nTarget uses mask vector')
box(ax,.52,.65,.215,.24,'3  Transform','12 relational blocks\nFour attention masks\nThen SwiGLU update')
box(ax,.77,.65,.215,.24,'4  Predict DNF','Final RMSNorm\nBoolean head: 256 → 1\nSigmoid → probability')
for x1,x2 in [(.235,.27),(.485,.52),(.735,.77)]:ax.annotate('',(x2,.77),(x1,.77),arrowprops=dict(arrowstyle='->',color=blue,lw=2))
ax.text(.03,.56,'Inside each relational block: four different attention masks',weight='bold',fontsize=13,color=ink)
labels=[('COLUMN','same table + column'),('FEATURE','same row or FK → PK'),('NEIGHBOR','PK → FK relation'),('FULL','all non-padding cells')]
for i,(title,body) in enumerate(labels):
 x=.03+i*.24;box(ax,x,.30,.215,.20,title,body+'\n8 attention heads',green)
 if i<3:ax.annotate('',(x+.24,.40),(x+.215,.40),arrowprops=dict(arrowstyle='->',color=green,lw=1.5))
ax.text(.04,.21,'Each attention: x ← x + Attention(RMSNorm(x)); then a 256 → 1,024 → 256 SwiGLU FFN.',fontsize=11)
ax.text(.04,.14,'A value mask hides the query answer. Attention masks specify which cells can exchange information.',fontsize=11)
ax.text(.04,.07,'Frozen parameters do not constrain what the sampler puts into the input.',color=red,weight='bold',fontsize=12)
save(fig,'architecture')
fig,ax=plt.subplots(figsize=(11.5,5.2));ax.set(xlim=(-40,50),ylim=(0,4));ax.set_yticks([]);ax.set_xlabel('Days relative to the prediction cutoff');ax.spines[['top','left','right']].set_visible(False)
ax.axvline(0,color=blue,lw=2);ax.text(1,3.7,'Predict here\nDay 0',color=blue,weight='bold')
for y,start,end,label,col in [(2.8,-35,-5,'Past label: outcome window complete',green),(1.8,-10,20,'Earlier row, but outcome still unavailable',red),(.8,0,30,'Same-time row: answer arrives in the future',red)]:
 ax.plot([start,end],[y,y],color=col,lw=6,solid_capstyle='round');ax.scatter([start,end],[y,y],color=col,s=45);ax.text(start,y+.24,label,color=col,fontsize=11)
ax.set_title('Row timestamp ≠ label availability',loc='left',fontweight='bold',fontsize=17,pad=18)
fig.subplots_adjust(bottom=.20)
fig.text(.12,.025,'Illustration for a 30-day horizon. Actual audit found zero labels with an unfinished window; ingestion delay is unknown.',fontsize=10,color=ink)
save(fig,'availability')
r=json.loads((P/'evidence/l175/verified-audit.json').read_text());fig,(ax,bx)=plt.subplots(1,2,figsize=(11.5,4.6),gridspec_kw={'width_ratios':[1,1.4]})
ax.bar([0,1,2],[x['future_queries'] for x in r['per_seed']],color=red,width=.55);ax.set(xticks=[0,1,2],xlabel='Context seed',ylabel='Queries with a future-dated cell',ylim=(0,702));ax.set_title('Complete population: 702 per seed',fontsize=12)
for x in r['per_seed']:ax.text(x['seed'],x['future_queries']+12,str(x['future_queries']),ha='center',color=red,weight='bold')
ax.spines[['top','right']].set_visible(False);bx.axis('off');bx.text(0,.96,'What was actually measured?',weight='bold',fontsize=15,color=ink)
for y,text,col in [(.78,'2,106 contexts independently audited',green),(.62,'385 future-dated cells / 77 contexts',red),(.46,'0 exposed query targets',green),(.30,'0 outcome labels with unfinished windows',green),(.14,'6 model evaluations: NOT_RUN',red)]:bx.text(0,y,text,fontsize=12,color=col)
fig.suptitle('The input audit stops the performance experiment',fontsize=17,weight='bold',x=.08,ha='left');fig.tight_layout(rect=(0,0,1,.9));save(fig,'audit')
