"""Checkpoint-specific evidence flow and measured results; portable SVG/PNG."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;D=P/'figures/l150';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l150','axes.spines.top':False,'axes.spines.right':False})
INK='#17324d';TEAL='#007f82';GOLD='#a35919'
def box(a,x,y,w,h,text,color=TEAL):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02',facecolor='#f0f7f6',edgecolor=color,linewidth=1.4);a.add_patch(p);t=a.text(x+w/2,y+h/2,text,ha='center',va='center',color=INK);a._pairs=getattr(a,'_pairs',[])+[(p,t)]
def arrow(a,x,y,u,v):a.annotate('',(u,v),(x,y),arrowprops={'arrowstyle':'->','color':INK,'lw':1.5})
def canvas(w,h):
 f,a=plt.subplots(figsize=(w,h));a.set(xlim=(0,w),ylim=(0,h));a.axis('off');return f,a
def save(f,name):
 f.canvas.draw();r=f.canvas.get_renderer()
 for a in f.axes:
  for p,t in getattr(a,'_pairs',[]):
   b=p.get_window_extent(r);q=t.get_window_extent(r);assert b.x0<=q.x0 and b.x1>=q.x1 and b.y0<=q.y0 and b.y1>=q.y1,(name,t.get_text())
 for ext in ['png','svg']:f.savefig(D/f'{name}.{ext}',dpi=160,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else None)
 plt.close(f)
 if (D/f'{name}.svg').exists():
  p=D/f'{name}.svg';p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')
f,a=canvas(9,6.7);a.text(.2,6.35,'ONE REPORT · FOUR INDEPENDENT GATES',weight='bold',fontsize=15,color=INK)
for x,y,title,body in [(.3,3.65,'1 · Protocol','Full fits + frozen selection\nKeys + source + temporal audit'),(4.7,3.65,'2 · Historical score','Five-seed mean vs 3.798\nFrozen descriptive ±0.20 band'),(.3,1.55,'3 · Current competition','Comparable metric and protocol\nRequires a separate audit'),(4.7,1.55,'4 · Learner defense','Explain choices and limits\nRequires reviewed written work')]:
 box(a,x,y,3.85,1.55,title+'\n\n'+body)
a.text(.35,.5,'A PASS in one box supplies no missing evidence to another.',color=TEAL,fontsize=12);save(f,'gates')
f,a=canvas(9,7.5);a.text(.2,7.15,'TEST CANNOT CHOOSE THE LEARNING RATE',weight='bold',fontsize=15,color=INK)
box(a,.3,5.3,8.3,1.2,'Three full candidates · same seed 100 · ten epochs\nLearning rates 0.001 / 0.003 / 0.005\nEach checkpoint selected by its first validation minimum')
arrow(a,4.45,5.3,4.45,4.7)
box(a,.3,3.55,8.3,1.15,'Validation-only selection packet\nLowest MAE wins; smaller learning rate breaks exact ties\nSearch loader rejects test access')
arrow(a,4.45,3.55,4.45,2.95)
box(a,.3,1.8,8.3,1.15,'Freeze chosen rate + candidate hashes + seeds 10–14\nThen run five fresh selected fits\nReport test mean and sample SD; do not retune')
a.text(.3,.7,'The reference track keeps seeds 0–4 and learning rate 0.005.',color=TEAL)
a.text(.3,.2,'Prior test exposure still makes this a course experiment, not a pristine holdout.',color=INK,fontsize=10);save(f,'selection')
s=json.loads((P/'evidence/l150/summary.json').read_text());f,a=plt.subplots(figsize=(9,5.4))
for offset,track,color in [(0,'reference',TEAL),(7,'selected',GOLD)]:
 r=s['tracks'][track];a.scatter([offset+i for i in range(5)],r['values'],color=color,s=55,label=track.title()+' seeds');a.errorbar(offset+5.2,r['mean'],yerr=r['sample_sd'],fmt='D',color=color,capsize=5)
a.axhspan(3.598,3.998,color=TEAL,alpha=.08,label='Historical ±0.20 tolerance')
a.axhline(3.798,color=INK,ls='--',label='Paper mean 3.798')
replay=next(r['test_mae'] for r in s['records'] if r['track']=='replay');a.axhline(replay,color='#8e6297',ls=':',label='Checkpoint replay')
a.set(xticks=list(range(5))+[5.2]+list(range(7,12))+[12.2],xticklabels=['0','1','2','3','4','Mean','10','11','12','13','14','Mean'],ylabel='Raw test MAE · lower is better',xlabel='Seed ID; diamonds show mean ± sample SD',title='Complete runs do not guarantee score reproduction')
a.legend(fontsize=9,loc='best');a.grid(axis='y',alpha=.18);f.tight_layout();save(f,'results')
