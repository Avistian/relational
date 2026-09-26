"""Computation diagrams and measured results; portable PNG and text SVG."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;O=P/'figures/l106';O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
TEAL='#087e82';ORANGE='#b65b2c';INK='#233b45';GRAY='#667879'
def canvas(title,sub,height=5):
 f,ax=plt.subplots(figsize=(10,height));f.patch.set_facecolor('#fbfcf9');ax.set_facecolor('#fbfcf9');ax.set_xlim(0,10);ax.set_ylim(0,height);ax.axis('off');ax.text(.25,height-.38,title,fontsize=19,weight='bold',color=INK);ax.text(.25,height-.78,sub,fontsize=11,color=GRAY);return f,ax

def box(ax,x,y,w,h,text,color=TEAL):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor='white',edgecolor=color,lw=1.5));ax.text(x+.15,y+h-.18,text,va='top',fontsize=12,color=INK,linespacing=1.8)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':TEAL,'lw':2})
def save(f,name):
 f.canvas.draw()
 for ax in f.axes:
  if not ax.axison:
   bounds=ax.get_window_extent();renderer=f.canvas.get_renderer()
   for text in ax.texts:
    b=text.get_window_extent(renderer)
    assert b.x0>=bounds.x0-2 and b.x1<=bounds.x1+2 and b.y0>=bounds.y0-2 and b.y1<=bounds.y1+2, (name,text.get_text(),'outside axes')
 f.savefig(O/(name+'.svg'),bbox_inches='tight');f.savefig(O/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(f)
f,a=canvas('EdgeBank: score first, then observe','Synthetic directed interactions · query just before t = 10 · unlimited memory',5.5)
box(a,.3,1.8,2.35,2.4,'History [3,3]\nAX @ 1\nAY @ 3\nBX @ 8')
box(a,3.15,1.8,2.55,2.4,'Set of pairs [3,2]\n{AX, AY, BX}\nDiscard multiplicity\nKeep directed identity')
box(a,6.35,1.8,3.25,2.4,'Candidates [2,2]\nAX ∈ memory → 1\nBY ∉ memory → 0\nOutput scores [2]')
arrow(a,(2.75,3),(3.05,3));arrow(a,(5.8,3),(6.25,3));a.text(.4,.98,'Only after observing BY @ 10: memory becomes {AX, AY, BX, BY}.',color=TEAL,fontsize=13);a.text(.4,.5,'Scored negative pairs never update the memory. Binary scores are not probabilities.',fontsize=11,color=GRAY);save(f,'architecture')
f,a=canvas('Which absent pairs become negatives?','Same query interval; change the eligible comparison population',5.4)
for x,title,body,col in [(.3,'Random','Possible pairs\nMostly never observed\nOften easy when sparse',GRAY),(3.55,'Historical','Previously seen pairs\nExclude current interval\nTest stale memory',TEAL),(6.8,'Inductive','Previously seen pairs\nExclude train/val pairs\nNew pair ≠ new node',ORANGE)]:
 box(a,x,1.5,2.85,2.7,title+'\n\n'+body,col)
a.text(.35,.85,'Historical / inductive fallback: fill a shortage with random pairs.',fontsize=12,color=INK);a.text(.35,.38,'Paper concept shown. Source replay separately audits interval boundaries and collisions.',fontsize=11,color=GRAY);save(f,'candidates')
f,a=canvas('AUROC counts cross-class pairs','Two positives score [1,0]. Every cell is one positive–negative comparison.',5.4)
for x,negative,title in [(.5,[0,0],'Unseen negatives'),(5.4,[1,1],'Remembered negatives')]:
 a.text(x,4,title,fontsize=14,weight='bold',color=INK)
 for j,n in enumerate(negative):a.text(x+1.5+j*1.2,3.4,'neg '+str(n),ha='center',fontsize=11)
 for i,p in enumerate([1,0]):
  y=2.5-i*.8;a.text(x,y+.22,'pos '+str(p),fontsize=11)
  for j,n in enumerate(negative):
   val=1 if p>n else .5 if p==n else 0
   a.add_patch(FancyBboxPatch((x+.95+j*1.2,y),1.05,.6,boxstyle='round,pad=.03',facecolor='#e3efea' if val else '#f5e7df',edgecolor='white'));a.text(x+1.47+j*1.2,y+.29,str(val),ha='center',va='center',fontsize=14,color=INK)
 a.text(x,1.05,'AUROC = '+('3 / 4 = 0.75' if negative[0]==0 else '1 / 4 = 0.25'),fontsize=13,color=TEAL)
a.text(.5,.4,'Win = 1 · tie = 0.5 · loss = 0. Never break score ties using row order.',fontsize=12,color=GRAY);save(f,'metrics')
path=P/'_analysis_l106_results.json'
if path.exists():
 r=json.loads(path.read_text());f,axes=plt.subplots(1,2,figsize=(10,4.7));f.patch.set_facecolor('#fbfcf9')
 for j,(ax,metric) in enumerate(zip(axes,['AP','AUROC'])):
  for mode,offset,color in [('unlimited',-.12,TEAL),('window',.12,ORANGE)]:
   rs=[r['conditions'][s]['summary'][mode] for s in ['rnd','hist_nre','induc_nre']]
   ax.errorbar([i+offset for i in range(3)],[x['mean'][j] for x in rs],yerr=[x['sd_ddof0'][j] for x in rs],fmt='o',color=color,capsize=4,label=mode+' replay ± SD')
   ax.scatter([i+offset for i in range(3)],[x['paper_target'][j] for x in rs],marker='x',s=65,color=color,label=mode+' paper')
  ax.set_xticks(range(3),['Random','Historical','Inductive']);ax.set_ylim(0,1);ax.set_ylabel(metric);ax.grid(axis='y',alpha=.2)
 axes[0].legend(fontsize=9,loc='lower left');f.suptitle('Wikipedia: released-code replay versus rounded paper targets',fontsize=15);f.text(.08,.02,'Batch means · five source iterations · SD is not a confidence interval · historical identity unestablished',fontsize=10);f.tight_layout(rect=(0,.06,1,.92));save(f,'results')
print('Figures written')
