"""Portable computation and evidence figures for the classification portfolio."""
from pathlib import Path
import json,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;D=P/'figures/l151';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'l151','axes.spines.top':False,'axes.spines.right':False})
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
 svg=D/f'{name}.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
f,a=canvas(9,7);a.text(.2,6.6,'ONE RESULT MUST CARRY ITS EVIDENCE',weight='bold',fontsize=15,color=INK)
for x,y,title,body in [(.3,4.5,'1 · Define','study + cutoff → binary label\nAUROC in [0,1]'),(4.7,4.5,'2 · Pin','source + archives + runtime\nfull model and recipe'),(.3,2.5,'3 · Freeze','validation choice + seeds\nhashes before refits'),(4.7,2.5,'4 · Collect','five complete fits per track\nmean + sample SD')]:box(a,x,y,3.85,1.5,title+'\n\n'+body)
box(a,.3,.35,8.25,1.35,'5 · Audit keys, labels, time and source   →   6 · Bound the claim\nProtocol PASS does not imply historical identity.\nA cited comparator does not establish a fresh comparison.');save(f,'contract')
f,a=canvas(9,8.4);a.text(.2,8.05,'ONE STUDY QUERY → ONE CLASSIFICATION SCORE',weight='bold',fontsize=14,color=INK)
box(a,.3,6.55,8.3,1.0,'Query (study 17, cutoff t) · released trial database\nSample 64 then 32 neighbors; each owner retains its own t')
arrow(a,4.45,6.55,4.45,6.15)
box(a,.3,4.8,8.3,1.3,'Typed row features → 128 coordinates + relative-time encoding\nTwo heterogeneous GraphSAGE layers\nWithin each relation: mean neighbors + transformed destination\nAcross relations: sum → batch norm → ReLU')
arrow(a,4.45,4.8,4.45,4.4)
box(a,.3,2.6,8.3,1.7,'Illustrative one-coordinate neighbor mean\nLegal neighbors [2, 6] → (2 + 6) / 2 = 4\nA future neighbor [20] is excluded by this query cutoff\nIncluding it would change the mean to 28 / 3 = 9.33',GOLD)
arrow(a,4.45,2.6,4.45,2.15)
box(a,.3,.55,8.3,1.45,'Root study vector [batch,128] → MLP → logit [batch,1]\nSigmoid at evaluation → ranking score [batch]\nTraining: binary cross-entropy on logits; Adam; 20 full epochs\nIllustrated mean is one component, not the complete learned layer');save(f,'architecture')
f,a=canvas(9,7.5);a.text(.2,7.15,'VALIDATION CHOOSES · TEST EVALUATES',weight='bold',fontsize=15,color=INK)
box(a,.3,5.3,8.3,1.2,'Three full candidates · seed 100 · twenty epochs\nLearning rates 0.00005 / 0.0001 / 0.0002\nEach checkpoint: first strict validation maximum')
arrow(a,4.45,5.3,4.45,4.7)
box(a,.3,3.55,8.3,1.15,'Nomination packet contains validation only\nLargest AUROC wins; smaller rate breaks an exact tie\nTest task table is not opened during search')
arrow(a,4.45,3.55,4.45,2.95)
box(a,.3,1.8,8.3,1.15,'Freeze candidate hashes + chosen rate + seeds 10–14\nFive fresh selected fits → test mean and sample SD\nFinal resampled validation may differ from selection validation')
a.text(.3,.7,'Reference track stays at rate 0.0001, seeds 0–4; never pool the tracks.',color=TEAL)
a.text(.3,.2,'Past test exposure remains disclosed; this course search is exploratory.',color=INK,fontsize=10);save(f,'selection')
s=json.loads((P/'evidence/l151/summary.json').read_text());f,a=plt.subplots(figsize=(9,5.4))
for offset,track,color in [(0,'reference',TEAL),(7,'selected',GOLD)]:
 r=s['tracks'][track];a.scatter([offset+i for i in range(5)],[100*v for v in r['values']],color=color,s=55,label=track.title()+' seeds');a.errorbar(offset+5.2,100*r['mean'],yerr=100*r['sample_sd'],fmt='D',color=color,capsize=5)
a.axhspan(67.6,69.6,color=TEAL,alpha=.08,label='Reference descriptive ±1pp band');a.axhline(68.6,color=INK,ls='--',label='Published RDL mean 68.60')
a.set(ylabel='Test AUROC · percentage points',xlabel='Reference seeds 0–4 | mean±SD     Selected seeds 10–14 | mean±SD',title='Full-data fits · individual seeds and sample SD')
a.set_xticks([0,1,2,3,4,5.2,7,8,9,10,11,12.2],['0','1','2','3','4','mean','10','11','12','13','14','mean']);a.legend(fontsize=9,loc='best');a.grid(axis='y',alpha=.2);f.tight_layout();save(f,'results')
print('Four portable portfolio figures generated')
