"""Domain-specific computation diagrams, deterministic SVG and portable PNG."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Circle,Rectangle
P=Path(__file__).resolve().parent;D=P/'figures/l138';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.hashsalt':'l138','axes.spines.top':False,'axes.spines.right':False})
blue='#17324d';teal='#007f82';orange='#b85d20';gray='#687780'
def save(f,n):
 for ext in ['svg','png']:f.savefig(D/f'{n}.{ext}',dpi=150,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else None)
 plt.close(f)
def box(a,x,y,w,h,text,color=teal):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02',edgecolor=color,facecolor='#f1f7f7',linewidth=1.5));a.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11,color=blue)
def arrow(a,x,y,u,v,color=teal):a.annotate('',xy=(u,v),xytext=(x,y),arrowprops=dict(arrowstyle='->',color=color,lw=2))
f,a=plt.subplots(figsize=(9,6.6));a.set(xlim=(0,10),ylim=(0,8.5));a.axis('off');a.set_title('From a book review to a customer churn score',loc='left',fontweight='bold',color=blue,pad=16)
for x,y,label,color in [(1,6.5,'Customer u',teal),(4.5,6.5,'Review r\nday −12',teal),(8,6.5,'Book p',teal),(4.5,7.9,'Future review\nday +1',orange)]:
 a.add_patch(Circle((x,y),.58,facecolor='#eef6f5',edgecolor=color,lw=2));a.text(x,y,label,ha='center',va='center',fontsize=10,color=color)
a.annotate('',xy=(3.9,6.5),xytext=(1.6,6.5),arrowprops=dict(arrowstyle='<->',color=teal,lw=2));a.annotate('',xy=(7.4,6.5),xytext=(5.1,6.5),arrowprops=dict(arrowstyle='<->',color=teal,lw=2));a.text(6.5,7.9,'Excluded at cutoff 0',color=orange,fontsize=10)
a.text(.2,5.55,'Keys make edges. Text, categories, price and rating make features.',fontsize=11,color=blue)
box(a,.1,4.25,9.5,1.0,'Per-table encoders → 128 features per sampled row\nGloVe text means (300) → learned projection; relative age → time encoding')
arrow(a,4.8,5.5,4.8,5.28)
box(a,.1,2.8,4.1,1.0,'Layer 1: product → review\nbook features enrich the review')
box(a,5.2,2.8,4.4,1.0,'Layer 2: review → customer\nhistorical messages enrich the root')
arrow(a,2.1,4.25,2.1,3.82);arrow(a,4.25,3.3,5.15,3.3)
a.text(.2,2.35,'Each relation: Σ neighbor vectors → learned transform + root transform',fontsize=11,color=blue)
a.text(.2,1.98,'Sum relations → per-node normalization → ReLU; keep the same cutoff.',fontsize=11,color=blue)
box(a,.1,.35,4.1,1.05,'Root customer [batch,128]\nlinear head → logit [batch,1]')
box(a,5.2,.35,4.4,1.05,'Sigmoid → churn probability\nBCE trains; validation AUROC selects')
a.plot([9.62,9.88,9.88,2.1],[3.3,3.3,1.65,1.65],color=teal,lw=2);arrow(a,2.1,1.65,2.1,1.43);arrow(a,4.25,.86,5.15,.86)
save(f,'architecture')
f,a=plt.subplots(figsize=(8.2,3.7));a.set(xlim=(-105,106),ylim=(-.8,2));a.set_yticks([]);a.set_xlabel('Days relative to query cutoff');a.set_xticks([-91,0,91]);a.spines[['left','right','top']].set_visible(False)
a.axvspan(-91,0,color=teal,alpha=.10);a.axvspan(0,91,color=orange,alpha=.10);a.axvline(0,color=blue,lw=1.5)
a.text(-45,1.65,'ELIGIBILITY',ha='center',color=teal,fontweight='bold');a.text(45,1.65,'LABEL',ha='center',color=orange,fontweight='bold')
a.plot([-91,0],[1,1],color=teal,lw=3);a.scatter([-91],[1],facecolors='white',edgecolors=teal,s=90,zorder=3);a.scatter([0],[1],color=teal,s=90,zorder=3)
a.plot([0,91],[.45,.45],color=orange,lw=3);a.scatter([0],[.45],facecolors='white',edgecolors=orange,s=90,zorder=3);a.scatter([91],[.45],color=orange,s=90,zorder=3)
a.text(-45,.58,'Any review in (−91, 0]',ha='center',fontsize=11);a.text(45,0,'No review in (0, 91] → churn 1',ha='center',fontsize=11)
a.text(0,-.57,'A review exactly at day 0 makes the customer eligible, not retained.',ha='center',fontsize=10,color=blue)
a.set_title('Two adjacent windows answer two different questions',loc='left',fontweight='bold',color=blue);save(f,'windows')
f,a=plt.subplots(figsize=(7.3,4));a.set(xlim=(0,5),ylim=(0,3));a.axis('off');a.set_title('AUROC counts positive–negative comparisons',loc='left',fontweight='bold',color=blue)
for i,(p,lab) in enumerate([(1,'Churn score 1'),(2,'Churn score 2')]):
 a.text(.02,1.65-i*.8,lab,va='center',fontsize=11)
 for j,n in enumerate([0,1]):
  v=float(p>n)+.5*float(p==n);x=1.6+j*1.3;y=1.3-i*.8
  a.add_patch(Rectangle((x,y),1.15,.65,facecolor='#d5eeea' if v==1 else '#f9e4c6',edgecolor='white'));a.text(x+.575,y+.325,f'{v:g}',ha='center',va='center',fontsize=18,color=blue)
for j,n in enumerate([0,1]):a.text(2.175+j*1.3,2.15,f'Non-churn score {n}',ha='center',fontsize=10)
a.text(2.5,.1,'(1 + ½ + 1 + 1) / 4 = 0.875',ha='center',fontsize=14,color=teal);save(f,'ranking')
