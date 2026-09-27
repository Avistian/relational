"""OGB information-flow contract; no new neural architecture is introduced."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D=Path(__file__).parent/'figures/l111';D.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none','svg.hashsalt':'l111'})
f,ax=plt.subplots(figsize=(10,7));ax.set(xlim=(0,10),ylim=(0,7));ax.axis('off')
ink='#18354b';teal='#087f82';amber='#ae571e'
def box(x,y,w,h,title,body,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',edgecolor=color,facecolor='#f5f8f7',lw=1.6));ax.text(x+.15,y+h-.25,title,weight='bold',color=color,va='top');ax.text(x+.15,y+.18,body,color=ink,va='bottom',linespacing=1.5)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.8,color=ink))
ax.text(.1,6.65,'One benchmark · separate fitting, selection and evaluation',fontsize=17,weight='bold',color=ink)
box(.2,4.7,9.5,1.25,'OFFICIAL GRAPH + IDENTITIES','169,343 papers · features [N,128] · labels [N,1]\nOfficial year split supplies node IDs; never re-split to improve a score.',teal)
for x,title,body in [(.2,'TRAIN · through 2017','90,941 labels\nFit the predictor'),(3.5,'VALIDATION · 2018','29,799 labels\nChoose model settings'),(6.8,'TEST · 2019 onward','48,603 labels\nScore the frozen choice')]:
 box(x,2.65,2.8,1.35,title,body,teal if x<6 else amber);arrow((x+1.4,4.65),(x+1.4,4.05))
box(.2,.35,4.4,1.4,'TODAY · majority baseline','Training labels → most frequent class\nRepeat that ID for every question\nEvaluator + independent correct / total',teal)
box(5.15,.35,4.55,1.4,'NEXT · GCN in Lesson 112','Full feature graph → graph network\nLoss uses training labels only\nGraph visibility needs its own contract',amber)
arrow((1.6,2.6),(1.6,1.85))
for ext in ['svg','png']:f.savefig(D/f'contract.{ext}',bbox_inches='tight',dpi=150,metadata={'Date':None} if ext=='svg' else {})
plt.close(f)
