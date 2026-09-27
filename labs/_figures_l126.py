"""Three deterministic computation diagrams; values come from the visible kernel."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from relkit.beta_l126 import average_precision
P=Path(__file__).resolve().parent/'figures/l126';P.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.hashsalt':'l126'})
INK='#193746';TEAL='#087e82';BG='#faf9f6';BLUE='#e0edf4';GOLD='#f6e6bc';GREEN='#dcefe3'
def canvas(title,h):
    f,a=plt.subplots(figsize=(11,h));f.patch.set_facecolor(BG);a.set(xlim=(0,11),ylim=(0,h));a.axis('off');a.set_title(title,loc='left',weight='bold',fontsize=17,color=INK,pad=16);return f,a

def box(a,x,y,w,h,text,color=BLUE):
    a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',fc=color,ec=INK,lw=1));a.text(x+w/2,y+h/2,text,ha='center',va='center',color=INK,fontsize=11)
def arrow(a,x,y,xx,yy):a.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':INK,'lw':1.8})
def save(f,name):
    f.tight_layout();f.savefig(P/(name+'.svg'),metadata={'Date':None});f.savefig(P/(name+'.png'),dpi=150,metadata={'Software':'L126'});plt.close(f)
f,a=canvas('One database, two different access boundaries',6)
a.text(.15,5.35,'Synthetic timeline · time units are days · query cutoff = 10 · test cap = 30',color=INK)
a.plot([.8,10.2],[3.95,3.95],color=INK)
for day,x in [(5,2),(10,3.5),(15,5),(25,8),(30,9.5)]:
    a.plot(x,3.95,'o',color=TEAL);a.text(x,4.25,str(day),ha='center',color=INK)
a.axvline(3.5,ymin=.34,ymax=.78,ls='--',color=TEAL);a.text(3.5,3.48,'query cutoff',ha='center',fontsize=11)
a.axvline(9.5,ymin=.34,ymax=.78,ls='--',color='#a77316');a.text(9.5,3.48,'test cap',ha='center',fontsize=11)
box(a,.3,2.15,10.2,.75,'Dataset cap allows all five rows: [5, 10, 15, 25, 30]',GOLD)
box(a,.3,1.03,10.2,.75,'Query features may use only [5, 10] → every hop inherits cutoff 10',GREEN)
a.text(.3,.2,'Future rows can construct labels in the offline evaluator. They cannot become query inputs.',fontsize=11,color=INK);save(f,'boundaries')
f,a=canvas('Identity survives batching only when scores carry query keys',5.5)
a.text(.2,4.9,'Worked example · same entity can appear at two prediction times',color=INK)
box(a,.2,3.75,3.1,.7,'Evaluator: (A,10), (B,10), (A,20)',BLUE)
box(a,6.4,3.75,4.1,.7,'Batch: (A,20), (A,10), (B,10)',GOLD)
box(a,6.4,2.35,4.1,.7,'Scores: 0.8, 0.2, 0.4',GOLD);arrow(a,8.4,3.65,8.4,3.15)
box(a,.2,2.35,4.3,.7,'Join on (entity,time): positions [1,2,0]',GREEN);arrow(a,6.25,2.7,4.65,2.7);arrow(a,1.75,3.65,1.75,3.15)
box(a,.2,.95,4.3,.7,'Evaluator receives [0.2, 0.4, 0.8]',GREEN);arrow(a,2.35,2.25,2.35,1.75)
box(a,6.4,.95,4.1,.7,'Joining only on A/B is ambiguous',GOLD)
a.text(.2,.15,'Reject duplicate, missing, extra and non-finite predictions before scoring.',fontsize=12,color=INK);save(f,'identity')
f,a=canvas('Average precision consumes equal-score groups together',6.2)
a.text(.15,5.55,'Worked example · labels [1,0,1,0,1] · scores [0.9,0.5,0.5,0.1,0.1]',fontsize=12,color=INK)
headers=['Threshold','Rows admitted','TP / rows','Recall gain','Contribution']
xs=[.25,2.15,4.65,6.8,8.8]
for x,h in zip(xs,headers):a.text(x,4.78,h,weight='bold',color=INK,fontsize=11)
vals=[['≥ 0.9','[1]','1 / 1','1 / 3','1 / 3'],['≥ 0.5','[1, 0, 1]','2 / 3','1 / 3','2 / 9'],['≥ 0.1','[1, 0, 1, 0, 1]','3 / 5','1 / 3','1 / 5']]
for i,row in enumerate(vals):
    yy=3.8-i*.95
    a.add_patch(FancyBboxPatch((.1,yy-.25),10.7,.7,boxstyle='round,pad=.04',fc=GREEN if i==1 else BLUE,ec='none'))
    for x,v in zip(xs,row):a.text(x,yy,v,color=INK,fontsize=12)
ap=average_precision([1,0,1,0,1],[.9,.5,.5,.1,.1])
a.text(.2,.72,f'AP = 1/3 + 2/9 + 1/5 = {ap:.6f}',color=TEAL,weight='bold',fontsize=16)
a.text(.2,.1,'Two rows tied at 0.5 enter at once. Reordering them cannot change the result.',color=INK,fontsize=11);save(f,'average-precision')
print('Built three L126 figures')
