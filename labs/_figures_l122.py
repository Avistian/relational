"""Deterministic row/coordinate and typed message diagrams for L122."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent/'figures/l122';P.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'l122','font.size':12})
INK='#19384a';BLUE='#dcecf5';GREEN='#dceee4'
def canvas(title):
 f,a=plt.subplots(figsize=(11,6));a.set(xlim=(0,11),ylim=(0,6));a.axis('off');f.patch.set_facecolor('#faf9f6');a.set_title(title,loc='left',weight='bold',fontsize=17,color=INK,pad=15);return f,a
def box(a,x,y,w,h,text,color=BLUE):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.10',fc=color,ec=INK,lw=1));a.text(x+w/2,y+h/2,text,ha='center',va='center',color=INK,fontsize=12)
def arrow(a,x,y,xx,yy):a.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':INK,'lw':1.6})
def save(f,name):
 f.tight_layout();f.savefig(P/(name+'.svg'),metadata={'Date':None});f.savefig(P/(name+'.png'),dpi=150,metadata={'Software':'L122'});plt.close(f)
f,a=canvas('Trace transfer7: a key is not a row coordinate')
box(a,.2,3.5,3.1,1.8,'person table\nrow0: id90, age40\nrow1: id10, age20\nrow2: id300, age60')
box(a,.2,.9,3.1,1.6,'transfer row0\nid7, sender10\nreceiver90, amount5')
box(a,4,3.5,2.8,1.8,'Build lookup\n90 → 0\n10 → 1\n300 → 2',GREEN);arrow(a,3.45,4.4,3.85,4.4)
box(a,7.5,3.5,3.1,1.8,'sender column\nsource row0 → row1\nedge_index[:, 0]\n= [0, 1]',GREEN)
box(a,7.5,.9,3.1,1.8,'receiver column\nsource row0 → row0\nedge_index[:, 0]\n= [0, 0]',GREEN)
arrow(a,6.95,4.4,7.35,4.4);arrow(a,6.8,3.6,7.35,2.5);arrow(a,3.45,1.7,4.15,3.3)
a.text(.2,.15,'Source and destination coordinates belong to different node types. [0, 0] is not a self-loop.',color=INK,fontsize=11);save(f,'mapping')
f,a=canvas('Preserve two roles, then trace the receiver sum')
for y,label in [(4.5,'transfer7 · amount5'),(2.8,'transfer8 · amount8'),(1.1,'transfer9 · amount2')]:box(a,.2,y,3,1,label)
for y,label in [(4.5,'person90 · row0\nreceived = 5 + 2 = 7'),(2.8,'person10 · row1\nreceived = 8'),(1.1,'person300 · row2\nreceived = 0')]:box(a,7.3,y,3.3,1,label,GREEN)
arrow(a,3.35,5,7.15,5);arrow(a,3.35,3.3,7.15,3.3);arrow(a,3.35,1.6,7.15,4.8)
a.text(4.5,5.3,'receiver',color=INK);a.text(4.5,3.6,'receiver',color=INK);a.text(4.6,1.1,'No row points to person300.\nThe node still exists.',fontsize=11,color=INK,ha='center')
a.text(.2,.25,'Sender edges and reverse edges live in separate typed stores (omitted here). No learned weights.',fontsize=11,color=INK);save(f,'routing')
for ext in ['svg','png']:(P/('paper.'+ext)).write_bytes((P.parent/'l117'/('architecture.'+ext)).read_bytes())
