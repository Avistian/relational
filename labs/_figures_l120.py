"""Deterministic diagrams specific to the exam's forward computation."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent/'figures/l120';P.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'l120','font.size':12})
INK='#19384a';BLUE='#dcecf5';GREEN='#dceee4'
def canvas(title):
 f,a=plt.subplots(figsize=(12,6));a.set(xlim=(0,12),ylim=(0,6));a.axis('off');f.patch.set_facecolor('#faf9f6');a.set_title(title,loc='left',weight='bold',fontsize=18,color=INK,pad=15);return f,a
def box(a,x,y,w,h,text,color=BLUE):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',fc=color,ec=INK,lw=1));a.text(x+w/2,y+h/2,text,ha='center',va='center',color=INK,fontsize=11)
def arrow(a,x,y,xx,yy):a.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':INK,'lw':1.5})
def save(f,name):
 f.tight_layout();f.savefig(P/(name+'.svg'),metadata={'Date':None});f.savefig(P/(name+'.png'),dpi=150,metadata={'Software':'L120'});plt.close(f)
f,a=canvas('One query, three typed inputs, two message rounds')
for y,label in [(4.3,'Person: constant pair\n[Np, 2]'),(2.5,'Event: amount + age\n[Ne, 2]'),(.7,'Merchant: constant pair\n[Nm, 2]')]:
 box(a,.2,y,2.4,1,label);box(a,3,y,2,1,'Table encoder\n[Ntype, 8]');arrow(a,2.7,y+.5,2.9,y+.5)
box(a,5.6,1.3,2.7,3.3,'Round 1 → Round 2\n\nWrelation × source\n→ typed receiver sum\n+ Wtype × root\n→ ReLU',GREEN)
for y in [4.8,3,1.2]:arrow(a,5.15,y,5.5,3)
box(a,8.9,3.6,2.7,1.3,'Person head → [Np]\nroot mask → [B]');arrow(a,8.45,3.6,8.8,4.1)
box(a,8.9,1.3,2.7,1.5,'Mature query labels [B]\nMAE → backward\n→ Adam update',GREEN);arrow(a,10.2,3.4,10.2,3)
a.text(.2,.1,'Original course model. Full published GraphSAGE reproduction is a separate, pinned execution.',fontsize=11,color=INK);save(f,'architecture')
f,a=canvas('The day-7 query cannot see a day-8 arrival')
box(a,.3,2.5,2.3,1.2,'person90\nquery cutoff = 7')
for y,text,col in [(4.4,'event1: event3 / arrival3\nLEGAL → merchant8',GREEN),(2.5,'event2: event5 / arrival8\nLATE → excluded','#f4ded9'),(.6,'event5: event11 / arrival11\nFUTURE → excluded','#f4ded9')]:
 box(a,3.3,y,4.1,1.2,text,col);arrow(a,2.8,3.1,3.15,y+.6)
box(a,8.2,2.5,3.2,1.5,'At cutoff 8:\nevent2 becomes legal\nmerchant4 enters',GREEN)
a.text(.3,.15,'Filter both clocks before traversal. Keep the same root cutoff at every hop.',color=INK);save(f,'cutoff')
# Portable copy of the actual full-data architecture, with its existing provenance.
for ext in ['svg','png']:(P/('paper.'+ext)).write_bytes((P.parent/'l117'/('architecture.'+ext)).read_bytes())
