"""Computation-exposing figures, exported as editable SVG and portable PNG."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
P=Path(__file__).resolve().parent;OUT=P/'figures/l091';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none'})
ink='#183447';teal='#087f8c';rust='#a34430';gold='#c89025'
def canvas(title,subtitle,size=(12,7)):
 f,ax=plt.subplots(figsize=size);f.patch.set_facecolor('#fbfcfa');ax.set_facecolor('#fbfcfa');ax.set_xlim(0,12);ax.set_ylim(0,7);ax.axis('off')
 ax.text(.25,6.7,title,fontsize=21,weight='bold',color=ink);ax.text(.25,6.3,subtitle,fontsize=11,color='#536b77');return f,ax
def box(ax,x,y,w,h,title,body,color=teal):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.09,rounding_size=.13',fc='white',ec=color,lw=1.7))
 ax.text(x+.16,y+h-.34,title,weight='bold',color=color,fontsize=12)
 ax.text(x+.16,y+h-.72,body,va='top',linespacing=1.6,fontsize=11,color=ink)
def arrow(ax,a,b,color=ink):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':color,'lw':1.8})
def save(f,name):
 f.savefig(OUT/f'{name}.png',dpi=170,bbox_inches='tight');f.savefig(OUT/f'{name}.svg',bbox_inches='tight');plt.close(f)
f,a=canvas('R-GCN · a different transform for each relation','AIFB release recipe: 8,285 nodes · 45 predicates · 90 directed supports + self')
box(a,.3,4.05,3.1,1.7,'01  TYPED GRAPH','Remove target predicates\nSᵣ[i,j]: message j → i\nEach row sums to 1 or 0')
box(a,4.1,4.05,3.2,1.7,'02  IDENTITY INPUT','H⁰ = I  [8,285 × 8,285]\nDo not allocate dense I\nSᵣ I Wᵣ = Sᵣ Wᵣ')
box(a,8,4.05,3.5,1.7,'03  FIRST LAYER','Σᵣ Sᵣ Wᵣ¹ → ReLU\n91 matrices [8,285 × 16]\nH¹: [8,285 × 16]')
arrow(a,(3.45,4.9),(3.95,4.9));arrow(a,(7.4,4.9),(7.85,4.9))
box(a,8,1.55,3.5,1.7,'04  SECOND LAYER','Σᵣ Sᵣ H¹ Wᵣ²\n91 matrices [16 × 4]\nLogits: [8,285 × 4]')
arrow(a,(9.7,4),(9.7,3.4))
box(a,4.1,1.55,3.2,1.7,'TRAIN · 140 LABELS','Masked cross-entropy\n50 full-batch Adam updates\nGradients → both layers',gold)
box(a,.3,1.55,3.1,1.7,'INFER · 36 ENTITIES','Fixed graph; learned weights\nArgmax over four logits\nLabels used only to score',rust)
arrow(a,(7.9,2.65),(7.4,2.65),gold)
a.plot([9.7,9.7,1.85],[1.45,1.1,1.1],color=rust,lw=1.8)
arrow(a,(1.85,1.1),(1.85,1.48),rust)
a.text(.3,.65,'Basis extension only: Wᵣˡ = Σᵦ aᵣᵦˡ Vᵦˡ  ·  B = 4  ·  separate bases per layer',color=teal,fontsize=13,weight='bold')
a.text(.3,.25,'Shared across edges of one relation; separate across layers. Test labels never supervise training.',color=ink,fontsize=11)
save(f,'architecture')
f,a=canvas('One typed update · follow the denominators','Scalar fixture; messages travel into T. Features, weights and self path are held fixed.')
box(a,.3,3.8,3.3,2,'BUYS · teal','A = 2, C = 8\nMean = (2 + 8) / 2 = 5\nWeight 2 → contribution 10')
box(a,4.25,3.8,3.3,2,'RETURNS · rust','C = 8\nMean = 8 / 1 = 8\nWeight −1 → contribution −8',rust)
box(a,8.2,3.8,3.3,2,'SELF · gold','T = 1\nNo neighbor average\nWeight 1 → contribution 1',gold)
for x in [2,6,10]:arrow(a,(x,3.65),(6,2.8))
box(a,2.5,1.6,7,1.1,'SUM → ReLU','10 − 8 + 1 = 3. Empty relation neighborhoods would contribute zero.')
a.text(.4,.7,'Intervention: move A to returns → buys 16, returns −5, self 1 → output 12.',color=ink,fontsize=13)
a.text(.4,.2,'Same multiset of incoming features; different typed partition and per-relation means.',color=ink,fontsize=11)
save(f,'trace')
f,a=canvas('Basis sharing · reconstruct before activation','Two 2×2 bases, one relation coefficient row, and a feature vector [3, 1].')
box(a,.3,3.65,2.5,2.1,'V₁ · identity','[ 1   0 ]\n[ 0   1 ]\nCoefficient: 2')
box(a,3.3,3.65,2.5,2.1,'V₂ · swap','[ 0   1 ]\n[ 1   0 ]\nCoefficient: −1',rust)
box(a,7,3.65,4.4,2.1,'Wᵣ = 2 V₁ − V₂','[  2  −1 ]\n[ −1   2 ]\nOne relation-specific matrix')
arrow(a,(5.95,4.7),(6.85,4.7))
a.text(.4,2.75,'[3, 1] × Wᵣ = [3×2 − 1, −3 + 1×2] = [5, −1] → ReLU → [5, 0]',color=ink,fontsize=14)
box(a,.4,.7,5.3,1.5,'INDEPENDENT RELATIONS','100 × 16 × 16 = 25,600 parameters',rust)
box(a,6.2,.7,5.1,1.5,'FOUR SHARED BASES','4 × 16 × 16 + 100 × 4 = 1,424 parameters')
save(f,'basis')
if (P/'_paper_l091_results.json').exists():
 r=json.loads((P/'_paper_l091_results.json').read_text());scores=np.array([x['test_accuracy'] for x in r['runs']])*100
 f,ax=plt.subplots(figsize=(11,4.5));f.patch.set_facecolor('#fbfcfa');ax.set_facecolor('#fbfcfa')
 ax.scatter(range(10),scores,s=75,c=teal,label='Fresh PyTorch runs on the fixed split')
 ax.axhline(95.83,color=rust,ls='--',label='Paper Table 2: 95.83%')
 ax.axhline(scores.mean(),color=teal,label=f'Port mean: {scores.mean():.2f}%')
 ax.set(ylim=(85,100),xticks=range(10),xlabel='Declared initialization seed',ylabel='Test accuracy (%)',title='Full AIFB experiment · ten runs, fifty updates each')
 ax.legend(loc='lower left',frameon=False);ax.spines[['top','right']].set_visible(False)
 f.text(.13,-.05,'Sample SD = %.2f pp. One split, 36 test entities. Historical exact parity: INCOMPARABLE.'%(scores.std(ddof=1)),fontsize=11,color=ink)
 save(f,'results')
