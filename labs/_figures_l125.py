"""Computation-specific, deterministic vector diagrams and portable PNGs."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent/'figures/l125';P.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'l125','font.size':11})
INK='#18384a';BLUE='#deebf4';GREEN='#dcefe3';GOLD='#f7e9cb';BG='#faf9f6'
def canvas(title,height=6):
 f,a=plt.subplots(figsize=(12,height));f.patch.set_facecolor(BG);a.set(xlim=(0,12),ylim=(0,height));a.axis('off');a.set_title(title,loc='left',color=INK,fontsize=18,weight='bold',pad=18);return f,a

def box(a,x,y,w,h,label,color=BLUE,fontsize=11):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',fc=color,ec=INK,lw=1));a.text(x+w/2,y+h/2,label,ha='center',va='center',color=INK,fontsize=fontsize)
def arrow(a,x,y,xx,yy,label=None):
 a.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','lw':1.7,'color':INK})
 if label:a.text((x+xx)/2,(y+yy)/2+.13,label,ha='center',va='bottom',fontsize=9,color=INK)
def save(f,name):
 f.tight_layout();f.savefig(P/(name+'.svg'),metadata={'Date':None});f.savefig(P/(name+'.png'),dpi=145,metadata={'Software':'L125'});plt.close(f)
f,a=plt.subplots(figsize=(10,10));f.patch.set_facecolor(BG);a.set(xlim=(0,12),ylim=(0,12));a.axis('off')
a.set_title('Six typed columns → six tokens → one row encoder',loc='left',color=INK,fontsize=17,weight='bold',pad=18)
a.text(.15,11.65,'Read each card downward: raw value → materialized representation → learned encoding.',fontsize=11,color=INK)
rows=[('01  NUMERICAL','30 or missing','float [B,1]','center + scale → affine'),('02  CATEGORICAL','north or unseen','integer index [B,1]','column offsets → lookup'),('03  MULTI-CATEGORY','[a,b] or [c]','ragged index lists','pool category embeddings'),('04  TIMESTAMP','a calendar date','calendar fields','year + periodic fields → project'),('05  EMBEDDING','vector [1,0,0]','dense values, width 3','column-specific projection'),('06  TEXT','note: red apple','fixed hash counts, width 16','learned projection')]
for i,(title,raw,mat,enc) in enumerate(rows):
 x=.18+(i%2)*6;y=8.30-(i//2)*3.35
 a.add_patch(FancyBboxPatch((x,y),5.45,2.95,boxstyle='round,pad=.08',facecolor='white',edgecolor='#bfd0d6',lw=1.5))
 a.text(x+.22,y+2.57,title,fontsize=13,weight='bold',color=INK)
 a.text(x+.22,y+2.07,'Raw: '+raw,fontsize=12,color='#93601a')
 a.annotate('',xy=(x+.4,y+1.58),xytext=(x+.4,y+1.88),arrowprops=dict(arrowstyle='->',color=INK))
 a.text(x+.22,y+1.29,mat,fontsize=12,color='#245878')
 a.annotate('',xy=(x+.4,y+.82),xytext=(x+.4,y+1.10),arrowprops=dict(arrowstyle='->',color=INK))
 a.text(x+.22,y+.54,enc,fontsize=12,color='#176c68')
 a.text(x+.22,y+.13,'Output: token [B, 1, 8]',fontsize=12,weight='bold',color='#176c68')
a.text(.2,.65,'Concatenate in TensorFrame column order: [B,6,8]. Here B = 2 rows.',fontsize=13,weight='bold',color=INK)
a.text(.2,.12,'Materialization is frozen. Encoders are trained. The text adapter is an illustrative hash encoder, not pretrained.',fontsize=10.5,color=INK)
save(f,'typed')
f,a=canvas('Model architecture — follow the loss back into both row encoders',8)
for x,label,cols in [(.2,'DRIVERS',6),(6.35,'RESULTS',10)]:
 box(a,x,6.2,5.35,.95,f'{label}: typed cells → column tokens\n[N,{cols},8] → flatten [N,{cols*8}]',BLUE)
 box(a,x,4.65,5.35,.95,'Two residual blocks, width 8\nLinear → LN → ReLU paths + shortcut',GREEN)
 box(a,x,3.1,5.35,.95,'LayerNorm → ReLU → Linear\nRow vectors [N,8]',GREEN)
 arrow(a,x+2.65,6.1,x+2.65,5.7);arrow(a,x+2.65,4.55,x+2.65,4.15)
box(a,.2,1.55,3.7,.9,'Root transform\n857 driver rows [857,8]',BLUE)
box(a,7.6,1.55,4.1,.9,'Mean neighbor transform\n18,389 eligible result rows',BLUE)
arrow(a,2.85,3,2.05,2.55);arrow(a,9,3,9.65,2.55)
box(a,4.3,1.55,2.8,.9,'Add → ReLU\nSelect 23 seed IDs',GOLD);arrow(a,4,2,4.2,2);arrow(a,7.5,2,7.2,2)
box(a,4.3,.15,2.8,.8,'Scalar head → L1 loss\n23 training labels',GOLD);arrow(a,5.7,1.45,5.7,1.05)
a.text(.2,.12,'Fit cutoff: 2004-09-03\nEvent edges ≤ the same cutoff\nStatic creation history unavailable',fontsize=10,color=INK)
a.text(8,.12,'One course gradient step\nSeparate table parameters\nNo benchmark score claim',fontsize=10,color=INK)
a.text(.2,7.65,'Solid arrows: forward. Autograd reverses this path. Frozen materialization has no gradient.',fontsize=11,color=INK);save(f,'architecture')
f,a=canvas('An embedding without its row ID is an unsafe graph feature',5)
colors=[GREEN,GOLD,BLUE]
for i,(key,value) in enumerate([(40,'[1,2]'),(7,'[3,4]'),(90,'[5,6]')]):
 box(a,.3,3.4-i,3,.65,f'entity {key} → {value}',colors[i])
for j,(key,value,src) in enumerate([(90,'[5,6]',2),(40,'[1,2]',0),(90,'[5,6]',2)]):
 box(a,8.3,3.4-j,3.2,.65,f'query {key} → {value}',colors[src]);arrow(a,3.4,3.725-src,8.2,3.725-j)
box(a,4.2,3.6,3.1,.65,'Gather positions [2,0,2]',GOLD)
a.text(.3,4.55,'ENCODED ENTITY ROWS',weight='bold',color=INK);a.text(8.3,4.55,'REQUESTED QUERY ORDER',weight='bold',color=INK)
a.text(.3,.35,'Duplicate source IDs are an error. Repeated query IDs are valid.\nBackward: entity 90 receives two contributions; entity 7 receives none.',color=INK,fontsize=12);save(f,'identity')
