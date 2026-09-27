"""Computation-first taxonomy, classification, metrics, time and result figures."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;OUT=P/'figures/l128';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'l128','font.size':12})
INK='#19384a';BLUE='#dfeaf2';GOLD='#f5e9c7';GREEN='#dceee4'
def save(f,name):
 f.tight_layout();f.savefig(OUT/(name+'.svg'),metadata={'Date':None});f.savefig(OUT/(name+'.png'),dpi=150,metadata={'Software':'L128'});plt.close(f)
def setup(title,h):
 f,a=plt.subplots(figsize=(10,h));a.axis('off');a.set(xlim=(0,10),ylim=(0,h));f.patch.set_facecolor('#faf9f6');a.set_title(title,loc='left',fontsize=17,weight='bold',color=INK,pad=15);return f,a
def box(a,x,y,w,h,text,color=BLUE):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04',fc=color,ec=INK));a.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=12,color=INK)
def arrow(a,x,y,xx,yy):a.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','lw':2,'color':INK})
f,a=setup('Same representation space; three different answers',7)
for y,title,op,out in [(5.35,'Binary query','h: 128 values → w·h + b','z=0 → sigmoid(z)=0.5\nOne probability per query'),(3.2,'Regression query','h: 128 values → w·h + b','Predictions [3,8], truth [5,7]\nMAE = (2+1)/2 = 1.5'),(1.05,'Recommendation query','Customer [1,2] · article vectors\nA=[2,0], B=[0,2], C=[1,1]','Scores A=2, B=4, C=3\nReturn IDs [B,C,A]')]:
 box(a,.15,y,4.1,1.25,title+'\n'+op);box(a,5,y,4.7,1.25,out,GREEN);arrow(a,4.35,y+.6,4.9,y+.6)
a.text(.15,.3,'Illustrative numbers. Real H&M evaluation requests 12 ranked article IDs per query.',fontsize=11,color=INK);save(f,'heads')
f,a=setup('F1 classification: from query to logit to selected checkpoint',9)
items=[('B (driver ID, cutoff) queries','Historical binary targets kept separate',GOLD),('Sample temporal FK / reverse edges','Two hops [128,64]; Nτ rows per table',BLUE),('Separate four-block Frame ResNet per table','Nτ × typed fields → Nτ × 128; add relative-time vectors',BLUE),('Two heterogeneous sum-GraphSAGE layers','Neighbor sum → relation sum → node norm → ReLU',BLUE),('Read first B driver vectors → linear head','B × 128 → B × 1 logits',BLUE)]
for i,(h,t,c) in enumerate(items):
 y=7.6-i*1.3;box(a,.6,y,8.8,.85,h+'\n'+t,c)
 if i<4:arrow(a,5,y-.06,5,y-.4)
box(a,.3,.55,4.35,1.1,'TRAIN: logits + B binary labels\nBCEWithLogits → Adam\nUpdate all trainable modules',GOLD)
box(a,5.2,.55,4.5,1.1,'EVALUATE: sigmoid probabilities\nFirst maximum validation AUROC\nFreeze checkpoint → score test',GREEN)
arrow(a,3,2.35,2.5,1.75);arrow(a,7,2.35,7.5,1.75);save(f,'architecture')
f,axs=plt.subplots(1,2,figsize=(10,4.5));f.patch.set_facecolor('#faf9f6')
ax=axs[0];ax.imshow([[1,.5],[1,1]],vmin=0,vmax=1,cmap='Greens')
for i,row in enumerate([[1,.5],[1,1]]):
 for j,v in enumerate(row):ax.text(j,i,str(v),ha='center',va='center',fontsize=22)
ax.set(xticks=[0,1],xticklabels=['negative .1','negative .4'],yticks=[0,1],yticklabels=['positive .4','positive .8'],title='AUROC: four pair credits\n(1 + ½ + 1 + 1)/4 = .875')
ax=axs[1];ax.bar([1,2,3],[1,0,2/3],color=['#39846b','#dddddd','#39846b']);ax.set(xticks=[1,2,3],xticklabels=['#1: item 2 ✓','#2: item 7 ×','#3: item 4 ✓'],ylim=(0,1.25),ylabel='Precision contribution at a hit',title='AP@3: relevant set {2,4}\n(1 + 0 + ⅔)/2 = ⅚');ax.text(1,1.08,'1/1',ha='center');ax.text(3,.75,'2/3',ha='center');save(f,'metrics')
f,a=setup('A future target and an autoregressive decoder differ',6.5)
box(a,.2,4.65,3.6,1,'Observed history through Monday\nReal past purchases',BLUE);box(a,5,4.65,4.7,1,'One future-week binary prediction\np(any purchase | history)',GREEN);arrow(a,3.9,5.15,4.9,5.15)
a.text(.2,4.1,'TEMPORAL TASK: one answer about a future window.',fontsize=12,color=INK)
box(a,.2,2.6,2.6,.95,'Monday history',BLUE);box(a,3.5,2.6,2.7,.95,'Generate Tuesday\ny₁',GOLD);box(a,7,2.6,2.7,.95,'Generate Wednesday\ny₂ given y₁',GREEN);arrow(a,2.9,3.05,3.4,3.05);arrow(a,6.3,3.05,6.9,3.05)
a.text(.2,1.7,'AUTOREGRESSIVE DECODER: the generated first output becomes context.',fontsize=12,color=INK)
a.text(.2,.55,'Training may use the true y₁ (teacher forcing). Inference uses generated y₁.\nChanging that input can change the next prediction. No such decoder is run here.',fontsize=12,color=INK);save(f,'time')
p=P/'evidence/l128/summary.json'
if p.exists():
 s=json.loads(p.read_text());f,axs=plt.subplots(1,2,figsize=(10,4.5));f.patch.set_facecolor('#faf9f6')
 for ax,split in zip(axs,['val','test']):
  m=s['metrics'][split];ax.scatter(range(5),[100*r[split] for r in s['seeds']],s=60,label='Fresh seed');ax.axhline(100*m['target'],ls='--',color='#a04c3e',label='Paper mean');ax.errorbar(5.4,100*m['mean'],yerr=100*m['sample_sd'],fmt='D',capsize=5,color=INK,label='Mean ± sample SD');ax.set(xticks=[0,1,2,3,4,5.4],xticklabels=['0','1','2','3','4','Mean'],ylabel='AUROC (%)',title=split+' · detail scale');ax.grid(axis='y',alpha=.2)
 axs[0].legend(fontsize=9);f.suptitle('Reconstructed historical labels · seed SD is not a confidence interval',fontsize=12);save(f,'scores')
