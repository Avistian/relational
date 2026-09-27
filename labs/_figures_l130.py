"""Model-specific computation, real query shapes, decision boundary and measured scores."""
import json,collections
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent;OUT=P/'figures/l130';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'l130','font.size':11})
INK='#19384a';GREEN='#dceee4';GOLD='#f5e9c7';BLUE='#dfeaf2'
def save(f,name):
 f.tight_layout();f.savefig(OUT/(name+'.svg'),metadata={'Date':None});f.savefig(OUT/(name+'.png'),dpi=150,metadata={'Software':'L130'});plt.close(f)
def setup(title,height=7):
 f,a=plt.subplots(figsize=(12,height));a.axis('off');a.set(xlim=(0,12),ylim=(0,height));f.patch.set_facecolor('#faf9f6');a.set_title(title,loc='left',fontsize=18,weight='bold',color=INK,pad=15);return f,a
def box(a,x,y,w,h,text,color=GREEN):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',fc=color,ec=INK));a.text(x+w/2,y+h/2,text,ha='center',va='center',color=INK,fontsize=11)
def arrow(a,x,y,xx,yy):a.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':INK,'lw':2})
f,a=setup('RelBench F1: one cutoff follows every message',8)
box(a,.2,6.2,2.5,1.2,'B task queries\n(driver ID, cutoff)\nTarget kept separate',GOLD)
box(a,3.2,6.2,3.5,1.2,'Temporal disjoint sampling\n9 table types + FK/reverse edges\nUniform fanouts [128, 64]',BLUE)
box(a,7.4,6.2,4.2,1.2,'Per-table TensorFrames\nNumerical · categorical · time · text\nNτ sampled rows, typed columns',BLUE)
arrow(a,2.8,6.8,3.1,6.8);arrow(a,6.8,6.8,7.3,6.8)
box(a,7.4,4.15,4.2,1.25,'Separate Frame ResNet per table\n4 blocks → Hτ: Nτ × 128\n+ encoding of (cutoff − row time)',BLUE);arrow(a,9.5,6.1,9.5,5.5)
box(a,2.7,4.15,3.95,1.25,'2 heterogeneous GraphSAGE layers\nSum neighbors within each relation\nSum relations → norm → ReLU',BLUE);arrow(a,7.3,4.8,6.75,4.8)
box(a,.2,2.05,3.2,1.25,'Read first B driver vectors\nB × 128 → scalar head\nB × 1 raw predictions',BLUE);arrow(a,3.9,4.05,2.0,3.4)
box(a,4.1,2.05,3.1,1.25,'Training: mean L1 loss\nB future target values\nAdam updates all modules',GOLD);arrow(a,3.5,2.7,4,2.7)
box(a,8,2.05,3.6,1.25,'Evaluation: clip predictions\nTraining-label q₂ / q₉₈\nValidation picks checkpoint',GREEN);a.plot([1.8,1.8,9.8],[1.95,1.4,1.4],color=INK,lw=2);arrow(a,9.8,1.4,9.8,1.97)
a.text(.2,.75,'Shape distinction: Nτ includes context copies; B counts supervised queries.\nNode date ≤ its own root cutoff. Missing ingestion/static creation histories remain unobserved.',color=INK,fontsize=12)
save(f,'architecture')
example=json.loads((P/'evidence/l124/example-subgraph.json').read_text());counts=collections.Counter(n[0] for n in example['input_nodes'])
f,a=setup('One real task row: graph size is not prediction size',6)
q=example['task_row'];box(a,.2,4.35,3.1,1.15,f"Driver {q['driverId']} · 2004-07-05\nFuture mean target: {q['position']}\nLabel horizon ends 2004-09-03",GOLD)
box(a,4,4.35,3.4,1.15,'Exhaustive two-hop context\n99 node occurrences\n245 directed edges',BLUE);arrow(a,3.4,4.9,3.9,4.9)
box(a,8.1,4.35,3.5,1.15,'Readout\n1 × 128 seed vector\n1 × 1 prediction',GREEN);arrow(a,7.5,4.9,8,4.9)
for i,(name,n) in enumerate(sorted(counts.items())):
 x=.25+(i%4)*2.95;y=2.7-(i//4)*1.25;box(a,x,y,2.65,.85,f'{name}: {n} '+('row' if n==1 else 'rows')+f'\nEncoded shape {n} × 128',BLUE)
a.text(.2,.25,'Structural trace inherited and verified from L124. Training uses bounded sampled neighborhoods.',color=INK,fontsize=11);save(f,'trace')
f,a=setup('Separate selection from the final comparison',5.5)
box(a,.2,3.65,3.25,1.05,'TRAIN × 10 epochs\n7,453 queries every epoch\nStore validation trace',BLUE)
box(a,4.25,3.65,3.25,1.05,'SELECT\nFirst minimum validation MAE\nKeep checkpoint hash',GOLD)
box(a,8.3,3.65,3.25,1.05,'SCORE\nFrozen checkpoint on 760 test rows\nSave IDs, times, labels, predictions',GREEN)
arrow(a,3.55,4.2,4.15,4.2);arrow(a,7.6,4.2,8.2,4.2)
box(a,1.3,1.25,4.1,1.2,'REPEAT with five fresh initializations\nExact seed set + unique run identities\nMissing run → incomplete experiment',BLUE)
box(a,6.6,1.25,4.6,1.2,'AGGREGATE\nMean and sample SD (divide by n − 1)\nCompare to named paper row + deviations',GREEN)
arrow(a,9.9,3.5,8.9,2.55);arrow(a,5.5,1.85,6.5,1.85)
a.text(.2,.35,'Test MAE has no arrow back into SELECT. A close mean is descriptive, not historical identity.',color=INK,fontsize=12);save(f,'selection')
s=json.loads((P/'evidence/l130/summary.json').read_text());f,axs=plt.subplots(1,2,figsize=(11,4.5));f.patch.set_facecolor('#faf9f6')
for ax,split,title in zip(axs,['val','test'],['Validation','Test']):
 m=s['metrics'][split];ax.scatter(range(5),[r[split] for r in s['seeds']],s=60,color='#176b79',label='Fresh seed');ax.axhline(m['target'],color='#a04c3e',ls='--',label='Paper mean');ax.errorbar(5.4,m['mean'],yerr=m['sample_sd'],fmt='D',color=INK,capsize=6,label='Mean ± sample SD');ax.set(xticks=list(range(5))+[5.4],xticklabels=['0','1','2','3','4','Mean'],ylabel='MAE (position units)',title=title+' · detail scale');ax.grid(axis='y',alpha=.2)
axs[0].legend(fontsize=9);f.suptitle('Five fresh complete fits · seed SD is not a confidence interval',fontsize=13);save(f,'scores')

f,a=setup('Identity first: follow one question through the join',5.8)
box(a,.2,3.8,3.2,1.15,'TASK ROWS\n(7,10) → target 2\n(7,20) → target 5',BLUE)
box(a,4.3,3.8,3.2,1.15,'RECEIVED PREDICTIONS\n(7,20) → prediction 4\n(7,10) → prediction 2',GOLD)
box(a,8.3,3.8,3.2,1.15,'JOIN ON BOTH KEYS\n(7,10): |2 − 2| = 0\n(7,20): |4 − 5| = 1',GREEN)
arrow(a,3.5,4.35,4.2,4.35);arrow(a,7.6,4.35,8.2,4.35)
box(a,.6,1.3,4.8,1.2,'Position alone\n(|4 − 2| + |2 − 5|) / 2 = 2.5\nWrong questions paired',GOLD)
box(a,6.3,1.3,5.1,1.2,'Exact key alignment\n(0 + 1) / 2 = 0.5\nMissing or duplicate key → reject',GREEN)
a.text(.2,.35,'Synthetic fixture. Only prediction order changed; correct MAE must remain invariant.',color=INK,fontsize=12)
save(f,'alignment')
