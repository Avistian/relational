"""Portable computational figures for historical v2; arithmetic is generated."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures/l064';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':12,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'#faf9f5'})
BLUE='#256b97';GREEN='#247052';ORANGE='#aa5722';INK='#24353c'
def canvas(height=5):
 fig,ax=plt.subplots(figsize=(10,height),facecolor='#faf9f5');ax.set(xlim=(0,10),ylim=(0,height));ax.axis('off');return fig,ax
def box(ax,x,y,w,h,text,color=BLUE,size=12):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor='white',edgecolor=color,lw=1.5));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color=INK,linespacing=1.5)
def arrow(ax,a,b,color=INK):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color=color,lw=1.4))
def save(fig,name):fig.savefig(OUT/(name+'-v2.png'),dpi=160,bbox_inches='tight');plt.close(fig)

def architecture():
 fig,ax=canvas(13)
 ax.text(.2,12.65,'Trace a query feature into its target prediction',fontsize=17,weight='bold')
 ax.text(.2,12.22,'Actual classifier: 12 blocks · D=192 · 6 heads × 32 · two features per group',fontsize=11)
 box(ax,.25,10.75,4.45,1.1,'Numeric x: B × (C+Q) × F\ncontext-fitted wrapper\nG = ceil(F/2) feature groups')
 box(ax,5.3,10.75,4.4,1.1,'Observed y: B × C\nappend Q unknown targets\ntrue query labels never enter',GREEN)
 arrow(ax,(2.5,10.72),(2.5,10.25));arrow(ax,(7.5,10.72),(7.5,10.25))
 box(ax,.25,8.85,4.45,1.25,'Pairs → impute → context z-score\nactive-group scaling + 2 flags\n4 → 192 linear; add 48 → 192 group ID')
 box(ax,5.3,8.85,4.4,1.25,'Mean-impute unknown y → class rank\n(rank, missing flag): 2 → 192\nB × (C+Q) × 192',GREEN)
 arrow(ax,(2.5,8.8),(5,8.35));arrow(ax,(7.5,8.8),(5,8.35))
 box(ax,1.4,7.65,7.2,.58,'Concatenate target token: B × N × (G+1) × 192',INK,13)
 arrow(ax,(5,7.58),(5,7.17))
 ax.add_patch(FancyBboxPatch((.2,2.7),9.5,4.35,boxstyle='round,pad=.1',facecolor='#eaf0f2',edgecolor=BLUE,lw=2))
 ax.text(.45,6.74,'REPEAT ×12 · distinct learned weights in every block',fontsize=13,weight='bold')
 box(ax,.55,5.77,8.8,.64,'1  Within each row: full feature + target attention → add input → norm')
 arrow(ax,(5,5.68),(5,5.48))
 ax.plot([2.65,7.25],[5.48,5.48],color=INK,lw=1.4)
 arrow(ax,(2.65,5.48),(2.65,5.22));arrow(ax,(7.25,5.48),(7.25,5.22))
 box(ax,.55,3.82,4.2,1.35,'2a  Context receivers\nat each token position\nC queries read C context senders\n6 Q heads; 6 K/V heads',BLUE,11)
 box(ax,5.13,3.82,4.2,1.35,'2b  Query receivers\nat each token position\nQ queries read C context senders\n6 Q heads; FIRST K/V head reused',GREEN,11)
 arrow(ax,(2.65,3.72),(2.65,3.47));arrow(ax,(7.25,3.72),(7.25,3.47))
 ax.plot([2.65,7.25],[3.47,3.47],color=INK,lw=1.4)
 arrow(ax,(5,3.47),(5,3.36))
 ax.text(5,3.57,'Both routes: add input → norm',ha='center',fontsize=9)
 box(ax,.55,2.91,8.8,.42,'3  Per token: 192 → GELU(768) → 192 → add input → norm',INK,11)
 arrow(ax,(5,2.57),(5,2.19))
 box(ax,1,1.35,8,.72,'Read query TARGET states: B × Q × 192\nhead 192 → GELU(768) → 10; retain K active classes',GREEN)
 arrow(ax,(5,1.25),(5,.91))
 ax.text(5,.61,'logits / 0.9 → softmax → B × Q × K probabilities',ha='center',fontsize=13,color=GREEN)
 ax.text(.2,.1,'Pretraining only: hidden synthetic targets → cross-entropy → update shared weights.\nThis lab: original frozen weights; no optimizer. Uncached explicit CPU computation.',fontsize=10)
 save(fig,'architecture')

def encoding():
 fig,ax=canvas(5.2);ax.text(.1,4.85,'One coherent two-feature group: values and flags stay distinct',fontsize=15,weight='bold')
 columns=[('Input group',[[1,10],[3,'NaN'],[5,14],[7,12]]),('After imputation',[[1,10],[3,12],[5,14],[7,12]]),('Normalized values',[[-1,-1],[0,0],[1,1],[2,0]]),('Flags from raw input',[[0,0],[0,-2],[0,0],[0,0]])]
 for c,(title,rows) in enumerate(columns):
  x=.25+c*2.48;ax.text(x+1,4.23,title,ha='center',fontsize=12,color=BLUE)
  for r,row in enumerate(rows):box(ax,x,3.36-r*.66,2,.46,str(row),GREEN if r==3 else BLUE,13)
  if c<2:arrow(ax,(x+2.05,3.58),(x+2.42,3.58))
 ax.text(.25,3.98,'context',fontsize=9);ax.text(.25,1.08,'query',fontsize=9,color=GREEN)
 ax.text(.3,.62,'Context mean (3,12), sample std (2,2); both values vary → sqrt(2/2)=1.',fontsize=12)
 ax.text(.3,.15,'Encoder sees [z₀,z₁,flag₀,flag₁]. Missing row: [0,0,0,−2]; query: [2,0,0,0].',fontsize=12)
 save(fig,'encoding')

def attention():
 fig,ax=canvas(5.4);ax.text(.1,5,'First-head K/V reuse changes the learned computation',fontsize=16,weight='bold')
 ax.text(.1,4.55,'Synthetic two-head trace · queries retain different scores · scalar values',fontsize=11)
 a=1/(1+np.exp(-1));w=np.array([[1-a,a],[a,1-a]])
 rows=[['head 0','(0, 1)',f'({1-a:.3f}, {a:.3f})','(1, 3)',f'{w[0]@[1,3]:.6f}',f'{w[0]@[1,3]:.6f}'],['head 1','(1, 0)',f'({a:.3f}, {1-a:.3f})','(10, 30)',f'{w[1]@[10,30]:.6f}',f'{w[1]@[1,3]:.6f}']]
 table=ax.table(cellText=rows,colLabels=['Q head','Scaled scores','Weights','Own-head V','Ordinary MHA','Reused-head V₀'],bbox=[.005,.44,.99,.3],cellLoc='center');table.auto_set_font_size(False);table.set_fontsize(10)
 ax.text(.2,1.74,'Ordinary context route: head 1 uses values (10,30).',fontsize=12)
 ax.text(.2,1.2,'Query route: head 1 keeps its query, but uses head 0 keys AND values.',fontsize=12,color=GREEN)
 ax.text(.2,.59,'The table isolates the value difference with specified scores. In the real layer,\nreplacing keys changes the scores too. Reuse does not mean identical Q heads.',fontsize=11)
 save(fig,'attention')

def cost():
 fig,axes=plt.subplots(1,2,figsize=(10,4.3));c=np.array([40,80,160]);q=c//4;g=11
 axes[0].plot(c,(c+q)*g*g,'o-',label='feature pairs');axes[0].plot(c,g*(c+q)*c,'s-',label='row pairs');axes[0].set(xlabel='Context rows C (Q=C/4)',ylabel='Score pairs / head / layer',title='Rows grow: G+1=11 fixed');axes[0].legend()
 groups=np.array([5,10,20]);axes[1].plot(groups,100*(groups+1)**2,'o-',label='feature pairs');axes[1].plot(groups,(groups+1)*100*80,'s-',label='row pairs');axes[1].set(xlabel='Feature groups G (two features each)',title='Groups grow: C=80, Q=20 fixed');axes[1].legend()
 fig.suptitle('Count actual receiver–sender pairs; these are not peak-memory measurements',fontsize=12);fig.tight_layout();save(fig,'cost')

def boundary():
 fig,ax=canvas(5.6);ax.text(.15,5.23,'Query coupling enters before the attention mask',fontsize=16,weight='bold')
 box(ax,.2,3.3,2.75,1.2,'Context: one varying feature;\nother observed values all 1\nwith one NaN retained',BLUE,11)
 box(ax,3.6,3.3,2.75,1.2,'Impute NaN → 1\noriginal query (0.3,1)\nsecond channel stays\nconstant',BLUE,11)
 box(ax,7,3.3,2.75,1.2,'Used count u=1\nvalue factor sqrt(2)\nexisting encoded rows',GREEN,11)
 arrow(ax,(3,3.9),(3.5,3.9));arrow(ax,(6.4,3.9),(6.9,3.9))
 box(ax,.2,1.1,2.75,1.2,'Append query (0,2)\nno query label supplied\nsame context + weights',ORANGE,11)
 box(ax,3.6,1.1,2.75,1.2,'Second channel now varies\nall-row active count u=2\nvalue factor becomes 1',ORANGE,11)
 box(ax,7,1.1,2.75,1.2,'Earlier encodings change\nthen SAME legal attention\nfirst probability can change',ORANGE,11)
 arrow(ax,(3,1.7),(3.5,1.7));arrow(ax,(6.4,1.7),(6.9,1.7));arrow(ax,(8.4,3.15),(8.4,2.45),ORANGE)
 ax.text(.25,.36,'Control: replace context NaN with 1 → wrapper drops constant column → coupling disappears.',fontsize=11)
 save(fig,'boundary')

def results():
 r=json.loads((ROOT/'_verify_l064_v2_results.json').read_text());fig,axes=plt.subplots(1,3,figsize=(10,4.3),sharey=True)
 for ax,name in zip(axes,['diabetes','blood_transfusion','wdbc']):
  for seed in [0,1,2]:
   values=[next(a['log_loss'] for a in r['records'] if a['dataset']==name and a['seed']==seed and a['condition']==condition) for condition in ['observed','shuffled']]
   ax.plot([0,1],values,'o-',label=f'split {seed}')
  ax.set(xticks=[0,1],xticklabels=['Observed','Shuffled'],title=name.replace('_',' '),ylim=(0,.75));ax.grid(axis='y',alpha=.2)
 axes[0].set_ylabel('Held-out log loss (lower is better)');axes[-1].legend(fontsize=9)
 fig.suptitle('Fresh author reference · frozen weights · same features and test rows per paired split',fontsize=12);fig.tight_layout();save(fig,'results')
if __name__=='__main__':architecture();encoding();attention();cost();boundary();results()
