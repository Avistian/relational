"""Portable computation figures; arithmetic generated from declared examples/evidence."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures/l067';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':12,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#faf9f6','axes.facecolor':'#faf9f6','savefig.facecolor':'#faf9f6'})
BLUE='#28658a';ORANGE='#b85c38';GREEN='#347260';GRAY='#555b61'
def save(fig,name):fig.savefig(OUT/(name+'-v2.png'),dpi=165,bbox_inches='tight');plt.close(fig)
def box(ax,x,y,w,h,text,color=BLUE):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.015',fc='white',ec=color,lw=1.5));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11,color='#202b33')
def arrow(ax,a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color=GRAY,lw=1.5))
def architecture():
 fig,ax=plt.subplots(figsize=(12,9));ax.set(xlim=(0,12),ylim=(-.8,9));ax.axis('off');ax.text(0,8.8,'Trace one query through the complete original checkpoint',fontsize=18,weight='bold')
 box(ax,.2,7.35,3.2,.95,'Outer train memory N × F\ntrain scaler + clip ±10');box(ax,4.3,7.35,3.2,.95,'Query q\nsame outer transform',ORANGE);box(ax,8.4,7.35,3.3,.95,'Exact k nearest rows\ncontext C=k; query Q=1')
 arrow(ax,(7.5,7.8),(8.35,7.8));ax.plot([1.8,1.8,10],[8.35,8.58,8.58],c=GRAY,lw=1.5);arrow(ax,(10,8.58),(10,8.35))
 box(ax,1.6,5.7,8.8,1.0,'Pad to 100 → contiguous T × B × 100 → context masked moments\nnormalize → clamp ±100 → divide(F/100) → Linear 100→512')
 arrow(ax,(10,7.3),(9,6.75));box(ax,.2,4.2,4.2,1.0,'Context: feature + Linear(y)\nB × C × 512');box(ax,6.7,4.2,4.8,1.0,'Query: feature only\nB × Q × 512',ORANGE)
 arrow(ax,(4,5.65),(2.5,5.25));arrow(ax,(8,5.65),(9,5.25))
 box(ax,1.0,1.85,10,1.85,'Repeat 12 complete postnorm blocks\nQ: all C+Q rows   ·   K,V: C context rows only\n4 heads × 128: softmax(QKᵀ/√128)V → output projection\nLN(h+attention) → LN(h+Linear 1024→512(GELU(Linear 512→1024(h))))')
 arrow(ax,(2.4,4.15),(3,3.75));arrow(ax,(9,4.15),(9,3.75))
 box(ax,1.4,.1,9.2,1.0,'Keep Q query rows → Linear 512→1024 → GELU → Linear 1024→10\nfirst K training classes → softmax temperature 1 → B × Q × K',GREEN);arrow(ax,(6,1.8),(6,1.15))
 ax.text(.2,-.47,'Frozen inference: weights θ stay fixed.  Adaptation: query cross entropy updates all encoders, blocks and head.',fontsize=11,color=GREEN)
 save(fig,'architecture')
def normalization():
 fig,axes=plt.subplots(1,3,figsize=(12,4.3));context=np.array([0.,2.,4.]);query=3.;mean=context.mean();std=context.std(ddof=1)
 vals=[[*context,query],list((np.r_[context,query]-mean)/(std+1e-6)),list((np.r_[context,query]-mean)/(std+1e-6)/(4/100))]
 for ax,v,title in zip(axes,vals,['1. Local raw coordinate','2. Context mean 2, sample std 2','3. F=4: divide by 4/100']):
  ax.bar(range(4),v,color=[BLUE]*3+[ORANGE]);ax.axhline(0,c=GRAY,lw=.8);ax.set_xticks(range(4),['c₁','c₂','c₃','query']);ax.set_title(title,fontsize=12)
  for i,z in enumerate(v):ax.annotate(f'{z:.2f}',(i,z),xytext=(0,5 if z>=0 else -17),textcoords='offset points',ha='center')
  ax.margins(y=.25)
 fig.suptitle('The query is transformed by context statistics; it never fits them',fontsize=16);fig.tight_layout();save(fig,'normalization')
def episodes():
 fig,axes=plt.subplots(3,1,figsize=(12,6.7),sharex=True)
 labels=[-3,-2,-1,0,1,2,3]
 for ax in axes:ax.set_xlim(-3.8,3.8);ax.set_ylim(-.6,.6);ax.set_yticks([]);ax.spines[['left','bottom']].set_visible(False);ax.axhline(0,color='#ccc',zorder=0)
 axes[0].scatter(labels,[0]*7,s=260,c=[BLUE]*3+[GRAY]+[BLUE]*3);axes[0].set_title('1. Anchor 0 retrieves six eligible neighbors; exclude its original ID',loc='left')
 for x in labels:axes[0].text(x,.22,str(x),ha='center')
 axes[1].scatter([-3,-1,2,3],[0]*4,s=260,c=BLUE,label='shared context');axes[1].scatter([-2,1],[0]*2,s=260,c=ORANGE,marker='s',label='queries');axes[1].set_title('2. Shuffle once, then split C=4 and Q=2; both queries read one context',loc='left');axes[1].legend(loc='lower right',ncol=2)
 axes[2].scatter([-3,-1,0,1],[0]*4,s=260,c=GREEN,label='exact neighbors of −2');axes[2].scatter([-2],[0],s=260,c=ORANGE,marker='s');axes[2].set_title('3. Exact query −2 neighborhood differs: 0 and 1 replace 2 and 3',loc='left');axes[2].legend(loc='lower right');axes[2].set_xticks(labels);axes[2].tick_params(labelbottom=True)
 fig.tight_layout();save(fig,'episodes')
def cost():
 k=100;r=32;b=2;exact=r*k*(k+1);shared=b*k*(k+r//b)
 fig,ax=plt.subplots(figsize=(10,4));ax.barh(['Exact: 32 contexts, 1 query each','Shared: 2 contexts, 16 queries each'],[exact,shared],color=[BLUE,ORANGE]);ax.invert_yaxis();ax.set_xlabel('Permitted attention pairs per head per layer');ax.set_xlim(0,exact*1.3)
 for i,v in enumerate([exact,shared]):ax.text(v+exact*.02,i,f'{v:,}',va='center')
 ax.set_title(f'R=32, k=100: context sharing reduces permitted pairs by {exact/shared:.2f}×',fontsize=14);fig.tight_layout();save(fig,'cost')
def selection():
 fig,axes=plt.subplots(1,2,figsize=(11,4.3));steps=[0,30]
 for ax,v,title in zip(axes,[[.82,.75],[.82,.85]],['Reject a harmful update','Retain an improved update']):
  ax.plot(steps,v,'o-',c=BLUE,ms=10);best=int(np.argmax(v));ax.scatter([steps[best]],[v[best]],s=260,facecolors='none',edgecolors=ORANGE,lw=2.5);ax.set_ylim(.7,.9);ax.set_xticks(steps);ax.set_xlabel('Gradient steps');ax.set_ylabel('Validation AUC');ax.set_title(title)
  ax.text(.5,.08,f'Selected step {steps[best]}',transform=ax.transAxes,ha='center')
 fig.suptitle('Illustration: the test labels are absent from checkpoint selection',fontsize=15);fig.tight_layout();save(fig,'selection')
def results():
 p=ROOT/'_analysis_l067_v2_results.json'
 if not p.exists():return
 a=json.loads(p.read_text());names=['global_all','random_k','local_frozen','local_ft_paper_lr','local_ft_release_lr'];labels=['Global all','Random k','Local frozen','Local FT .01','Local FT 1e−5']
 fig,axes=plt.subplots(1,3,figsize=(13,5),sharey=True)
 for ax,d in zip(axes,a['datasets']):
  rows=[next(r for r in a['summary'] if r['dataset']==d and r['arm']==n) for n in names]
  ax.errorbar(range(5),[r['auc_mean'] for r in rows],yerr=[r['auc_sd'] for r in rows],fmt='o',capsize=4,color=BLUE);ax.set_xticks(range(5),labels,rotation=40,ha='right');ax.set_title(d.replace('_',' '));ax.set_ylim(.45,1.02);ax.grid(axis='y',alpha=.2)
 axes[0].set_ylabel('Test AUC, mean ± sample SD over 3 split seeds');fig.suptitle('Fresh full-checkpoint CPU panel; error bars do not measure dataset uncertainty',fontsize=15);fig.tight_layout();save(fig,'results')
def ranks():
 p=ROOT/'_analysis_l067_v2_results.json'
 if not p.exists():return
 a=json.loads(p.read_text());pairs=list(a['mean_ranks'].items());fig,ax=plt.subplots(figsize=(10,4.3))
 ax.scatter([v for n,v in pairs],range(len(pairs)),s=100,c=BLUE);ax.set_yticks(range(len(pairs)),[n.replace('_',' ') for n,v in pairs]);ax.invert_yaxis();ax.set_xlim(.7,5.3);ax.set_xticks(range(1,6));ax.set_xlabel('Mean rank of dataset mean AUC (1 is best)');ax.grid(axis='x',alpha=.2)
 for i,(n,v) in enumerate(pairs):ax.annotate(f'{v:.3f}',(v,i),xytext=(10,0),textcoords='offset points',va='center')
 cd=a['nemenyi_cd'];ax.plot([1,1+cd],[-.65,-.65],color=ORANGE,lw=2);ax.plot([1,1],[-.76,-.54],color=ORANGE);ax.plot([1+cd,1+cd],[-.76,-.54],color=ORANGE);ax.text(1+cd/2,-.88,'5% Nemenyi critical difference',ha='center',fontsize=10,color=ORANGE);ax.set_ylim(4.55,-1.1);ax.set_title(f"Three datasets, five arms: Friedman p={a['friedman']['pvalue']:.4f}; Nemenyi CD={cd:.3f}",fontsize=13)
 ax.text(.02,-.29,'CD is a rank-distance threshold, not a confidence interval.\nNo general superiority or equivalence follows from this tiny panel.',transform=ax.transAxes,fontsize=11)
 fig.tight_layout();save(fig,'ranks')
if __name__=='__main__':
 for f in [architecture,normalization,episodes,cost,selection,results,ranks]:f()
